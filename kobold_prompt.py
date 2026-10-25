"""Generate image prompts with a local KoboldCpp server, then review/edit them before they're used.

KoboldCpp's API is stateless: the Memory and chat history in its Kobold Lite page are never
seen by API calls, so the node sends its context block as the system message on every request.
"""

import json
import logging
import hashlib
import math
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request

from aiohttp import web

import comfy.model_management
import folder_paths
from server import PromptServer

DEFAULT_URL = "http://localhost:5001"
CONTEXT_DIR = os.path.join(folder_paths.get_user_directory(), "prompt_contexts")
ROUTE = "/kobold_prompt"


# ---------------------------------------------------------------- KoboldCpp client

def _base(url):
    url = (url or DEFAULT_URL).strip().rstrip("/")
    if not re.match(r"^https?://", url):
        url = "http://" + url
    for suffix in ("/v1", "/api"):
        if url.endswith(suffix):
            url = url[: -len(suffix)]
    return url


def _request(url, payload=None, timeout=10):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def model_status(url):
    """{'ok', 'model', 'context', 'url'} for the KoboldCpp server at url."""
    base = _base(url)
    try:
        model = _request(f"{base}/api/v1/model", timeout=3).get("result") or ""
    except Exception as e:
        return {"ok": False, "running": False, "url": base, "error": str(getattr(e, "reason", e))}
    if model.lower() in ("", "inactive", "koboldcpp/inactive", "none"):  # started without a model (admin mode)
        return {"ok": False, "running": True, "url": base, "error": "no model loaded"}
    context = None
    try:
        context = _request(f"{base}/api/extra/true_max_context_length", timeout=3).get("value")
    except Exception:
        pass
    return {"ok": True, "url": base, "model": model, "context": context}


def chat(url, messages, max_tokens, temperature, seed):
    base = _base(url)
    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed,
        "sampler_seed": seed,  # KoboldCpp's native name; harmless to OpenAI-style parsing
    }
    try:
        resp = _request(f"{base}/v1/chat/completions", payload, timeout=600)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"KoboldCpp returned HTTP {e.code} from {base}: {e.read().decode('utf-8', 'replace')[:300]}")
    except (urllib.error.URLError, OSError) as e:
        raise RuntimeError(f"KoboldCpp isn't running at {base} ({getattr(e, 'reason', e)}). Start KoboldCpp and load a text model, then queue again.")
    try:
        return resp["choices"][0]["message"]["content"] or "", resp.get("model") or ""
    except (KeyError, IndexError, TypeError):
        raise RuntimeError(f"Unexpected reply from KoboldCpp: {json.dumps(resp)[:300]}")


# ---------------------------------------------------------------- reply cleanup

_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_FENCE = re.compile(r"```[^\n`]*\n?(.*?)```", re.S)
# "**Prompt:**", "### Final Prompt", "Image prompt -" ... at the very start of the reply
_LABEL = re.compile(r"^\s*(?:#+\s*)?[*_]{0,2}\s*(?:final |image |positive |generated |your )?prompt\b"
                    r"(?:[ \t]*[*_]{0,2}[ \t]*[:\-–—][ \t]*[*_]{0,2}[ \t]*\n?|[ \t]*[*_]{0,2}[ \t]*\n|[*_]{1,2}[ \t]*)", re.I)
_QUOTES = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"})


def clean_reply(text):
    """Keep only the prompt: drop <think> blocks, unwrap a code block, drop 'Here's your prompt:' / '**Prompt:**' lead-ins."""
    text = _THINK.sub("", text)
    low = text.lower()
    if "</think>" in low:  # chat template opened the <think> block itself
        text = text[low.rindex("</think>") + len("</think>"):]
    elif "<think>" in low:  # reasoning was cut off by max_tokens; there is no answer after it
        text = text[: low.index("<think>")]
    blocks = [b.strip() for b in _FENCE.findall(text) if b.strip()]
    if blocks:
        text = max(blocks, key=len)
    text = text.strip()
    lines = text.split("\n")
    if len(lines) > 1 and lines[0].rstrip(" *_").endswith(":") and len(lines[0]) < 120 and not _LABEL.match(lines[0] + "\n"):
        text = "\n".join(lines[1:]).strip()
    text = _LABEL.sub("", text, count=1).strip()
    text = text.replace("**", "")  # markdown bold means nothing to the text encoder
    if len(text) > 1 and text[0] == text[-1] and text[0] in "\"'" and text.count(text[0]) == 2:
        text = text[1:-1].strip()
    return text


def _norm(s):
    return re.sub(r"\s+", " ", s.translate(_QUOTES)).strip().lower()


def ensure_phrase(text, phrase, position):
    phrase = phrase.strip()
    if not phrase or _norm(phrase) in _norm(text):
        return text
    return f"{phrase} {text}" if position == "start" else f"{text} {phrase}"


# ---------------------------------------------------------------- context presets

def _preset_path(name):
    name = (name or "").strip()
    if name.lower().endswith(".txt"):
        name = name[:-4]
    if not name or not re.fullmatch(r"[\w\- .()]+", name) or name.startswith("."):
        raise ValueError("Use letters, numbers, spaces, - _ . ( ) only")
    return os.path.join(CONTEXT_DIR, name + ".txt")


def list_presets():
    try:
        return sorted((f[:-4] for f in os.listdir(CONTEXT_DIR) if f.lower().endswith(".txt")), key=str.lower)
    except FileNotFoundError:
        return []


# ---------------------------------------------------------------- randomized slots

# (input name, label in the request, default options). One option per line; "text|weight" sets a weight.
SLOTS = [
    ("angle", "camera angle", [
        "Eye level, facing the camera directly or turned slightly|3",
        "Camera slightly above eye level, she looks up into the lens|2",
        "Camera slightly below eye level, looking up at her face|2",
        "Three-quarter view or side profile|1",
    ]),
    ("expression", "expression", [
        "neutral and relaxed", "calm", "smirking or playful", "angry", "quiet half-smile", "looking away thoughtfully",
        "tired or sleepy", "skeptical", "flirtatious", "determined", "laughing|0.5", "crying|0.5",
    ]),
    ("lighting", "lighting", [
        "warm lamp light", "golden hour window light", "night street light", "neon", "backlit", "hard side light",
        "overcast daylight", "harsh midday sun", "cool bluish light",
    ]),
    ("clothing_color", None, ["red", "white", "black", "grey", "navy", "yellow", "burgundy", "denim blue", "cream|0.5"]),
    ("clothing_type", None, [
        "t-shirt", "hoodie", "sweater", "button-up shirt", "leather jacket", "denim jacket", "tank top", "turtleneck",
        "blouse", "workout top",
    ]),
    ("hairstyle", "hairstyle", ["loose and long|4", "ponytail", "low bun", "updo", "windblown", "tucked behind one ear", "sleek"]),
    ("setting", "setting", [
        "city street", "cafe", "kitchen", "park", "beach", "office", "car interior", "rooftop", "subway platform",
        "bedroom", "gym", "rainy window", "forest trail", "bar", "bookstore|0.5",
    ]),
]
SLOT_NAMES = [name for name, _, _ in SLOTS]
COMBO_SLOTS = ("angle", "expression", "lighting", "setting")  # this combination isn't repeated within RECENT_RUNS
RECENT_RUNS = 8
REROLLS = 6
_NODE_DIR = os.path.dirname(os.path.abspath(__file__))
PICKS_FILE = os.path.join(_NODE_DIR, "koboldcpp_recent_picks.json")
CONVERSATION_FILE = os.path.join(_NODE_DIR, "koboldcpp_conversation.json")
_file_lock = threading.Lock()


def _load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default
    except Exception as e:
        logging.warning(f"[KoboldCpp Prompt] ignoring unreadable {os.path.basename(path)}: {e}")
        return default


def _save_json(path, data):
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        os.replace(tmp, path)
    except OSError as e:
        logging.warning(f"[KoboldCpp Prompt] couldn't write {os.path.basename(path)}: {e}")


def parse_options(text):
    """[(option, weight)] from one-option-per-line text; 'text|2' weights an option, weight 0 turns it off."""
    options = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        weight = 1.0
        if "|" in line:
            head, tail = line.rsplit("|", 1)
            try:
                weight, line = float(tail), head.strip()
            except ValueError:
                pass
        if line and weight > 0:
            options.append((line, weight))
    return options


def _signature(slot_options):
    return hashlib.sha1(json.dumps(slot_options, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def pick_slots(seed, slot_texts):
    """Weighted picks per slot from random.Random(seed), avoiding the previous run's values and the
    angle/expression/lighting/setting combinations of the last RECENT_RUNS runs (kept on disk).
    Re-running a seed that's still in the recent list gives back exactly the same picks."""
    slot_options = {name: parse_options(slot_texts.get(name, "")) for name in SLOT_NAMES}
    slot_options = {k: v for k, v in slot_options.items() if v}
    if not slot_options:
        return {}
    sig = _signature(slot_options)
    with _file_lock:
        recent = _load_json(PICKS_FILE, [])
        if not isinstance(recent, list):
            recent = []
        for entry in recent:
            if entry.get("seed") == seed and entry.get("options") == sig:
                return entry["picks"]

        rng = random.Random(seed)
        draw = lambda name: rng.choices([o for o, _ in slot_options[name]], [w for _, w in slot_options[name]])[0]
        last = recent[-1]["picks"] if recent else {}
        combos = {tuple(e["picks"].get(k) for k in COMBO_SLOTS) for e in recent[-RECENT_RUNS:]}

        def roll(name):
            value = draw(name)
            for _ in range(REROLLS):
                if value != last.get(name) or len({o for o, _ in slot_options[name]}) < 2:
                    break
                value = draw(name)
            return value

        picks = {name: roll(name) for name in slot_options}
        combo_slots = [k for k in COMBO_SLOTS if k in picks]
        for _ in range(REROLLS * 2):
            if not combo_slots or tuple(picks.get(k) for k in COMBO_SLOTS) not in combos:
                break
            name = rng.choice(combo_slots)
            picks[name] = roll(name)

        recent.append({"seed": seed, "options": sig, "picks": picks, "time": time.strftime("%Y-%m-%d %H:%M:%S")})
        _save_json(PICKS_FILE, recent[-RECENT_RUNS:])
    return picks


def choices_sentence(picks):
    parts = []
    for name, label, _ in SLOTS:
        if name == "clothing_type":
            continue
        if name == "clothing_color":
            clothing = " ".join(picks[k] for k in ("clothing_color", "clothing_type") if k in picks)
            if clothing:
                parts.append(f"clothing: {clothing}")
        elif name in picks:
            parts.append(f"{label}: {picks[name]}")
    if not parts:
        return ""
    return f"Use exactly these choices: {'; '.join(parts)}. Do not substitute other options."


def picks_line(picks):
    return " | ".join(f"{name}={picks[name]}" for name in SLOT_NAMES if name in picks)


# ---------------------------------------------------------------- review pause

class _Pending:
    def __init__(self, text):
        self.text = text
        self.action = None
        self.event = threading.Event()


_pending = {}  # node_id -> _Pending
_pending_lock = threading.Lock()


def wait_for_review(node_id, text, title):
    item = _Pending(text)
    with _pending_lock:
        _pending[node_id] = item
    try:
        PromptServer.instance.send_sync("kobold_prompt.review", {"node_id": node_id, "text": text, "title": title})
        while not item.event.wait(0.25):
            comfy.model_management.throw_exception_if_processing_interrupted()
    finally:
        with _pending_lock:
            _pending.pop(node_id, None)
        PromptServer.instance.send_sync("kobold_prompt.review_done", {"node_id": node_id})
    if item.action == "cancel":
        raise comfy.model_management.InterruptProcessingException()
    return item.text


# ---------------------------------------------------------------- routes

routes = PromptServer.instance.routes


@routes.get(ROUTE + "/status")
async def _status(request):
    import asyncio
    url = request.rel_url.query.get("url", DEFAULT_URL)
    return web.json_response(await asyncio.get_running_loop().run_in_executor(None, model_status, url))


@routes.get(ROUTE + "/contexts")
async def _contexts(request):
    return web.json_response({"names": list_presets(), "folder": CONTEXT_DIR})


@routes.get(ROUTE + "/contexts/{name}")
async def _context_load(request):
    try:
        with open(_preset_path(request.match_info["name"]), "r", encoding="utf-8") as f:
            return web.json_response({"text": f.read()})
    except (ValueError, OSError) as e:
        return web.json_response({"error": str(e)}, status=404)


@routes.post(ROUTE + "/contexts")
async def _context_save(request):
    body = await request.json()
    try:
        path = _preset_path(body.get("name"))
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=400)
    if os.path.exists(path) and not body.get("overwrite"):
        return web.json_response({"error": "exists"}, status=409)
    os.makedirs(CONTEXT_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(body.get("text") or "")
    return web.json_response({"names": list_presets(), "name": os.path.basename(path)[:-4]})


@routes.get(ROUTE + "/review")
async def _review_pending(request):
    with _pending_lock:
        return web.json_response({"pending": [{"node_id": k, "text": v.text} for k, v in _pending.items()]})


@routes.post(ROUTE + "/review/{node_id}")
async def _review_answer(request):
    body = await request.json()
    with _pending_lock:
        item = _pending.get(request.match_info["node_id"])
    if item is None:
        return web.json_response({"error": "not waiting"}, status=404)
    if body.get("text") is not None:
        item.text = body["text"]
    item.action = body.get("action", "continue")
    item.event.set()
    return web.json_response({"ok": True})


# ---------------------------------------------------------------- nodes

# node_id -> {"context": sha1 of the context, "turns": [[user, assistant], ...]}; kept on disk so it survives restarts
_history = _load_json(CONVERSATION_FILE, {})
if not isinstance(_history, dict):
    _history = {}


def _context_key(context):
    return hashlib.sha1(context.encode("utf-8")).hexdigest()


class KoboldCppPrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "server_url": ("STRING", {"default": DEFAULT_URL, "tooltip": "KoboldCpp address (the one Kobold Lite opens on)."}),
                "context": ("STRING", {"default": "", "multiline": True, "placeholder": "Guidelines sent with every request (system message). Save/load as .txt below.", "tooltip": "Your standing guidelines. Sent as the system message on every request, since KoboldCpp's API doesn't remember Kobold Lite's Memory."}),
                "request": ("STRING", {"default": "", "multiline": True, "placeholder": "What you want this time, e.g. rooftop cafe at dusk, laughing", "tooltip": "This run's request (user message)."}),
                "required_phrase": ("STRING", {"default": "", "multiline": True, "placeholder": "Optional: exact text the prompt must contain", "tooltip": "Inserted word for word if the model leaves it out."}),
                "phrase_position": (["start", "end"], {"default": "start", "tooltip": "Where the required phrase goes when it has to be added."}),
                "max_tokens": ("INT", {"default": 512, "min": 16, "max": 8192, "tooltip": "Longest reply allowed."}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0x7FFFFFFF, "control_after_generate": True, "tooltip": "Same seed + same inputs reuses the last prompt instead of asking again. Set to 'fixed' to keep a prompt while you change image settings."}),
                "clean_output": ("BOOLEAN", {"default": True, "tooltip": "Keep only the prompt: unwrap code blocks, drop 'Here's your prompt:' lines and <think> reasoning."}),
                "keep_conversation": ("BOOLEAN", {"default": False, "tooltip": "Send recent requests/replies too, so follow-ups like 'same scene at dawn' work. Cleared when the context changes."}),
                "history_turns": ("INT", {"default": 3, "min": 1, "max": 20, "tooltip": "How many earlier request/reply pairs to include when keep_conversation is on."}),
            },
            # optional, after the old inputs, so saved workflows keep their widget order
            "optional": {
                "randomize_slots": ("BOOLEAN", {"default": True, "tooltip": "Pick an angle, expression, lighting, clothing, hairstyle and setting from the lists below (using the seed) and tell the model to use exactly those. Avoids repeating the last runs, even after a restart."}),
                **{name: ("STRING", {"default": "\n".join(default), "multiline": True,
                                     "tooltip": f"{name.replace('_', ' ').capitalize()} options, one per line. Add |weight to make one more or less likely, e.g. 'neon|2' or 'neon|0.5'. Empty = leave this slot to the model."})
                   for name, _, default in SLOTS},
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt", "model", "raw_reply", "picks")
    FUNCTION = "generate"
    CATEGORY = "text"
    DESCRIPTION = "Asks a local KoboldCpp server to write an image prompt from your request and standing guidelines."

    def generate(self, server_url, context, request, required_phrase, phrase_position, max_tokens, temperature, seed,
                 clean_output, keep_conversation, history_turns, unique_id=None, randomize_slots=False, **slot_texts):
        if not request.strip() and not context.strip():
            raise ValueError("KoboldCpp Prompt: enter a request (and/or a context block).")
        # randomize_slots defaults to off here so an API-format prompt saved before it existed runs as before
        picks = pick_slots(seed, {n: slot_texts.get(n, "\n".join(d)) for n, _, d in SLOTS}) if randomize_slots else {}
        user_text = request.strip()
        sentence = choices_sentence(picks)
        if sentence:
            user_text = f"{user_text}\n\n{sentence}" if user_text else sentence

        key = str(unique_id)
        ctx_key = _context_key(context)
        hist = _history.get(key)
        if not keep_conversation or not isinstance(hist, dict) or hist.get("context") != ctx_key:
            hist = {"context": ctx_key, "turns": []}
        messages = []
        if context.strip():
            messages.append({"role": "system", "content": context})
        if keep_conversation:
            for user, assistant in hist["turns"][-history_turns:]:
                messages += [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
        messages.append({"role": "user", "content": user_text or "Write the prompt."})

        raw, model = chat(server_url, messages, max_tokens, temperature, seed)
        if not model or model.lower() in ("koboldcpp", "gpt-3.5-turbo"):
            model = model_status(server_url).get("model") or model
        text = clean_reply(raw) if clean_output else raw.strip()
        if not text:
            raise RuntimeError("KoboldCpp returned an empty prompt. Try raising max_tokens (reasoning models use many tokens before answering).")
        text = ensure_phrase(text, required_phrase, phrase_position)

        if keep_conversation:
            hist["turns"].append([user_text, text])
            hist["turns"] = hist["turns"][-20:]
            _history[key] = hist
            changed = True
        else:
            changed = _history.pop(key, None) is not None
        if changed:
            with _file_lock:
                _save_json(CONVERSATION_FILE, _history)
        picks_text = picks_line(picks)
        if picks_text:
            logging.info(f"[KoboldCpp Prompt] picks: {picks_text}")
        logging.info(f"[KoboldCpp Prompt] {model}: {text[:200]}")
        return {"ui": {"model": [model], "text": [text], "picks": [picks_text]}, "result": (text, model, raw, picks_text)}


class ReviewEditPrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
                "pause": (["when text changes", "every run", "off"], {"default": "when text changes", "tooltip": "'when text changes': pauses for new prompts; reruns with the same prompt reuse your edit. 'every run': always pauses. 'off': passes text straight through."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "review"
    CATEGORY = "text"
    DESCRIPTION = "Pauses the run so you can read and edit the prompt, then continues with your version. Cancel stops the run."

    @classmethod
    def IS_CHANGED(s, text=None, pause="when text changes", **kwargs):
        return math.nan if pause == "every run" else pause

    def review(self, text, pause, unique_id=None):
        if pause != "off":
            text = wait_for_review(str(unique_id), text, "Review & Edit Prompt")
        return {"ui": {"text": [text]}, "result": (text,)}


NODE_CLASS_MAPPINGS = {"KoboldCppPrompt": KoboldCppPrompt, "ReviewEditPrompt": ReviewEditPrompt}
NODE_DISPLAY_NAME_MAPPINGS = {"KoboldCppPrompt": "KoboldCpp Prompt", "ReviewEditPrompt": "Review & Edit Prompt"}
