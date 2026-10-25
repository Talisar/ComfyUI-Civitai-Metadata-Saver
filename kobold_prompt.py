"""Generate image prompts with a local KoboldCpp server, then review/edit them before they're used.

KoboldCpp's API is stateless: the Memory and chat history in its Kobold Lite page are never
seen by API calls, so the node sends its context block as the system message on every request.
"""

import json
import logging
import math
import os
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
_QUOTES = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"})


def clean_reply(text):
    """Keep only the prompt: drop <think> blocks, unwrap a code block, drop a 'Here's your prompt:' lead-in."""
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
    if len(lines) > 1 and lines[0].rstrip().endswith(":") and len(lines[0]) < 120:
        text = "\n".join(lines[1:]).strip()
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

_history = {}  # node_id -> {"context": str, "turns": [(user, assistant), ...]}


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
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt", "model", "raw_reply")
    FUNCTION = "generate"
    CATEGORY = "text"
    DESCRIPTION = "Asks a local KoboldCpp server to write an image prompt from your request and standing guidelines."

    def generate(self, server_url, context, request, required_phrase, phrase_position, max_tokens, temperature, seed,
                 clean_output, keep_conversation, history_turns, unique_id=None):
        if not request.strip() and not context.strip():
            raise ValueError("KoboldCpp Prompt: enter a request (and/or a context block).")
        key = str(unique_id)
        hist = _history.get(key)
        if not keep_conversation or hist is None or hist["context"] != context:
            hist = {"context": context, "turns": []}
        messages = []
        if context.strip():
            messages.append({"role": "system", "content": context})
        if keep_conversation:
            for user, assistant in hist["turns"][-history_turns:]:
                messages += [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
        messages.append({"role": "user", "content": request.strip() or "Write the prompt."})

        raw, model = chat(server_url, messages, max_tokens, temperature, seed)
        if not model or model.lower() in ("koboldcpp", "gpt-3.5-turbo"):
            model = model_status(server_url).get("model") or model
        text = clean_reply(raw) if clean_output else raw.strip()
        if not text:
            raise RuntimeError("KoboldCpp returned an empty prompt. Try raising max_tokens (reasoning models use many tokens before answering).")
        text = ensure_phrase(text, required_phrase, phrase_position)

        if keep_conversation:
            hist["turns"].append((request.strip(), text))
            hist["turns"] = hist["turns"][-20:]
            _history[key] = hist
        else:
            _history.pop(key, None)
        logging.info(f"[KoboldCpp Prompt] {model}: {text[:200]}")
        return {"ui": {"model": [model], "text": [text]}, "result": (text, model, raw)}


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
