"""Record the real prompt text and sampler settings as nodes execute.

Static graph reading can't know text produced at runtime (LLM prompt expanders,
wildcards, random seeds resolved in Python). Wrapping CLIP.tokenize and
comfy.sample.sample/sample_custom lets us capture the values that were actually
used, keyed by (prompt_id, node_id, list_index) of the node that was executing.
list_index tells apart the items of a list run (e.g. a list of prompts), where
ComfyUI calls each node once per item within one prompt.
"""

import functools
import logging
from collections import OrderedDict

MAX_PROMPTS = 32

texts = {}    # (prompt_id, node_id, list_index) -> [str, ...]
samples = {}  # (prompt_id, node_id, list_index) -> dict
_prompt_order = OrderedDict()


def _context():
    try:
        from comfy_execution.utils import get_executing_context
        ctx = get_executing_context()
    except Exception:
        return None
    if ctx is None or ctx.node_id is None:
        return None
    return str(ctx.prompt_id), str(ctx.node_id), getattr(ctx, "list_index", None) or 0


def _touch(prompt_id):
    _prompt_order[prompt_id] = True
    _prompt_order.move_to_end(prompt_id)
    while len(_prompt_order) > MAX_PROMPTS:
        old, _ = _prompt_order.popitem(last=False)
        for store in (texts, samples):
            for key in [k for k in store if k[0] == old]:
                del store[key]


def _record_text(text):
    key = _context()
    if key is None or not isinstance(text, str):
        return
    _touch(key[0])
    found = texts.setdefault(key, [])
    if text not in found:
        found.append(text)


def _record_sample(**values):
    key = _context()
    if key is None:
        return
    _touch(key[0])
    rec = samples.setdefault(key, {})
    for k, v in values.items():
        if v is not None:
            rec[k] = v


def install():
    import comfy.sd
    import comfy.sample

    if getattr(comfy.sd.CLIP.tokenize, "_civitai_meta", False):
        return

    orig_tokenize = comfy.sd.CLIP.tokenize

    @functools.wraps(orig_tokenize)
    def tokenize(self, text, *args, **kwargs):
        try:
            _record_text(text)
        except Exception:
            logging.exception("[Civitai Metadata] text capture failed")
        return orig_tokenize(self, text, *args, **kwargs)

    tokenize._civitai_meta = True
    comfy.sd.CLIP.tokenize = tokenize

    orig_sample = comfy.sample.sample

    @functools.wraps(orig_sample)
    def sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0, *args, **kwargs):
        try:
            _record_sample(steps=steps, cfg=cfg, sampler_name=sampler_name, scheduler=scheduler,
                           denoise=denoise, seed=kwargs.get("seed"))
        except Exception:
            logging.exception("[Civitai Metadata] sampler capture failed")
        return orig_sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise, *args, **kwargs)

    comfy.sample.sample = sample

    orig_sample_custom = comfy.sample.sample_custom

    @functools.wraps(orig_sample_custom)
    def sample_custom(*args, **kwargs):
        try:
            _record_sample(seed=kwargs.get("seed"))
        except Exception:
            logging.exception("[Civitai Metadata] sampler capture failed")
        return orig_sample_custom(*args, **kwargs)

    comfy.sample.sample_custom = sample_custom
