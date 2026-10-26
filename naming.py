"""Filename / folder naming compatible with Save Image Extended's conventions.

A name is a prefix followed by comma-separated keys, joined with a one-character delimiter:
  - "%F", "%Y-%m-%d %H-%M-%S" ...   strftime date/time
  - "./key"                          subfolder named by the key's value (e.g. ./lora, ./%F)
  - "sub/", "a/b"                    literal subfolder (no delimiter around it)
  - "123.widget_name"                widget value from node #123
  - "widget_name"                    widget value from the last node that has it
  - "seed", "steps", "cfg", "sampler_name", "scheduler", "denoise",
    "model", "lora", "loras"         values traced for the saved image (see metadata.extract)
  - "some.text" / unknown names      used literally
"""

import os
import re
from datetime import datetime

from .metadata import GraphReader, is_link, short_name

MODEL_EXTS = (".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".sft")
INFO_KEYS = ("seed", "steps", "cfg", "sampler_name", "scheduler", "denoise", "guidance")
INVALID_CHARS = re.compile(r'[<>:"|?*\x00-\x1f]')


def _clean(value):
    if isinstance(value, float):
        value = round(value, 2)
    value = str(value)
    if value.lower().endswith(MODEL_EXTS):
        value = short_name(value)
    return INVALID_CHARS.sub("", value.replace("\\", "/"))


def _info_value(key, info):
    if key in INFO_KEYS:
        return info.get(key)
    if key in ("model", "ckpt_name", "unet_name") and info.get("model"):
        return short_name(info["model"])
    if key in ("lora", "loras") and info.get("loras"):
        names = [short_name(n) for n, _ in info["loras"]]
        return names[0] if key == "lora" else "+".join(names)
    return None


def _widget_value(graph, key, node_id=None):
    nodes = [node_id] if node_id is not None else list(graph.prompt)
    found = None
    for nid in nodes:
        node = graph.node(nid)
        if node and key in node.get("inputs", {}):
            found = node["inputs"][key]  # keep the last match
    if found is None:
        return None
    value = graph.resolve(found) if is_link(found) else found
    if isinstance(value, (dict, list)):
        return None
    return value


def build_name(prefix, keys, delimiter, prompt, info, timestamp=None):
    timestamp = timestamp or datetime.now()
    graph = GraphReader(prompt)
    name = timestamp.strftime(prefix) if "%" in prefix else prefix

    for key in (k.strip() for k in keys.split(",")):
        if not key:
            continue
        subfolder = key.startswith("./") and "/" not in key[2:]
        if subfolder:
            key = key[2:]
        if key.startswith("%") and _info_value(key[1:], info) is not None:
            value = _info_value(key[1:], info)  # "%lora" means "lora", not a date code
        elif "%" in key:
            try:
                value = timestamp.strftime(key)
            except ValueError:  # Windows rejects unknown codes; keep the text instead
                value = key.replace("%", "")
        elif "/" in key:
            value = key
        else:
            value = _info_value(key, info)
            parts = key.split(".")
            if value is None and len(parts) == 2 and all(parts) and parts[0].isdecimal():
                value = _widget_value(graph, parts[1], parts[0])
                if value is None:
                    value = _widget_value(graph, parts[1])
                if value is None:
                    value = parts[1]
            elif value is None and len(parts) == 1:
                value = _widget_value(graph, key)
                if value is None:
                    value = key
            elif value is None:
                value = key
        value = _clean(value)
        if value == "":
            continue
        if subfolder:
            value = "/" + value.replace("/", "-")

        no_delim = not name or name.endswith("/") or value.startswith("/") or value.startswith(".") \
            or ("/" in value and not value.endswith("/"))
        name += ("" if no_delim else delimiter) + value

    name = re.sub(r"(^|/)\./", r"\1", name.replace("\\", "/"))
    parts = [p.strip().strip(delimiter).strip() for p in name.split("/")]
    return "/".join(p for p in parts if p and p not in (".", ".."))


def next_counter(folder, stem, ext, digits, position, delimiter, one_counter_per_folder):
    """Highest existing counter + 1 (does not fill gaps)."""
    if not os.path.isdir(folder):
        return 1
    best = 0
    for f in os.listdir(folder):
        if not f.lower().endswith(ext):
            continue
        base = f[: -len(ext)]
        if position == "last":
            num, rest = base[-digits:], base[: -digits].rstrip(delimiter)
        else:
            num, rest = base[:digits], base[digits:].lstrip(delimiter)
        if num.isdecimal() and (one_counter_per_folder or rest == stem):
            best = max(best, int(num))
    return best + 1


def resolve_paths(output_dir, filename_prefix, filename_keys, foldername_prefix, foldername_keys,
                  delimiter, prompt, info, timestamp=None):
    """Returns (absolute folder, filename stem). Filename subfolders are folded into the folder."""
    timestamp = timestamp or datetime.now()
    folder = build_name(foldername_prefix, foldername_keys, delimiter, prompt, info, timestamp)
    filename = build_name(filename_prefix, filename_keys, delimiter, prompt, info, timestamp)
    if not filename and not filename_keys.strip():
        filename = "ComfyUI"
    sub, _, stem = filename.rpartition("/")
    full = os.path.abspath(os.path.join(output_dir, folder, sub))
    root = os.path.abspath(output_dir)
    if os.path.commonpath([full, root]) != root:
        raise ValueError(f"Save path escapes the output directory: {full}")
    return full, stem or "ComfyUI"
