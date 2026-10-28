"""Load one text or image file from a folder per queued run.

Unlike list loaders, each run gets a single item, so the whole workflow
(generate, save, upscale, save) finishes and frees memory before the next run.
"""

import hashlib
import logging
import os
import re

import numpy as np
import torch
from PIL import Image, ImageOps

import folder_paths

SORT_MODES = ["natural name", "name", "modified date"]
AT_END_MODES = ["stop", "wrap"]


def resolve_folder(folder):
    """Empty: ComfyUI output folder. Relative name/path: inside output. Absolute path: as is."""
    folder = folder.strip().strip('"').strip("'")
    output = folder_paths.get_output_directory()
    if not folder:
        return output
    folder = os.path.expanduser(folder)
    return folder if os.path.isabs(folder) else os.path.join(output, folder)


def _natural_key(name):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]


def list_files(folder, extensions):
    """Files directly in folder (no subfolders) whose extension is in extensions, unsorted."""
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")
    files = []
    for entry in os.scandir(folder):
        if entry.is_file() and os.path.splitext(entry.name)[1].lower().lstrip(".") in extensions:
            files.append(entry.path)
    return files


def sort_files(files, sort):
    if sort == "name":
        return sorted(files, key=lambda p: os.path.basename(p).lower())
    if sort == "modified date":
        return sorted(files, key=lambda p: (os.path.getmtime(p), _natural_key(os.path.basename(p))))
    return sorted(files, key=lambda p: _natural_key(os.path.basename(p)))


def pick_file(folder, extensions, index, sort, at_end, kind):
    """Returns (path, index_used, total). index is 1-based."""
    path = resolve_folder(folder)
    files = sort_files(list_files(path, extensions), sort)
    total = len(files)
    if total == 0:
        raise FileNotFoundError(f"No {kind} files in {path}")
    if index > total:
        if at_end == "stop":
            hint = " (the index control looks set to randomize; set it to increment)" if index > 1_000_000 else ""
            raise IndexError(f"index {index} > {total} files, batch finished{hint}")
        index = (index - 1) % total + 1
    return files[index - 1], index, total


def parse_extensions(extensions):
    return {e.strip().lower().lstrip(".") for e in extensions.split(",") if e.strip()}


def file_state(folder, extensions, index, sort, at_end, kind):
    """IS_CHANGED value: index plus the chosen file and its mtime, so a new index always re-runs."""
    try:
        path, used, total = pick_file(folder, extensions, index, sort, at_end, kind)
        return f"{index}|{used}/{total}|{path}|{os.path.getmtime(path)}"
    except Exception as e:
        # Let the run itself raise the error.
        return f"{index}|error|{e}"


def read_text(path):
    with open(path, "rb") as f:
        data = f.read()
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        logging.warning(f"[Folder loaders] {os.path.basename(path)} is not UTF-8; reading it as cp1252")
        return data.decode("cp1252", errors="replace")


def read_parameters(img):
    """A1111 'parameters' text from a PNG text chunk, or EXIF UserComment (JPG/WebP)."""
    text = img.info.get("parameters")
    if isinstance(text, str):
        return text
    try:
        raw = img.getexif().get_ifd(0x8769).get(0x9286)
    except Exception:
        return ""
    if isinstance(raw, str):
        return raw
    if not isinstance(raw, bytes) or not raw:
        return ""
    code, body = raw[:8], raw[8:]
    if code.startswith(b"UNICODE"):
        # Big-endian is what A1111 and our save node write; guess little-endian from the zero bytes.
        le = len(body) > 1 and body[1:2] == b"\x00" and body[0:1] != b"\x00"
        return body.decode("utf-16-le" if le else "utf-16-be", errors="replace").rstrip("\x00")
    if code.startswith(b"ASCII") or code == b"\x00" * 8:
        return body.decode("utf-8", errors="replace").rstrip("\x00")
    return raw.decode("utf-8", errors="replace").rstrip("\x00")


def open_image(path):
    """Returns (image, mask, parameters) for one image file."""
    with Image.open(path) as img:
        parameters = read_parameters(img)
        img = ImageOps.exif_transpose(img)
        if img.mode == "I":
            img = img.point(lambda v: v * (1 / 255))
        has_alpha = "A" in img.getbands() or (img.mode == "P" and "transparency" in img.info)
        alpha = img.convert("RGBA").getchannel("A") if has_alpha else None
        rgb = img.convert("RGB")

    image = torch.from_numpy(np.array(rgb).astype(np.float32) / 255.0)[None,]
    # Same convention as core Load Image: 1 where the image is transparent.
    if alpha is not None:
        mask = 1.0 - torch.from_numpy(np.array(alpha).astype(np.float32) / 255.0)[None,]
    else:
        mask = torch.zeros((1, rgb.height, rgb.width), dtype=torch.float32)
    return image, mask, parameters


def _common_inputs():
    return {
        "folder": ("STRING", {"default": "", "tooltip": "Empty: ComfyUI's output folder. A name: a folder inside output. Or a full path. Subfolders are skipped."}),
        "index": ("INT", {"default": 1, "min": 1, "max": 0xFFFFFFFF, "control_after_generate": True, "tooltip": "Which file to load (1 = first). New nodes start on increment: set Run count to the number of files to go through the folder one file per run. Do not use randomize."}),
        "sort": (SORT_MODES, {"default": "natural name", "tooltip": "'natural name' puts 2.txt before 10.txt. 'modified date' is oldest first."}),
        "at_end": (AT_END_MODES, {"default": "stop", "tooltip": "'stop': an index past the last file raises 'batch finished', so extra queued runs fail fast. 'wrap': start over at the first file."}),
    }


class LoadTextByIndex:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": _common_inputs()}

    RETURN_TYPES = ("STRING", "STRING", "INT", "INT")
    RETURN_NAMES = ("text", "filename", "index", "total")
    FUNCTION = "load"
    CATEGORY = "text"
    DESCRIPTION = "Loads ONE .txt file from a folder per run, picked by index. Set index to increment and Run count to the number of files, so each prompt runs through the whole workflow and saves before the next starts."

    @classmethod
    def IS_CHANGED(s, folder, index, sort, at_end):
        return file_state(folder, {"txt"}, index, sort, at_end, ".txt")

    def load(self, folder, index, sort, at_end):
        path, used, total = pick_file(folder, {"txt"}, index, sort, at_end, ".txt")
        name = os.path.basename(path)
        print(f"[LoadTextByIndex] {used}/{total}: {name}")
        return (read_text(path), os.path.splitext(name)[0], used, total)


class LoadImageByIndex:
    @classmethod
    def INPUT_TYPES(s):
        inputs = _common_inputs()
        inputs["extensions"] = ("STRING", {"default": "png,jpg,jpeg,webp", "tooltip": "Comma-separated image extensions to include."})
        return {"required": inputs}

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING", "INT", "INT", "STRING")
    RETURN_NAMES = ("image", "mask", "filename", "caption", "index", "total", "parameters")
    FUNCTION = "load"
    CATEGORY = "image"
    DESCRIPTION = "Loads ONE image from a folder per run, picked by index, plus its matching .txt caption if there is one and its A1111 'parameters' metadata. Set index to increment and Run count to the number of images."

    @classmethod
    def IS_CHANGED(s, folder, index, sort, at_end, extensions):
        return file_state(folder, parse_extensions(extensions), index, sort, at_end, "image")

    def load(self, folder, index, sort, at_end, extensions):
        exts = parse_extensions(extensions)
        if not exts:
            raise ValueError("Load Image by Index: extensions is empty.")
        path, used, total = pick_file(folder, exts, index, sort, at_end, "image")
        name = os.path.basename(path)
        stem = os.path.splitext(name)[0]
        print(f"[LoadImageByIndex] {used}/{total}: {name}")

        image, mask, parameters = open_image(path)
        caption_path = os.path.join(os.path.dirname(path), stem + ".txt")
        caption = read_text(caption_path) if os.path.isfile(caption_path) else ""
        return (image, mask, stem, caption, used, total, parameters)


class LoadImageWithParameters:
    """Core Load Image (pick or drag & drop into ComfyUI's input folder) plus the A1111 parameters."""

    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["image"])
        return {"required": {"image": (sorted(files), {"image_upload": True})}}

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "mask", "parameters")
    FUNCTION = "load"
    CATEGORY = "image"
    DESCRIPTION = "Like core Load Image (drag & drop works), plus a 'parameters' output with the image's A1111 generation data. Wire it to Save Image (Civitai Metadata) 'source_parameters' to keep that data through an upscale."

    @classmethod
    def IS_CHANGED(s, image):
        path = folder_paths.get_annotated_filepath(image)
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return f"Invalid image file: {image}"
        return True

    def load(self, image):
        return open_image(folder_paths.get_annotated_filepath(image))


NODE_CLASS_MAPPINGS = {"LoadTextByIndex": LoadTextByIndex, "LoadImageByIndex": LoadImageByIndex,
                       "LoadImageWithParameters": LoadImageWithParameters}
NODE_DISPLAY_NAME_MAPPINGS = {"LoadTextByIndex": "Load Text by Index (from folder)", "LoadImageByIndex": "Load Image by Index (from folder)",
                              "LoadImageWithParameters": "Load Image (with Parameters)"}
