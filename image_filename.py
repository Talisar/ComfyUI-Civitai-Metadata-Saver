"""Get the source filename of an image from whatever node loaded it.

IMAGE tensors carry no filename, so this node looks at the workflow instead:
it follows its image input back through the graph (through upscales, resizes
and other image nodes) to the first node with an image file widget, such as
core Load Image's 'image', and returns that file's name.
"""

import logging
import os
from collections import deque

IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp", "gif", "tif", "tiff", "avif", "jxl", "heic", "heif"}
# Inputs that usually carry the image itself; followed before other links (e.g. a model or VAE).
IMAGE_INPUT_NAMES = ("image", "images", "pixels", "samples", "latent_image", "latent")
MAX_NODES = 200


def _is_link(value):
    return isinstance(value, list) and len(value) == 2 and isinstance(value[0], (str, int))


def file_stem(value):
    """Stem of an image filename/path widget value, or None if it isn't one.

    Handles core annotations ('a/b.png [input]') and Windows paths.
    """
    if not isinstance(value, str):
        return None
    value = value.strip().strip('"').strip("'")
    for tag in (" [input]", " [output]", " [temp]"):
        if value.endswith(tag):
            value = value[: -len(tag)]
    name = value.replace("\\", "/").rsplit("/", 1)[-1]
    stem, ext = os.path.splitext(name)
    if stem and ext.lower().lstrip(".") in IMAGE_EXTENSIONS:
        return stem
    return None


def _by_index_stem(inputs):
    """Load Image by Index computes its file from folder + index; pick the same one."""
    from .folder_loaders import parse_extensions, pick_file

    try:
        path, _, _ = pick_file(inputs["folder"], parse_extensions(inputs["extensions"]), inputs["index"],
                               inputs["sort"], inputs["at_end"], "image")
    except Exception:
        return None
    return os.path.splitext(os.path.basename(path))[0]


def _node_stem(node):
    inputs = node.get("inputs", {})
    if node.get("class_type") == "LoadImageByIndex":
        stem = _by_index_stem(inputs)
        if stem:
            return stem
    # A widget literally named 'image' first (core Load Image and most copies of it), then any other.
    names = sorted(inputs, key=lambda k: (k != "image", k not in ("image_path", "path", "filename", "file")))
    for name in names:
        stem = file_stem(inputs[name])
        if stem:
            return stem
    return None


def trace_filename(prompt, unique_id, input_name="image"):
    """Breadth-first walk from this node's input back to the nearest node with an image file widget."""
    node = prompt.get(str(unique_id))
    if node is None:
        return None
    start = node.get("inputs", {}).get(input_name)
    if not _is_link(start):
        return None

    queue = deque([str(start[0])])
    seen = set()
    while queue and len(seen) < MAX_NODES:
        node_id = queue.popleft()
        if node_id in seen or node_id not in prompt:
            continue
        seen.add(node_id)
        node = prompt[node_id]
        stem = _node_stem(node)
        if stem:
            return stem
        inputs = node.get("inputs", {})
        links = [k for k in inputs if _is_link(inputs[k])]
        links.sort(key=lambda k: k not in IMAGE_INPUT_NAMES)
        queue.extend(str(inputs[k][0]) for k in links)
    return None


class ImageFilename:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Any image that came from a load-image node, directly or after upscales/resizes."}),
                "fallback": ("STRING", {"default": "image", "tooltip": "Used when no image file is found upstream (e.g. a generated image)."}),
            },
            "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "filename")
    FUNCTION = "get_filename"
    CATEGORY = "image"
    DESCRIPTION = "Outputs the filename (without extension) of the image file this image was loaded from, by following the image wire back to its load node. Works with core Load Image and most other loaders. The image passes through unchanged."

    def get_filename(self, image, fallback="image", prompt=None, unique_id=None):
        stem = trace_filename(prompt or {}, unique_id) if unique_id is not None else None
        if not stem:
            logging.warning(f"[Image Filename] no image file found upstream; using fallback '{fallback}'")
            stem = fallback
        return (image, stem)


NODE_CLASS_MAPPINGS = {"ImageFilename": ImageFilename}
NODE_DISPLAY_NAME_MAPPINGS = {"ImageFilename": "Image Filename (from loader)"}
