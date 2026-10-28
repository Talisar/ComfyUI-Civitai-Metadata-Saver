import hashlib
import json
import logging
import os
import threading

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

import folder_paths
from comfy.cli_args import args

from . import hooks, naming
from .metadata import Runtime, build_parameters, extract, parse_source_fields, rescale_parameters

HASH_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hash_cache.json")
_hash_lock = threading.Lock()


def _load_hash_cache():
    try:
        with open(HASH_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def autov2_hash(folder, name):
    """Civitai's AutoV2 hash (first 10 hex chars of the file's sha256), cached by path/size/mtime."""
    if not name:
        return None
    path = folder_paths.get_full_path(folder, name)
    if path is None and folder == "diffusion_models":
        path = folder_paths.get_full_path("checkpoints", name)
    if path is None or not os.path.isfile(path):
        return None
    st = os.stat(path)
    with _hash_lock:
        cache = _load_hash_cache()
        entry = cache.get(path)
        if entry and entry.get("size") == st.st_size and entry.get("mtime") == st.st_mtime:
            return entry["sha256"][:10]
        logging.info(f"[Civitai Metadata] hashing {path} (first time only)")
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024 * 8), b""):
                h.update(chunk)
        cache[path] = {"size": st.st_size, "mtime": st.st_mtime, "sha256": h.hexdigest()}
        tmp = HASH_CACHE_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=1)
        os.replace(tmp, HASH_CACHE_PATH)
        return cache[path]["sha256"][:10]


def exif_user_comment(text):
    exif = Image.Exif()
    exif.get_ifd(0x8769)[0x9286] = b"UNICODE\x00" + text.encode("utf-16-be")
    return exif


class SaveImageCivitai:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "ComfyUI", "tooltip": "Start of the filename. May contain strftime dates (%F) and subfolders (a/b)."}),
                "filename_keys": ("STRING", {"default": "", "multiline": True, "tooltip": "Comma-separated keys appended to the filename: %F / %H-%M-%S dates, ./subfolder, 123.widget_name, widget_name, or traced values seed, steps, cfg, sampler_name, scheduler, model, lora, loras. Unknown keys are used literally."}),
                "foldername_prefix": ("STRING", {"default": "", "tooltip": "Folder (inside the output directory) to save into. Same key/date syntax."}),
                "foldername_keys": ("STRING", {"default": "", "multiline": True, "tooltip": "Comma-separated keys appended to the folder name. Prefix a key with ./ to make it a subfolder."}),
                "delimiter": ("STRING", {"default": "_", "tooltip": "Text placed between prefix, keys and counter."}),
                "counter_digits": ("INT", {"default": 4, "min": 1, "max": 9, "tooltip": "Zero-padded counter width. Continues from the highest number in the folder; does not fill gaps."}),
                "counter_position": (["last", "first"], {"default": "last"}),
                "one_counter_per_folder": ("BOOLEAN", {"default": True, "tooltip": "On: one counter for the whole folder. Off: separate counter per filename."}),
                "file_format": (["png", "jpg", "webp"], {"default": "png", "tooltip": "PNG keeps the ComfyUI workflow too. JPG/WebP carry only the Civitai parameters (EXIF UserComment)."}),
                "quality": ("INT", {"default": 95, "min": 1, "max": 100, "tooltip": "JPG/WebP quality."}),
                "embed_workflow": ("BOOLEAN", {"default": True, "tooltip": "Also embed the ComfyUI workflow (PNG only) so the image still drags back into ComfyUI."}),
                "lora_tags_in_prompt": ("BOOLEAN", {"default": True, "tooltip": "Append <lora:name:strength> tags to the positive prompt like A1111 does."}),
                "hash_models": ("BOOLEAN", {"default": True, "tooltip": "Compute model/LoRA hashes so Civitai can link the resources. Cached after the first time."}),
            },
            "optional": {
                "positive_override": ("STRING", {"forceInput": True, "tooltip": "Use this text as the positive prompt instead of the one traced from the graph."}),
                "negative_override": ("STRING", {"forceInput": True}),
                "extra_parameters": ("STRING", {"default": "", "multiline": False, "tooltip": "Extra A1111 fields appended verbatim, e.g. 'Shift: 4'."}),
                "source_parameters": ("STRING", {"forceInput": True, "tooltip": "Parameters read from the source image (Load Image by Index 'parameters' output). When not empty, they are reused as-is instead of tracing this workflow; Size is updated and 'Hires upscale' added if the image was resized."}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("parameters",)
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "image"
    DESCRIPTION = "Saves images with A1111-style 'parameters' metadata (prompts, seed, sampler, model, LoRAs + hashes) that Civitai reads automatically."

    def trace(self, positive_override, negative_override, prompt, unique_id):
        info = {}
        try:
            node = (prompt or {}).get(str(unique_id), {})
            image_link = node.get("inputs", {}).get("images")
            ctx_prompt_id, list_index = None, 0
            try:
                from comfy_execution.utils import get_executing_context
                ctx = get_executing_context()
                if ctx:
                    ctx_prompt_id = str(ctx.prompt_id)
                    list_index = getattr(ctx, "list_index", None) or 0
            except Exception:
                pass
            info = extract(prompt, image_link, Runtime(ctx_prompt_id, hooks.texts, hooks.samples, list_index))
        except Exception:
            logging.exception("[Civitai Metadata] failed to read workflow graph")

        if positive_override is not None:
            info["positive"] = positive_override
        if negative_override is not None:
            info["negative"] = negative_override
        return info

    def build(self, images, info, lora_tags_in_prompt, hash_models, extra_parameters):
        model_hash, lora_hashes = None, {}
        if hash_models:
            try:
                model_hash = autov2_hash(info.get("model_folder") or "checkpoints", info.get("model"))
                for name, _ in info.get("loras", []):
                    lora_hashes[name] = autov2_hash("loras", name)
            except Exception:
                logging.exception("[Civitai Metadata] hashing failed")

        h, w = images.shape[1], images.shape[2]
        return build_parameters(info, w, h, model_hash, lora_hashes, lora_tags_in_prompt, extra_parameters or "")

    def save_images(self, images, filename_prefix="ComfyUI", filename_keys="", foldername_prefix="", foldername_keys="",
                    delimiter="_", counter_digits=4, counter_position="last", one_counter_per_folder=True,
                    file_format="png", quality=95, embed_workflow=True,
                    lora_tags_in_prompt=True, hash_models=True, positive_override=None, negative_override=None,
                    extra_parameters="", source_parameters=None, prompt=None, extra_pnginfo=None, unique_id=None):
        if source_parameters and source_parameters.strip():
            # Upscale-only workflow: no sampler here, so carry the original generation's data over.
            info = parse_source_fields(source_parameters)
            h, w = images.shape[1], images.shape[2]
            parameters = rescale_parameters(source_parameters, w, h, extra_parameters)
        else:
            info = self.trace(positive_override, negative_override, prompt, unique_id)
            parameters = self.build(images, info, lora_tags_in_prompt, hash_models, extra_parameters)

        ext = f".{file_format}"
        folder, stem = naming.resolve_paths(self.output_dir, filename_prefix, filename_keys, foldername_prefix,
                                            foldername_keys, delimiter, prompt, info)
        os.makedirs(folder, exist_ok=True)
        counter = naming.next_counter(folder, stem, ext, counter_digits, counter_position, delimiter, one_counter_per_folder)
        subfolder = os.path.relpath(folder, self.output_dir)
        subfolder = "" if subfolder == "." else subfolder

        results = []
        for image in images:
            img = Image.fromarray(np.clip(255. * image.cpu().numpy(), 0, 255).astype(np.uint8))
            num = f"{counter:0{counter_digits}}"
            file = f"{stem}{delimiter}{num}{ext}" if counter_position == "last" else f"{num}{delimiter}{stem}{ext}"
            path = os.path.join(folder, file)

            if file_format == "png":
                metadata = None
                if not args.disable_metadata:
                    metadata = PngInfo()
                    metadata.add_text("parameters", parameters)
                    if embed_workflow:
                        if prompt is not None:
                            metadata.add_text("prompt", json.dumps(prompt))
                        if extra_pnginfo is not None:
                            for x in extra_pnginfo:
                                metadata.add_text(x, json.dumps(extra_pnginfo[x]))
                img.save(path, pnginfo=metadata, compress_level=self.compress_level)
            else:
                kwargs = {"quality": quality}
                if not args.disable_metadata:
                    kwargs["exif"] = exif_user_comment(parameters)
                if file_format == "jpg":
                    img.convert("RGB").save(path, "JPEG", **kwargs)
                else:
                    img.save(path, "WEBP", **kwargs)

            results.append({"filename": file, "subfolder": subfolder, "type": self.type})
            counter += 1

        return {"ui": {"images": results}, "result": (parameters,)}


NODE_CLASS_MAPPINGS = {"SaveImageCivitai": SaveImageCivitai}
NODE_DISPLAY_NAME_MAPPINGS = {"SaveImageCivitai": "Save Image (Civitai Metadata)"}
