import logging

from . import hooks
from .civitai_saver import NODE_CLASS_MAPPINGS as _SAVER_NODES, NODE_DISPLAY_NAME_MAPPINGS as _SAVER_NAMES
from .kobold_prompt import NODE_CLASS_MAPPINGS as _KOBOLD_NODES, NODE_DISPLAY_NAME_MAPPINGS as _KOBOLD_NAMES
from .folder_loaders import NODE_CLASS_MAPPINGS as _FOLDER_NODES, NODE_DISPLAY_NAME_MAPPINGS as _FOLDER_NAMES

try:
    hooks.install()
except Exception:
    logging.exception("[Civitai Metadata] could not install runtime capture hooks; falling back to graph-only metadata")

NODE_CLASS_MAPPINGS = {**_SAVER_NODES, **_KOBOLD_NODES, **_FOLDER_NODES}
NODE_DISPLAY_NAME_MAPPINGS = {**_SAVER_NAMES, **_KOBOLD_NAMES, **_FOLDER_NAMES}
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
