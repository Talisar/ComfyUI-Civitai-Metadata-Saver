import logging

from . import hooks
from .civitai_saver import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

try:
    hooks.install()
except Exception:
    logging.exception("[Civitai Metadata] could not install runtime capture hooks; falling back to graph-only metadata")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
