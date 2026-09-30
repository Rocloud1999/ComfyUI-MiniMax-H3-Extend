"""MiniMax H3 continuation nodes and an Autogrow Ref2VA endpoint node.

The original continuation nodes and model patches are unchanged. A mixed
registry keeps their classic registration while exposing the endpoint as a
real V3 ComfyNode. Do not add a parallel comfy_entrypoint: ComfyUI selects
NODE_CLASS_MAPPINGS first and would ignore it.
"""

import logging

from .nodes import NODE_CLASS_MAPPINGS as EXTEND_CLASS_MAPPINGS
from .nodes import NODE_DISPLAY_NAME_MAPPINGS as EXTEND_DISPLAY_NAME_MAPPINGS
from .nodes import inject_into_native

NODE_CLASS_MAPPINGS = dict(EXTEND_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS = dict(EXTEND_DISPLAY_NAME_MAPPINGS)
inject_into_native()

try:
    from comfy_api.latest import io
except ImportError as exc:
    # Only an unavailable V3 API is optional. Do not hide a broken dependency
    # inside an otherwise installed API, which needs a real error report.
    if exc.name not in ("comfy_api", "comfy_api.latest"):
        raise
    io = None

if io is not None and hasattr(io, "ComfyNode") and hasattr(io, "Autogrow"):
    from .endpoint_nodes import MiniMaxH3ReferenceToVideoWithEndpoints

    schema = MiniMaxH3ReferenceToVideoWithEndpoints.GET_SCHEMA()
    NODE_CLASS_MAPPINGS[schema.node_id] = MiniMaxH3ReferenceToVideoWithEndpoints
    NODE_DISPLAY_NAME_MAPPINGS[schema.node_id] = schema.display_name
else:
    # No fallback endpoint implementation. Existing continuation nodes remain
    # available when the installation is too old to expose Autogrow inputs.
    logging.warning("[ComfyUI-MiniMax-H3-Extend] Autogrow endpoint node requires "
                    "ComfyUI's V3 Autogrow API; update ComfyUI and its frontend. "
                    "The original Extend/Encode AV nodes remain registered.")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
