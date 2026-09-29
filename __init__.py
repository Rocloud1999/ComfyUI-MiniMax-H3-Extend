"""ComfyUI-MiniMax-H3-Extend -- continuation and Ref2VA endpoint nodes.

Nothing in ComfyUI's model code is touched at import time. On ComfyUI >= 0.34
these nodes use native keyframe support. On older ComfyUI, patches are
installed on first execution and gated to this pack's context/endpoint
markers. The native-namespace injection only adds genuinely absent names.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, inject_into_native
from .endpoint_nodes import (
    NODE_CLASS_MAPPINGS as ENDPOINT_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as ENDPOINT_DISPLAY_NAME_MAPPINGS,
)

# Copy rather than mutate nodes.py's registries on import.
NODE_CLASS_MAPPINGS = {**NODE_CLASS_MAPPINGS, **ENDPOINT_CLASS_MAPPINGS}
NODE_DISPLAY_NAME_MAPPINGS = {**NODE_DISPLAY_NAME_MAPPINGS, **ENDPOINT_DISPLAY_NAME_MAPPINGS}

inject_into_native()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
