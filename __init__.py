"""ComfyUI-MiniMax-H3-Extend -- video-extend (continuation) nodes for stock/
public ComfyUI. See patch.py's module docstring for the mechanism.

Nothing in ComfyUI's model code is touched at import time. On ComfyUI >= 0.34
the nodes run on stock's own keyframe support and nothing is ever patched; on
older ComfyUI, patch.apply() runs the first time an extend node executes, and
its patches pass straight through to stock code for any conditioning this
pack didn't build. The native-namespace injection below only adds names that
are genuinely absent -- it never overrides an existing native class.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, inject_into_native

inject_into_native()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
