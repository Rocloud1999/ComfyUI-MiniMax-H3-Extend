"""ComfyUI-MiniMax-H3-Extend -- backports video-extend (continuation) support
onto stock/public ComfyUI installs that don't have it natively. See
patch.py's module docstring for the mechanism and README.md for status.

Both the model-level patches and the native-namespace injection are skipped
outright if this ComfyUI already has MiniMaxH3VideoExtend natively (i.e. it's
already on a fork with real support) -- this pack only ever fills a gap,
never overrides something better that's already there.
"""

from . import patch

_patched = patch.apply()

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, inject_into_native  # noqa: E402

inject_into_native()

if _patched:
    print("[ComfyUI-MiniMax-H3-Extend] Patched PackedLayout + MiniMaxH3.extra_conds "
          "(no native video-extend support found) -- see this pack's README.md for verification status.")
else:
    print("[ComfyUI-MiniMax-H3-Extend] Native MiniMaxH3VideoExtend already present -- patches skipped, not needed.")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
