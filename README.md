# ComfyUI-MiniMax-H3-Extend


## What this does

1. **`MiniMaxH3VideoExtendPatched`** node (`nodes.py`) -- a standalone,
   directly-usable node exposing the above, plus injection of
   `MiniMaxH3VideoExtend` into `comfy_extras.nodes_minimax_h3`'s own
   namespace (again, only if absent) so anything that looks it up by that
   name there -- e.g. `ComfyUI-H3-Cast`'s `H3CastToVideoExtend` -- finds a
   working implementation transparently, no changes of its own needed.

2. **`MiniMaxH3EncodeAVPatched`** node (`nodes.py`) -- `vae`-encodes video
   frames (+ optional audio) into the AV latent `MiniMaxH3VideoExtendPatched`'s
   (or the native node's) `context_latent` input needs, for feeding an
   externally-loaded prior clip (e.g. `VHS_LoadVideo`) into either. Vendored
   from the native `MiniMaxH3EncodeAV`, which isn't part of stock/public
   support either. Lives here rather than in `ComfyUI-H3-Cast` since it's an
   extend/continuation concern, not a cast/character one -- also injected
   into the native namespace the same way as `MiniMaxH3VideoExtend` above.


Confirmed working via live testing (2026-08-11) across text-to-video,
reference-to-video, and cast-to-video continuation. Recommended starting
settings: `context_frames` 2, `ref_spacing` 1-2, `ref_decay` 0.3, `ref_ramp`
3-4 (5-6 if the prior clip had more motion than usual).

## Install

Same as any custom node pack -- clone into `custom_nodes/`, restart ComfyUI.
No core ComfyUI files are edited; everything is monkey-patched in memory at
import time. To remove: delete this folder, restart.
