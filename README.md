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

3. **`MiniMaxH3ReferenceToVideoWithEndpoints`** (`endpoint_nodes.py`) --
   official-style V3 Autogrow references plus `first_frame` / `last_frame`,
   without a previous clip. Supports up to 9 independent pictures, 3 videos,
   3 paired video soundtracks and 3 standalone audio clips. Outputs
   `positive` and `latent`. This replaces the previous endpoint interface;
   there is no classic endpoint fallback or alias. Re-add the endpoint node
   in existing workflows. The original continuation nodes and `patch.py`
   are unchanged. See [usage, migration and test limitations](docs/ref2va-endpoints.md).

The following live-test status concerns the original continuation nodes,
not the Autogrow endpoint node:

Confirmed working via live testing (2026-08-11) across text-to-video,
reference-to-video, and cast-to-video continuation. Recommended starting
settings: `context_frames` 2, `ref_spacing` 1-2, `ref_decay` 0.3, `ref_ramp`
3-4 (5-6 if the prior clip had more motion than usual).

## Install

Same as any custom node pack -- clone into `custom_nodes/`, restart ComfyUI.
No core ComfyUI files are edited, and nothing in ComfyUI's model code is
touched at import time:

- **ComfyUI >= 0.34**: the extend node runs entirely on stock's own keyframe
  support (context frames become ordinary keyframes at negative frame
  indices). Nothing is patched, ever.
- **ComfyUI <= 0.33.x**: stock can't anchor keyframes anywhere but the first/
  last frame, so `PackedLayout`/`MiniMaxH3.extra_conds` patches are installed
  the first time an extend or endpoint node actually runs. Even then they
  pass straight through to stock code for any conditioning this pack didn't
  build, so other H3 workflows retain the existing pass-through behavior.

The Autogrow endpoint additionally needs ComfyUI's V3 Autogrow API and a
compatible frontend. If that API is unavailable, only the new endpoint
node is skipped; the original continuation registrations remain available.
Audio references require an audio VAE; endpoint images require a video VAE.

To remove: delete this folder, restart.

## Tests

```bash
python -m unittest discover -s tests -v
python tests/check_history_parity.py
```

Requires PyTorch. The second command reads the former endpoint implementation
from Git history (fetch the baseline commit first for a shallow checkout).
Tests cover the actual reference builder and legacy patch, using mocked
ComfyUI API/CLIP/VAE components. They do not validate frontend interaction,
real model inference or pixel-identical endpoints.
