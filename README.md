# ComfyUI-MiniMax-H3-Extend

Backports MiniMax H3 **video-extend** (continuation) support onto stock/public
ComfyUI installs. The native `MiniMaxH3VideoExtend` node only exists on
[kat3ri/ComfyUI](https://github.com/kat3ri/ComfyUI)'s
`feat/minimax-h3-video-extend` branch (or later) -- confirmed 2026-08-11
against a real "clean" master-branch ComfyUI that continuation isn't a
missing node so much as a missing capability in the DiT model's own
positional-encoding code, which stock's `PackedLayout` explicitly rejects:

```python
if pixel_index == 0:
    cond_t = ...
elif frame_count is not None and pixel_index == frame_count - 1:
    cond_t = ...
else:
    raise ValueError("only first/last keyframe anchors are supported")
```

## What this does

Monkey-patches (only applied if the native class is genuinely absent -- never
overrides a real fork's better native support):

1. **`comfy.ldm.minimax.model.PackedLayout.__init__`** -- adds handling for
   keyframe dicts carrying `kind="context"` (trailing video latent frames of
   a prior clip, placed at negative RoPE-time positions counting backward
   from the target's own frame 0) and `kind="context_audio"` (the audio
   counterpart). Also generalizes keyframe anchoring from a hardcoded
   `text_len` origin to one that accounts for `ref_images`/etc. being present
   in the *same* call -- stock's own two node types never combine keyframes
   and refs, so it never needed this; when refs is empty, the new origin
   equals the old one exactly, so stock's existing call patterns are
   unaffected bit-for-bit.

2. **`comfy.model_base.MiniMaxH3.extra_conds`** -- fixes a real bug this
   surfaced: stock overwrites `cond_video_latents`/`cond_audio_latents` from
   `refs` instead of appending to what keyframes already contributed. Harmless
   for stock's own node types (which never combine the two), but silently
   drops the context frames' actual latent data the moment both are present
   in one call -- which continuation-with-references does, by design.

3. **`MiniMaxH3VideoExtendPatched`** node (`nodes.py`) -- a standalone,
   directly-usable node exposing the above, plus injection of
   `MiniMaxH3VideoExtend` into `comfy_extras.nodes_minimax_h3`'s own
   namespace (again, only if absent) so anything that looks it up by that
   name there -- e.g. `ComfyUI-H3-Cast`'s `H3CastToVideoExtend` -- finds a
   working implementation transparently, no changes of its own needed.

4. **`MiniMaxH3EncodeAVPatched`** node (`nodes.py`) -- `vae`-encodes video
   frames (+ optional audio) into the AV latent `MiniMaxH3VideoExtendPatched`'s
   (or the native node's) `context_latent` input needs, for feeding an
   externally-loaded prior clip (e.g. `VHS_LoadVideo`) into either. Vendored
   from the native `MiniMaxH3EncodeAV`, which isn't part of stock/public
   support either. Lives here rather than in `ComfyUI-H3-Cast` since it's an
   extend/continuation concern, not a cast/character one -- also injected
   into the native namespace the same way as `MiniMaxH3VideoExtend` above.

## Deliberately NOT ported

The fork has diverged substantially from upstream ComfyUI (72 files,
+2616/-8194 lines as of 2026-08-11 -- much of that unrelated drift, not this
feature). Scoped down to just what's needed for "continue a prior clip,
optionally with reference images/audio at the same time":

- **`temporal_stretch`** -- spreads target frames' positions apart for a
  slow-motion-of-a-longer-clip effect. Separate mechanism, not needed for
  extend to work at all.
- **scene3d / world-latent grounding** -- a spatial-reference mechanism
  (e.g. a rendered room's own latent) layered on the same positional
  machinery. Also separate.
- **Per-block noise-augmentation strength (`aug`)** -- the fork threads a
  strength value through every segment (`(kind, n_rows, aug)` 3-tuples
  instead of stock's `(kind, n_rows)`). Dropped entirely here to avoid
  touching every downstream consumer of `PackedLayout.segments` (several
  unpacking sites elsewhere in `model.py` all expect the 2-element stock
  shape) -- context frames always run at full strength (no noise blending).
- **`context_static_time`** -- pins all context frames to the same zero-
  distance point instead of stepping them back through real time. That
  option exists specifically for the scene3d/world case ("this isn't really
  a prior moment, it's a spatial reference"), which isn't ported.

If you need any of these, the honest answer is the fork, not this pack.

## Verification status

**Ported from the fork's actual model code and reasoned through carefully,
but not yet confirmed against a live reference render.** Positional-encoding
math that's subtly wrong produces plausible-looking but *incorrect* output,
not an error -- confirm this actually matches expected continuation behavior
before trusting it for real work.

The rigorous way to check (same class as the correctness gates a colleague's
separate multi-GPU MiniMax H3 project uses for its own changes): render the
same shot's extend step on both this patch (stock ComfyUI) and the real fork,
same seed/settings, and compare. Until that's been done, treat output from
this as unverified.

## Install

Same as any custom node pack -- clone into `custom_nodes/`, restart ComfyUI.
No core ComfyUI files are edited; everything is monkey-patched in memory at
import time. To remove: delete this folder, restart.
