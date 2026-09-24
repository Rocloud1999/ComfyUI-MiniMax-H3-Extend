"""Backports MiniMax H3 video-extend (continuation) support onto stock/public
ComfyUI, which doesn't have it -- only kat3ri/ComfyUI's
feat/minimax-h3-video-extend branch does (confirmed 2026-08-11 against a
real "clean" master-branch ComfyUI: stock's PackedLayout hard-rejects any
keyframe anchor except exactly frame 0 or frame_count-1, and stock's
MiniMaxH3.extra_conds drops keyframe-contributed cond latents entirely
whenever ref_images are ALSO present in the same call, since it overwrites
cond_video_latents from refs instead of appending to it).

ComfyUI >= 0.34 needs none of this: its PackedLayout places keyframes at any
(even negative) resolved_frame_index, carries multi-frame and audio keyframes,
and appends ref latents after keyframe ones. There nodes.py expresses context
as plain native keyframes and apply() is never called.

On older ComfyUI (<= 0.33.x) two patches are installed lazily, the first time
MiniMaxH3VideoExtendPatched actually runs -- never at import -- and both pass
straight through to stock's own code unless the keyframes carry this pack's
kind="context"/"context_audio" markers, so workflows that don't use this pack
are unaffected even after installation. Skipped outright if the native
MiniMaxH3VideoExtend class already exists (a fork with real support):

1. comfy.ldm.minimax.model.PackedLayout.__init__ -- adds handling for
   keyframe dicts carrying kind="context" (trailing video latent frames of a
   prior clip, placed at negative RoPE-time positions counting backward from
   the target's own frame 0 via the same FRAME_PER_TOKEN cycle ordinary
   frame-to-frame spacing already uses) and kind="context_audio" (the audio
   counterpart). Ported from the fork's real PackedLayout, but deliberately
   narrower: no temporal_stretch, no scene3d/world-latent grounding, no
   per-block noise-augmentation strength override -- those are separate
   fork features layered on top of the same mechanism, not needed for basic
   "continue a prior clip" to work. The one non-context-specific change is
   generalizing keyframe anchoring from a hardcoded text_len origin to a
   target_origin that also accounts for ref_images/ref_videos/ref_audios
   being present in the SAME call (stock's own two node types never combine
   keyframes and refs, so it never needed this; when refs is empty/None,
   target_origin == text_len exactly, so stock's own existing call patterns
   are unaffected bit-for-bit).

2. comfy.model_base.MiniMaxH3.extra_conds -- fixes the overwrite-not-append
   bug above so a call combining context keyframes AND refs (which extend
   with cast/character references does, by design) actually embeds BOTH
   contributions' real latent data instead of silently dropping the
   keyframes' half.

STATUS: ported from the fork's actual model code and reasoned through
carefully (see this pack's README.md for the full trace), but not yet
verified against a live reference render. Positional-encoding math that's
subtly wrong produces plausible-looking but incorrect output, not an error --
confirm this actually matches expected behavior before trusting it for real
work, the same way everything else in this project needed live verification
rather than being trusted from source-reading alone.
"""

import inspect

import torch

import comfy.ldm.minimax.model as h3model
import comfy.model_base as model_base
import comfy.conds

# captured before apply() can replace them, so detection and pass-through
# always see stock's own code
_stock_packed_layout_init = h3model.PackedLayout.__init__
_stock_extra_conds = model_base.MiniMaxH3.extra_conds
_applied = False

CONTEXT_KINDS = ("context", "context_audio")


def _native_has_video_extend() -> bool:
    """A real native class, not this pack's own inject_into_native() shim --
    that runs at import, before apply() ever gets a chance to look."""
    import comfy_extras.nodes_minimax_h3 as native
    cls = getattr(native, "MiniMaxH3VideoExtend", None)
    return cls is not None and not getattr(cls, "_h3_extend_shim", False)


def native_keyframes_supported() -> bool:
    """True on ComfyUI >= 0.34, whose PackedLayout anchors keyframes at an
    arbitrary resolved_frame_index; older layouts take a frame_count argument
    and hard-reject anything but the first/last frame."""
    return "frame_count" not in inspect.signature(_stock_packed_layout_init).parameters


def _is_extend_call(keyframes) -> bool:
    return bool(keyframes) and any(kf.get("kind") in CONTEXT_KINDS for kf in keyframes)


def _refs_cursor_delta(refs):
    """Mirrors the cursor math the (otherwise unmodified) refs loop below
    will apply, computed ahead of time so context keyframes -- built earlier
    in row order -- can anchor at the target's true eventual origin instead
    of the pre-refs text_len."""
    delta = 0.0
    for blk in refs:
        kind = blk["kind"]
        if kind == "image":
            delta += 1.0
        elif kind == "audio":
            delta += float(blk["ref_audio_t"])
        elif kind in ("video", "video_audio"):
            delta += max(float(blk["ref_audio_t"]), sum(h3model._video_t_spans(blk["latent_t"])))
    return delta


def _context_k_distance(k):
    """Distance from target_origin for context slot k (k<=0): k=0 is the
    hard zero-RoPE-distance anchor (the same trick that makes first-frame
    keyframes lock identity so reliably), k=-m sums the natural per-step
    FRAME_PER_TOKEN cycle for m steps back."""
    if k >= 0:
        return 0.0
    m = -k
    return sum(h3model.FRAME_RESCALE * h3model.FRAME_PER_TOKEN[(-i) % 5] for i in range(1, m + 1))


def _patched_packed_layout_init(self, text_len, latent_t, latent_h, latent_w, audio_t,
                                 keyframes=None, refs=None, frame_count=None):
    if not _is_extend_call(keyframes):
        return _stock_packed_layout_init(self, text_len, latent_t, latent_h, latent_w, audio_t,
                                         keyframes=keyframes, refs=refs, frame_count=frame_count)
    frame, w_grid = h3model._frame_grid(latent_h, latent_w)
    frame_rows = frame.shape[0]

    segments = [("text", text_len)]  # (kind, n_rows) -- stock's original 2-tuple shape, unchanged
    g = torch.zeros(text_len, 3, dtype=torch.float64)
    g[:, 0] = torch.arange(text_len, dtype=torch.float64)
    pos = [g]

    img_pos, img_update = [], []
    audio_pos, audio_update = [], []
    cursor = text_len
    row = text_len

    # generalization of stock's hardcoded text_len keyframe origin: equals
    # text_len exactly when refs is empty/None, so stock's own existing call
    # patterns (keyframes XOR refs, never both) are unaffected bit-for-bit
    target_origin = float(text_len) + (_refs_cursor_delta(refs) if refs else 0.0)
    context_k_cursor = 0
    audio_context_cursor = target_origin

    if keyframes:
        for kf in keyframes:
            if kf.get("kind") == "context":
                n_frames = kf["num_frames"]
                ks = range(context_k_cursor - n_frames + 1, context_k_cursor + 1)
                t_grid = torch.tensor([target_origin - _context_k_distance(k) for k in ks], dtype=torch.float64)
                context_k_cursor -= n_frames
                g = torch.empty(n_frames, frame_rows, 3, dtype=torch.float64)
                g[:, :, 0] = t_grid[:, None]
                g[:, :, 1:] = frame[None]
                n = n_frames * frame_rows
                segments.append(("cond", n))
                pos.append(g.reshape(-1, 3))
                img_pos.append(torch.arange(row, row + n))
                img_update.append(torch.zeros(n, dtype=torch.bool))
                row += n
                continue
            if kf.get("kind") == "context_audio":
                rt = kf["num_frames"]
                segments.append(("ref_audio", rt * 2))
                pos.append(h3model._audio_grid(audio_context_cursor - rt, rt, float(w_grid[0]), float(w_grid[-1])))
                audio_pos.append(torch.arange(row, row + rt * 2))
                audio_update.append(torch.zeros(rt * 2, dtype=torch.bool))
                audio_context_cursor -= rt
                row += rt * 2
                continue
            pixel_index = kf["resolved_frame_index"]
            if pixel_index == 0:
                cond_t = target_origin
            elif frame_count is not None and pixel_index == frame_count - 1:
                cond_t = target_origin + sum(h3model._video_t_spans(latent_t)) - h3model.FRAME_RESCALE
            else:
                raise ValueError("only first/last keyframe anchors are supported")
            g = torch.empty(frame_rows, 3, dtype=torch.float64)
            g[:, 0] = cond_t
            g[:, 1:] = frame
            segments.append(("cond", frame_rows))
            pos.append(g)
            img_pos.append(torch.arange(row, row + frame_rows))
            img_update.append(torch.zeros(frame_rows, dtype=torch.bool))
            row += frame_rows

    # everything below is stock's original refs/audio/video logic, untouched
    target_audio_w = (float(w_grid[0]), float(w_grid[-1]))
    if refs:
        cursor = float(text_len)
        for blk in refs:
            kind = blk["kind"]
            if kind == "image":
                r_frame, _ = h3model._frame_grid(blk["latent_h"], blk["latent_w"])
                n = r_frame.shape[0]
                g = torch.empty(n, 3, dtype=torch.float64)
                g[:, 0] = cursor
                g[:, 1:] = r_frame
                segments.append(("ref_img", n))
                pos.append(g)
                img_pos.append(torch.arange(row, row + n))
                img_update.append(torch.zeros(n, dtype=torch.bool))
                row += n
                cursor += 1.0
            elif kind == "audio":
                rt = blk["ref_audio_t"]
                if rt > 0:
                    segments.append(("ref_audio", rt * 2))
                    pos.append(h3model._audio_grid(cursor, rt, *target_audio_w))
                    audio_pos.append(torch.arange(row, row + rt * 2))
                    audio_update.append(torch.zeros(rt * 2, dtype=torch.bool))
                    row += rt * 2
                cursor += float(rt)
            elif kind in ("video", "video_audio"):
                rt = blk["ref_audio_t"]
                vt = blk["latent_t"]
                r_frame, r_w_grid = h3model._frame_grid(blk["latent_h"], blk["latent_w"])
                if rt > 0:
                    segments.append(("ref_audio", rt * 2))
                    pos.append(h3model._audio_grid(cursor, rt, float(r_w_grid[0]), float(r_w_grid[-1])))
                    audio_pos.append(torch.arange(row, row + rt * 2))
                    audio_update.append(torch.zeros(rt * 2, dtype=torch.bool))
                    row += rt * 2
                n = vt * r_frame.shape[0]
                segments.append(("ref_img", n))
                pos.append(h3model._video_grid(vt, r_frame, cursor))
                img_pos.append(torch.arange(row, row + n))
                img_update.append(torch.zeros(n, dtype=torch.bool))
                row += n
                cursor += max(float(rt), sum(h3model._video_t_spans(vt)))

    segments.append(("audio", audio_t * 2))
    pos.append(h3model._audio_grid(cursor, audio_t, *target_audio_w))
    audio_pos.append(torch.arange(row, row + audio_t * 2))
    audio_update.append(torch.ones(audio_t * 2, dtype=torch.bool))
    row += audio_t * 2

    n_video = latent_t * frame_rows
    segments.append(("video", n_video))
    pos.append(h3model._video_grid(latent_t, frame, cursor))
    img_pos.append(torch.arange(row, row + n_video))
    img_update.append(torch.ones(n_video, dtype=torch.bool))
    row += n_video

    self.seq_len = row
    self.position_ids = torch.cat(pos)
    self.img_pos = torch.cat(img_pos)
    self.img_update = torch.cat(img_update)
    self.audio_pos = torch.cat(audio_pos)
    self.audio_update = torch.cat(audio_update)
    self.signature = (text_len, latent_t, latent_h, latent_w, audio_t)
    seg_abs = []
    off = 0
    for kind, n in segments:
        seg_abs.append((off, off + n, kind))
        off += n
    self.segments = seg_abs


def _patched_extra_conds(self, **kwargs):
    if not _is_extend_call(kwargs.get("minimax_keyframes", None)):
        return _stock_extra_conds(self, **kwargs)
    out = model_base.BaseModel.extra_conds(self, **kwargs)
    cross_attn = kwargs.get("cross_attn", None)
    if cross_attn is not None:
        cross_attn = self.diffusion_model.preprocess_text_embeds(
            cross_attn.to(device=kwargs["device"], dtype=self.get_dtype_inference()))
        out['c_crossattn'] = comfy.conds.CONDRegular(cross_attn)

    latent_shapes = kwargs.get("latent_shapes", None)
    if latent_shapes is not None:
        out['latent_shapes'] = comfy.conds.CONDConstant(latent_shapes)

    payload = {}
    tags = kwargs.get("minimax_token_tags", None)
    if tags is not None:
        payload["text_token_tags"] = tags
    keyframes = kwargs.get("minimax_keyframes", None)
    if keyframes is not None:
        payload["keyframes"] = keyframes
        payload["frame_count"] = kwargs.get("minimax_frame_count", None)
        # not every keyframe kind carries both -- context is video-only,
        # context_audio is audio-only (unlike stock's own keyframes, which
        # always carry "latent" and never "audio_latent")
        payload["cond_video_latents"] = [kf["latent"] for kf in keyframes if "latent" in kf]
        kf_audio = [kf["audio_latent"] for kf in keyframes if "audio_latent" in kf]
        if kf_audio:
            payload["cond_audio_latents"] = kf_audio
    refs = kwargs.get("minimax_refs", None)
    if refs is not None:
        payload["refs"] = refs
        # THE FIX: append to whatever keyframes already contributed, not
        # overwrite -- stock's original always did a plain `=` assignment
        # here, which silently drops keyframe-contributed cond latents the
        # moment refs are also present in the same call.
        payload["cond_video_latents"] = payload.get("cond_video_latents", []) + \
            [r["latent"] for r in refs if "latent" in r]
        payload["cond_audio_latents"] = payload.get("cond_audio_latents", []) + \
            [r["audio_latent"] for r in refs if r.get("audio_latent") is not None]
    if kwargs.get("minimax_visual_cond_noise_aug", None) is not None:
        payload["visual_cond_noise_aug"] = kwargs["minimax_visual_cond_noise_aug"]
    if kwargs.get("minimax_audio_cond_noise_aug", None) is not None:
        payload["audio_cond_noise_aug"] = kwargs["minimax_audio_cond_noise_aug"]
    payload["seed"] = kwargs.get("seed", 0)
    payload["audio_scale"] = self.audio_scale()
    if cross_attn is not None and latent_shapes is not None and len(latent_shapes) > 1:
        vs = latent_shapes[0]
        payload["layout"] = h3model.PackedLayout(
            cross_attn.shape[1], vs[2], (vs[3] + 1) // 2 * 2, (vs[4] + 1) // 2 * 2,
            latent_shapes[1][-1], keyframes=payload.get("keyframes"),
            refs=payload.get("refs"), frame_count=payload.get("frame_count"))
    out['minimax_payload'] = comfy.conds.CONDConstant(payload)
    return out


def apply() -> bool:
    """Idempotent. Returns True if the (pass-through-gated) patches are
    installed, False if not needed: native keyframe support (ComfyUI >= 0.34)
    or a native MiniMaxH3VideoExtend class already present."""
    global _applied
    if _applied:
        return True
    if native_keyframes_supported() or _native_has_video_extend():
        return False
    h3model.PackedLayout.__init__ = _patched_packed_layout_init
    model_base.MiniMaxH3.extra_conds = _patched_extra_conds
    _applied = True
    print("[ComfyUI-MiniMax-H3-Extend] Installed legacy PackedLayout/extra_conds patches "
          "(ComfyUI <= 0.33.x); non-extend H3 workflows pass through to stock code unchanged.")
    return True
