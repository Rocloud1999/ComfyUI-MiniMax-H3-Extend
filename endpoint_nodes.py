"""Ref2VA endpoint conditioning without a previous clip or context latent."""

import torch


def _check_image(image, name):
    if image is None:
        return
    if (not isinstance(image, torch.Tensor) or image.ndim != 4
            or min(image.shape[:3]) < 1 or image.shape[-1] not in (3, 4)):
        raise ValueError(f"{name} must be a non-empty IMAGE tensor [N, H, W, 3 or 4]")


class MiniMaxH3ReferenceToVideoWithEndpointsPatched:
    """Combine Ref2VA references and target-timeline image anchors.

    Anchors condition generation; they do not overwrite sampled frames or
    guarantee pixel-identical endpoints. No synthetic continuation is used.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "width": ("INT", {"default": 1344, "min": 32, "max": 16384, "step": 32}),
                "height": ("INT", {"default": 768, "min": 32, "max": 16384, "step": 32}),
                "length": ("INT", {"default": 124, "min": 5, "max": 3600, "step": 17,
                                   "tooltip": "Requested frames at 24 fps; rounded up to 17k+5. frame_count reports the actual length."}),
                "ref_image_size": (["match", "max"], {"default": "match"}),
            },
            "optional": {
                "first_frame": ("IMAGE", {"tooltip": "Anchor frame 0. Uses only the first image in a batch; stretches to width/height."}),
                "last_frame": ("IMAGE", {"tooltip": "Anchor the actual final frame after length alignment. Uses only the first image; center cover-crops to width/height."}),
                "ref_images": ("IMAGE", {"tooltip": "Each image in this batch is one <Picture i>. Endpoint images do not consume reference numbers."}),
                "ref_video": ("IMAGE", {"tooltip": "One reference video as an IMAGE batch at 24 fps (at least 5 frames), not a previous clip to extend."}),
                "ref_video_audio": ("AUDIO", {"tooltip": "Optional soundtrack for ref_video; requires ref_video and audio_vae."}),
                "ref_audio": ("AUDIO", {"tooltip": "One standalone reference audio clip; requires audio_vae."}),
                "audio_vae": ("VAE",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT", "INT")
    RETURN_NAMES = ("positive", "latent", "frame_count")
    FUNCTION = "run"
    CATEGORY = "model/conditioning/minimax"
    DESCRIPTION = (
        "Ref2VA + first/last target-frame anchors, with no context_latent. "
        "Connect at least one endpoint. Conditioning is not a pixel-exact lock."
    )

    def run(self, clip, vae, prompt, width, height, length, ref_image_size="match",
            first_frame=None, last_frame=None, ref_images=None, ref_video=None,
            ref_video_audio=None, ref_audio=None, audio_vae=None):
        if first_frame is None and last_frame is None:
            raise ValueError("Connect at least first_frame or last_frame")
        for name, value in (("width", width), ("height", height)):
            if not isinstance(value, int) or isinstance(value, bool) or not 32 <= value <= 16384 or value % 32:
                raise ValueError(f"{name} must be a multiple of 32 between 32 and 16384")
        if not isinstance(length, int) or isinstance(length, bool) or not 5 <= length <= 3600:
            raise ValueError("length must be an integer between 5 and 3600")
        if ref_image_size not in ("match", "max"):
            raise ValueError("ref_image_size must be 'match' or 'max'")
        for name, image in (("first_frame", first_frame), ("last_frame", last_frame),
                            ("ref_images", ref_images), ("ref_video", ref_video)):
            _check_image(image, name)
        if ref_video is not None and ref_video.shape[0] < 5:
            raise ValueError("ref_video needs at least 5 frames at 24 fps")
        if ref_video_audio is not None and ref_video is None:
            raise ValueError("ref_video_audio requires ref_video")
        if audio_vae is None and (ref_video_audio is not None or ref_audio is not None):
            raise ValueError("audio_vae is required for reference audio")

        import node_helpers
        import comfy_extras.nodes_minimax_h3 as native
        from . import patch
        from .nodes import _build_ref_blocks

        # Keep the existing pack's feature detection and native-fork guard.
        # Both legacy dispatchers must recognize our endpoints, not just
        # extra_conds; otherwise the references shift the target time origin.
        legacy = not patch.native_keyframes_supported()
        if legacy:
            patch.apply()

        latent, frame_count = native._empty_av_latent(width, height, length)
        keyframes = []
        for image, frame_index, crop in (
            (first_frame, 0, "disabled"),
            (last_frame, frame_count - 1, "center"),
        ):
            if image is None:
                continue
            resized = native._resize(image[:1], width, height, crop)
            keyframe = {"resolved_frame_index": frame_index, "latent": vae.encode(resized)}
            if legacy:
                # A private ownership marker, NOT a fabricated context keyframe.
                keyframe[patch.ENDPOINT_MARKER] = True
            keyframes.append(keyframe)

        # The classic node API supplies tensors, while the shared helper
        # expects numbered dictionaries. Never test a multi-element tensor
        # for truthiness. Keep video frames together as ONE video reference.
        image_refs = ({f"ref_image_{i + 1}": ref_images[i:i + 1]
                       for i in range(ref_images.shape[0])} if ref_images is not None else None)
        video_refs = {"ref_video_1": ref_video} if ref_video is not None else None
        video_audio_refs = {"ref_video_audio_1": ref_video_audio} if ref_video_audio is not None else None
        audio_refs = {"ref_audio_1": ref_audio} if ref_audio is not None else None
        ref_items, ref_blocks = [], []
        if image_refs or video_refs or audio_refs:
            ref_items, ref_blocks = _build_ref_blocks(
                vae, audio_vae, width, height, frame_count, ref_image_size,
                image_refs, video_refs, video_audio_refs, audio_refs,
            )

        # Endpoints stay in target keyframes, not in the numbered Picture list.
        tokens = clip.tokenize(prompt, minimax_ref_items=ref_items)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        values = {"minimax_keyframes": keyframes, "minimax_frame_count": frame_count}
        if ref_blocks:
            values["minimax_refs"] = ref_blocks
        positive = node_helpers.conditioning_set_values(conditioning, values)
        return positive, latent, frame_count


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3ReferenceToVideoWithEndpointsPatched": MiniMaxH3ReferenceToVideoWithEndpointsPatched,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3ReferenceToVideoWithEndpointsPatched": "MiniMax H3 Reference to Video + First/Last (Backported)",
}
