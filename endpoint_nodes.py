"""Official-style Autogrow Ref2VA inputs plus first/last target anchors."""

from collections.abc import Mapping

import torch
from comfy_api.latest import io


def _check_image(image, name):
    if image is None:
        return
    if (not isinstance(image, torch.Tensor) or image.ndim != 4
            or min(image.shape[:3]) < 1 or image.shape[-1] not in (3, 4)):
        raise ValueError(f"{name} must be a non-empty IMAGE tensor [N, H, W, 3 or 4]")


def _reference_group(value, name, prefix, limit):
    """Validate Autogrow dictionaries without renumbering or reordering slots."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an Autogrow input group, not an IMAGE batch")
    if len(value) > limit:
        raise ValueError(f"{name} accepts at most {limit} inputs")
    for key in value:
        if (not isinstance(key, str) or not key.startswith(prefix)
                or not key[len(prefix):].isdigit()):
            raise ValueError(f"{name} contains an invalid input name: {key!r}")
    # Preserve the framework's input order, as the official reference node does.
    # Drop disconnected slots but retain each connected slot's original suffix.
    return {key: item for key, item in value.items() if item is not None}


class MiniMaxH3ReferenceToVideoWithEndpoints(io.ComfyNode):
    """Ref2VA references and target-timeline anchors, without continuation.

    This replaces the classic endpoint node; there is no legacy input adapter.
    The existing reference encoder and legacy model patch remain unchanged.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3ReferenceToVideoWithEndpoints",
            display_name="MiniMax H3 Reference to Video + First/Last",
            category="model/conditioning/minimax",
            description=(
                "Ref2VA with Autogrow references and first/last target anchors. "
                "Connect at least one endpoint. No context_latent is needed; "
                "conditioning does not guarantee pixel-identical endpoints."
            ),
            inputs=[
                io.Clip.Input("clip"),
                io.Vae.Input("vae", tooltip="H3 video VAE, required to encode endpoint and reference latents."),
                io.Vae.Input("audio_vae", optional=True, tooltip="Required when any reference soundtrack or standalone audio is connected."),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Int.Input("width", default=1344, min=32, max=16384, step=32),
                io.Int.Input("height", default=768, min=32, max=16384, step=32),
                io.Int.Input("length", default=124, min=5, max=3600, step=17,
                             tooltip="Frames at 24 fps; rounded up to 17k+5. The last anchor uses the actual aligned final frame."),
                io.Combo.Input("ref_image_size", options=["match", "max"], default="match"),
                io.Image.Input("first_frame", optional=True,
                               tooltip="Anchor frame 0. Uses the first image only, stretched to the output canvas."),
                io.Image.Input("last_frame", optional=True,
                               tooltip="Anchor the aligned final frame. Uses the first image only, with center cover-crop."),
                io.Autogrow.Input("ref_images", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_image", tooltip="One reference picture per socket; only the first image in each input batch is used."),
                        prefix="ref_image_", min=0, max=9)),
                io.Autogrow.Input("ref_videos", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_video", tooltip="One independent reference video as an IMAGE batch at 24 fps, at least 5 frames."),
                        prefix="ref_video_", min=0, max=3)),
                io.Autogrow.Input("ref_video_audios", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input("ref_video_audio", tooltip="Soundtrack of the same-numbered reference video; requires audio_vae."),
                        prefix="ref_video_audio_", min=0, max=3)),
                io.Autogrow.Input("ref_audios", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input("ref_audio", tooltip="One standalone reference audio; requires audio_vae."),
                        prefix="ref_audio_", min=0, max=3)),
            ],
            outputs=[io.Conditioning.Output(display_name="positive"), io.Latent.Output()],
        )

    @classmethod
    def execute(cls, clip, vae, prompt, width, height, length, ref_image_size="match",
                first_frame=None, last_frame=None, audio_vae=None,
                ref_images=None, ref_videos=None, ref_video_audios=None,
                ref_audios=None) -> io.NodeOutput:
        if first_frame is None and last_frame is None:
            raise ValueError("Connect at least first_frame or last_frame")
        if vae is None:
            raise ValueError("vae is required to encode endpoint and reference latents")
        for name, value in (("width", width), ("height", height)):
            if not isinstance(value, int) or isinstance(value, bool) or not 32 <= value <= 16384 or value % 32:
                raise ValueError(f"{name} must be a multiple of 32 between 32 and 16384")
        if not isinstance(length, int) or isinstance(length, bool) or not 5 <= length <= 3600:
            raise ValueError("length must be an integer between 5 and 3600")
        if ref_image_size not in ("match", "max"):
            raise ValueError("ref_image_size must be 'match' or 'max'")
        _check_image(first_frame, "first_frame")
        _check_image(last_frame, "last_frame")
        image_refs = _reference_group(ref_images, "ref_images", "ref_image_", 9)
        video_refs = _reference_group(ref_videos, "ref_videos", "ref_video_", 3)
        video_audio_refs = _reference_group(ref_video_audios, "ref_video_audios", "ref_video_audio_", 3)
        audio_refs = _reference_group(ref_audios, "ref_audios", "ref_audio_", 3)
        for name, image in image_refs.items():
            _check_image(image, name)
        for name, video in video_refs.items():
            _check_image(video, name)
            if video.shape[0] < 5:
                raise ValueError(f"{name} needs at least 5 frames at 24 fps")
        for name in video_audio_refs:
            video_name = "ref_video_" + name.rsplit("_", 1)[-1]
            if video_name not in video_refs:
                raise ValueError(f"{name} requires its matching {video_name}")
        if audio_vae is None and (video_audio_refs or audio_refs):
            raise ValueError("audio_vae is required for reference audio")

        import node_helpers
        import comfy_extras.nodes_minimax_h3 as native
        from . import patch
        from .nodes import _build_ref_blocks

        # Preserve the previous endpoint node's inference path: only the input
        # collection/API and NodeOutput wrapper change, not the payload math.
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
                keyframe[patch.ENDPOINT_MARKER] = True
            keyframes.append(keyframe)

        ref_items, ref_blocks = [], []
        if image_refs or video_refs or audio_refs:
            ref_items, ref_blocks = _build_ref_blocks(
                vae, audio_vae, width, height, frame_count, ref_image_size,
                image_refs, video_refs, video_audio_refs, audio_refs,
            )
        # Endpoints never consume numbered Picture/Video/Audio references.
        tokens = clip.tokenize(prompt, minimax_ref_items=ref_items)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        values = {"minimax_keyframes": keyframes, "minimax_frame_count": frame_count}
        if ref_blocks:
            values["minimax_refs"] = ref_blocks
        positive = node_helpers.conditioning_set_values(conditioning, values)
        return io.NodeOutput(positive, latent)
