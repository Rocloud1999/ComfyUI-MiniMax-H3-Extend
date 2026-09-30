"""CPU contract/regression tests, not a real H3 render or frontend test.

ComfyUI's API, geometry helpers, CLIP and VAEs are mocked. The repository's
actual reference builder, original continuation nodes and legacy patches run.
Run: python -m unittest discover -s tests -v
"""

import importlib.util
import inspect
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch as mock_patch

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_h3_autogrow_test_pack"


def module(name):
    result = types.ModuleType(name)
    result.__path__ = []
    return result


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    if "." in name:
        parent, child = name.rsplit(".", 1)
        setattr(sys.modules[parent], child, result)
    return result


def fake_io():
    """Small API double: tests schema arguments, not frontend expansion."""
    class Field:
        def __init__(self, id=None, **kwargs):
            self.id = id
            self.__dict__.update(kwargs)

    class ComfyNode:
        @classmethod
        def GET_SCHEMA(cls):
            return cls.define_schema()

    class NodeOutput:
        def __init__(self, *args):
            self.result = args

    result = types.SimpleNamespace(ComfyNode=ComfyNode, Schema=Field, NodeOutput=NodeOutput)
    for name in ("Clip", "Vae", "String", "Int", "Combo", "Image", "Audio", "Conditioning", "Latent"):
        setattr(result, name, types.SimpleNamespace(Input=Field, Output=Field))
    result.Autogrow = types.SimpleNamespace(Input=type("AutogrowInput", (Field,), {}), TemplatePrefix=Field)
    return result


class EndpointTests(unittest.TestCase):
    def setUp(self):
        self.make_environment()

    def make_environment(self, modern=False):
        names = ("comfy", "comfy.ldm", "comfy.ldm.minimax", "comfy.ldm.minimax.model",
                 "comfy.model_base", "comfy.conds", "comfy.nested_tensor", "comfy_extras",
                 "comfy_extras.nodes_minimax_h3", "comfy_api", "comfy_api.latest", "node_helpers", PACKAGE)
        modules = {name: module(name) for name in names}
        for name, value in modules.items():
            if "." in name:
                parent, child = name.rsplit(".", 1)
                setattr(modules[parent], child, value)
        modules[PACKAGE].__path__ = [str(ROOT)]
        modules[PACKAGE].__package__ = PACKAGE
        cleanup = mock_patch.dict(sys.modules, modules)
        cleanup.start()
        self.addCleanup(cleanup.stop)
        self.modules = modules
        self.io = modules["comfy_api.latest"].io = fake_io()
        h3 = self.h3 = modules["comfy.ldm.minimax.model"]
        h3.FRAME_RESCALE = 5 / 3
        h3.FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
        h3._video_t_spans = lambda n: [h3.FRAME_RESCALE * h3.FRAME_PER_TOKEN[i % 5] for i in range(n)]

        def frame_grid(h, w):
            return torch.cartesian_prod(torch.arange(h // 2), torch.arange(w // 2)).double(), torch.arange(w // 2).double()

        def video_grid(n, frame, cursor):
            starts = torch.tensor([0.0] + h3._video_t_spans(n - 1), dtype=torch.float64).cumsum(0) + cursor
            result = torch.zeros(n, len(frame), 3, dtype=torch.float64)
            result[:, :, 0], result[:, :, 1:] = starts[:, None], frame[None]
            return result.reshape(-1, 3)

        def audio_grid(cursor, n, low, high):
            result = torch.zeros(n * 2, 3, dtype=torch.float64)
            result[:, 0] = (torch.arange(n) + cursor).repeat(2)
            result[:n, 2], result[n:, 2] = low, high
            return result

        h3._frame_grid, h3._video_grid, h3._audio_grid = frame_grid, video_grid, audio_grid

        class PackedLayout:
            def __init__(self, text_len, latent_t, latent_h, latent_w, audio_t,
                         keyframes=None, refs=None, frame_count=None):
                self.used_stock = True

        if modern:
            def modern_init(self, text_len, latent_t, latent_h, latent_w, audio_t, keyframes=None, refs=None):
                self.used_stock = True
            PackedLayout.__init__ = modern_init
        h3.PackedLayout = PackedLayout
        self.stock_layout = PackedLayout.__init__

        class BaseModel:
            def extra_conds(self, **kwargs):
                return {}

        class MiniMaxH3(BaseModel):
            diffusion_model = types.SimpleNamespace(preprocess_text_embeds=lambda x: x)

            def extra_conds(self, **kwargs):
                return {"used_stock": True}

            def audio_scale(self):
                return 1.0

            def get_dtype_inference(self):
                return torch.float32

        self.model_class, self.stock_extra = MiniMaxH3, MiniMaxH3.extra_conds
        modules["comfy.model_base"].BaseModel = BaseModel
        modules["comfy.model_base"].MiniMaxH3 = MiniMaxH3

        class Condition:
            def __init__(self, cond):
                self.cond = cond
        modules["comfy.conds"].CONDRegular = modules["comfy.conds"].CONDConstant = Condition
        native = self.native = modules["comfy_extras.nodes_minimax_h3"]

        def empty_av(width, height, length):
            count = length + (5 - length) % 17
            samples = types.SimpleNamespace(is_nested=True, tensors=(
                torch.zeros(1, 24, 2 + (count - 5) // 17 * 5, height // 16, width // 16),
                torch.zeros(1, 32, 2, round(count / 24 * 40)),
            ))
            return {"samples": samples}, count

        def resize(image, width, height, crop):
            return F.interpolate(image[..., :3].movedim(-1, 1), (height, width)).movedim(1, -1)

        def encode(image):
            n, h, w, _ = image.shape
            t = 1 if n == 1 else 2 + (n - 5) // 17 * 5
            return torch.full((1, 24, t, h // 16, w // 16), float(image.mean()))

        native._empty_av_latent = Mock(side_effect=empty_av)
        native._resize = Mock(side_effect=resize)
        native.adapt_canvas = lambda w, h: (w, h)
        native._encode_ref_audio = Mock(side_effect=lambda vae, audio: (audio["z"], audio["z"].shape[-1]))
        native.MiniMaxH3ReferenceToVideo = type("OfficialReferenceNode", (), {"_encode_ref_audio": staticmethod(native._encode_ref_audio)})
        modules["node_helpers"].conditioning_set_values = lambda cond, values: [
            [embedding, {**metadata, **values}] for embedding, metadata in cond
        ]
        self.vae = types.SimpleNamespace(encode=Mock(side_effect=encode))
        self.clip = types.SimpleNamespace(
            tokenize=Mock(return_value="tokens"),
            encode_from_tokens_scheduled=Mock(return_value=[[torch.zeros(1, 3, 4), {"preserved": 7}]]),
        )
        # Use real repository code, including its reference loops and legacy dispatchers.
        self.nodes = load(PACKAGE + ".nodes", ROOT / "nodes.py")
        self.patch = load(PACKAGE + ".patch", ROOT / "patch.py")
        self.endpoint_module = load(PACKAGE + ".endpoint_nodes", ROOT / "endpoint_nodes.py")
        self.node = self.endpoint_module.MiniMaxH3ReferenceToVideoWithEndpoints
        self.first, self.last = self.image(0.2), self.image(0.8)

    @staticmethod
    def image(value, frames=1, height=32, width=32):
        return torch.full((frames, height, width, 3), value)

    @staticmethod
    def audio(value, length=8):
        return {"z": torch.full((1, 32, 2, length), value)}

    def run_node(self, **kwargs):
        inputs = dict(clip=self.clip, vae=self.vae, prompt="A continuous shot",
                      width=64, height=32, length=124, first_frame=self.first, last_frame=self.last)
        inputs.update(kwargs)
        result = self.node.execute(**inputs)
        self.assertIsInstance(result, self.io.NodeOutput)
        self.assertEqual(len(result.result), 2)
        return result.result

    def payload(self, positive, latent):
        return self.model_class().extra_conds(
            **positive[0][1], cross_attn=positive[0][0],
            latent_shapes=[tuple(t.shape) for t in latent["samples"].tensors], device="cpu",
        )["minimax_payload"].cond

    def import_registry(self):
        package = sys.modules[PACKAGE]
        exec(compile((ROOT / "__init__.py").read_text(encoding="utf-8"), "__init__.py", "exec"), package.__dict__)
        return package

    def test_schema_is_autogrow_with_official_limits(self):
        schema = self.node.GET_SCHEMA()
        groups = {field.id: field for field in schema.inputs if isinstance(field, self.io.Autogrow.Input)}
        expected = {"ref_images": ("ref_image_", 9), "ref_videos": ("ref_video_", 3),
                    "ref_video_audios": ("ref_video_audio_", 3), "ref_audios": ("ref_audio_", 3)}
        self.assertEqual(set(groups), set(expected))
        for name, (prefix, limit) in expected.items():
            self.assertEqual((groups[name].template.prefix, groups[name].template.min, groups[name].template.max), (prefix, 0, limit))
            self.assertTrue(groups[name].optional)
        self.assertEqual(len(schema.outputs), 2)
        fields = {field.id for field in schema.inputs}
        self.assertTrue({"first_frame", "last_frame"}.issubset(fields))
        self.assertTrue({"context_latent", "ref_video", "ref_audio", "ref_video_audio"}.isdisjoint(fields))
        self.assertNotIn("INPUT_TYPES", self.node.__dict__)
        self.assertNotIn("run", self.node.__dict__)

    def test_mixed_registry_preserves_original_nodes_without_old_endpoint_alias(self):
        originals = dict(self.nodes.NODE_CLASS_MAPPINGS)
        registry = self.import_registry().NODE_CLASS_MAPPINGS
        self.assertEqual(set(registry), set(originals) | {self.node.GET_SCHEMA().node_id})
        for name, node in originals.items():
            self.assertIs(registry[name], node)
        self.assertEqual(self.nodes.NODE_CLASS_MAPPINGS, originals)
        self.assertIs(self.native.MiniMaxH3VideoExtend, self.nodes._NativeShim)
        self.assertIs(self.h3.PackedLayout.__init__, self.stock_layout)
        self.assertIs(self.model_class.extra_conds, self.stock_extra)

    def test_missing_autogrow_does_not_remove_original_nodes(self):
        del self.io.Autogrow
        with self.assertLogs(level="WARNING"):
            registry = self.import_registry().NODE_CLASS_MAPPINGS
        self.assertEqual(registry, self.nodes.NODE_CLASS_MAPPINGS)
        self.assertIs(self.h3.PackedLayout.__init__, self.stock_layout)

    def test_no_context_and_two_endpoints_without_references(self):
        positive, latent = self.run_node()
        metadata = positive[0][1]
        self.assertEqual([k["resolved_frame_index"] for k in metadata["minimax_keyframes"]], [0, 123])
        self.assertEqual(metadata["minimax_frame_count"], 124)
        self.assertEqual(metadata["preserved"], 7)
        self.assertNotIn("minimax_refs", metadata)
        self.assertTrue(all("kind" not in k for k in metadata["minimax_keyframes"]))
        self.assertEqual(latent["samples"].tensors[0].shape, (1, 24, 37, 2, 4))
        self.clip.tokenize.assert_called_once_with("A continuous shot", minimax_ref_items=[])

    def test_length_alignment_and_single_endpoints(self):
        for requested, actual in ((5, 5), (6, 22), (124, 124), (125, 141), (362, 362)):
            for first, last in ((self.first, None), (None, self.last)):
                with self.subTest(length=requested, first=first is not None):
                    positive, _ = self.run_node(length=requested, first_frame=first, last_frame=last)
                    meta = positive[0][1]
                    self.assertEqual(meta["minimax_frame_count"], actual)
                    self.assertEqual([k["resolved_frame_index"] for k in meta["minimax_keyframes"]], [0 if first is not None else actual - 1])

    def test_endpoint_first_rgb_and_resize_modes_are_unchanged(self):
        rgba = torch.cat([self.first, self.first[..., :1]], dim=-1).repeat(2, 1, 1, 1)
        self.run_node(first_frame=rgba)
        calls = self.native._resize.call_args_list
        self.assertEqual([call.args[3] for call in calls], ["disabled", "center"])
        self.assertEqual(calls[0].args[0].shape[0], 1)
        self.assertEqual(self.vae.encode.call_args_list[0].args[0].shape, (1, 32, 64, 3))

    def test_multiple_images_keep_independent_sizes_and_take_one_per_slot(self):
        refs = {"ref_image_0": torch.cat([self.first, self.last]),
                "ref_image_1": self.image(0.6, height=64, width=32)}
        positive, _ = self.run_node(ref_images=refs)
        blocks = positive[0][1]["minimax_refs"]
        self.assertEqual([b["kind"] for b in blocks], ["image", "image"])
        self.assertEqual([(b["latent_h"], b["latent_w"]) for b in blocks], [(2, 2), (4, 2)])
        self.assertAlmostEqual(float(blocks[0]["latent"].mean()), 0.2, places=5)
        self.assertAlmostEqual(float(blocks[1]["latent"].mean()), 0.6, places=5)
        self.assertEqual(len(self.clip.tokenize.call_args.kwargs["minimax_ref_items"]), 2)

    def test_three_videos_are_encoded_and_trimmed_independently(self):
        refs = {f"ref_video_{i}": self.image(value, frames=n) for i, (value, n) in enumerate(((0.1, 5), (0.4, 26), (0.9, 41)))}
        positive, _ = self.run_node(ref_videos=refs)
        blocks = positive[0][1]["minimax_refs"]
        self.assertEqual([b["latent_t"] for b in blocks], [2, 7, 12])
        self.assertEqual([b["kind"] for b in blocks], ["video"] * 3)
        for block, value in zip(blocks, (0.1, 0.4, 0.9)):
            self.assertAlmostEqual(float(block["latent"].mean()), value, places=5)
        items = self.clip.tokenize.call_args.kwargs["minimax_ref_items"]
        self.assertEqual([len(item["data"]) for item in items], [1, 2, 4])
        self.assertEqual(items[2]["timestamps"], [0.0, 0.5, 1.0, 1.5])

    def test_each_video_is_capped_to_actual_generation_length(self):
        positive, _ = self.run_node(length=6, ref_videos={"ref_video_0": self.image(0.3, frames=39), "ref_video_1": self.image(0.7, frames=56)})
        self.assertEqual([b["latent_t"] for b in positive[0][1]["minimax_refs"]], [7, 7])
        self.assertEqual(positive[0][1]["minimax_keyframes"][-1]["resolved_frame_index"], 21)

    def test_video_soundtracks_match_suffixes_not_dictionary_order(self):
        a, b = self.audio(0.25, 8), self.audio(0.75, 9)
        positive, _ = self.run_node(
            ref_videos={"ref_video_0": self.image(0.1, frames=5), "ref_video_2": self.image(0.2, frames=22)},
            ref_video_audios={"ref_video_audio_2": b, "ref_video_audio_0": a}, audio_vae=object(),
        )
        blocks = positive[0][1]["minimax_refs"]
        self.assertIs(blocks[0]["audio_latent"], a["z"])
        self.assertIs(blocks[1]["audio_latent"], b["z"])
        self.assertEqual([i["type"] for i in self.clip.tokenize.call_args.kwargs["minimax_ref_items"]], ["audio", "video", "audio", "video"])

    def test_multiple_standalone_audio_and_empty_slots(self):
        a, b = self.audio(0.25), self.audio(0.75)
        positive, _ = self.run_node(ref_images={"ref_image_0": None}, ref_videos={"ref_video_0": None},
                                   ref_audios={"ref_audio_0": a, "ref_audio_1": None, "ref_audio_2": b}, audio_vae=object())
        blocks = positive[0][1]["minimax_refs"]
        self.assertEqual([b["kind"] for b in blocks], ["audio", "audio"])
        self.assertIs(blocks[0]["audio_latent"], a["z"])
        self.assertIs(blocks[1]["audio_latent"], b["z"])

    def test_full_reference_capacity_reaches_legacy_payload_and_layout(self):
        kwargs = dict(
            ref_images={f"ref_image_{i}": self.image((i + 1) / 10) for i in range(9)},
            ref_videos={f"ref_video_{i}": self.image((i + 1) / 10, frames=n) for i, n in enumerate((5, 22, 39))},
            ref_video_audios={f"ref_video_audio_{i}": self.audio((i + 1) / 10, 8 + i) for i in range(3)},
            ref_audios={f"ref_audio_{i}": self.audio((i + 1) / 10, 12 + i) for i in range(3)},
            audio_vae=object(),
        )
        positive, latent = self.run_node(**kwargs)
        payload = self.payload(positive, latent)
        blocks = positive[0][1]["minimax_refs"]
        self.assertEqual(len(blocks), 15)
        self.assertEqual(len(payload["cond_video_latents"]), 14)  # 2 anchors + 9 pictures + 3 videos
        self.assertEqual(len(payload["cond_audio_latents"]), 6)
        expected = [k["latent"] for k in positive[0][1]["minimax_keyframes"]] + [b["latent"] for b in blocks if "latent" in b]
        for actual, want in zip(payload["cond_video_latents"], expected):
            self.assertIs(actual, want)
        layout = payload["layout"]
        self.assertFalse(getattr(layout, "used_stock", False))
        cond_starts = [start for start, _, kind in layout.segments if kind == "cond"]
        origin = 3 + 9 + (5 + 22 + 39) * self.h3.FRAME_RESCALE + 12 + 13 + 14
        self.assertAlmostEqual(float(layout.position_ids[cond_starts[0], 0]), origin)
        self.assertAlmostEqual(float(layout.position_ids[cond_starts[1], 0]), origin + 123 * self.h3.FRAME_RESCALE)
        video_rows = sum(z.shape[2] * (z.shape[3] // 2) * (z.shape[4] // 2) for z in expected)
        self.assertEqual(int((~layout.img_update).sum()), video_rows)
        self.assertEqual(int((~layout.audio_update).sum()), sum(z.shape[-1] * 2 for z in payload["cond_audio_latents"]))
        self.assertEqual([kind for _, _, kind in layout.segments][-2:], ["audio", "video"])

    def test_request_order_and_reference_dictionary_are_not_mutated(self):
        refs = {"ref_image_2": self.last, "ref_image_0": self.first}
        positive, _ = self.run_node(ref_images=refs)
        self.assertEqual(list(refs), ["ref_image_2", "ref_image_0"])
        self.assertIs(refs["ref_image_2"], self.last)
        self.assertAlmostEqual(float(positive[0][1]["minimax_refs"][0]["latent"].mean()), 0.8, places=5)

    def test_invalid_inputs_fail_before_encoding_or_patching(self):
        cases = (
            {"first_frame": None, "last_frame": None}, {"vae": None}, {"width": 63}, {"height": 0},
            {"length": 4}, {"length": True}, {"ref_image_size": "other"},
            {"first_frame": torch.zeros(0, 32, 32, 3)}, {"last_frame": torch.zeros(32, 32, 3)},
            {"ref_images": self.first}, {"ref_videos": self.first},
            {"ref_images": {"ref_image_0": torch.zeros(1, 32, 32, 1)}},
            {"ref_videos": {"ref_video_0": self.image(0.3, frames=4)}},
            {"ref_video_audios": {"ref_video_audio_1": self.audio(0.2)}, "audio_vae": object()},
            {"ref_audios": {"ref_audio_0": self.audio(0.2)}},
            {"ref_images": {"wrong": self.first}},
            {"ref_images": {f"ref_image_{i}": self.first for i in range(10)}},
            {"ref_videos": {f"ref_video_{i}": self.image(0.3, frames=5) for i in range(4)}},
        )
        for inputs in cases:
            with self.subTest(inputs=list(inputs)), self.assertRaises(ValueError):
                self.run_node(**inputs)
        self.vae.encode.assert_not_called()
        self.assertIs(self.h3.PackedLayout.__init__, self.stock_layout)

    def test_legacy_single_input_names_are_not_accepted(self):
        self.assertTrue({"ref_video", "ref_video_audio", "ref_audio"}.isdisjoint(inspect.signature(self.node.execute).parameters))
        with self.assertRaises(TypeError):
            self.run_node(ref_video=self.image(0.1, frames=5))

    def test_legacy_dispatch_is_scoped_and_idempotent(self):
        gate = self.patch._is_extend_call
        self.assertFalse(gate(None))
        self.assertFalse(gate([{"resolved_frame_index": 0}]))
        self.assertTrue(gate([{"kind": "context"}]))
        self.assertTrue(gate([{"kind": "context_audio"}]))
        self.run_node()
        self.assertTrue(gate([{self.patch.ENDPOINT_MARKER: True}]))
        self.assertEqual(self.model_class().extra_conds(minimax_keyframes=[{"resolved_frame_index": 0}]), {"used_stock": True})
        self.assertTrue(self.h3.PackedLayout(3, 37, 2, 4, 207, keyframes=[{"resolved_frame_index": 0}], frame_count=124).used_stock)
        installed = self.h3.PackedLayout.__init__
        self.assertTrue(self.patch.apply())
        self.assertIs(installed, self.h3.PackedLayout.__init__)

    def test_modern_path_does_not_install_legacy_patches(self):
        self.make_environment(modern=True)
        with mock_patch.object(self.patch, "apply", wraps=self.patch.apply) as apply:
            positive, _ = self.run_node(ref_images={"ref_image_0": self.first}, ref_videos={"ref_video_0": self.image(0.3, frames=5)})
            apply.assert_not_called()
        self.assertIs(self.h3.PackedLayout.__init__, self.stock_layout)
        self.assertIs(self.model_class.extra_conds, self.stock_extra)
        self.assertTrue(all(self.patch.ENDPOINT_MARKER not in k for k in positive[0][1]["minimax_keyframes"]))

    def test_native_fork_guard_is_preserved(self):
        self.native.MiniMaxH3VideoExtend = type("NativeExtend", (), {})
        self.assertFalse(self.patch.apply())
        self.assertIs(self.h3.PackedLayout.__init__, self.stock_layout)
        self.assertIs(self.model_class.extra_conds, self.stock_extra)

    def test_legacy_reference_audio_encoder_fallback(self):
        encoder = self.native._encode_ref_audio
        del self.native._encode_ref_audio
        positive, _ = self.run_node(ref_audios={"ref_audio_0": self.audio(0.2)}, audio_vae=object())
        encoder.assert_called_once()
        self.assertEqual(positive[0][1]["minimax_refs"][0]["kind"], "audio")

    def test_original_continuation_node_still_uses_context_and_references(self):
        context, _ = self.native._empty_av_latent(64, 32, 124)
        positive, latent = self.nodes.MiniMaxH3VideoExtendPatched().run(
            self.clip, self.vae, context, "Continue", 124, pin_last_frame=False,
            first_frame=self.first, last_frame=self.last, ref_images=torch.cat([self.first, self.last]),
        )
        keys = positive[0][1]["minimax_keyframes"]
        self.assertEqual([k.get("kind") for k in keys[:2]], ["context", "context_audio"])
        self.assertEqual([k["resolved_frame_index"] for k in keys[2:]], [0, 123])
        payload = self.payload(positive, latent)
        self.assertEqual(len(payload["cond_video_latents"]), 5)
        self.assertEqual(len(payload["cond_audio_latents"]), 1)

    def test_original_encode_av_node_still_returns_video_latent(self):
        result, = self.nodes.MiniMaxH3EncodeAVPatched().run(self.vae, self.image(0.3, frames=22))
        self.assertEqual(result["samples"].shape, (1, 24, 7, 2, 2))


if __name__ == "__main__":
    unittest.main()
