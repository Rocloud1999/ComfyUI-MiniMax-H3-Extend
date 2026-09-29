"""CPU unit tests. ComfyUI, tokenizers and VAEs are mocked; no render is tested.

Run from the repository root: python -m unittest discover -s tests -v
Requires PyTorch, but no ComfyUI installation, model weights or GPU.
"""

import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch as mock_patch

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_h3_endpoint_test_pack"


def module(name):
    result = types.ModuleType(name)
    result.__path__ = []
    return result


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


class EndpointTests(unittest.TestCase):
    def setUp(self):
        self.make_environment()

    def make_environment(self, modern=False):
        # All global monkey patches are isolated to fake modules per test.
        modules = {name: module(name) for name in (
            "comfy", "comfy.ldm", "comfy.ldm.minimax", "comfy.ldm.minimax.model",
            "comfy.model_base", "comfy.conds", "comfy_extras",
            "comfy_extras.nodes_minimax_h3", "node_helpers", PACKAGE, PACKAGE + ".nodes",
        )}
        for name, value in modules.items():
            if "." in name:
                parent, child = name.rsplit(".", 1)
                setattr(modules[parent], child, value)
        modules[PACKAGE].__path__ = [str(ROOT)]
        modules[PACKAGE].__package__ = PACKAGE
        cleanup = mock_patch.dict(sys.modules, modules)
        cleanup.start()
        self.addCleanup(cleanup.stop)
        h3 = modules["comfy.ldm.minimax.model"]
        self.h3 = h3
        h3.FRAME_RESCALE = 40 / 24
        h3.FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
        h3._video_t_spans = lambda n: [h3.FRAME_RESCALE * h3.FRAME_PER_TOKEN[i % 5] for i in range(n)]

        def frame_grid(h, w):
            rows = torch.cartesian_prod(torch.arange(h // 2), torch.arange(w // 2)).double()
            return rows, torch.arange(w // 2).double()

        def video_grid(n, frame, cursor):
            starts = torch.tensor([0.0] + h3._video_t_spans(n - 1)).cumsum(0).double() + cursor
            result = torch.zeros(n, len(frame), 3, dtype=torch.float64)
            result[:, :, 0] = starts[:, None]
            result[:, :, 1:] = frame[None]
            return result.reshape(-1, 3)

        def audio_grid(cursor, n, w0, w1):
            result = torch.zeros(n * 2, 3, dtype=torch.float64)
            result[:, 0] = torch.arange(n).repeat_interleave(2) + cursor
            return result

        h3._frame_grid, h3._video_grid, h3._audio_grid = frame_grid, video_grid, audio_grid

        class PackedLayout:
            def __init__(self, text_len, latent_t, latent_h, latent_w, audio_t,
                         keyframes=None, refs=None, frame_count=None):
                self.used_stock = True

        if modern:
            def modern_init(self, text_len, latent_t, latent_h, latent_w, audio_t,
                            keyframes=None, refs=None):
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

        self.model_class = MiniMaxH3
        self.stock_extra = MiniMaxH3.extra_conds
        modules["comfy.model_base"].BaseModel = BaseModel
        modules["comfy.model_base"].MiniMaxH3 = MiniMaxH3

        class Condition:
            def __init__(self, cond):
                self.cond = cond

        modules["comfy.conds"].CONDRegular = Condition
        modules["comfy.conds"].CONDConstant = Condition
        native = modules["comfy_extras.nodes_minimax_h3"]
        self.native = native

        def empty_av(width, height, length):
            count = length + (5 - length) % 17
            t = 2 + (count - 5) // 17 * 5
            samples = types.SimpleNamespace(is_nested=True, tensors=(
                torch.zeros(1, 24, t, height // 16, width // 16),
                torch.zeros(1, 32, 2, round(count / 24 * 40)),
            ))
            return {"samples": samples}, count

        def resize(image, width, height, crop):
            # Deliberately not a test of ComfyUI's Lanczos/cover-crop implementation.
            return F.interpolate(image[..., :3].movedim(-1, 1), (height, width)).movedim(1, -1)

        native._empty_av_latent = Mock(side_effect=empty_av)
        native._resize = Mock(side_effect=resize)
        modules["node_helpers"].conditioning_set_values = lambda cond, values: [
            [embedding, {**metadata, **values}] for embedding, metadata in cond
        ]
        self.vae = types.SimpleNamespace(encode=Mock(side_effect=lambda image: torch.full(
            (1, 24, 1, image.shape[1] // 16, image.shape[2] // 16), float(image.mean())
        )))
        self.clip = types.SimpleNamespace(
            tokenize=Mock(return_value="tokens"),
            encode_from_tokens_scheduled=Mock(return_value=[[torch.zeros(1, 3, 4), {"preserved": 7}]]),
        )
        # Test the wrapper's contract with the unchanged shared ref helper,
        # rather than reimplementing/testing the upstream reference encoder.
        self.ref_blocks = [{"kind": "image", "latent": torch.ones(1, 24, 1, 2, 2),
                            "latent_h": 2, "latent_w": 2}]
        self.ref_items = [{"type": "image", "data": torch.ones(1, 32, 32, 3)}]
        nodes = modules[PACKAGE + ".nodes"]
        nodes._build_ref_blocks = Mock(return_value=(self.ref_items, self.ref_blocks))
        nodes.NODE_CLASS_MAPPINGS = {"MiniMaxH3VideoExtendPatched": object(), "MiniMaxH3EncodeAVPatched": object()}
        nodes.NODE_DISPLAY_NAME_MAPPINGS = {key: key for key in nodes.NODE_CLASS_MAPPINGS}
        nodes.inject_into_native = Mock()
        self.nodes = nodes
        self.patch = load(PACKAGE + ".patch", ROOT / "patch.py")
        self.endpoint_module = load(PACKAGE + ".endpoint_nodes", ROOT / "endpoint_nodes.py")
        self.node = self.endpoint_module.MiniMaxH3ReferenceToVideoWithEndpointsPatched()
        self.first = torch.full((1, 32, 32, 3), 0.2)
        self.last = torch.full((1, 32, 32, 3), 0.8)

    def run_node(self, **kwargs):
        inputs = dict(clip=self.clip, vae=self.vae, prompt="A continuous shot",
                      width=64, height=32, length=124, first_frame=self.first, last_frame=self.last)
        inputs.update(kwargs)
        return self.node.run(**inputs)

    def test_no_context_socket_and_registry_import(self):
        schema = self.node.INPUT_TYPES()
        self.assertNotIn("context_latent", schema["required"] | schema["optional"])
        package = sys.modules[PACKAGE]
        original_keys = set(self.nodes.NODE_CLASS_MAPPINGS)
        exec(compile((ROOT / "__init__.py").read_text(), "__init__.py", "exec"), package.__dict__)
        self.assertIn("MiniMaxH3ReferenceToVideoWithEndpointsPatched", package.NODE_CLASS_MAPPINGS)
        self.assertTrue(original_keys.issubset(package.NODE_CLASS_MAPPINGS))
        self.assertEqual(set(self.nodes.NODE_CLASS_MAPPINGS), original_keys)
        self.nodes.inject_into_native.assert_called_once()
        self.assertIs(self.h3.PackedLayout.__init__, self.stock_layout)

    def test_two_endpoints_without_references(self):
        positive, latent, count = self.run_node()
        meta = positive[0][1]
        self.assertEqual(count, 124)
        self.assertEqual([k["resolved_frame_index"] for k in meta["minimax_keyframes"]], [0, 123])
        self.assertEqual(meta["minimax_frame_count"], count)
        self.assertEqual(meta["preserved"], 7)
        self.assertNotIn("minimax_refs", meta)
        self.assertTrue(all("kind" not in k for k in meta["minimax_keyframes"]))
        self.assertEqual(latent["samples"].tensors[0].shape, (1, 24, 37, 2, 4))
        self.nodes._build_ref_blocks.assert_not_called()
        self.clip.tokenize.assert_called_once_with("A continuous shot", minimax_ref_items=[])

    def test_length_alignment_and_single_endpoints(self):
        for requested, expected in ((5, 5), (6, 22), (124, 124), (125, 141), (362, 362)):
            for first, last in ((self.first, None), (None, self.last)):
                with self.subTest(length=requested, first=first is not None):
                    positive, _, count = self.run_node(length=requested, first_frame=first, last_frame=last)
                    self.assertEqual(count, expected)
                    keyframes = positive[0][1]["minimax_keyframes"]
                    self.assertEqual(len(keyframes), 1)
                    self.assertEqual(keyframes[0]["resolved_frame_index"], 0 if first is not None else count - 1)

    def test_endpoint_batch_uses_first_rgb_and_expected_resize_modes(self):
        batch = torch.cat([torch.cat([self.first, self.first[..., :1]], -1)] * 2)
        self.run_node(first_frame=batch)
        calls = self.native._resize.call_args_list
        self.assertEqual(calls[0].args[0].shape[0], 1)
        self.assertEqual([call.args[3] for call in calls], ["disabled", "center"])
        self.assertEqual(self.vae.encode.call_args_list[0].args[0].shape, (1, 32, 64, 3))

    def test_numbered_refs_and_video_audio_are_forwarded_separately(self):
        images = torch.cat([self.first, self.last])
        video = self.first.repeat(5, 1, 1, 1)
        soundtrack, audio, audio_vae = {"track": "video"}, {"track": "standalone"}, object()
        positive, _, count = self.run_node(ref_images=images, ref_video=video,
                                         ref_video_audio=soundtrack, ref_audio=audio, audio_vae=audio_vae)
        args = self.nodes._build_ref_blocks.call_args.args
        self.assertEqual(args[4], count)
        self.assertEqual(list(args[6]), ["ref_image_1", "ref_image_2"])
        self.assertTrue(torch.equal(args[6]["ref_image_2"], images[1:2]))
        self.assertIs(args[7]["ref_video_1"], video)
        self.assertIs(args[8]["ref_video_audio_1"], soundtrack)
        self.assertIs(args[9]["ref_audio_1"], audio)
        self.assertIs(positive[0][1]["minimax_refs"], self.ref_blocks)
        self.assertIs(self.clip.tokenize.call_args.kwargs["minimax_ref_items"], self.ref_items)

    def test_invalid_inputs_fail_before_encoding_or_patching(self):
        cases = (
            {"first_frame": None, "last_frame": None}, {"width": 63}, {"height": 0},
            {"length": 4}, {"length": True}, {"ref_image_size": "other"},
            {"first_frame": torch.zeros(0, 32, 32, 3)}, {"last_frame": torch.zeros(32, 32, 3)},
            {"ref_images": torch.zeros(1, 32, 32, 1)}, {"ref_audio": {}},
            {"ref_video_audio": {}, "audio_vae": object()},
            {"ref_video": self.first.repeat(4, 1, 1, 1)},
            {"ref_video": self.first.repeat(5, 1, 1, 1), "ref_video_audio": {}},
        )
        for inputs in cases:
            with self.subTest(inputs=list(inputs)), self.assertRaises(ValueError):
                self.run_node(**inputs)
        self.vae.encode.assert_not_called()
        self.assertIs(self.h3.PackedLayout.__init__, self.stock_layout)

    def test_legacy_gate_is_scoped_to_pack_owned_keyframes(self):
        gate = self.patch._is_extend_call
        self.assertFalse(gate(None))
        self.assertFalse(gate([]))
        self.assertFalse(gate([{"resolved_frame_index": 0}]))
        self.assertFalse(gate([{self.patch.ENDPOINT_MARKER: False}]))
        self.assertTrue(gate([{"kind": "context"}]))
        self.assertTrue(gate([{"kind": "context_audio"}]))
        self.assertTrue(gate([{self.patch.ENDPOINT_MARKER: True, "resolved_frame_index": 0}]))

    def test_legacy_both_dispatchers_and_latent_order(self):
        positive, latent, _ = self.run_node(ref_images=self.first)
        metadata = positive[0][1]
        model = self.model_class()
        shapes = [tuple(t.shape) for t in latent["samples"].tensors]
        payload = model.extra_conds(**metadata, cross_attn=torch.zeros(1, 3, 4),
                                   latent_shapes=shapes, device="cpu")["minimax_payload"].cond
        self.assertEqual(len(payload["cond_video_latents"]), 3)
        self.assertIs(payload["cond_video_latents"][0], metadata["minimax_keyframes"][0]["latent"])
        self.assertIs(payload["cond_video_latents"][1], metadata["minimax_keyframes"][1]["latent"])
        self.assertIs(payload["cond_video_latents"][2], self.ref_blocks[0]["latent"])
        layout = payload["layout"]
        self.assertFalse(getattr(layout, "used_stock", False))
        cond_rows = [(start, end) for start, end, kind in layout.segments if kind == "cond"]
        self.assertEqual(len(cond_rows), 2)
        target_origin = 4.0  # text length 3 + one reference image
        self.assertTrue(torch.allclose(layout.position_ids[cond_rows[0][0]:cond_rows[0][1], 0],
                                       torch.full((2,), target_origin, dtype=torch.float64)))
        expected_tail = target_origin + 123 * self.h3.FRAME_RESCALE
        self.assertAlmostEqual(float(layout.position_ids[cond_rows[1][0], 0]), expected_tail)
        self.assertFalse(bool(layout.img_update[:4].any()))
        self.assertTrue(bool(layout.img_update[-74:].all()))

    def test_legacy_unmarked_calls_still_use_stock(self):
        self.run_node()
        self.assertEqual(self.model_class().extra_conds(minimax_keyframes=[{"resolved_frame_index": 0}],
                                                       minimax_refs=self.ref_blocks), {"used_stock": True})
        layout = self.h3.PackedLayout(3, 37, 2, 4, 207, keyframes=[{"resolved_frame_index": 0}],
                                      refs=self.ref_blocks, frame_count=124)
        self.assertTrue(layout.used_stock)
        first_patch = self.h3.PackedLayout.__init__
        self.assertTrue(self.patch.apply())
        self.assertIs(self.h3.PackedLayout.__init__, first_patch)

    def test_reference_audio_offsets_and_audio_latents_are_preserved(self):
        positive, latent, _ = self.run_node()
        audio_z = torch.zeros(1, 32, 2, 8)
        metadata = positive[0][1]
        metadata["minimax_refs"] = [{"kind": "audio", "ref_audio_t": 8, "audio_latent": audio_z}]
        payload = self.model_class().extra_conds(**metadata, cross_attn=torch.zeros(1, 3, 4),
            latent_shapes=[tuple(t.shape) for t in latent["samples"].tensors], device="cpu")["minimax_payload"].cond
        self.assertEqual(len(payload["cond_video_latents"]), 2)
        self.assertIs(payload["cond_audio_latents"][0], audio_z)
        start = next(start for start, _, kind in payload["layout"].segments if kind == "cond")
        self.assertEqual(float(payload["layout"].position_ids[start, 0]), 11.0)

    def test_modern_path_does_not_install_legacy_patches(self):
        self.make_environment(modern=True)
        with mock_patch.object(self.patch, "apply", wraps=self.patch.apply) as apply:
            positive, _, _ = self.run_node(ref_images=self.first)
            apply.assert_not_called()
        self.assertIs(self.h3.PackedLayout.__init__, self.stock_layout)
        self.assertIs(self.model_class.extra_conds, self.stock_extra)
        self.assertTrue(all(self.patch.ENDPOINT_MARKER not in k for k in positive[0][1]["minimax_keyframes"]))

    def test_native_fork_guard_is_preserved(self):
        self.native.MiniMaxH3VideoExtend = type("NativeExtend", (), {})
        self.assertFalse(self.patch.apply())
        self.assertIs(self.h3.PackedLayout.__init__, self.stock_layout)
        self.assertIs(self.model_class.extra_conds, self.stock_extra)


if __name__ == "__main__":
    unittest.main()
