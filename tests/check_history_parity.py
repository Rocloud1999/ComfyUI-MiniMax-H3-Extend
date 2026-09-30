"""Compare prepared inference inputs with the endpoint node in Git history.

No old node implementation is stored in the current tree. This reads the
historical blob and runs it in the same mocked environment as the new node.
It checks conditioning, AV latents, tokenizer items, and (legacy path only)
the real repository patch's packed payload/layout. No learned model forward
or real ComfyUI frontend is executed.

Run from the repository root: python tests/check_history_parity.py
A shallow checkout must first fetch the baseline commit. --baseline-file
also accepts an externally retrieved copy with the exact historical blob SHA.
"""

import argparse
import contextlib
import hashlib
import io
import itertools
from pathlib import Path
import subprocess
import types

import torch

from test_endpoints import EndpointTests, PACKAGE, ROOT

BASE_REF = "eeb2d66004c9d52995c7561e4860a3651d5fe72a"
BASE_BLOB = "b2887f17663d283674de41a910bb01e6fdc0b02a"


def assert_same(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_same(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assert_same(left, right)
    elif hasattr(expected, "__dict__"):
        assert_same(vars(actual), vars(expected))
    else:
        assert actual == expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-file", type=Path)
    args = parser.parse_args()
    if args.baseline_file is not None:
        source = args.baseline_file.read_bytes()
    else:
        result = subprocess.run(["git", "show", f"{BASE_REF}:endpoint_nodes.py"], cwd=ROOT,
                                check=False, capture_output=True)
        if result.returncode:
            raise SystemExit(f"Cannot read baseline from Git history. Fetch commit {BASE_REF} first.")
        source = result.stdout
    digest = hashlib.sha1(b"blob " + str(len(source)).encode() + b"\0" + source).hexdigest()
    if digest != BASE_BLOB:
        raise SystemExit("Baseline file does not match the historical Git blob; comparison aborted.")

    count = 0
    cases = itertools.product((False, True), (5, 6, 124, 125), ("both", "first", "last"),
                              ("none", "images", "video", "all"))
    for modern, length, endpoints, references in cases:
        test = EndpointTests(methodName="runTest")
        try:
            test.make_environment(modern=modern)
            baseline = types.ModuleType(PACKAGE + ".history_endpoint")
            baseline.__package__ = PACKAGE
            exec(compile(source, "git-history:endpoint_nodes.py", "exec"), baseline.__dict__)
            common = dict(clip=test.clip, vae=test.vae, prompt="Equivalent inputs", width=64, height=32,
                          length=length, first_frame=test.first if endpoints != "last" else None,
                          last_frame=test.last if endpoints != "first" else None)
            old_inputs, new_inputs = {}, {}
            if references in ("images", "all"):
                old_inputs["ref_images"] = torch.cat([test.first, test.last])
                new_inputs["ref_images"] = {"ref_image_1": test.first, "ref_image_2": test.last}
            if references in ("video", "all"):
                video = test.image(0.45, frames=39)
                old_inputs["ref_video"] = video
                new_inputs["ref_videos"] = {"ref_video_1": video}
            if references == "all":
                a, b = test.audio(0.25, 8), test.audio(0.75, 9)
                common["audio_vae"] = object()
                old_inputs.update(ref_video_audio=a, ref_audio=b)
                new_inputs.update(ref_video_audios={"ref_video_audio_1": a}, ref_audios={"ref_audio_1": b})
            with contextlib.redirect_stdout(io.StringIO()):
                old_positive, old_latent, old_count = baseline.MiniMaxH3ReferenceToVideoWithEndpointsPatched().run(**common, **old_inputs)
                old_items = test.clip.tokenize.call_args.kwargs["minimax_ref_items"]
                new_positive, new_latent = test.node.execute(**common, **new_inputs).result
                new_items = test.clip.tokenize.call_args.kwargs["minimax_ref_items"]
                assert_same(new_positive, old_positive)
                assert_same(new_latent, old_latent)
                assert_same(new_items, old_items)
                assert new_positive[0][1]["minimax_frame_count"] == old_count
                if not modern:
                    assert_same(test.payload(new_positive, new_latent), test.payload(old_positive, old_latent))
            count += 1
        except Exception as exc:
            raise AssertionError(f"Parity failed: modern={modern}, length={length}, endpoints={endpoints}, references={references}") from exc
        finally:
            test.doCleanups()
    print(f"PASS: {count} historical parity cases; mocked CLIP/VAE/API, no learned model forward.")


if __name__ == "__main__":
    main()
