"""Checks for the batch runner: sampling, per-image isolation, and both lines."""

import tempfile
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

import torch
from PIL import Image
from spandrel import ModelTiling

from metrics import load_metric_tensor
from runner import (BLUR_METRICS, ImageFailure, ImageResult, process_image,
                    release_device_memory, run_batch, run_ordered_batch, select_sources)
from runs import allocate_run_directory


def _write_source(directory, name, width=64, height=48, uniform=None):
    path = directory / name
    image = Image.new("RGB", (width, height))
    if uniform is not None:
        image.paste(uniform, (0, 0, width, height))
    else:
        image.putdata(
            [((x * 5) % 256, (y * 9) % 256, (x * y) % 256) for y in range(height) for x in range(width)]
        )
    image.save(path, format="PNG")
    return path


class _BicubicStubLine:
    """Stands in for the SR model: same interface, a plain resize instead."""

    scale = 4

    def __init__(self, failing=()):
        self.failing = set(failing)
        self.device = "stub"

    def run(self, lr_path, destination, expected_size):
        if lr_path.name in self.failing:
            raise RuntimeError("stub model refused this image")
        with Image.open(lr_path) as low:
            low.load()
            low.resize(expected_size, resample=Image.Resampling.BICUBIC).save(destination, format="PNG")


class SelectSourcesTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        for index in range(6):
            _write_source(self.root, f"img{index}.png", 16, 16)

    def test_takes_the_first_n_in_stable_order_by_default(self):
        selected = select_sources(self.root, limit=3)

        self.assertEqual([p.name for p in selected], ["img0.png", "img1.png", "img2.png"])

    def test_a_seed_samples_randomly_but_reproducibly(self):
        first = select_sources(self.root, limit=3, seed=42)
        again = select_sources(self.root, limit=3, seed=42)

        self.assertEqual([p.name for p in first], [p.name for p in again])
        self.assertEqual(len(first), 3)
        self.assertEqual(len(set(p.name for p in first)), 3)

    def test_a_limit_beyond_the_directory_returns_everything(self):
        self.assertEqual(len(select_sources(self.root, limit=99)), 6)


class RunBatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from perceptual import PerceptualMetric

        cls.perceptual = PerceptualMetric(device="cpu")

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.sources = self.root / "sources"
        self.sources.mkdir()
        self.run_dir = allocate_run_directory(self.root / "runs")

    def test_one_image_produces_scores_on_both_lines(self):
        source = _write_source(self.sources, "a.png", 67, 51)

        result = process_image(source, self.run_dir, _BicubicStubLine(), self.perceptual)

        self.assertIsInstance(result, ImageResult)
        self.assertEqual(result.original, (67, 51))
        self.assertEqual(result.cropped, (64, 48))
        self.assertEqual(result.low, (16, 12))
        for line in (result.sr, result.bicubic):
            self.assertGreater(line.psnr, 0)
            self.assertLessEqual(line.ssim, 1.0)
            self.assertGreaterEqual(line.lpips, 0.0)

    def test_both_lines_read_the_same_low_resolution_file(self):
        source = _write_source(self.sources, "a.png", 64, 48)

        process_image(source, self.run_dir, _BicubicStubLine(), self.perceptual)

        produced = sorted(p.relative_to(self.run_dir).as_posix() for p in self.run_dir.rglob("*.png"))
        self.assertEqual(produced, ["bicubic/a.png", "hr/a.png", "lr/a.png", "sr/a.png"])

    def test_a_failing_image_is_isolated_and_the_batch_continues(self):
        sources = [
            _write_source(self.sources, "good1.png"),
            _write_source(self.sources, "bad.png"),
            _write_source(self.sources, "good2.png"),
        ]

        results, failures = run_batch(sources, self.run_dir, _BicubicStubLine(failing={"bad.png"}), self.perceptual)

        self.assertEqual([r.source_name for r in results], ["good1.png", "good2.png"])
        self.assertEqual(len(failures), 1)
        self.assertIsInstance(failures[0], ImageFailure)
        self.assertEqual(failures[0].source_name, "bad.png")
        self.assertEqual(failures[0].stage, "sr")
        self.assertIn("refused", failures[0].reason)

    def test_an_unreadable_source_is_recorded_rather_than_crashing_the_batch(self):
        deep = self.sources / "deep.png"
        Image.new("I;16", (32, 32), 1000).save(deep, format="PNG")
        good = _write_source(self.sources, "ok.png")

        results, failures = run_batch(sorted([deep, good]), self.run_dir, _BicubicStubLine(), self.perceptual)

        self.assertEqual([r.source_name for r in results], ["ok.png"])
        self.assertEqual([(f.source_name, f.stage) for f in failures], [("deep.png", "decode")])
        self.assertIn("16", failures[0].reason)

    def test_a_flat_image_makes_the_bicubic_line_exact_and_psnr_infinite(self):
        # A constant image survives downscale and bicubic upscale unchanged, so
        # its MSE is 0 and PSNR must be recorded as inf rather than dividing.
        source = _write_source(self.sources, "flat.png", 64, 48, uniform=(120, 130, 140))

        result = process_image(source, self.run_dir, _BicubicStubLine(), self.perceptual)

        self.assertEqual(result.bicubic.psnr, float("inf"))
        self.assertAlmostEqual(result.bicubic.ssim, 1.0, delta=1e-6)

    def test_the_real_sr_line_wires_up(self):
        from sr_line import SuperResolutionLine

        source = _write_source(self.sources, "real.png", 64, 48)

        result = process_image(source, self.run_dir, SuperResolutionLine(), self.perceptual)

        with Image.open(self.run_dir / "sr" / "real.png") as restored:
            self.assertEqual(restored.size, (64, 48))
        self.assertEqual(load_metric_tensor(self.run_dir / "sr" / "real.png").shape, (1, 3, 48, 64))
        self.assertGreater(result.sr.psnr, 0)


class DeviceMemoryTests(unittest.TestCase):
    def test_the_cache_is_emptied_when_a_gpu_is_present(self):
        with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.empty_cache") as empty:
            release_device_memory()

        empty.assert_called_once_with()

    def test_nothing_is_called_on_cpu(self):
        with patch("torch.cuda.is_available", return_value=False), patch("torch.cuda.empty_cache") as empty:
            release_device_memory()

        empty.assert_not_called()


class _OrderedDescriptor:
    """No checkpoint or downloaded parameters; uses the real ordered inference."""

    def __init__(self, role, calls=None, operation=None):
        self.role = role
        self.scale = 2 if role == "sr" else 1
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.architecture = SimpleNamespace(id="Synthetic")
        self.tiling = ModelTiling.SUPPORTED
        self.size_requirements = SimpleNamespace(minimum=1, multiple_of=1, square=False)
        self.calls = calls
        self.operation = operation

    def __call__(self, image):
        if self.calls is not None:
            self.calls.append(self.role)
        if self.operation is not None:
            return self.operation(image)
        if self.role == "sr":
            return image.repeat_interleave(2, -1).repeat_interleave(2, -2) * 0.5
        return image + 0.1


class OrderedBatchTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.inputs = self.root / "input"
        self.inputs.mkdir()
        self.checkpoints = {}
        for role in ("sr", "deblur"):
            path = self.root / f"{role}.pth"
            path.write_bytes(role.encode())
            self.checkpoints[role] = path
        self.run_dir = allocate_run_directory(self.root / "runs", subdirectories=())

    def test_four_routes_keep_full_input_and_order_without_legacy_degradation(self):
        source = _write_source(self.inputs, "odd.png", 7, 5, (40, 80, 120))
        original = source.read_bytes()
        baselines = {}
        outputs = {}
        for order in (["sr"], ["deblur"], ["sr", "deblur"], ["deblur", "sr"]):
            calls = []
            with (patch("runner.load_model", side_effect=lambda path, *, role: _OrderedDescriptor(role, calls)) as loader,
                  patch("runner.decode_source", side_effect=AssertionError("legacy decode forbidden")),
                  patch("runner.downscale", side_effect=AssertionError("legacy degradation forbidden"))):
                result = run_ordered_batch([source], self.run_dir, order, self.checkpoints, baselines)
            self.assertIsNone(result["error"])
            self.assertEqual([call.kwargs["role"] for call in loader.call_args_list], order)
            self.assertEqual(calls, order)
            row = result["rows"][0]
            self.assertEqual(row["status"], "success", row)
            self.assertEqual(row["input_size"], (7, 5))
            self.assertEqual(row["output_size"], (14, 10) if "sr" in order else (7, 5))
            self.assertIs(row["before"], baselines[str(source)]["before"])
            for name in BLUR_METRICS:
                self.assertNotEqual(row["after"][name]["status"], "pending")
            self.assertEqual(row["after"]["laplacian_variance"]["value"], 0.0)
            self.assertTrue(row["after"]["laplacian_variance"]["valid"])
            if "sr" in order:
                self.assertTrue(all(item["reason"] == "跨尺寸不適用" for item in row["changes"].values()))
            expected = torch.tensor([40, 80, 120], dtype=torch.float32) / 255
            for role in order:
                expected = expected * 0.5 if role == "sr" else expected + 0.1
            with Image.open(row["output"]) as output:
                outputs[tuple(order)] = output.getpixel((0, 0))
                self.assertEqual(outputs[tuple(order)], tuple(expected.mul(255).round().int().tolist()))
            self.assertEqual(result["models"][0]["architecture"], "Synthetic")
            self.assertEqual(len(result["models"][0]["sha256"]), 64)
            self.assertGreaterEqual(result["elapsed_seconds"], 0)
            self.assertIsNone(result["resources"]["cuda_peak_allocated_bytes"])
        self.assertNotEqual(outputs[("sr", "deblur")], outputs[("deblur", "sr")])
        self.assertEqual(source.read_bytes(), original)
        self.assertEqual(set(baselines[str(source)]), {"input_size", "before"})

    def test_decode_failure_is_isolated_and_input_safety_is_preserved(self):
        bad = self.inputs / "bad.png"
        bad.write_bytes(b"broken PNG")
        good = _write_source(self.inputs, "good.png", 7, 5)
        before = {path: path.read_bytes() for path in [bad, good, *self.checkpoints.values()]}
        with patch("runner.load_model", side_effect=lambda path, *, role: _OrderedDescriptor(role)):
            result = run_ordered_batch([bad, good], self.run_dir, ["deblur"], self.checkpoints, {})
        self.assertEqual([row["status"] for row in result["rows"]], ["failed", "success"])
        self.assertEqual(result["rows"][0]["failure_stage"], "decode")
        self.assertIsNone(result["rows"][0]["output"])
        for path, contents in before.items():
            self.assertEqual(path.read_bytes(), contents)

    def test_all_same_stem_images_fail_and_other_image_runs(self):
        sources = [_write_source(self.inputs, name, 7, 5) for name in ("same.png", "same.jpg", "ok.png")]
        with patch("runner.load_model", side_effect=lambda path, *, role: _OrderedDescriptor(role)):
            result = run_ordered_batch(sources, self.run_dir, ["deblur"], self.checkpoints, {})
        self.assertEqual([row["status"] for row in result["rows"]], ["failed", "failed", "success"])
        self.assertEqual([row["failure_stage"] for row in result["rows"][:2]], ["output", "output"])
        self.assertFalse((self.run_dir / result["id"] / "same.png").exists())

    def test_id_uses_checkpoint_path_and_content_and_directory_is_never_reused(self):
        source = _write_source(self.inputs, "a.png", 7, 5)
        other = self.root / "other" / "deblur.pth"
        other.parent.mkdir()
        other.write_bytes(self.checkpoints["deblur"].read_bytes())
        with patch("runner.load_model", side_effect=lambda path, *, role: _OrderedDescriptor(role)) as loader:
            first = run_ordered_batch([source], self.run_dir, ["deblur"], self.checkpoints, {})
            saved = Path(first["rows"][0]["output"]).read_bytes()
            collision = run_ordered_batch([source], self.run_dir, ["deblur"], self.checkpoints, {})
            self.assertEqual(loader.call_count, 1)
            second = run_ordered_batch([source], self.run_dir, ["deblur"], {"deblur": other}, {})
            other.write_bytes(b"changed checkpoint bytes")
            third = run_ordered_batch([source], self.run_dir, ["deblur"], {"deblur": other}, {})
        self.assertIsNotNone(collision["error"])
        self.assertEqual(collision["rows"][0]["failure_stage"], "output")
        self.assertEqual(len({first["id"], second["id"], third["id"]}), 3)
        self.assertEqual(Path(first["rows"][0]["output"]).read_bytes(), saved)

    def test_model_load_failure_has_identity_failed_rows_and_releases_earlier_model(self):
        source = _write_source(self.inputs, "a.png", 7, 5)
        references = []

        def load(path, *, role):
            if role == "deblur":
                raise RuntimeError("bad checkpoint")
            descriptor = _OrderedDescriptor(role)
            references.append(weakref.ref(descriptor))
            return descriptor

        with patch("runner.load_model", side_effect=load):
            result = run_ordered_batch([source], self.run_dir, ["sr", "deblur"], self.checkpoints, {})
        self.assertIn("bad checkpoint", result["error"])
        self.assertEqual(result["rows"][0]["failure_stage"], "model_load")
        self.assertEqual(result["models"][1]["path"], str(self.checkpoints["deblur"]))
        self.assertEqual(len(result["models"][1]["sha256"]), 64)
        self.assertIsNone(references[0]())

    def test_inference_and_write_failures_are_distinct_and_later_images_continue(self):
        from drone_sr.image_io import write_png

        sources = [_write_source(self.inputs, f"{i}.png", 7, 5) for i in range(3)]
        calls = 0

        def fail_once(image):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("synthetic model failure")
            return image

        def write(output, destination, source):
            if source == sources[1]:
                raise OSError("synthetic disk full")
            write_png(output, destination, source)

        with (patch("runner.load_model", side_effect=lambda path, *, role: _OrderedDescriptor(role, operation=fail_once)),
              patch("runner.write_png", side_effect=write)):
            result = run_ordered_batch(sources, self.run_dir, ["deblur"], self.checkpoints, {})
        self.assertEqual([row["failure_stage"] for row in result["rows"]], ["inference", "write", None])
        self.assertEqual([row["status"] for row in result["rows"]], ["failed", "failed", "success"])

    def test_invalid_mode_is_rejected_before_model_work(self):
        with patch("runner.load_model") as loader:
            for order in ([], ["deblur", "deblur"], ["unknown"]):
                with self.assertRaises(ValueError):
                    run_ordered_batch([], self.run_dir, order, {}, {})
        loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
