"""Tiny deterministic checks of the vendored metric core and validity contract."""

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

import blur_metrics as metrics


def _tensor(rgb):
    return torch.from_numpy(np.array(rgb, dtype=np.uint8)).permute(2, 0, 1).unsqueeze(0).float() / 255


def _textured_rgb():
    y, x = np.indices((96, 96))
    return np.stack(((x * 3 + y * 5 + (x // 13) * 35) % 256,
                     (x * 7 + y * 2 + (y // 11) * 45) % 256,
                     (x * 2 + y * 9) % 256), axis=-1).astype(np.uint8)


def _known(values):
    return {name: {"value": float(value), "valid": True, "status": "valid", "reason": None,
                   "debug": {"cpbd_edge_count": 2, "cpbd_valid_blocks": 1} if name == "cpbd" else {}}
            for name, value in zip(metrics.BLUR_METRICS, values)}


class CoreTests(unittest.TestCase):
    def test_fixed_laplacian_and_sobel_values(self):
        # An isolated impulse has one -1020 and four 255 Laplacian responses.
        # Sobel has four axial 510 responses and eight diagonal 255 components.
        gray = np.zeros((5, 5), dtype=np.float64)
        gray[2, 2] = 255
        self.assertEqual(metrics.laplacian_variance(gray), 52020.0)
        self.assertEqual(metrics.tenengrad(gray), 62424.0)

    def test_rgb_order_and_writer_quantization(self):
        rgb = np.tile(np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8), (3, 1, 1))
        gray = metrics._tensor_gray(_tensor(rgb))
        np.testing.assert_array_equal(gray, np.tile([76., 150., 29.], (3, 1)))
        tensor = torch.full((1, 3, 3, 3), 0.5)
        np.testing.assert_array_equal(metrics._tensor_gray(tensor), np.full((3, 3), 128.0))
        self.assertEqual(gray.dtype, np.float64)

    def test_identity_all_four_metrics(self):
        tensor = _tensor(_textured_rgb())
        before = metrics.measure_tensor(tensor)
        after = metrics.measure_tensor(tensor.clone())
        self.assertEqual(tuple(before), metrics.BLUR_METRICS)
        self.assertEqual(before, after)
        for name, change in metrics.compare_measurements(before, after).items():
            self.assertTrue(change["valid"], (name, before[name]))
            self.assertEqual(change["value"], 1.0 if change["kind"] == "ratio" else 0.0)

    def test_flat_image_distinguishes_zero_and_unmeasurable(self):
        result = metrics.measure_tensor(torch.zeros((1, 3, 64, 64)))
        for name in ("laplacian_variance", "tenengrad"):
            self.assertTrue(result[name]["valid"])
            self.assertEqual(result[name]["value"], 0.0)
        self.assertEqual(result["cpbd"]["value"], 0.0)
        self.assertFalse(result["cpbd"]["valid"])
        self.assertEqual(result["cpbd"]["status"], "unmeasurable")
        self.assertEqual(result["cpbd"]["debug"]["cpbd_edge_count"], 0)
        self.assertEqual(result["cpbd"]["debug"]["cpbd_valid_blocks"], 0)
        self.assertIsNone(result["cpbd"]["debug"]["cpbd_mean_edge_width"])
        # skimage 0.26 floors Sobel responses internally and returns finite 1.
        # This is distinct from the source wrapper's removed non-finite fallback.
        self.assertEqual(result["crete_roffet_blur"]["value"], 1.0)
        self.assertTrue(result["crete_roffet_blur"]["valid"])
        json.dumps(result, allow_nan=False)

    def test_crete_empty_inner_area_is_really_nonfinite(self):
        result = metrics.measure_tensor(torch.zeros((1, 3, 3, 3)))
        self.assertTrue(math.isnan(metrics.crete_roffet_blur(np.zeros((3, 3)))))
        self.assertIsNone(result["crete_roffet_blur"]["value"])
        self.assertEqual(result["crete_roffet_blur"]["status"], "unmeasurable")
        json.dumps(result, allow_nan=False)

    def test_cpbd_narrow_edge_scores_one(self):
        profile = np.zeros(128, dtype=np.uint8)
        profile[63] = 128
        profile[64:] = 255
        rgb = np.repeat(np.tile(profile, (128, 1))[..., None], 3, axis=2)
        measurement = metrics.measure_tensor(_tensor(rgb))["cpbd"]
        self.assertGreater(measurement["debug"]["cpbd_edge_count"], 0)
        self.assertEqual(measurement["value"], 1.0)
        self.assertTrue(measurement["valid"])

    def test_cpbd_zero_with_measured_edges_is_valid(self):
        # A broad smooth vertical transition yields widths above the CPBD cutoff.
        x = np.arange(128, dtype=np.float64)
        profile = np.rint(255 / (1 + np.exp(-(x - 63) / 5))).astype(np.uint8)
        rgb = np.repeat(np.tile(profile, (128, 1))[..., None], 3, axis=2)
        measurement = metrics.measure_tensor(_tensor(rgb))["cpbd"]
        self.assertGreater(measurement["debug"]["cpbd_edge_count"], 0)
        self.assertGreater(measurement["debug"]["cpbd_valid_blocks"], 0)
        self.assertEqual(measurement["value"], 0.0)
        self.assertTrue(measurement["valid"])

    def test_cpbd_sub_block_size_retains_debug(self):
        result = metrics.measure_tensor(_tensor(_textured_rgb()[:32, :32]))["cpbd"]
        self.assertEqual(result["value"], 0.0)
        self.assertEqual(result["debug"]["cpbd_valid_blocks"], 0)
        self.assertEqual(result["debug"]["cpbd_edge_count"], 0)
        self.assertFalse(result["valid"])

    def test_crete_nonfinite_does_not_become_valid_one(self):
        with patch.object(metrics, "blur_effect", return_value=math.nan):
            self.assertTrue(math.isnan(metrics.crete_roffet_blur(np.zeros((64, 64)))))
            result = metrics.measure_tensor(_tensor(_textured_rgb()))
        self.assertIsNone(result["crete_roffet_blur"]["value"])
        self.assertEqual(result["crete_roffet_blur"]["status"], "unmeasurable")
        self.assertTrue(result["laplacian_variance"]["valid"])

    def test_one_metric_failure_preserves_other_values(self):
        with patch.object(metrics, "tenengrad", side_effect=ValueError("single metric failure")):
            result = metrics.measure_tensor(_tensor(_textured_rgb()))
        self.assertEqual(result["tenengrad"]["status"], "failed")
        self.assertIn("single metric failure", result["tenengrad"]["reason"])
        for name in ("laplacian_variance", "cpbd", "crete_roffet_blur"):
            self.assertTrue(result[name]["valid"], (name, result[name]))

    def test_common_preprocessing_failure_is_explicit(self):
        for tensor in (torch.zeros((3, 64, 64)), torch.full((1, 3, 64, 64), math.nan),
                       torch.zeros((1, 3, 2, 2))):
            with self.subTest(shape=tuple(tensor.shape)):
                result = metrics.measure_tensor(tensor)
                self.assertEqual(set(result), set(metrics.BLUR_METRICS))
                for measurement in result.values():
                    self.assertEqual(measurement["status"], "failed")
                    self.assertFalse(measurement["valid"])
                    self.assertIsNone(measurement["value"])
                    self.assertIn("preprocessing:", measurement["reason"])

    def test_failed_measurements_have_independent_records(self):
        result = metrics.failed_measurements("output absent")
        result["cpbd"]["debug"]["note"] = 1
        self.assertEqual(result["laplacian_variance"]["debug"], {})
        self.assertEqual(result["tenengrad"]["reason"], "output absent")


class ComparisonTests(unittest.TestCase):
    def test_ratio_and_delta_are_per_image(self):
        a = metrics.compare_measurements(_known((1, 4, 0.3, 0.7)), _known((3, 2, 0.5, 0.4)))
        b = metrics.compare_measurements(_known((10, 2, 0.5, 0.3)), _known((10, 4, 0.2, 0.5)))
        self.assertEqual(a["laplacian_variance"]["value"], 3.0)
        self.assertEqual(b["laplacian_variance"]["value"], 1.0)
        self.assertEqual(a["tenengrad"]["value"], 0.5)
        self.assertEqual(b["tenengrad"]["value"], 2.0)
        self.assertAlmostEqual(a["cpbd"]["value"], 0.2)
        self.assertAlmostEqual(b["cpbd"]["value"], -0.3)
        self.assertAlmostEqual(a["crete_roffet_blur"]["value"], -0.3)

    def test_zero_baselines_are_na_without_epsilon(self):
        before = _known((0, 0, 0, 0.5))
        after = _known((10, 0, 0.2, 0.25))
        result = metrics.compare_measurements(before, after)
        for name in ("laplacian_variance", "tenengrad"):
            self.assertIsNone(result[name]["value"])
            self.assertFalse(result[name]["valid"])
            self.assertIn("0", result[name]["reason"])
        self.assertTrue(result["cpbd"]["valid"])
        self.assertEqual(before["laplacian_variance"]["value"], 0)
        self.assertEqual(after["laplacian_variance"]["value"], 10)

    def test_missing_edges_on_either_side_reject_only_cpbd(self):
        for side in (0, 1):
            values = [_known((1, 1, 0, 0.5)), _known((2, 2, 0.3, 0.4))]
            values[side]["cpbd"]["debug"]["cpbd_edge_count"] = 0
            result = metrics.compare_measurements(*values)
            self.assertFalse(result["cpbd"]["valid"])
            self.assertIsNone(result["cpbd"]["value"])
            self.assertTrue(result["laplacian_variance"]["valid"])

    def test_metric_failure_remains_a_failure_in_comparison(self):
        before, after = _known((1, 1, 0, 0.5)), _known((2, 2, 0.3, 0.4))
        after["tenengrad"] = metrics.failed_measurements("broken metric")["tenengrad"]
        result = metrics.compare_measurements(before, after)
        self.assertEqual(result["tenengrad"]["status"], "failed")
        self.assertIn("after: broken metric", result["tenengrad"]["reason"])
        self.assertTrue(result["cpbd"]["valid"])

    def test_cross_size_is_not_a_change(self):
        result = metrics.compare_measurements(_known((1, 1, 0, 0.5)), _known((2, 2, 0.3, 0.4)), applicable=False)
        for change in result.values():
            self.assertFalse(change["valid"])
            self.assertIsNone(change["value"])
            self.assertEqual(change["status"], "unmeasurable")
            self.assertEqual(change["reason"], "跨尺寸不適用")


class SourceEquivalenceTests(unittest.TestCase):
    def test_optional_original_core_matches_without_csv_api_import(self):
        source = Path("/mnt/c/Users/garyc/Downloads/metrics/blur_metrics")
        if not (source / "metrics.py").exists():
            self.skipTest("Optional original Downloads core is not present; fixed numeric checks still run")
        expected = {"metrics": "ccdc3be71793d2098225978150e5310f7079b5530ae0ce8894d6bfdbe1096a37",
                    "preprocessing": "d7c643a61519d372edadabe97e75acce8c3607cd92a7cfc5ac6ada900a4cfbbc"}
        for name, digest in expected.items():
            self.assertEqual(hashlib.sha256((source / f"{name}.py").read_bytes()).hexdigest(), digest)
        package_name = "_blur_original_numeric_core"
        package = types.ModuleType(package_name)
        package.__path__ = [str(source)]
        modules = {package_name: package}
        with patch.dict(sys.modules, modules):
            for name in ("preprocessing", "metrics"):
                fullname = f"{package_name}.{name}"
                spec = importlib.util.spec_from_file_location(fullname, source / f"{name}.py")
                module = importlib.util.module_from_spec(spec)
                sys.modules[fullname] = module
                spec.loader.exec_module(module)
            original = sys.modules[f"{package_name}.metrics"]
            gray = cv2.cvtColor(_textured_rgb(), cv2.COLOR_RGB2GRAY).astype(np.float64)
            for name in ("laplacian_variance", "tenengrad", "cpbd_score", "crete_roffet_blur"):
                self.assertEqual(getattr(metrics, name)(gray), getattr(original, name)(gray), name)
            self.assertEqual(metrics._cpbd(gray), original._cpbd(gray))
            self.assertNotIn(f"{package_name}.api", sys.modules)


if __name__ == "__main__":
    unittest.main()
