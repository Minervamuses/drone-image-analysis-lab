"""Checks for the averaging rules fixed by GOALS.md item 5 and the inf rule in item 2."""

import math
import unittest

from runner import ImageFailure, ImageResult, LineScores
from summary import BLUR_METRICS, summarise, summarise_mode, summarise_mode_comparisons


def _result(name, sr, bicubic):
    return ImageResult(
        source_name=name,
        original=(64, 48),
        cropped=(64, 48),
        low=(16, 12),
        sr=LineScores(*sr),
        bicubic=LineScores(*bicubic),
    )


def _by_name(summary):
    return {metric.name: metric for metric in summary.metrics}


class SummaryTests(unittest.TestCase):
    def test_averages_are_arithmetic_means_over_the_included_images(self):
        results = [
            _result("a", sr=(30.0, 0.80, 0.20), bicubic=(32.0, 0.90, 0.30)),
            _result("b", sr=(20.0, 0.60, 0.10), bicubic=(28.0, 0.70, 0.50)),
        ]

        metrics = _by_name(summarise(results, []))

        self.assertEqual(metrics["PSNR"].sr_mean, 25.0)
        self.assertEqual(metrics["PSNR"].bicubic_mean, 30.0)
        self.assertAlmostEqual(metrics["SSIM"].sr_mean, 0.70)
        self.assertAlmostEqual(metrics["LPIPS"].bicubic_mean, 0.40)

    def test_failed_images_are_counted_but_never_averaged(self):
        results = [_result("a", sr=(30.0, 0.8, 0.2), bicubic=(32.0, 0.9, 0.3))]
        failures = [ImageFailure("b", "sr", "model refused"), ImageFailure("c", "decode", "16 bits")]

        summary = summarise(results, failures)

        self.assertEqual(summary.included, 1)
        self.assertEqual(summary.failed, 2)
        self.assertEqual(_by_name(summary)["PSNR"].counted, 1)

    def test_an_infinite_psnr_drops_that_image_from_the_psnr_mean_only(self):
        results = [
            _result("a", sr=(30.0, 0.80, 0.20), bicubic=(float("inf"), 1.00, 0.00)),
            _result("b", sr=(20.0, 0.60, 0.10), bicubic=(28.0, 0.70, 0.50)),
        ]

        metrics = _by_name(summarise(results, []))

        self.assertEqual(metrics["PSNR"].counted, 1)
        self.assertEqual(metrics["PSNR"].excluded_infinite, 1)
        self.assertEqual(metrics["PSNR"].sr_mean, 20.0)
        self.assertEqual(metrics["PSNR"].bicubic_mean, 28.0)
        # SSIM and LPIPS still see both images.
        self.assertEqual(metrics["SSIM"].counted, 2)
        self.assertAlmostEqual(metrics["SSIM"].sr_mean, 0.70)
        self.assertEqual(metrics["LPIPS"].counted, 2)
        self.assertAlmostEqual(metrics["LPIPS"].bicubic_mean, 0.25)

    def test_an_image_infinite_on_either_line_is_dropped_from_psnr(self):
        results = [
            _result("a", sr=(float("inf"), 1.0, 0.0), bicubic=(31.0, 0.9, 0.3)),
            _result("b", sr=(20.0, 0.6, 0.1), bicubic=(28.0, 0.7, 0.5)),
        ]

        self.assertEqual(_by_name(summarise(results, []))["PSNR"].counted, 1)

    def test_every_psnr_infinite_leaves_no_mean_rather_than_a_wrong_one(self):
        results = [_result("a", sr=(float("inf"), 1.0, 0.0), bicubic=(float("inf"), 1.0, 0.0))]

        psnr = _by_name(summarise(results, []))["PSNR"]

        self.assertEqual(psnr.counted, 0)
        self.assertIsNone(psnr.sr_mean)
        self.assertEqual(psnr.winner, "n/a")

    def test_the_winner_follows_each_metric_own_direction(self):
        results = [_result("a", sr=(30.0, 0.80, 0.20), bicubic=(32.0, 0.90, 0.30))]

        metrics = _by_name(summarise(results, []))

        self.assertTrue(metrics["PSNR"].higher_is_better)
        self.assertTrue(metrics["SSIM"].higher_is_better)
        self.assertFalse(metrics["LPIPS"].higher_is_better)
        self.assertEqual(metrics["PSNR"].winner, "bicubic")
        self.assertEqual(metrics["SSIM"].winner, "bicubic")
        # Lower LPIPS is better, so SR wins here even though its number is smaller.
        self.assertEqual(metrics["LPIPS"].winner, "SR")
        self.assertAlmostEqual(metrics["LPIPS"].margin, 0.10)

    def test_equal_means_are_a_tie(self):
        results = [_result("a", sr=(30.0, 0.8, 0.2), bicubic=(30.0, 0.8, 0.2))]

        self.assertEqual(_by_name(summarise(results, []))["PSNR"].winner, "tie")

    def test_no_successful_image_produces_a_summary_rather_than_an_error(self):
        summary = summarise([], [ImageFailure("a", "decode", "broken")])

        self.assertEqual(summary.included, 0)
        self.assertEqual(summary.failed, 1)
        for metric in summary.metrics:
            self.assertIsNone(metric.sr_mean)
            self.assertEqual(metric.winner, "n/a")

    def test_the_three_metrics_are_reported_in_a_fixed_order(self):
        summary = summarise([_result("a", sr=(30.0, 0.8, 0.2), bicubic=(30.0, 0.8, 0.2))], [])

        self.assertEqual([m.name for m in summary.metrics], ["PSNR", "SSIM", "LPIPS"])

    def test_means_ignore_nothing_else(self):
        results = [_result(str(i), sr=(float(i), 0.5, 0.5), bicubic=(float(i), 0.5, 0.5)) for i in range(1, 5)]

        psnr = _by_name(summarise(results, []))["PSNR"]

        self.assertEqual(psnr.counted, 4)
        self.assertEqual(psnr.sr_mean, 2.5)
        self.assertFalse(math.isnan(psnr.sr_mean))


def _mode_metric(value, *, valid=True, reason=None, debug=None):
    return {"value": value, "valid": valid, "status": "valid" if valid else "unmeasurable",
            "reason": reason, "debug": debug or {}}


def _mode_row(source, before=1.0, after=1.0):
    row = {"input": source, "status": "success", "before": {}, "after": {}, "changes": {}}
    for name in BLUR_METRICS:
        row["before"][name] = _mode_metric(before)
        row["after"][name] = _mode_metric(after)
        ratio = name in ("laplacian_variance", "tenengrad")
        row["changes"][name] = _mode_metric(after / before if ratio else after - before)
        row["changes"][name]["kind"] = "ratio" if ratio else "delta"
    return row


def _mode_combo(identifier, rows, order=("deblur",)):
    return {"id": identifier, "order": list(order), "rows": rows}


class ModeSummaryTests(unittest.TestCase):
    def test_median_uses_per_image_ratios_and_ties_stay_in_valid_denominator(self):
        rows = [_mode_row("a", 10, 20), _mode_row("b", 100, 100), _mode_row("c", 1000, 500)]
        summary = summarise_mode(_mode_combo("A", rows))
        metric = summary["metrics"]["laplacian_variance"]

        self.assertEqual(metric["median_change"], 1.0)
        self.assertNotEqual(metric["median_change"], (20 + 100 + 500) / (10 + 100 + 1000))
        self.assertEqual((metric["improved"], metric["tied"], metric["reversed"]), (1, 1, 1))
        self.assertEqual(metric["improvement_proportion"], 1 / 3)
        self.assertEqual((metric["valid"], metric["total"]), (3, 3))
        self.assertEqual(list(summary["metrics"]), list(BLUR_METRICS))

    def test_delta_direction_and_per_metric_failure_do_not_drop_good_output(self):
        rows = [_mode_row("a", 0.8, 0.4), _mode_row("b", 0.3, 0.3), _mode_row("c", 0.1, 0.2)]
        rows[0]["changes"]["cpbd"] = _mode_metric(None, valid=False, reason="no measurable edges")
        summary = summarise_mode(_mode_combo("A", rows))

        self.assertEqual(summary["success"], 3)
        self.assertEqual(summary["failed"], 0)
        crete = summary["metrics"]["crete_roffet_blur"]
        self.assertEqual((crete["improved"], crete["tied"], crete["reversed"]), (1, 1, 1))
        self.assertEqual(crete["median_change"], 0.0)
        self.assertEqual(summary["metrics"]["cpbd"]["valid"], 2)
        self.assertEqual(summary["metrics"]["cpbd"]["exclusions"], {"no measurable edges": 1})
        self.assertEqual(summary["metrics"]["tenengrad"]["valid"], 3)

    def test_empty_invalid_missing_and_nonfinite_never_create_a_score(self):
        empty = summarise_mode(_mode_combo("A", []))
        for metric in empty["metrics"].values():
            self.assertIsNone(metric["median_change"])
            self.assertIsNone(metric["improvement_proportion"])
        row = {"input": "missing", "status": "success"}
        bad = _mode_row("nonfinite")
        bad["changes"]["cpbd"] = _mode_metric(float("nan"))
        summary = summarise_mode(_mode_combo("A", [row, bad]))
        self.assertEqual(summary["metrics"]["cpbd"]["valid"], 0)
        self.assertIn("變化值非有限數值", summary["metrics"]["cpbd"]["exclusions"])
        self.assertEqual(summary["metrics"]["tenengrad"]["valid"], 1)

    def test_processing_failure_is_excluded_even_if_stale_changes_exist(self):
        row = _mode_row("broken", 1, 10)
        row.update(status="failed", failure_stage="write", reason="disk full")
        summary = summarise_mode(_mode_combo("A", [row]))
        self.assertEqual(summary["failed"], 1)
        for metric in summary["metrics"].values():
            self.assertEqual(metric["valid"], 0)
            self.assertIn("處理失敗 (write): disk full", metric["exclusions"])

    def test_cpbd_disappeared_edges_are_counted_without_discarding_other_metrics(self):
        row = _mode_row("blurred")
        row["before"]["cpbd"]["debug"] = {"cpbd_edge_count": 50, "cpbd_valid_blocks": 1}
        row["after"]["cpbd"]["debug"] = {"cpbd_edge_count": 0, "cpbd_valid_blocks": 0}
        row["changes"]["cpbd"] = _mode_metric(None, valid=False, reason="no measurable edges after")
        summary = summarise_mode(_mode_combo("A", [row]))
        self.assertEqual(summary["metrics"]["cpbd"]["cpbd_edges_disappeared"], 1)
        self.assertEqual(summary["metrics"]["laplacian_variance"]["valid"], 1)

    def test_cross_size_modes_never_use_changes_even_if_records_are_valid(self):
        for order in (("sr",), ("sr", "deblur"), ("deblur", "sr")):
            with self.subTest(order=order):
                combo = _mode_combo("upscale", [_mode_row("a", 1, 4)], order)
                for metric in summarise_mode(combo)["metrics"].values():
                    self.assertIsNone(metric["median_change"])
                    self.assertEqual(metric["exclusions"], {"跨尺寸不適用": 1})
                self.assertEqual(summarise_mode_comparisons([combo, combo])["global"], [])

    def test_bad_third_model_does_not_clear_good_pair_common_sets(self):
        first = _mode_combo("A", [_mode_row("/input/a", 1, 2), _mode_row("/input/b", 1, 10), _mode_row("/input/c", 1, 4)])
        second = _mode_combo("B", [_mode_row("/input/b", 1, 1), _mode_row("/input/c", 1, 5), _mode_row("/input/d", 1, 7)])
        third = _mode_combo("C", [{"input": "/input/b", "status": "failed", "reason": "bad model"}])
        comparisons = summarise_mode_comparisons([first, second, third])
        self.assertEqual(comparisons["global"], [])
        self.assertEqual(comparisons["global_unavailable"], list(BLUR_METRICS))
        laplacian = next(item for item in comparisons["pairwise"] if item["metric"] == "laplacian_variance")
        self.assertEqual(laplacian["models"], ["A", "B"])
        self.assertEqual(laplacian["inputs"], ["/input/b", "/input/c"])
        self.assertEqual((laplacian["valid"], laplacian["total"], laplacian["coverage"]), (2, 4, 0.5))
        self.assertEqual(laplacian["median_changes"], {"A": 7, "B": 3})

    def test_common_sets_are_metric_specific_and_global_is_only_valid_intersection(self):
        first = _mode_combo("A", [_mode_row("a", 1, 2), _mode_row("b", 1, 4)])
        second = _mode_combo("B", [_mode_row("a", 1, 3), _mode_row("b", 1, 7)])
        first["rows"][0]["changes"]["cpbd"] = _mode_metric(None, valid=False, reason="no edges")
        comparisons = summarise_mode_comparisons([first, second])
        by_metric = {item["metric"]: item for item in comparisons["global"]}
        self.assertEqual(by_metric["laplacian_variance"]["inputs"], ["a", "b"])
        self.assertEqual(by_metric["cpbd"]["inputs"], ["b"])
        self.assertEqual(by_metric["cpbd"]["median_changes"], {"A": 3, "B": 6})


if __name__ == "__main__":
    unittest.main()
