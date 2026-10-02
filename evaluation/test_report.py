"""Checks checkpoint attribution and data-only evaluation reports."""

import hashlib
import re
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from report import REQUIRED_HEADER_FIELDS, RunEnvironment, describe_environment, render_failure_report, render_report
from runner import ImageFailure, ImageResult, LineScores
from summary import summarise


def _environment(run_dir):
    return RunEnvironment(
        started=datetime(2026, 9, 19, 2, 30, 0, tzinfo=timezone.utc),
        run_dir=run_dir,
        git_head="0123456789abcdef0123456789abcdef01234567",
        worktree_clean=True,
        source_directory=Path("input"),
        discovery_rule="direct children with suffix .png/.jpg/.jpeg",
        sampling="sequential",
        limit=5,
        discovered=738,
        selected=2,
        pillow_version="12.3.0",
        model_target="realesr-general-x4v3.pth",
        model_sha256="8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292",
        architecture="RealESRGAN Compact",
        model_scale=4,
        tile_size=512,
        sr_device="cpu",
        lpips_device="cpu",
        cudnn_benchmark=False,
        lpips_version="0.1.4",
        lpips_net="alex",
        lpips_linear_sha256="df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0",
        lpips_backbone_url="https://download.pytorch.org/models/alexnet-owt-7be5be79.pth",
        lpips_backbone_sha256="7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02",
    )


def _result(name, sr, bicubic):
    return ImageResult(
        source_name=name,
        original=(4056, 3040),
        cropped=(4056, 3040),
        low=(1014, 760),
        sr=LineScores(*sr),
        bicubic=LineScores(*bicubic),
    )


class ReportTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.run_dir = Path(directory.name) / "20260919T023000Z"
        self.results = [
            _result("a.JPG", sr=(27.5, 0.78, 0.21), bicubic=(29.1, 0.84, 0.36)),
            _result("b.JPG", sr=(26.0, 0.74, 0.25), bicubic=(28.4, 0.81, 0.39)),
        ]
        self.failures = [ImageFailure("c.JPG", "sr", "out of memory"), ImageFailure("d.png", "decode", "16 bits per sample")]
        self.text = render_report(_environment(self.run_dir), self.results, self.failures, summarise(self.results, self.failures))

    def test_every_required_header_field_is_present_with_a_value(self):
        for field in REQUIRED_HEADER_FIELDS:
            with self.subTest(field=field):
                match = re.search(rf"^\| {re.escape(field)} \| (.+?) \|$", self.text, re.MULTILINE)
                self.assertIsNotNone(match, f"header field missing: {field}")
                value = match.group(1).strip()
                self.assertNotIn(value, ("", "None", "nan", "—", "-"), f"header field empty: {field}")

    def test_the_header_records_what_makes_a_number_traceable(self):
        self.assertIn("0123456789abcdef0123456789abcdef01234567", self.text)
        self.assertIn(str(self.run_dir), self.text)
        self.assertIn("2026-09-19", self.text)

    def test_the_header_separates_the_device_each_metric_runs_on(self):
        self.assertIn("| PSNR／SSIM device | CPU; float64", self.text)
        self.assertIn("| LPIPS device | cpu", self.text)

    def test_the_report_contains_only_headings_and_data_tables(self):
        for line in self.text.splitlines():
            self.assertTrue(not line or line.startswith(("#", "|")), line)
        self.assertNotIn("解讀前提", self.text)
        self.assertNotIn("結論", self.text)
        self.assertNotIn("GAN 類", self.text)

    def test_checkpoint_filename_and_hash_identify_the_run(self):
        self.assertIn("| checkpoint | realesr-general-x4v3.pth |", self.text)
        self.assertIn("| 模型 SHA-256 | " + _environment(self.run_dir).model_sha256 + " |", self.text)

    def test_environment_describes_the_selected_checkpoint(self):
        checkpoint = self.run_dir.parent / "alternate.pth"
        sr_line = SimpleNamespace(model_path=checkpoint, architecture="SwinIR", scale=4, device="cpu")
        with patch("report._git", side_effect=["", "selected-commit"]), patch(
            "report._sha256", return_value="selected-sha256"
        ) as sha256:
            environment = describe_environment(
                started=_environment(self.run_dir).started,
                run_dir=self.run_dir,
                project_root=self.run_dir.parent,
                source_directory=Path("input"),
                sampling="sequential",
                limit=2,
                discovered=2,
                selected=2,
                sr_line=sr_line,
                lpips_device="cpu",
            )

        self.assertEqual(environment.model_target, str(checkpoint))
        self.assertEqual(environment.model_sha256, "selected-sha256")
        self.assertEqual(environment.architecture, "SwinIR")
        sha256.assert_any_call(checkpoint)

    def test_there_is_one_row_per_successful_image(self):
        rows = [line for line in self.text.splitlines() if line.startswith("| a.JPG |") or line.startswith("| b.JPG |")]

        self.assertEqual(len(rows), len(self.results))
        for row in rows:
            self.assertIn("4056×3040", row)
            self.assertIn("1014×760", row)

    def test_the_average_section_states_counts_and_a_winner_per_metric(self):
        for metric in ("PSNR", "SSIM", "LPIPS"):
            with self.subTest(metric=metric):
                row = re.search(rf"^\| {metric} \| .+ \|$", self.text, re.MULTILINE)
                self.assertIsNotNone(row, f"no average row for {metric}")
                self.assertIn(row.group(0).split("|")[5].strip(), ("SR", "bicubic", "tie"))
        self.assertIn("| 納入張數 | 排除張數 |\n|---|---|\n| 2 | 2 |", self.text)

    def test_failures_are_listed_with_stage_and_reason(self):
        self.assertIn("| c.JPG | sr | out of memory |", self.text)
        self.assertIn("| d.png | decode | 16 bits per sample |", self.text)

    def test_nothing_renders_as_none_or_nan(self):
        for token in ("None", "nan", "NaN"):
            self.assertNotIn(token, self.text)

    def test_a_run_with_no_successful_image_still_produces_a_report(self):
        text = render_report(_environment(self.run_dir), [], self.failures, summarise([], self.failures))

        self.assertIn("| 納入張數 | 排除張數 |\n|---|---|\n| 0 | 2 |", text)
        self.assertIn("| c.JPG | sr | out of memory |", text)
        self.assertNotIn("nan", text)

    def test_an_infinite_psnr_is_shown_with_the_exclusion_count(self):
        results = [
            _result("flat.JPG", sr=(30.0, 0.9, 0.1), bicubic=(float("inf"), 1.0, 0.0)),
            _result("b.JPG", sr=(26.0, 0.74, 0.25), bicubic=(28.4, 0.81, 0.39)),
        ]

        text = render_report(_environment(self.run_dir), results, [], summarise(results, []))

        self.assertIn("inf", text)
        self.assertIn("| PSNR | ↑ | 26.0000 | 28.4000 | bicubic | 2.4000 | 1 | 1 |", text)

    def test_checkpoint_initialization_failure_has_data_only_report(self):
        checkpoint = self.run_dir.parent / "broken|weights.pth"
        checkpoint.write_bytes(b"invalid weights")
        text = render_failure_report(checkpoint, selected=2, error="Unsupported model\nno descriptor")

        self.assertIn("| checkpoint | broken&#124;weights.pth |", text)
        self.assertIn("| 模型 SHA-256 | " + hashlib.sha256(b"invalid weights").hexdigest() + " |", text)
        self.assertIn("| 選取張數 | 2 |", text)
        self.assertIn("| 狀態 | failed |", text)
        self.assertIn("| 原因 | Unsupported model no descriptor |", text)
        for line in text.splitlines():
            self.assertTrue(not line or line.startswith(("#", "|")), line)

    def test_unreadable_checkpoint_failure_can_still_be_reported(self):
        text = render_failure_report(self.run_dir.parent / "missing.pth", selected=2, error="not found")
        self.assertIn("| checkpoint | missing.pth |", text)
        self.assertIn("| 模型 SHA-256 | n/a |", text)



class ModeSkeletonReportTests(unittest.TestCase):
    def test_two_views_link_sources_outputs_models_and_failures_without_fake_metrics(self):
        from report import write_mode_reports
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            combo = {"id": "deblur-abc", "order": ["deblur"], "error": None,
                     "elapsed_seconds": 0.1, "models": [
                         {"role": "deblur", "path": "/models/deblur/model.pth",
                          "sha256": "checksum", "architecture": "Synthetic", "scale": 1, "device": "cpu"}],
                     "rows": [
                         {"input": "/input/good.png", "input_size": (65, 67), "status": "success",
                          "output": str(run / "deblur-abc/good.png"), "output_size": (65, 67)},
                         {"input": "/input/bad.png", "status": "failed", "failure_stage": "decode",
                          "reason": "cannot decode"}]}
            write_mode_reports(run, {"mode": "deblur", "selected": ["good.png", "bad.png"]}, [combo])
            main = (run / "report.md").read_text()
            detail = (run / "per_image.md").read_text()
            self.assertIn("(per_image.md)", main)
            self.assertIn("(report.md)", detail)
            self.assertIn("成功 1 / 2", main)
            self.assertIn("checksum", main)
            self.assertIn("deblur-abc/good.png", detail)
            self.assertIn("cannot decode", detail)
            self.assertIn("待量測", detail)
            for name in ("laplacian_variance", "tenengrad", "cpbd", "crete_roffet_blur"):
                self.assertEqual(detail.count(f"| {name} |"), 2)


if __name__ == "__main__":
    unittest.main()
