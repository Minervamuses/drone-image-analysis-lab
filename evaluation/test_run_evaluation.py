"""Checkpoint selection and the shared sample across evaluation runs."""

import contextlib
import io
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import run_evaluation


class EvaluationSelectionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.models = self.root / "models"
        self.models.mkdir()
        self.sources = self.root / "input"
        self.sources.mkdir()
        self.runs = self.root / "runs"
        self.model_patch = patch.object(run_evaluation, "MODELS_ROOT", self.models)
        self.model_patch.start()
        self.addCleanup(self.model_patch.stop)

    def checkpoint(self, name):
        path = self.models / name
        path.write_bytes(b"checkpoint selection test")
        return path

    def test_default_and_named_checkpoint(self):
        default = self.checkpoint("model.pth")
        alternate = self.checkpoint("alternate.safetensors")
        arguments = run_evaluation._parse(["--legacy-sr"])
        self.assertEqual(run_evaluation.select_checkpoints(arguments.model, arguments.all), [default])
        self.assertEqual(run_evaluation.select_checkpoints(alternate.name, False), [alternate])
        for name in ("missing.pth", "../outside.pth", str(alternate), ""):
            with self.subTest(name=name), self.assertRaises(ValueError):
                run_evaluation.select_checkpoints(name, False)

    def test_all_uses_supported_files_in_order_without_repeating_symlink_target(self):
        first = self.checkpoint("a.pth")
        second = self.checkpoint("b.PT")
        third = self.checkpoint("c.ckpt")
        fourth = self.checkpoint("d.safetensors")
        self.checkpoint("notes.txt")
        (self.models / "nested.pth").mkdir()
        (self.models / "model.pth").symlink_to(first.name)
        self.assertEqual(run_evaluation.select_checkpoints("model.pth", True), [first, second, third, fourth])

    def test_empty_all_and_conflicting_arguments_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "No checkpoints"):
            run_evaluation.select_checkpoints("model.pth", True)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            run_evaluation._parse(["--legacy-sr", "--model", "a.pth", "--all"])
        self.assertEqual(raised.exception.code, 2)

    def test_all_reuses_one_seeded_sample_and_keeps_reports_and_outputs_separate(self):
        checkpoints = [self.checkpoint("a.pth"), self.checkpoint("b.pth")]
        for index in range(5):
            (self.sources / f"{index}.png").touch()
        observed = []

        def evaluate(arguments, selected, discovered, checkpoint, run_dir):
            observed.append((selected, checkpoint, run_dir))
            (run_dir / "report.md").write_text(checkpoint.name)
            return 0

        with (
            patch.object(run_evaluation, "select_sources", wraps=run_evaluation.select_sources) as select,
            patch.object(run_evaluation, "_evaluate_checkpoint", side_effect=evaluate),
            patch.object(run_evaluation, "release_device_memory"),
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            status = run_evaluation.main([
                "--legacy-sr", "--all", "--input", str(self.sources), "--runs-root", str(self.runs),
                "--limit", "2", "--seed", "37",
            ])
        self.assertEqual(status, 0)
        self.assertEqual(output.getvalue().splitlines()[-1], "checkpoints attempted: a.pth, b.pth")
        select.assert_called_once_with(self.sources, limit=2, seed=37)
        self.assertEqual([row[1] for row in observed], checkpoints)
        self.assertIs(observed[0][0], observed[1][0])
        self.assertEqual(len(observed[0][0]), 2)
        self.assertNotEqual(observed[0][2], observed[1][2])
        for _, checkpoint, run_dir in observed:
            self.assertEqual((run_dir / "report.md").read_text(), checkpoint.name)
            self.assertTrue((run_dir / "sr").is_dir())

    def test_load_failure_is_reported_and_does_not_skip_the_next_checkpoint(self):
        self.checkpoint("a-broken.pth")
        self.checkpoint("b-working.pth")
        (self.sources / "source.png").touch()
        with (
            patch.object(run_evaluation, "_evaluate_checkpoint", side_effect=[ValueError("unsupported scale"), 0]) as evaluate,
            patch.object(run_evaluation, "release_device_memory"),
            contextlib.redirect_stdout(io.StringIO()) as output,
            contextlib.redirect_stderr(io.StringIO()),
        ):
            status = run_evaluation.main([
                "--legacy-sr", "--all", "--input", str(self.sources), "--runs-root", str(self.runs),
            ])
        self.assertEqual(status, 1)
        self.assertEqual(output.getvalue().splitlines()[-1],
                         "checkpoints attempted: a-broken.pth, b-working.pth")
        self.assertEqual(evaluate.call_count, 2)
        failed_dir = evaluate.call_args_list[0].args[-1]
        text = (failed_dir / "report.md").read_text()
        self.assertIn("a-broken.pth", text)
        self.assertIn("unsupported scale", text)


class EvaluationModeTests(unittest.TestCase):
    setUp = EvaluationSelectionTests.setUp
    checkpoint = EvaluationSelectionTests.checkpoint

    def deblur_folder(self):
        folder = self.root / "deblur"
        folder.mkdir(exist_ok=True)
        model_patch = patch.object(run_evaluation, "DEBLUR_MODELS_ROOT", folder)
        model_patch.start()
        self.addCleanup(model_patch.stop)
        return folder

    def test_flags_preserve_order_and_legacy_is_explicit(self):
        sr = self.checkpoint("sr.pth")
        for flags, order in (
            (["--sr"], ["sr"]),
            (["--deblur"], ["deblur"]),
            (["--sr", "--deblur"], ["sr", "deblur"]),
            (["--deblur", "--sr"], ["deblur", "sr"]),
        ):
            with self.subTest(flags=flags):
                extra = ["--sr-model", str(sr)] if "sr" in order else []
                arguments = run_evaluation._parse([*flags, *extra])
                self.assertEqual(arguments.stages, order)
                self.assertEqual(arguments.limit, 1)
        self.assertEqual(run_evaluation._parse(["--legacy-sr"]).limit, 5)
        invalid = (
            [], ["--sr"], ["--all"], ["--model", "old.pth"],
            ["--deblur", "--deblur"],
            ["--sr", "--sr", "--sr-model", str(sr)],
            ["--legacy-sr", "--deblur"],
            ["--legacy-sr", "--sr-model", str(sr)],
            ["--legacy-sr", "--deblur-model", str(sr)],
            ["--deblur", "--all"],
            ["--deblur", "--model", "old.pth"],
        )
        for flags in invalid:
            with self.subTest(invalid=flags), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    run_evaluation._parse(flags)
                self.assertEqual(raised.exception.code, 2)

    def test_two_deblur_checkpoints_produce_only_the_selected_combinations(self):
        deblur = self.deblur_folder()
        first, second = deblur / "a.pth", deblur / "b.PT"
        for path in (first, second):
            path.write_bytes(b"stub checkpoint")
        sr = self.checkpoint("chosen.pth")
        cases = (
            (["--deblur"], [{"deblur": first}, {"deblur": second}]),
            (["--sr", "--sr-model", str(sr)], [{"sr": sr}]),
            (["--sr", "--deblur", "--sr-model", str(sr)],
             [{"sr": sr, "deblur": first}, {"sr": sr, "deblur": second}]),
            (["--deblur", "--sr", "--sr-model", str(sr)],
             [{"sr": sr, "deblur": first}, {"sr": sr, "deblur": second}]),
        )
        for flags, expected in cases:
            with self.subTest(flags=flags):
                combinations = run_evaluation.select_mode_checkpoints(run_evaluation._parse(flags))
                self.assertEqual(combinations, expected)

    def test_deblur_scan_is_flat_sorted_and_resolve_deduplicated(self):
        deblur = self.deblur_folder()
        expected = [deblur / name for name in ("a.pth", "b.PT", "c.ckpt", "d.safetensors")]
        for path in reversed(expected):
            path.write_bytes(b"not a real weight")
        (deblur / "notes.txt").touch()
        (deblur / "nested.pth").mkdir()
        (deblur / "z-alias.pth").symlink_to(expected[0].name)
        arguments = run_evaluation._parse(["--deblur"])
        self.assertEqual(run_evaluation.select_mode_checkpoints(arguments),
                         [{"deblur": path} for path in expected])
        single = run_evaluation._parse(["--deblur", "--deblur-model", str(expected[2])])
        self.assertEqual(run_evaluation.select_mode_checkpoints(single), [{"deblur": expected[2]}])

    def test_empty_deblur_scan_is_an_error_without_sr_fallback(self):
        self.deblur_folder()
        self.checkpoint("model.pth")
        with self.assertRaisesRegex(ValueError, "[Nn]o checkpoints"):
            run_evaluation.select_mode_checkpoints(run_evaluation._parse(["--deblur"]))

    def test_new_modes_share_one_sample_and_run_without_overwriting_prior_run(self):
        deblur = self.deblur_folder()
        checkpoints = [deblur / "same.pt", deblur / "same.pth"]
        for checkpoint in checkpoints:
            checkpoint.write_bytes(checkpoint.name.encode())
        for index in range(5):
            Image.new("RGB", (7, 5), (index, 20, 30)).save(self.sources / f"{index}.png")
        observed = []

        def evaluate(selected, run_dir, order, models, baselines):
            observed.append((selected, run_dir, order, models, baselines))
            return {"id": str(len(observed)), "order": order, "models": {},
                    "rows": [{"status": "success"}], "error": None, "elapsed_seconds": 0.0}

        def report(run_dir, environment, combinations):
            (run_dir / "report.md").write_text("new suite " + str(len(observed)))

        args = ["--deblur", "--input", str(self.sources), "--runs-root", str(self.runs),
                "--limit", "2", "--seed", "37"]
        with (
            patch.object(run_evaluation, "select_sources", wraps=run_evaluation.select_sources) as select,
            patch.object(run_evaluation, "run_ordered_batch", side_effect=evaluate),
            patch.object(run_evaluation, "write_mode_reports", side_effect=report),
            patch.object(run_evaluation, "release_device_memory"),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(run_evaluation.main(args), 0)
            first_report = observed[0][1] / "report.md"
            original_report = first_report.read_bytes()
            self.assertEqual(run_evaluation.main(args), 0)
        self.assertEqual(select.call_count, 2)
        select.assert_called_with(self.sources, limit=2, seed=37)
        self.assertEqual(len(observed), 4)
        self.assertIs(observed[0][0], observed[1][0])
        self.assertIs(observed[0][4], observed[1][4])
        self.assertEqual(len(observed[0][0]), 2)
        self.assertEqual(observed[0][1], observed[1][1])
        self.assertNotEqual(observed[0][1], observed[2][1])
        self.assertEqual([row[3] for row in observed[:2]], [{"deblur": path} for path in checkpoints])
        self.assertTrue(all(row[2] == ["deblur"] for row in observed))
        self.assertEqual(first_report.read_bytes(), original_report)

    def test_model_or_image_failure_keeps_next_combination_and_nonzero_status(self):
        deblur = self.deblur_folder()
        for name in ("a.pth", "b.pth"):
            (deblur / name).write_bytes(b"fake checkpoint")
        Image.new("RGB", (7, 5)).save(self.sources / "source.png")
        for failure in (
            {"id": "a", "error": "unsupported checkpoint", "rows": []},
            {"id": "a", "error": None, "rows": [{"status": "failed", "reason": "cannot decode"}]},
        ):
            combinations = [failure, {"id": "b", "error": None, "rows": [{"status": "success"}]}]
            with (
                self.subTest(failure=failure),
                patch.object(run_evaluation, "run_ordered_batch", side_effect=combinations) as evaluate,
                patch.object(run_evaluation, "write_mode_reports") as report,
                patch.object(run_evaluation, "release_device_memory"),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                status = run_evaluation.main([
                    "--deblur", "--input", str(self.sources), "--runs-root", str(self.runs),
                ])
            self.assertEqual(status, 1)
            self.assertEqual(evaluate.call_count, 2)
            self.assertEqual(report.call_args.args[-1], combinations)

    def test_import_does_not_initialize_lpips_or_sr_line(self):
        project = Path(run_evaluation.__file__).resolve().parents[1]
        environment = dict(os.environ, PYTHONPATH=os.pathsep.join((str(project / "src"),
                                                                  str(project / "evaluation"))))
        result = subprocess.run([
            sys.executable, "-c", "import sys; import run_evaluation; "
            "assert 'perceptual' not in sys.modules; assert 'lpips' not in sys.modules; "
            "assert 'sr_line' not in sys.modules",
        ], cwd=project, env=environment, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)


class LabArgumentsTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        (self.root / "lab").mkdir()
        shutil.copyfile(Path(run_evaluation.__file__).resolve().parents[1] / "lab" / "run.sh",
                        self.root / "lab" / "run.sh")
        python = self.root / ".venv" / "bin" / "python"
        python.parent.mkdir(parents=True)
        self.log = self.root / "python-argv.txt"
        self.preflight = self.root / "preflight.py"
        python.write_text(
            "#!/usr/bin/env bash\n"
            "printf 'CALL\\n' >> \"$LAB_STUB_LOG\"\n"
            "printf '%s\\n' \"$@\" >> \"$LAB_STUB_LOG\"\n"
            "if [[ \"${1-}\" == - ]]; then cat > \"$LAB_STUB_PREFLIGHT\"; fi\n"
        )
        python.chmod(0o755)

    def invoke(self, *arguments):
        result = subprocess.run(
            ["bash", str(self.root / "lab" / "run.sh"), *arguments],
            env=dict(os.environ, LAB_STUB_LOG=str(self.log), LAB_STUB_PREFLIGHT=str(self.preflight)),
            capture_output=True, text=True, timeout=10,
        )
        calls = [] if not self.log.exists() else [
            call.splitlines() for call in self.log.read_text().split("CALL\n")[1:]
        ]
        return result, calls

    def test_default_is_only_deblur_one_image_with_preflight(self):
        result, calls = self.invoke()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(["-"], calls)
        self.assertIn(["-m", "pip", "check"], calls)
        self.assertIn("torch.cuda.is_available()", self.preflight.read_text())
        self.assertIn('device="cuda"', self.preflight.read_text())
        arguments = calls[-1]
        self.assertEqual(arguments[0], "evaluation/run_evaluation.py")
        self.assertIn("--deblur", arguments)
        self.assertEqual(arguments[arguments.index("--input") + 1],
                         str(self.root / "evaluation" / "data" / "input"))
        self.assertEqual(arguments[arguments.index("--limit") + 1], "1")
        self.assertNotIn("--all", arguments)
        self.assertNotIn("--sr", arguments)
        self.assertNotIn("--legacy-sr", arguments)

    def test_explicit_checkpoint_and_sampling_are_preserved(self):
        supplied = ["--deblur-model", "models/deblur/model with spaces.pth", "--input", "photos",
                    "--limit", "2", "--seed", "17", "--runs-root", "custom-runs"]
        result, calls = self.invoke(*supplied)
        self.assertEqual(result.returncode, 0, result.stderr)
        parsed = run_evaluation._parse(calls[-1][1:])
        self.assertEqual(parsed.stages, ["deblur"])
        self.assertEqual(parsed.deblur_model.name, "model with spaces.pth")
        self.assertEqual((parsed.input.name, parsed.limit, parsed.seed, parsed.runs_root.name),
                         ("photos", 2, 17, "custom-runs"))

    def test_other_modes_legacy_and_abbreviations_are_rejected(self):
        for argument in ("--sr", "--legacy-sr", "--all", "--deblur", "--deb", "--sr-model=x.pth"):
            with self.subTest(argument=argument):
                self.log.unlink(missing_ok=True)
                result, calls = self.invoke(argument)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(any(call and call[0] == "evaluation/run_evaluation.py" for call in calls))

    def test_help_skips_gpu_preflight(self):
        result, calls = self.invoke("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn(["-"], calls)
        self.assertTrue("--help" in result.stdout or any("--help" in call for call in calls))


if __name__ == "__main__":
    unittest.main()
