"""Checkpoint selection and the shared sample across evaluation runs."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
        arguments = run_evaluation._parse([])
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
            run_evaluation._parse(["--model", "a.pth", "--all"])
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
                "--all", "--input", str(self.sources), "--runs-root", str(self.runs),
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
            status = run_evaluation.main(["--all", "--input", str(self.sources), "--runs-root", str(self.runs)])
        self.assertEqual(status, 1)
        self.assertEqual(output.getvalue().splitlines()[-1],
                         "checkpoints attempted: a-broken.pth, b-working.pth")
        self.assertEqual(evaluate.call_count, 2)
        failed_dir = evaluate.call_args_list[0].args[-1]
        text = (failed_dir / "report.md").read_text()
        self.assertIn("a-broken.pth", text)
        self.assertIn("unsupported scale", text)


if __name__ == "__main__":
    unittest.main()
