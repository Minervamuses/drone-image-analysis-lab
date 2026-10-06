import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

from drone_sr.__main__ import main
from test_inference import synthetic_descriptor


class CLITests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def picture(self, path, color=(12, 34, 56)):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (3, 2), color).save(path)
        return path.read_bytes()

    def run_cli(self, args=(), *, cwd=None, model="models/selected.pth", model_error=None,
                modes=("--sr",)):
        output = io.StringIO()
        descriptor = SimpleNamespace(device=torch.device("cpu"), scale=2)
        model_args = [] if model is None else ["--model", str(model)]
        with (
            contextlib.chdir(cwd or self.root),
            contextlib.redirect_stdout(output),
            contextlib.redirect_stderr(output),
            patch("sys.argv", ["drone_sr", *modes, *model_args, *args]),
            patch("drone_sr.inference.load_model", return_value=descriptor,
                  side_effect=model_error) as loader,
            patch("drone_sr.inference.upscale", side_effect=lambda image, model:
                  image.repeat_interleave(2, -2).repeat_interleave(2, -1)) as upscale,
        ):
            try:
                code = main()
            except SystemExit as error:
                code = error.code
        if loader.called:
            loader.assert_called_once_with(Path(model), role="sr")
        return code, output.getvalue(), loader.call_count, upscale.call_count

    def test_model_is_required_even_when_default_checkpoint_exists(self):
        self.picture(self.root / "input" / "sample.png")
        checkpoint = self.root / "models" / "model.pth"
        checkpoint.parent.mkdir()
        checkpoint.touch()
        code, text, loads, calls = self.run_cli(model=None)
        self.assertEqual(code, 2, text)
        self.assertIn("--sr-model", text)
        self.assertEqual((loads, calls), (0, 0))
        self.assertFalse((self.root / "output").exists())

    def test_relative_and_absolute_checkpoint_paths_are_passed_to_loader(self):
        self.picture(self.root / "input" / "sample.png")
        for model in (Path("checkpoints/selected model.safetensors"), self.root / "another.pth"):
            with self.subTest(model=model):
                code, text, loads, calls = self.run_cli(model=model)
                self.assertEqual(code, 0, text)
                self.assertEqual((loads, calls), (1, 1))
                self.assertIn(f"Model: {model}\n", text.split("Finished\n", 1)[1])

    def test_default_and_independently_optional_folder_arguments(self):
        cases = [
            ([], "input", "output"),
            (["--input", "photos here"], "photos here", "output"),
            (["--output", "results here"], "input", "results here"),
            (["--input", "photos here", "--output", "results here"],
             "photos here", "results here"),
        ]
        absolute = self.root / "absolute paths"
        cases.append((["--input", str(absolute / "photos"), "--output",
                       str(absolute / "results")],
                      absolute / "photos", absolute / "results"))
        for index, (args, source_dir, output_dir) in enumerate(cases):
            with self.subTest(args=args):
                cwd = self.root / str(index)
                cwd.mkdir()
                source = cwd / source_dir / "sample.png"
                before = self.picture(source)
                code, text, loads, calls = self.run_cli(args, cwd=cwd)
                self.assertEqual(code, 0, text)
                self.assertEqual((loads, calls), (1, 1))
                with Image.open(cwd / output_dir / "sample.png") as result:
                    self.assertEqual((result.format, result.mode, result.size),
                                     ("PNG", "RGB", (6, 4)))
                self.assertEqual(source.read_bytes(), before)
                self.assertIn("Processed: 1", text)
                self.assertIn("Failed: 0", text)

    def test_missing_or_non_directory_input_stops_before_model(self):
        for as_file in (False, True):
            with self.subTest(as_file=as_file):
                if as_file:
                    (self.root / "input").write_text("not a directory")
                code, text, loads, calls = self.run_cli()
                self.assertNotEqual(code, 0)
                self.assertIn("input", text.lower())
                self.assertEqual((loads, calls), (0, 0))
                self.assertFalse((self.root / "output").exists())

    def test_empty_input_has_no_model_work(self):
        source = self.root / "empty folder"
        source.mkdir()
        (source / "notes.txt").write_text("not an image")
        code, text, loads, calls = self.run_cli(["--input", "empty folder"])
        self.assertEqual(code, 0, text)
        self.assertIn("No supported images found", text)
        self.assertIn("empty folder", text)
        self.assertEqual((loads, calls), (0, 0))

    def test_output_file_is_rejected_before_processing(self):
        self.picture(self.root / "input" / "sample.png")
        target = self.root / "output"
        target.write_bytes(b"preserve this file")
        code, text, loads, calls = self.run_cli()
        self.assertNotEqual(code, 0)
        self.assertIn("output", text.lower())
        self.assertEqual((loads, calls), (0, 0))
        self.assertEqual(target.read_bytes(), b"preserve this file")

    def test_formats_stable_order_and_corrupt_image_continuation(self):
        names = ["A.JPG", "b_bad.PNG", "c.JPEG", "d.png", "e.TIF", "f.tiff"]
        for name in reversed(names):
            if name != "b_bad.PNG":
                self.picture(self.root / "input" / name)
        (self.root / "input" / "b_bad.PNG").write_bytes(b"broken image")
        (self.root / "input" / "notes.txt").write_text("ignore")
        self.picture(self.root / "input" / "nested" / "ignore.png")
        originals = {p.name: p.read_bytes() for p in (self.root / "input").iterdir()
                     if p.is_file()}
        code, text, loads, calls = self.run_cli()
        self.assertNotEqual(code, 0)
        self.assertEqual((loads, calls), (1, 5))
        self.assertIn("Processed: 5", text)
        self.assertIn("Failed: 1", text)
        self.assertIn("Model: models/selected.pth\n", text.split("Finished\n", 1)[1])
        positions = [text.index(f"[{index}/6] {name}")
                     for index, name in enumerate(names, 1)]
        self.assertEqual(positions, sorted(positions))
        self.assertEqual({p.name for p in (self.root / "output").iterdir()},
                         {f"{Path(name).stem}.png" for name in names if name != "b_bad.PNG"})
        for name, before in originals.items():
            self.assertEqual((self.root / "input" / name).read_bytes(), before)

    def test_success_numbers_its_result_and_preserves_other_output(self):
        source = self.root / "input" / "sample.png"
        self.picture(source)
        self.picture(self.root / "output" / "sample.png", (255, 0, 0))
        unrelated = self.root / "output" / "other.txt"
        unrelated.write_bytes(b"unrelated")
        code, text, _, _ = self.run_cli()
        self.assertEqual(code, 0, text)
        with Image.open(self.root / "output" / "sample.png") as previous:
            self.assertEqual(previous.getpixel((0, 0)), (255, 0, 0))
        with Image.open(self.root / "output" / "sample(2).png") as result:
            self.assertEqual(result.getpixel((0, 0)), (12, 34, 56))
        self.assertIn("sample.png -> sample(2).png", text)
        self.assertEqual(unrelated.read_bytes(), b"unrelated")

    def test_same_stem_inputs_are_numbered_after_existing_results(self):
        for name in ("same.jpg", "same.png", "other.png"):
            self.picture(self.root / "input" / name)
        old = self.root / "output" / "same.png"
        before = self.picture(old, (255, 0, 0))
        code, text, loads, calls = self.run_cli()
        self.assertEqual(code, 0, text)
        self.assertEqual((loads, calls), (1, 3))
        self.assertIn("Processed: 3", text)
        self.assertIn("Failed: 0", text)
        self.assertEqual(old.read_bytes(), before)
        self.assertEqual({p.name for p in old.parent.iterdir()},
                         {"same.png", "same(2).png", "same(3).png", "other.png"})

    def test_same_input_output_directory_and_symlink_alias_are_rejected(self):
        source = self.root / "input" / "sample.png"
        before = self.picture(source)
        alias = self.root / "alias"
        alias.symlink_to(self.root / "input", target_is_directory=True)
        for destination in ("input", "input/../input", "alias"):
            with self.subTest(destination=destination):
                code, text, loads, calls = self.run_cli(["--output", destination])
                self.assertNotEqual(code, 0)
                self.assertNotIn("unrecognized arguments", text)
                self.assertEqual((loads, calls), (0, 0))
                self.assertEqual(source.read_bytes(), before)

    def test_output_alias_of_a_different_input_is_preserved_and_numbered(self):
        for hardlink in (False, True):
            with self.subTest(hardlink=hardlink):
                cwd = self.root / str(hardlink)
                first = cwd / "input" / "a.jpg"
                second = cwd / "input" / "b.png"
                first_before = self.picture(first)
                second_before = self.picture(second, (78, 90, 12))
                target = cwd / "output" / "a.png"
                target.parent.mkdir()
                target.hardlink_to(second) if hardlink else target.symlink_to(second)
                code, text, _, calls = self.run_cli(cwd=cwd)
                self.assertEqual(code, 0, text)
                self.assertEqual(calls, 2)
                self.assertIn("Processed: 2", text)
                self.assertIn("Failed: 0", text)
                self.assertTrue((target.parent / "a(2).png").is_file())
                self.assertEqual(first.read_bytes(), first_before)
                self.assertEqual(second.read_bytes(), second_before)
                self.assertTrue(target.samefile(second))
                self.assertTrue((cwd / "output" / "b.png").is_file())

    def test_storage_failure_preserves_old_result_and_later_image_continues(self):
        for name in ("a.png", "b.png"):
            self.picture(self.root / "input" / name)
        old = self.root / "output" / "a.png"
        before = self.picture(old, (255, 0, 0))
        original_save = Image.Image.save
        saves = 0

        def fail_once(image, *args, **kwargs):
            nonlocal saves
            saves += 1
            if saves == 1:
                raise OSError("simulated disk full")
            return original_save(image, *args, **kwargs)

        with patch.object(Image.Image, "save", fail_once):
            code, text, _, _ = self.run_cli()
        self.assertNotEqual(code, 0)
        self.assertIn("simulated disk full", text)
        self.assertIn("Processed: 1", text)
        self.assertIn("Failed: 1", text)
        self.assertEqual(old.read_bytes(), before)
        self.assertEqual({p.name for p in old.parent.iterdir()}, {"a.png", "b.png"})

    def test_fatal_model_error_does_not_process_images(self):
        self.picture(self.root / "input" / "sample.png")
        code, text, loads, calls = self.run_cli(
            model_error=RuntimeError("Unable to load SR model: invalid checkpoint"))
        self.assertNotEqual(code, 0)
        self.assertIn("Unable to load SR model: invalid checkpoint", text)
        self.assertEqual((loads, calls), (1, 0))
        self.assertFalse((self.root / "output").exists())

    def test_help_exposes_model_and_folder_options_without_requiring_a_model(self):
        code, text, loads, calls = self.run_cli(["--help"], model=None)
        self.assertEqual(code, 0)
        for option in ("--sr", "--deblur", "--sr-model", "--model", "--deblur-model",
                       "--input", "--output"):
            self.assertIn(option, text)
        for option in ("--device", "--tile", "--scale", "--batch", "--overwrite"):
            self.assertNotIn(option, text)
        self.assertEqual((loads, calls), (0, 0))

    def test_no_mode_and_repeated_modes_fail_before_model_loading(self):
        self.picture(self.root / "input" / "sample.png")
        for modes in ((), ("--sr", "--sr"), ("--deblur", "--deblur"),
                      ("--sr", "--deblur", "--sr")):
            with self.subTest(modes=modes):
                code, text, loads, calls = self.run_cli(modes=modes)
                self.assertEqual(code, 2, text)
                self.assertEqual((loads, calls), (0, 0))
                self.assertFalse((self.root / "output").exists())

    def invoke_real_inference(self, args, descriptors):
        output = io.StringIO()

        def load(path, *, role):
            self.assertEqual(path, Path(f"{role}.pth"))
            return descriptors[role]

        with (
            contextlib.chdir(self.root),
            contextlib.redirect_stdout(output),
            contextlib.redirect_stderr(output),
            patch("sys.argv", ["drone_sr", *args]),
            patch("drone_sr.inference.load_model", side_effect=load) as loader,
        ):
            try:
                code = main()
            except SystemExit as error:
                code = error.code
        return code, output.getvalue(), loader

    def test_four_routes_keep_flag_order_and_require_only_active_checkpoints(self):
        source = self.root / "input" / "sample.png"
        before = self.picture(source, (40, 80, 120))
        outputs = {}
        routes = (("sr",), ("deblur",), ("sr", "deblur"), ("deblur", "sr"))
        for roles in routes:
            with self.subTest(roles=roles):
                calls = []

                def sr(image):
                    calls.append("sr")
                    return image * 0.5

                def deblur(image):
                    calls.append("deblur")
                    return image + 0.1

                descriptors = {
                    "sr": synthetic_descriptor(operation=sr),
                    "deblur": synthetic_descriptor(scale=1, operation=deblur),
                }
                destination = "-".join(roles)
                args = [f"--{role}" for role in roles]
                for role in roles:
                    args.extend((f"--{role}-model", f"{role}.pth"))
                code, text, loader = self.invoke_real_inference(
                    [*args, "--output", destination], descriptors)
                self.assertEqual(code, 0, text)
                self.assertEqual(calls, list(roles))
                self.assertEqual([call.kwargs["role"] for call in loader.call_args_list],
                                 list(roles))
                expected = torch.tensor([40, 80, 120], dtype=torch.float32) / 255
                for role in roles:
                    expected = expected * 0.5 if role == "sr" else expected + 0.1
                with Image.open(self.root / destination / "sample.png") as result:
                    self.assertEqual(result.size, (6, 4) if "sr" in roles else (3, 2))
                    outputs[roles] = result.getpixel((0, 0))
                    self.assertEqual(outputs[roles], tuple(expected.mul(255).round().int().tolist()))
                self.assertEqual(source.read_bytes(), before)
        self.assertNotEqual(outputs[("sr", "deblur")], outputs[("deblur", "sr")])

    def test_enabled_stage_without_its_checkpoint_fails_before_loading(self):
        self.picture(self.root / "input" / "sample.png")
        for args in (["--deblur"], ["--sr", "--deblur", "--sr-model", "sr.pth"],
                     ["--deblur", "--sr", "--deblur-model", "deblur.pth"]):
            with self.subTest(args=args):
                code, text, loader = self.invoke_real_inference(args, {})
                self.assertEqual(code, 2, text)
                loader.assert_not_called()
                self.assertFalse((self.root / "output").exists())

    def test_second_stage_failure_keeps_old_output_and_continues_next_image(self):
        for name in ("a.png", "b.png"):
            self.picture(self.root / "input" / name)
        old = self.root / "output" / "a.png"
        before = self.picture(old, (255, 0, 0))
        calls = 0

        def fail_once(image):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("synthetic second-stage failure")
            return image

        code, text, loader = self.invoke_real_inference(
            ["--sr", "--deblur", "--sr-model", "sr.pth", "--deblur-model", "deblur.pth"],
            {"sr": synthetic_descriptor(),
             "deblur": synthetic_descriptor(scale=1, operation=fail_once)},
        )
        self.assertNotEqual(code, 0)
        self.assertEqual(loader.call_count, 2)
        self.assertEqual(calls, 2)
        self.assertIn("deblur", text)
        self.assertIn("synthetic second-stage failure", text)
        self.assertIn("Processed: 1", text)
        self.assertIn("Failed: 1", text)
        self.assertEqual(old.read_bytes(), before)
        with Image.open(self.root / "output" / "b.png") as result:
            self.assertEqual(result.size, (6, 4))

    def test_output_alias_cannot_overwrite_selected_checkpoint(self):
        self.picture(self.root / "input" / "sample.png")
        checkpoint = self.root / "selected.pth"
        checkpoint.write_bytes(b"preserve checkpoint")
        for hardlink in (False, True):
            with self.subTest(hardlink=hardlink):
                target = self.root / str(hardlink) / "sample.png"
                target.parent.mkdir()
                target.hardlink_to(checkpoint) if hardlink else target.symlink_to(checkpoint)
                code, text, _, calls = self.run_cli(
                    ["--output", str(target.parent)], model=checkpoint)
                self.assertEqual(code, 0, text)
                self.assertEqual(calls, 1)
                self.assertEqual(checkpoint.read_bytes(), b"preserve checkpoint")
                self.assertTrue(target.samefile(checkpoint))
                self.assertTrue((target.parent / "sample(2).png").is_file())
                self.assertIn("Failed: 0", text)


if __name__ == "__main__":
    unittest.main()
