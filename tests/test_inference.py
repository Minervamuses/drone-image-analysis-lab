import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch
from spandrel import Architecture, ImageModelDescriptor, ModelTiling, SizeRequirements

from drone_sr.inference import load_model, run_stages, upscale


def synthetic_descriptor(*, wrong_size=False, channels=3, scale=2, purpose=None,
                         output_channels=None, operation=None,
                         tiling=ModelTiling.SUPPORTED, observed_shapes=None):
    """An untrained toy model exercises the real descriptor, not SR quality."""
    model = torch.nn.Conv2d(channels, channels, 1)

    def forward(module, tensor):
        if torch.is_grad_enabled():
            raise AssertionError("Inference must disable gradients")
        if module.training:
            raise AssertionError("Inference must use eval mode")
        parameter = next(module.parameters())
        if tensor.device != parameter.device or tensor.dtype != parameter.dtype:
            raise AssertionError("Tensor and model device/dtype must match")
        if observed_shapes is not None:
            observed_shapes.append(tuple(tensor.shape))
        result = tensor.repeat_interleave(scale, -2).repeat_interleave(scale, -1)
        if operation is not None:
            result = operation(result)
        return result[..., :-1, :] if wrong_size else result

    return ImageModelDescriptor(
        model,
        model.state_dict(),
        architecture=Mock(spec=Architecture),
        purpose=purpose or ("Restoration" if scale == 1 else "SR"),
        tags=[],
        supports_half=False,
        supports_bfloat16=False,
        scale=scale,
        input_channels=channels,
        output_channels=channels if output_channels is None else output_channels,
        size_requirements=SizeRequirements(minimum=4, multiple_of=4),
        tiling=tiling,
        call_fn=forward,
    )


class InferenceTests(unittest.TestCase):
    def test_real_descriptor_pads_odd_input_and_removes_padding(self):
        descriptor = synthetic_descriptor()
        source = torch.arange(45, dtype=torch.float32).reshape(1, 3, 3, 5) / 44
        output = upscale(source, descriptor)
        self.assertEqual(output.shape, (1, 3, 6, 10))
        torch.testing.assert_close(output, source.repeat_interleave(2, -2).repeat_interleave(2, -1))

    def test_wrong_output_size_is_rejected(self):
        descriptor = synthetic_descriptor(wrong_size=True)
        with self.assertRaisesRegex(ValueError, "output shape"):
            upscale(torch.zeros(1, 3, 4, 4), descriptor)

    def test_missing_checkpoint_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("drone_sr.inference.MODEL_PATH", Path(directory) / "model.pth"):
                with self.assertRaisesRegex(FileNotFoundError, "SR model not found: models/sr/model.pth"):
                    load_model()

    def test_invalid_checkpoint_fails_through_real_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "invalid.pth"
            checkpoint.write_bytes(b"not a checkpoint")
            with patch("drone_sr.inference.MODEL_PATH", checkpoint):
                with self.assertRaisesRegex(RuntimeError, "Unable to load SR model"):
                    load_model()

    def test_loader_selects_cpu_and_converts_descriptor_to_float32(self):
        descriptor = synthetic_descriptor().to(dtype=torch.float64)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pth"
            checkpoint.touch()
            with (
                patch("drone_sr.inference.MODEL_PATH", checkpoint),
                patch("drone_sr.inference.ModelLoader") as loader,
                patch("drone_sr.inference.torch.cuda.is_available", return_value=False),
            ):
                loader.return_value.load_from_file.return_value = descriptor
                loaded = load_model()
                loader.return_value.load_from_file.assert_called_once_with(checkpoint)
        self.assertEqual(loaded.device, torch.device("cpu"))
        self.assertEqual(loaded.dtype, torch.float32)
        self.assertFalse(loaded.model.training)

    def test_non_image_and_non_rgb_descriptors_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pth"
            checkpoint.touch()
            for descriptor in (object(), synthetic_descriptor(channels=1)):
                with self.subTest(descriptor=type(descriptor).__name__):
                    with (
                        patch("drone_sr.inference.MODEL_PATH", checkpoint),
                        patch("drone_sr.inference.ModelLoader") as loader,
                    ):
                        loader.return_value.load_from_file.return_value = descriptor
                        with self.assertRaisesRegex(ValueError, "RGB super-resolution"):
                            load_model()

    def test_explicit_checkpoint_is_loaded_without_changing_the_default(self):
        from drone_sr.inference import MODEL_PATH

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "alternate.pth"
            checkpoint.touch()
            with (
                patch("drone_sr.inference.ModelLoader") as loader,
                patch("drone_sr.inference.torch.cuda.is_available", return_value=False),
            ):
                loader.return_value.load_from_file.return_value = synthetic_descriptor()
                load_model(checkpoint)
                loader.return_value.load_from_file.assert_called_once_with(checkpoint)
        import drone_sr.inference

        self.assertEqual(drone_sr.inference.MODEL_PATH, MODEL_PATH)

    def test_deblur_loader_requires_rgb_restoration_at_scale_one(self):
        cases = (
            synthetic_descriptor(),
            synthetic_descriptor(scale=1, channels=1),
            synthetic_descriptor(scale=1, output_channels=1),
            synthetic_descriptor(scale=1, purpose="SR"),
            object(),
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "deblur.pth"
            checkpoint.touch()
            for descriptor in cases:
                with self.subTest(descriptor=type(descriptor).__name__):
                    with patch("drone_sr.inference.ModelLoader") as loader:
                        loader.return_value.load_from_file.return_value = descriptor
                        with self.assertRaisesRegex(ValueError, "[Rr]estoration|[Dd]eblur"):
                            load_model(checkpoint, role="deblur")
            descriptor = synthetic_descriptor(scale=1).to(dtype=torch.float64)
            with (
                patch("drone_sr.inference.ModelLoader") as loader,
                patch("drone_sr.inference.torch.cuda.is_available", return_value=False),
            ):
                loader.return_value.load_from_file.return_value = descriptor
                loaded = load_model(checkpoint, role="deblur")
            self.assertIs(loaded, descriptor)
            self.assertEqual(loaded.device, torch.device("cpu"))
            self.assertEqual(loaded.dtype, torch.float32)
            self.assertFalse(loaded.model.training)

    def test_sr_loader_rejects_restoration_and_wrong_output_channels(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "sr.pth"
            checkpoint.touch()
            for descriptor in (synthetic_descriptor(scale=1),
                               synthetic_descriptor(scale=1, purpose="SR"),
                               synthetic_descriptor(output_channels=1)):
                with self.subTest(purpose=descriptor.purpose, scale=descriptor.scale):
                    with patch("drone_sr.inference.ModelLoader") as loader:
                        loader.return_value.load_from_file.return_value = descriptor
                        with self.assertRaisesRegex(ValueError, "RGB super-resolution"):
                            load_model(checkpoint)

    def test_x1_descriptor_padding_is_removed_for_direct_and_tiled_images(self):
        for height, width in ((3, 5), (1, 513)):
            with self.subTest(height=height, width=width):
                shapes = []
                descriptor = synthetic_descriptor(scale=1, observed_shapes=shapes)
                source = torch.arange(height * width * 3).reshape(1, 3, height, width).float()
                source /= source.numel()
                output = upscale(source, descriptor)
                self.assertEqual(output.shape, source.shape)
                self.assertEqual(output.device.type, "cpu")
                torch.testing.assert_close(output, source, rtol=0, atol=0)
                self.assertEqual(len(shapes), 1 if width <= 512 else 2)
                self.assertTrue(all(shape[-1] % 4 == shape[-2] % 4 == 0 for shape in shapes))

    def test_descriptor_tiling_restrictions_use_single_direct_call(self):
        for tiling in (ModelTiling.DISCOURAGED, ModelTiling.INTERNAL):
            with self.subTest(tiling=tiling):
                shapes = []
                descriptor = synthetic_descriptor(scale=1, tiling=tiling, observed_shapes=shapes)
                source = torch.full((1, 3, 5, 513), 0.4)
                with patch("drone_sr.inference.upscale_tiled") as tiled:
                    result = upscale(source, descriptor)
                tiled.assert_not_called()
                self.assertEqual(len(shapes), 1)
                torch.testing.assert_close(result, source, rtol=0, atol=0)

    def test_existing_sr_tiling_is_preserved_for_discouraged_descriptors(self):
        shapes = []
        descriptor = synthetic_descriptor(tiling=ModelTiling.DISCOURAGED, observed_shapes=shapes)
        source = torch.full((1, 3, 5, 513), 0.4)
        result = upscale(source, descriptor)
        self.assertEqual(len(shapes), 2)
        torch.testing.assert_close(result, source.repeat_interleave(2, -2).repeat_interleave(2, -1))

    def test_stage_boundaries_reject_nonfinite_output_with_role(self):
        for value in (float("nan"), float("inf")):
            with self.subTest(value=value):
                descriptor = synthetic_descriptor(scale=1)
                # The real descriptor clamps infinity, so inject it at the stage boundary.
                with patch("drone_sr.inference.upscale", return_value=torch.full((1, 3, 3, 5), value)):
                    with self.assertRaisesRegex((ValueError, RuntimeError), "deblur.*finite"):
                        run_stages(torch.ones(1, 3, 3, 5), [("deblur", descriptor)])
        descriptor = synthetic_descriptor(scale=1, operation=lambda image: image * float("nan"))
        with self.assertRaisesRegex((ValueError, RuntimeError), "deblur.*finite"):
            run_stages(torch.ones(1, 3, 3, 5), [("deblur", descriptor)])

    def test_wrong_stage_output_shape_and_forward_failure_include_role(self):
        def fail(image):
            raise RuntimeError("synthetic forward failure")

        cases = (
            (synthetic_descriptor(scale=1, wrong_size=True), "shape"),
            (synthetic_descriptor(scale=1, operation=fail), "synthetic forward failure"),
        )
        for descriptor, reason in cases:
            with self.subTest(reason=reason):
                with self.assertRaisesRegex((ValueError, RuntimeError), f"deblur.*{reason}"):
                    run_stages(torch.ones(1, 3, 4, 4), [("deblur", descriptor)])

    def test_intermediate_tensor_keeps_precision_and_second_stage_gets_scaled_size(self):
        shapes = []
        first = synthetic_descriptor(operation=lambda image: image * 0.5)
        second = synthetic_descriptor(scale=1, operation=lambda image: image + 0.1,
                                      observed_shapes=shapes)
        source = torch.full((1, 3, 4, 4), 0.1234)
        result = run_stages(source, [("sr", first), ("deblur", second)])
        expected = (source * 0.5 + 0.1).repeat_interleave(2, -2).repeat_interleave(2, -1)
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        self.assertEqual(shapes, [(1, 3, 8, 8)])

    def test_loader_registers_all_five_candidates_without_checkpoint_weights(self):
        from spandrel import MAIN_REGISTRY

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "placeholder.pth"
            checkpoint.touch()
            with (
                patch("drone_sr.inference.ModelLoader") as loader,
                patch("drone_sr.inference.torch.cuda.is_available", return_value=False),
            ):
                loader.return_value.load_from_file.return_value = synthetic_descriptor(scale=1)
                load_model(checkpoint, role="deblur")
        for architecture in ("FFTformer", "NAFNet", "Restormer", "Uformer", "MPRNet"):
            with self.subTest(architecture=architecture):
                self.assertIn(architecture, MAIN_REGISTRY)


if __name__ == "__main__":
    unittest.main()
