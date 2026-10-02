"""One architecture-independent Spandrel image inference path."""

from pathlib import Path

import torch
from spandrel import ImageModelDescriptor, ModelLoader, ModelTiling
import spandrel_extra_arches

from .tiling import TILE_SIZE, upscale_tiled


# Registration is idempotent and does not load or download checkpoints.
spandrel_extra_arches.install(ignore_duplicates=True)

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "sr" / "model.pth"


def load_model(model_path: Path | None = None, *, role: str = "sr") -> ImageModelDescriptor:
    if role not in {"sr", "deblur"}:
        raise ValueError(f"Unknown model role: {role}")
    if role == "deblur" and model_path is None:
        raise ValueError("deblur requires an explicit checkpoint")
    label_role = "SR" if role == "sr" else "deblur"
    path = MODEL_PATH if model_path is None else Path(model_path)
    if not path.is_file():
        label = "models/sr/model.pth" if model_path is None else str(path)
        raise FileNotFoundError(f"{label_role} model not found: {label}")
    try:
        descriptor = ModelLoader().load_from_file(path)
    except Exception as error:
        raise RuntimeError(f"Unable to load {label_role} model: {error}") from error
    if (
        not isinstance(descriptor, ImageModelDescriptor)
        or descriptor.purpose != ("SR" if role == "sr" else "Restoration")
        or descriptor.input_channels != 3
        or descriptor.output_channels != 3
        or (descriptor.scale <= 1 if role == "sr" else descriptor.scale != 1)
    ):
        expected = "RGB super-resolution model (scale > 1)" if role == "sr" else "RGB Restoration model (scale = 1)"
        raise ValueError(f"The {role} checkpoint must describe an {expected}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return descriptor.to(device=device, dtype=torch.float32).eval()


@torch.inference_mode()
def _upscale_direct(image: torch.Tensor, descriptor: ImageModelDescriptor) -> torch.Tensor:
    image = image.to(device=descriptor.device, dtype=descriptor.dtype)
    output = descriptor(image)
    expected = (1, 3, image.shape[-2] * descriptor.scale, image.shape[-1] * descriptor.scale)
    if tuple(output.shape) != expected:
        raise ValueError(f"Unexpected model output shape: {tuple(output.shape)}; expected {expected}")
    return output


@torch.inference_mode()
def upscale(image: torch.Tensor, descriptor: ImageModelDescriptor) -> torch.Tensor:
    if max(image.shape[-2:]) <= TILE_SIZE or (
        descriptor.scale == 1 and descriptor.tiling != ModelTiling.SUPPORTED
    ):
        return _upscale_direct(image, descriptor)
    return upscale_tiled(image, descriptor.scale, lambda tile: _upscale_direct(tile, descriptor))


@torch.inference_mode()
def run_stages(
    image: torch.Tensor, stages: list[tuple[str, ImageModelDescriptor]]
) -> torch.Tensor:
    """Run the selected order in memory, quantizing only at the final PNG."""
    if not stages:
        raise ValueError("At least one stage is required")
    for role, descriptor in stages:
        try:
            if image.ndim != 4 or image.shape[:2] != (1, 3) or not torch.isfinite(image).all():
                raise ValueError("Expected a finite RGB NCHW input tensor")
            expected = (1, 3, image.shape[-2] * descriptor.scale, image.shape[-1] * descriptor.scale)
            image = upscale(image, descriptor)
            if tuple(image.shape) != expected:
                raise ValueError(f"Unexpected output shape: {tuple(image.shape)}; expected {expected}")
            if not torch.isfinite(image).all():
                raise ValueError("Non-finite output")
        except Exception as error:
            raise RuntimeError(f"{role} stage failed: {error}") from error
    return image
