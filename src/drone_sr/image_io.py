"""RGB tensor conversion and PNG output that preserves the source image."""

import os
import tempfile
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torchvision.transforms.functional import pil_to_tensor, to_pil_image


def _source_bits_per_sample(image: Image.Image, path: Path) -> int:
    """Largest bits per sample in the file itself; image.mode is already mapped."""
    if image.format == "TIFF":
        return max(image.tag_v2.get(258, (8,)))
    if image.format == "PNG":
        with path.open("rb") as stream:
            header = stream.read(26)
        # Bit depth sits at a fixed offset in IHDR, the first chunk of every PNG.
        return header[24] if header[12:16] == b"IHDR" else 8
    return 8


def read_image(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        if image.mode not in {"1", "L", "LA", "P", "RGB", "RGBA", "CMYK"}:
            raise ValueError(f"Unsupported image mode: {image.mode}")
        if image.format == "MPO":
            # DJI MPO JPEGs store the main image in frame 0.
            image.seek(0)
        elif getattr(image, "n_frames", 1) != 1:
            raise ValueError("Only single-frame images are supported")
        # Before anything decodes: Pillow maps 16-bit colour onto RGB/RGBA and
        # then truncates it to 8-bit, so by that point the evidence is gone.
        bits = _source_bits_per_sample(image, path)
        if bits > 8:
            raise ValueError(f"Unsupported source bit depth: {bits} bits per sample")
        # After the checks above: exif_transpose() returns a plain Image whose
        # n_frames is always 1, which would silently disable the check above it.
        oriented = ImageOps.exif_transpose(image)
        return pil_to_tensor(oriented.convert("RGB")).to(torch.float32).div_(255).unsqueeze(0)


def write_png(tensor: torch.Tensor, destination: Path, source: Path) -> None:
    if tensor.ndim != 4 or tensor.shape[:2] != (1, 3):
        raise ValueError(f"Expected RGB BCHW output, got {tuple(tensor.shape)}")
    if not torch.isfinite(tensor).all():
        raise ValueError("Model output contains non-finite values")
    if destination.resolve() == source.resolve() or (
        destination.exists() and destination.samefile(source)
    ):
        raise ValueError(f"Refusing to overwrite input image: {source}")

    pixels = tensor.detach().cpu().squeeze(0).clamp(0, 1).mul(255).round().to(torch.uint8)
    image = to_pil_image(pixels, mode="RGB")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        # Encode completely before replacing a previous successful result.
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".png", delete=False) as stream:
            temporary = Path(stream.name)
            image.save(stream, format="PNG")
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
