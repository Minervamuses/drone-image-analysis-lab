"""Running both upscale lines over a bounded sample of source images.

One image at a time: decode, mod-crop to the ground truth, downscale to the LR
PNG, then upscale that same file twice - once with bicubic, once with the
unmodified SR pipeline - and measure both against the ground truth. Nothing here
changes the degradation contract or the metric conventions; those are settled in
degradation.py, bicubic.py, metrics.py and perceptual.py and were verified
before this module existed.

A failure on one image is recorded and the batch carries on. Losing a whole run
to one unreadable file would be worse than a report that says which file dropped
out and why.
"""

import hashlib
import json
import random
import resource
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from drone_sr.image_io import read_image, write_png
from drone_sr.inference import load_model, run_stages

from bicubic import upscale_bicubic
from degradation import downscale, mod_crop, save_png
from metrics import load_metric_tensor, psnr, ssim, to_metric_tensor
from sources import decode_source, discover_sources

DEFAULT_LIMIT = 5
from blur_metrics import BLUR_METRICS, compare_measurements, failed_measurements, measure_tensor


def prepare_baselines(sources, baselines: dict) -> None:
    """Measure each decoded original once, even if the first checkpoint fails."""
    for source in sources:
        key = str(source.resolve())
        if key in baselines:
            continue
        try:
            image = read_image(source)
        except Exception as error:
            baselines[key] = {"input_size": None, "decode_error": f"{type(error).__name__}: {error}",
                              "before": failed_measurements(f"input decode: {error}")}
            continue
        try:
            measured = measure_tensor(image)
        except Exception as error:
            measured = failed_measurements(f"input metrics: {type(error).__name__}: {error}")
        baselines[key] = {"input_size": (image.shape[-1], image.shape[-2]), "before": measured}
        del image


def _model_record(role: str, path: Path) -> dict:
    record = {"role": role, "path": str(path.resolve()), "sha256": None,
              "architecture": None, "scale": None, "device": None,
              "tiling": None, "size_requirements": None}
    try:
        with path.open("rb") as stream:
            record["sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    except (OSError, ValueError) as error:
        record["error"] = f"{type(error).__name__}: {error}"
    return record


def _describe_model(record: dict, descriptor) -> None:
    requirements = descriptor.size_requirements
    record.update(
        architecture=str(descriptor.architecture.id), scale=descriptor.scale,
        device=str(descriptor.device), tiling=descriptor.tiling.name,
        size_requirements={key: getattr(requirements, key)
                           for key in ("minimum", "multiple_of", "square")},
    )


def run_ordered_batch(sources, run_dir: Path, order, checkpoints: dict, baselines: dict) -> dict:
    """Run one selected checkpoint combination on the suite's unchanged sample.

    All model/tensor references stay local to this call. Only small image metadata
    and measurements are shared in ``baselines``; the final PNG is the delivered
    image. Legacy synthetic SR evaluation continues through ``run_batch``.
    """
    order = list(order)
    if not order or len(order) != len(set(order)) or any(role not in {"sr", "deblur"} for role in order):
        raise ValueError("Select sr, deblur, or one order containing each once")
    sources = [Path(source).resolve() for source in sources]
    prepare_baselines(sources, baselines)
    models = [_model_record(role, Path(checkpoints[role])) for role in order]
    identity = [{key: model[key] for key in ("role", "path", "sha256")} for model in models]
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    combo_id = f"{'-then-'.join(order)}-{digest}"
    combo = {"id": combo_id, "order": order, "models": models, "rows": [],
             "error": None, "elapsed_seconds": None,
             "resources": {"cuda_peak_allocated_bytes": None, "cuda_peak_reserved_bytes": None}}
    started = time.perf_counter()
    stages = []
    descriptor = image = output = None
    cuda_devices = set()
    for source in sources:
        baseline = baselines.get(str(source), {})
        combo["rows"].append({
            "input": str(source), "input_size": baseline.get("input_size"),
            "output": None, "output_size": None, "status": "failed",
            "failure_stage": None, "reason": None, "elapsed_seconds": None,
            "before": baseline.get("before", failed_measurements("input unavailable")),
            "after": failed_measurements("output unavailable"),
            "changes": failed_measurements("output unavailable"),
        })
    try:
        output_dir = run_dir / combo_id
        try:
            output_dir.mkdir(exist_ok=False)
        except OSError as error:
            combo["error"] = f"Output directory: {type(error).__name__}: {error}"
            for row in combo["rows"]:
                row.update(failure_stage="output", reason=combo["error"])
            return combo

        try:
            for model in models:
                if model.get("error"):
                    raise ValueError(f"{model['role']} checkpoint: {model['error']}")
                descriptor = load_model(Path(model["path"]), role=model["role"])
                stages.append((model["role"], descriptor))
                _describe_model(model, descriptor)
                if descriptor.device.type == "cuda" and descriptor.device not in cuda_devices:
                    cuda_devices.add(descriptor.device)
                    torch.cuda.reset_peak_memory_stats(descriptor.device)
        except Exception as error:
            combo["error"] = f"Model load: {type(error).__name__}: {error}"
            for row in combo["rows"]:
                row.update(failure_stage="model_load", reason=combo["error"])
            return combo

        names = Counter(f"{source.stem}.png" for source in sources)
        protected = sources + [Path(model["path"]) for model in models]
        for source, row in zip(sources, combo["rows"]):
            row_started = time.perf_counter()
            stage = "output"
            destination = output_dir / f"{source.stem}.png"
            try:
                if names[destination.name] > 1:
                    raise ValueError(f"Multiple input files map to output: {destination.name}")
                if destination.exists() or destination.is_symlink():
                    raise ValueError(f"Refusing to reuse existing output: {destination}")
                if any(destination.resolve() == path for path in protected):
                    raise ValueError(f"Refusing to overwrite input or checkpoint: {destination}")
                stage = "decode"
                baseline = baselines[str(source)]
                if baseline.get("decode_error"):
                    raise ValueError(baseline["decode_error"])
                image = read_image(source)
                row["input_size"] = (image.shape[-1], image.shape[-2])
                row["before"] = baseline["before"]
                stage = "inference"
                output = run_stages(image, stages)
                stage = "write"
                write_png(output, destination, source)
                row.update(status="success", output=str(destination.resolve()),
                           output_size=(output.shape[-1], output.shape[-2]))
                # Score the delivered, quantized PNG rather than an in-memory prediction.
                delivered = None
                try:
                    delivered = read_image(destination)
                    row["output_size"] = (delivered.shape[-1], delivered.shape[-2])
                    row["after"] = measure_tensor(delivered)
                except Exception as error:
                    row["after"] = failed_measurements(f"output metrics: {type(error).__name__}: {error}")
                finally:
                    delivered = None
                row["changes"] = compare_measurements(row["before"], row["after"], applicable=order == ["deblur"])
            except Exception as error:
                row.update(failure_stage=stage, reason=f"{type(error).__name__}: {error}")
            finally:
                row["elapsed_seconds"] = time.perf_counter() - row_started
                image = output = None
        return combo
    finally:
        if cuda_devices:
            combo["resources"] = {
                "cuda_peak_allocated_bytes": sum(torch.cuda.max_memory_allocated(device) for device in cuda_devices),
                "cuda_peak_reserved_bytes": sum(torch.cuda.max_memory_reserved(device) for device in cuda_devices),
            }
        combo["resources"]["process_max_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        stages.clear()
        descriptor = image = output = None
        release_device_memory()
        combo["elapsed_seconds"] = time.perf_counter() - started


@dataclass(frozen=True)
class LineScores:
    psnr: float
    ssim: float
    lpips: float


@dataclass(frozen=True)
class ImageResult:
    """One image measured successfully on both lines. Sizes are (width, height)."""

    source_name: str
    original: tuple[int, int]
    cropped: tuple[int, int]
    low: tuple[int, int]
    sr: LineScores
    bicubic: LineScores


@dataclass(frozen=True)
class ImageFailure:
    source_name: str
    stage: str
    reason: str


def select_sources(directory: Path, limit: int = DEFAULT_LIMIT, seed: int | None = None) -> list[Path]:
    """The first *limit* sources in name order, or a reproducible random sample.

    Sequential by default so that a rerun with the same arguments compares the
    same images; a seed is required for the random path for the same reason.
    """
    found = discover_sources(directory)
    if limit is not None and limit < len(found):
        if seed is None:
            found = found[:limit]
        else:
            found = sorted(random.Random(seed).sample(found, limit), key=lambda path: path.name)
    return found


def release_device_memory() -> None:
    """Hand freed GPU blocks back to the driver. A no-op on CPU."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _score(truth, candidate, perceptual) -> LineScores:
    return LineScores(psnr=psnr(truth, candidate), ssim=ssim(truth, candidate), lpips=perceptual(truth, candidate))


def process_image(source: Path, run_dir: Path, sr_line, perceptual) -> ImageResult:
    """Decode, degrade, upscale both ways, measure. Raises _StageError on failure."""
    name = source.stem + ".png"
    hr_path, lr_path = run_dir / "hr" / name, run_dir / "lr" / name
    bicubic_path, sr_path = run_dir / "bicubic" / name, run_dir / "sr" / name

    with _stage("decode", source):
        image = decode_source(source)
    with _stage("degrade", source):
        cropped, record = mod_crop(image)
        save_png(cropped, hr_path)
        save_png(downscale(cropped), lr_path)
        low_size = _size_of(lr_path)
    with _stage("bicubic", source):
        upscale_bicubic(lr_path, record.cropped, bicubic_path)
    with _stage("sr", source):
        sr_line.run(lr_path, sr_path, record.cropped)
    # GOALS.md asks for this explicitly: on a 12227 MiB card, SR activations
    # still held while LPIPS allocates its own is the shape of an OOM.
    release_device_memory()
    with _stage("metrics", source):
        truth = to_metric_tensor(cropped)
        scores = {
            "bicubic": _score(truth, load_metric_tensor(bicubic_path), perceptual),
            "sr": _score(truth, load_metric_tensor(sr_path), perceptual),
        }

    return ImageResult(
        source_name=source.name,
        original=record.original,
        cropped=record.cropped,
        low=low_size,
        sr=scores["sr"],
        bicubic=scores["bicubic"],
    )


def run_batch(sources, run_dir: Path, sr_line, perceptual, progress=None):
    """Every source in order; returns (results, failures) and never raises for one image."""
    results, failures = [], []
    for index, source in enumerate(sources, start=1):
        if progress is not None:
            progress(index, len(sources), source.name)
        try:
            results.append(process_image(source, run_dir, sr_line, perceptual))
        except _StageError as error:
            failures.append(ImageFailure(source.name, error.stage, error.reason))
        release_device_memory()
    return results, failures


class _StageError(Exception):
    def __init__(self, stage: str, reason: str):
        super().__init__(f"{stage}: {reason}")
        self.stage = stage
        self.reason = reason


class _stage:
    """Turns whatever a step raises into a recordable (stage, reason) pair."""

    def __init__(self, name: str, source: Path):
        self.name = name
        self.source = source

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        if value is None or isinstance(value, _StageError):
            return False
        raise _StageError(self.name, f"{type(value).__name__}: {value}") from value


def _size_of(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size
