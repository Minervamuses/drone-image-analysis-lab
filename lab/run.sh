#!/usr/bin/env bash
# One deblur experiment; use the existing lab environment and prepared weights.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="$ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "error: prepare the existing project .venv and evaluation dependencies first." >&2
  exit 2
fi
export DEBLUR_REPO_ROOT="$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export TORCH_HOME="${TORCH_HOME:-$ROOT/models/torch-cache}"
exec "$PYTHON" -u - "$@" <<'PY'
import argparse
import csv
import gc
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.environ["DEBLUR_REPO_ROOT"])
INPUT = ROOT / "evaluation/data/input"
MODELS = ROOT / "models/deblur"
RUNS = ROOT / "evaluation/runs/deblur"
SEED = 923
CONDITIONS = {
    "none": {},
    "linear": {"length_px": 8, "angle_deg": 45},
    "trajectory": {"arc_deg": 90, "max_xy_span_px": 8},
    "gaussian": {"sigma_px": 2, "kernel_size": 13},
}
SHARPNESS_METRICS = ("laplacian_variance", "tenengrad", "cpbd", "crete_roffet_blur")
SHARPNESS_FIELDS = (
    "image", "kind", "model", "condition", "path", *SHARPNESS_METRICS,
    "status", "invalid_metrics", "error",
)
REFERENCE_FIELDS = (
    "image", "condition", "blur_params", "model", "original_path", "input_path", "output_path",
    "psnr", "ssim", "lpips", "status", "error",
)
SUMMARY_FIELDS = (
    "model", "n_expected", "n_success", "n_failed", "psnr_mean", "ssim_mean", "lpips_mean",
    "psnr_inf_count", "status",
)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="bash lab/run.sh",
        description="Assign each original-size image to one of four groups; run all non-excluded deblur checkpoints.",
        epilog=(
            "Prepare PNG/JPG/JPEG files in evaluation/data/input/ (first level), "
            "checkpoints in models/deblur/ (recursive), and existing project/evaluation dependencies. "
            "NAFNet-GoPro-width64.pth is skipped due to corrupted outputs, as authorized by the user. "
            "LPIPS requires the AlexNet cache at TORCH_HOME/hub/checkpoints/"
            "alexnet-owt-7be5be79.pth (default TORCH_HOME: models/torch-cache). "
            "No automatic installation, download, resizing, or CPU inference. "
            "New outputs: evaluation/runs/deblur/<UTC time>/, with full_reference.csv, "
            "summary.csv and raw sharpness.csv. --limit is applied after the seed=923 shuffle."
        ),
    )
    parser.add_argument("--limit", type=int, help="Use the first N shuffled images for a short lab run")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    return args


def rasterize(points):
    # Bilinear splatting preserves the uniform trajectory's time-weighted centroid.
    points = points - points.mean(axis=0)
    radius = int(np.ceil(np.max(np.abs(points)))) + 1
    kernel = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=np.float64)
    coords = points + radius
    base = np.floor(coords).astype(int)
    fraction = coords - base
    for dy, dx in ((0, 0), (0, 1), (1, 0), (1, 1)):
        weight = ((fraction[:, 0] if dx else 1 - fraction[:, 0])
                  * (fraction[:, 1] if dy else 1 - fraction[:, 1]))
        np.add.at(kernel, (base[:, 1] + dy, base[:, 0] + dx), weight)
    return kernel / kernel.sum()


def make_kernels():
    line = CONDITIONS["linear"]
    offsets = np.linspace(-line["length_px"] / 2, line["length_px"] / 2, 257)
    theta = np.deg2rad(line["angle_deg"])
    # As in the existing synthesis: angle from image up, x right and y down.
    linear = rasterize(np.column_stack((offsets * np.sin(theta), -offsets * np.cos(theta))))
    arc = CONDITIONS["trajectory"]
    theta = np.linspace(0, np.deg2rad(arc["arc_deg"]), 257)
    points = np.column_stack((np.cos(theta), np.sin(theta)))
    points *= arc["max_xy_span_px"] / np.ptp(points, axis=0).max()
    gaussian = CONDITIONS["gaussian"]
    line = cv2.getGaussianKernel(gaussian["kernel_size"], gaussian["sigma_px"], cv2.CV_64F)
    gaussian = line @ line.T
    return {"linear": linear, "trajectory": rasterize(points), "gaussian": gaussian / gaussian.sum()}


def blur_image(image, condition, kernels):
    if condition == "none":
        return image
    value = image.squeeze(0).permute(1, 2, 0).numpy().astype(np.float64)
    light = np.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055) ** 2.4)
    # filter2D is correlation; flip the PSF for convolution, reflecting the full image boundary.
    blurred = cv2.filter2D(light, -1, kernels[condition][::-1, ::-1], borderType=cv2.BORDER_REFLECT)
    blurred = np.clip(blurred, 0, 1)
    encoded = np.where(blurred <= 0.0031308, 12.92 * blurred, 1.055 * blurred ** (1 / 2.4) - 0.055)
    return torch.from_numpy(encoded.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)


def sharpness_row(record, kind, model, path, image=None, error=""):
    row = dict(image=record["image"], kind=kind, model=model, condition=record["condition"], path=str(path))
    if image is None:
        row.update({name: None for name in SHARPNESS_METRICS})
        row.update(status="failed", invalid_metrics=";".join(SHARPNESS_METRICS), error=error)
        return row
    try:
        measurements = measure_tensor(image)
    except Exception as exc:
        return sharpness_row(record, kind, model, path, error=f"sharpness: {type(exc).__name__}: {exc}")
    invalid = [name for name in SHARPNESS_METRICS if not measurements[name]["valid"]]
    row.update({name: measurements[name]["value"] for name in SHARPNESS_METRICS})
    row.update(
        status="ok" if not invalid else ("failed" if len(invalid) == len(SHARPNESS_METRICS) else "partial"),
        invalid_metrics=";".join(invalid),
        error="; ".join(f"{name}: {measurements[name]['status']}: {measurements[name]['reason']}" for name in invalid),
    )
    return row


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prepare_inputs(sources, run, sharpness):
    kernels = make_kernels()
    records = []
    reserved = set()
    for index, source in enumerate(sources):
        condition = tuple(CONDITIONS)[index % len(CONDITIONS)]
        destination = unique_output_path(run / "inputs" / f"{source.stem}.png", reserved)
        record = dict(
            image=source.name, condition=condition,
            blur_params=json.dumps(CONDITIONS[condition], sort_keys=True),
            original_path=str(source), input_path=str(destination), error="",
        )
        image = prepared = None
        try:
            image = read_image(source)
            print(f"Input {source.name}: {image.shape[-1]}x{image.shape[-2]}", flush=True)
            sharpness.append(sharpness_row(record, "original", "", source, image))
            prepared = blur_image(image, condition, kernels)
            write_png(prepared, destination, source)
        except Exception as exc:
            record["error"] = f"preparation: {type(exc).__name__}: {exc}"
            if image is None:
                sharpness.append(sharpness_row(record, "original", "", source, error=record["error"]))
            print(f"FAIL {source.name}: {record['error']}", flush=True)
        finally:
            del image, prepared
        records.append(record)
        write_csv(run / "sharpness.csv", sharpness, SHARPNESS_FIELDS)
        print(f"Prepared {index + 1}/{len(sources)}: {source.name} -> {condition}", flush=True)
    return records


def infer_model(checkpoint, records, run):
    model = checkpoint.relative_to(MODELS).as_posix()
    descriptor, load_error = None, ""
    rows = []
    try:
        descriptor = load_model(checkpoint, role="deblur")
        if descriptor.device.type != "cuda":
            raise RuntimeError("deblur must run on CUDA; CPU fallback is disabled")
        print(f"Model: {model}; architecture: {descriptor.architecture.name}; device: {descriptor.device}", flush=True)
    except Exception as exc:
        load_error = f"model load: {type(exc).__name__}: {exc}"
        print(f"FAIL {model}: {load_error}", flush=True)
    try:
        for index, record in enumerate(records):
            destination = unique_output_path(run / "outputs" / model / Path(record["input_path"]).name)
            row = dict(record, model=model, output_path=str(destination), psnr=None, ssim=None, lpips=None,
                       status="failed", error="; ".join(filter(None, (record["error"], load_error))))
            image = result = None
            try:
                if not row["error"]:
                    image = read_image(Path(record["input_path"]))
                    result = upscale(image, descriptor)
                    if result.shape != image.shape:
                        raise ValueError(f"Expected original-size {tuple(image.shape)}, got {tuple(result.shape)}")
                    write_png(result, destination, Path(record["input_path"]))
                    row["status"] = "saved"
            except Exception as exc:
                row["error"] = f"inference/save: {type(exc).__name__}: {exc}"
            finally:
                del image, result
                if row["status"] == "failed":
                    torch.cuda.empty_cache()
            rows.append(row)
            print(f"Inference {model} {index + 1}/{len(records)}: {row['status']} {row['error']}", flush=True)
    finally:
        # LPIPS is loaded only after the restoration model and its tensors are gone.
        del descriptor
        gc.collect()
        torch.cuda.empty_cache()
    return rows


def score_outputs(rows, sharpness):
    perceptual, lpips_error = None, ""
    if any(row["status"] == "saved" for row in rows):
        try:
            perceptual = PerceptualMetric(device="cuda:0")
        except Exception as exc:
            lpips_error = f"LPIPS load: {type(exc).__name__}: {exc}"
            gc.collect()
            torch.cuda.empty_cache()
            print(f"FAIL {lpips_error}", flush=True)
    try:
        for row in rows:
            if row["status"] != "saved":
                sharpness.append(sharpness_row(row, "model_output", row["model"], row["output_path"], error=row["error"]))
                continue
            original = restored = None
            sharpness_written = False
            errors = []
            try:
                # Measure the delivered PNG; GT is always the EXIF/MPO-decoded original.
                restored = read_image(Path(row["output_path"]))
                sharpness.append(sharpness_row(row, "model_output", row["model"], row["output_path"], restored))
                sharpness_written = True
                original = read_image(Path(row["original_path"]))
                # Match metrics.py's integer RGB levels [0,255], without float32 decode roundoff.
                original.mul_(255).round_()
                restored.mul_(255).round_()
                for name, metric in (("psnr", psnr), ("ssim", ssim), ("lpips", perceptual)):
                    try:
                        if metric is None:
                            raise RuntimeError(lpips_error)
                        value = float(metric(original, restored))
                        if not math.isfinite(value) and not (name == "psnr" and value == math.inf):
                            raise ValueError("non-finite metric result")
                        row[name] = value
                    except Exception as exc:
                        errors.append(f"{name}: {type(exc).__name__}: {exc}")
            except Exception as exc:
                errors.append(f"metric input: {type(exc).__name__}: {exc}")
            finally:
                del original, restored
                if errors:
                    torch.cuda.empty_cache()
            row.update(status="failed" if errors else "ok", error="; ".join(errors))
            if not sharpness_written:
                sharpness.append(sharpness_row(row, "model_output", row["model"], row["output_path"], error=row["error"]))
            print(f"Metrics {row['model']} {row['image']}: {row['status']} {row['error']}", flush=True)
    finally:
        del perceptual
        gc.collect()
        torch.cuda.empty_cache()


def summarize(model, rows):
    successful = [row for row in rows if row["status"] == "ok"]
    count = len(successful)
    summary = dict(model=model, n_expected=len(rows), n_success=count, n_failed=len(rows) - count,
                   psnr_inf_count=sum(row["psnr"] == math.inf for row in successful),
                   status="partial" if count < len(rows) else "ok")
    for name in ("psnr", "ssim", "lpips"):
        summary[name + "_mean"] = sum(row[name] for row in successful) / count if count else None
    return summary


def main():
    args = parse_args()  # --help exits before importing ML packages or requiring data/weights.
    started = time.perf_counter()
    sources = sorted(path for path in INPUT.iterdir()
                     if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}) if INPUT.is_dir() else []
    random.Random(SEED).shuffle(sources)
    if args.limit is not None:
        sources = sources[:args.limit]
    checkpoints = sorted((path for path in MODELS.rglob("*")
                          if path.is_file() and path.suffix.lower() in {".pth", ".pt", ".ckpt", ".safetensors"}),
                         key=lambda path: path.relative_to(MODELS).as_posix())
    # The user authorized skipping this checkpoint after visible color/checkerboard artifacts.
    excluded_checkpoint = "NAFNet-GoPro-width64.pth"
    for checkpoint in checkpoints:
        if checkpoint.name == excluded_checkpoint:
            print(f"SKIP {checkpoint.relative_to(MODELS).as_posix()}: known corrupted output; "
                  "user authorized skipping NAFNet on 2026-10-04", flush=True)
    checkpoints = [path for path in checkpoints if path.name != excluded_checkpoint]
    if not sources:
        raise SystemExit(f"error: no PNG/JPG/JPEG inputs in {INPUT}; prepare the lab data, no download attempted.")
    if not checkpoints:
        raise SystemExit(f"error: no runnable deblur checkpoints in {MODELS} after the authorized NAFNet exclusion; "
                         "prepare the lab weights, no download attempted.")
    global np, cv2, torch, read_image, unique_output_path, write_png, measure_tensor, load_model, upscale, psnr, ssim, PerceptualMetric
    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "evaluation")]
    try:
        import numpy as np
        import cv2
        import torch
        import lpips
        from drone_sr.image_io import read_image, unique_output_path, write_png
        from drone_sr.inference import load_model, upscale
        from blur_metrics import measure_tensor
        from metrics import psnr, ssim
        from perceptual import PerceptualMetric
    except Exception as exc:
        raise SystemExit(f"error: existing lab dependencies are unavailable: {type(exc).__name__}: {exc}; no installation attempted.") from exc

    def no_download(*args, **kwargs):
        raise RuntimeError("Automatic weight downloads are disabled; prepare the existing lab cache manually")

    torch.hub.download_url_to_file = no_download
    cache = Path(torch.hub.get_dir()) / "checkpoints/alexnet-owt-7be5be79.pth"
    calibration = Path(lpips.__file__).resolve().parent / "weights/v0.1/alex.pth"
    for path in (cache, calibration):
        if not path.is_file():
            raise SystemExit(f"error: required existing LPIPS weights missing: {path}; no download attempted.")
    if not torch.cuda.is_available():
        raise SystemExit("error: CUDA is unavailable; fix lab GPU access. CPU fallback is disabled.")
    try:
        torch.ones(1, device="cuda:0").sum().item()
        free, total = torch.cuda.mem_get_info()
    except Exception as exc:
        raise SystemExit(f"error: lab CUDA preflight failed: {type(exc).__name__}: {exc}") from exc
    print(f"torch: {torch.__version__}; CUDA runtime: {torch.version.cuda}; GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}; free/total VRAM bytes: {free}/{total}", flush=True)
    print(f"PSNR/SSIM: CPU float64; LPIPS: cuda:0; cache: {cache}", flush=True)
    print(Path("/proc/meminfo").read_text().splitlines()[:3], flush=True)
    for name in ("memory.max", "memory.current"):
        path = Path("/sys/fs/cgroup") / name
        if path.is_file():
            print(f"cgroup {name}: {path.read_text().strip()}", flush=True)
    cv2.setNumThreads(1)
    run = RUNS / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run.mkdir(parents=True, exist_ok=False)
    print(f"Run: {run}; images: {len(sources)}; checkpoints: {len(checkpoints)}; seed: {SEED}", flush=True)
    sharpness, reference, summaries = [], [], []
    write_csv(run / "full_reference.csv", reference, REFERENCE_FIELDS)
    write_csv(run / "summary.csv", summaries, SUMMARY_FIELDS)
    write_csv(run / "sharpness.csv", sharpness, SHARPNESS_FIELDS)
    records = prepare_inputs(sources, run, sharpness)
    print(f"Prepared {sum(not record['error'] for record in records)}/{len(records)} model inputs.", flush=True)
    for checkpoint in checkpoints:
        model_started = time.perf_counter()
        rows = infer_model(checkpoint, records, run)
        reference.extend(rows)
        write_csv(run / "full_reference.csv", reference, REFERENCE_FIELDS)
        score_outputs(rows, sharpness)
        summary = summarize(checkpoint.relative_to(MODELS).as_posix(), rows)
        summaries.append(summary)
        write_csv(run / "full_reference.csv", reference, REFERENCE_FIELDS)
        write_csv(run / "summary.csv", summaries, SUMMARY_FIELDS)
        write_csv(run / "sharpness.csv", sharpness, SHARPNESS_FIELDS)
        print(f"Summary {summary['model']}: {summary['n_success']}/{summary['n_expected']} complete scores; "
              f"{summary['status']}; {time.perf_counter() - model_started:.1f}s", flush=True)
    print(f"Finished: {run}; total elapsed: {time.perf_counter() - started:.1f}s", flush=True)
    return 1 if any(summary["n_failed"] for summary in summaries) else 0


if __name__ == "__main__":
    raise SystemExit(main())
PY
