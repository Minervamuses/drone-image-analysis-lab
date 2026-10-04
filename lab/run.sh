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
import json
import os
import random
import sys
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


def parse_args():
    parser = argparse.ArgumentParser(
        prog="bash lab/run.sh",
        description="Assign each original-size image to one of four groups; run every deblur checkpoint.",
        epilog=(
            "Prepare PNG/JPG/JPEG files in evaluation/data/input/ (first level), "
            "checkpoints in models/deblur/ (recursive), and existing project/evaluation dependencies. "
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
    return {"linear": linear, "trajectory": rasterize(points), "gaussian": line @ line.T}


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
    measurements = measure_tensor(image)
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
    for index, source in enumerate(sources):
        condition = tuple(CONDITIONS)[index % len(CONDITIONS)]
        destination = run / "inputs" / (source.name + ".png")
        record = dict(
            image=source.name, condition=condition,
            blur_params=json.dumps(CONDITIONS[condition], sort_keys=True),
            original_path=str(source), input_path=str(destination), error="",
        )
        image = prepared = None
        try:
            image = read_image(source)
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


def main():
    args = parse_args()  # --help exits before importing ML packages or requiring data/weights.
    sources = sorted(path for path in INPUT.iterdir()
                     if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}) if INPUT.is_dir() else []
    random.Random(SEED).shuffle(sources)
    if args.limit is not None:
        sources = sources[:args.limit]
    checkpoints = sorted((path for path in MODELS.rglob("*")
                          if path.is_file() and path.suffix.lower() in {".pth", ".pt", ".ckpt", ".safetensors"}),
                         key=lambda path: path.relative_to(MODELS).as_posix())
    if not sources:
        raise SystemExit(f"error: no PNG/JPG/JPEG inputs in {INPUT}; prepare the lab data, no download attempted.")
    if not checkpoints:
        raise SystemExit(f"error: no deblur checkpoints in {MODELS}; prepare the lab weights, no download attempted.")
    global np, cv2, torch, read_image, write_png, measure_tensor
    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "evaluation")]
    try:
        import numpy as np
        import cv2
        import torch
        from drone_sr.image_io import read_image, write_png
        from blur_metrics import measure_tensor
    except ImportError as exc:
        raise SystemExit(f"error: existing lab dependencies are missing: {exc}; no installation attempted.") from exc
    run = RUNS / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run.mkdir(parents=True, exist_ok=False)
    print(f"Run: {run}; images: {len(sources)}; checkpoints: {len(checkpoints)}; seed: {SEED}", flush=True)
    sharpness = []
    records = prepare_inputs(sources, run, sharpness)
    print(f"Prepared {sum(not record['error'] for record in records)}/{len(records)} model inputs.", flush=True)


if __name__ == "__main__":
    main()
PY
