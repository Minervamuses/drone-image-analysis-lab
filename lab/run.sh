#!/usr/bin/env bash
# Server-only deblur benchmark. All implementation lives in this entry point.
# No installation/downloads: use the existing project + evaluation dependencies.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export DEBLUR_REPO_ROOT="$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export TORCH_HOME="${TORCH_HOME:-$ROOT/models/torch-cache}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  cat <<'HELP'
Usage (on the lab server, after git pull):
  bash lab/run.sh --prepare       # organize data only; no imports of ML packages
  bash lab/run.sh --smoke         # 2 GT photos x 1 crop x C0/LIN-8/TRAJ-8
  bash lab/run.sh                 # smoke first, then full Part A + Part B

Place the original folder, including blur_metadata.csv, at:
  evaluation/data/923海上正攝/
The script creates evaluation/data/input/ as relative symlinks and copies the
metadata to evaluation/data/blur_metadata.csv. Existing flattened input + CSV
are also supported. Originals and previous experiment runs are never replaced.

Place checkpoints (no automatic downloads) at:
  models/deblur/                  # .pth/.pt/.ckpt/.safetensors, including subdirs
LPIPS also needs an existing AlexNet cache:
  models/torch-cache/hub/checkpoints/alexnet-owt-7be5be79.pth
Set TORCH_HOME to use another prepared cache. LPIPS calibration comes from lpips.

Options:
  --source DIR          Original folder (default: evaluation/data/923海上正攝)
  --metadata CSV        CSV to organize (default: source/blur_metadata.csv)
  --models DIR          Checkpoint folder (default: models/deblur)
  --runs-root DIR       Output parent (default: evaluation/runs/deblur_benchmark)
  --seed N              Random seed (default: 923)
  --k N                 Part A crops per GT photo (default: 6)
  --prepare | --smoke   Default runs smoke then full experiment
  --help                This help

Use the existing .venv; PYTHON=/path/to/python can select the server environment.
Each phase writes a new UTC directory, CSVs, summary.md, kernels and samples.
A failed checkpoint is recorded and other checkpoints continue; failed smoke
models are excluded from the full run. Nonzero exit means a recorded failure.
Full run is intentionally long (spec estimates 3-4 hours on A6000; unverified).
HELP
  exit 0
fi
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  if [[ " $* " == *" --prepare "* ]]; then
    PYTHON="$(command -v python3)"
  else
    echo "error: prepare the existing .venv and evaluation/requirements.txt on the server first." >&2
    exit 2
  fi
fi
exec "$PYTHON" -u - "$@" <<'PY'
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.environ["DEBLUR_REPO_ROOT"])
SIZE, PAD, BORDER = 512, 64, 32
LENGTHS = (4, 8, 16, 32, 64)
BALANCES = (0.002, 0.005, 0.01, 0.02, 0.05, 0.1)
NR = ("laplacian_variance", "tenengrad", "cpbd", "crete_roffet_blur", "cpbd_min_dir")
CONDITIONS = ["C0"] + [f"{kind}-{length}" for kind in ("LIN", "TRAJ") for length in LENGTHS]
FIELDS = [
    "block_id", "source", "source_path", "point", "alt_folder", "iso", "x", "y",
    "glint_frac", "condition", "L", "angle_deg", "group", "method", "status", "error",
    "seconds", "metric_seconds", "psnr", "ssim", "lpips", "balance", "nr_errors",
] + [f"{prefix}_{metric}" for prefix in ("gt", "input", "output", "delta_gt", "delta_input") for metric in NR]


def parse_args():
    parser = argparse.ArgumentParser(description="Server deblur benchmark; bash lab/run.sh --help for placement.")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--smoke", action="store_true")
    parser.add_argument("--source", type=Path, default=ROOT / "evaluation/data/923海上正攝")
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--models", type=Path, default=ROOT / "models/deblur")
    parser.add_argument("--runs-root", type=Path, default=ROOT / "evaluation/runs/deblur_benchmark")
    parser.add_argument("--seed", type=int, default=923)
    parser.add_argument("--k", type=int, default=6)
    args = parser.parse_args()
    if args.k < 1:
        parser.error("--k must be positive")
    return args


def number(row, key, required=True):
    try:
        value = float(row[key])
        if not math.isfinite(value):
            raise ValueError("non-finite")
        return value
    except (KeyError, ValueError, TypeError):
        if required:
            raise ValueError(f"{row.get('name', '?')}: missing/invalid {key}")
        return None


def prepare_data(args):
    """Resolve every CSV row before writing; symlink JPGs without moving originals."""
    data = ROOT / "evaluation/data"
    target_csv, flat = data / "blur_metadata.csv", data / "input"
    candidate = args.source / "blur_metadata.csv"
    metadata = args.metadata or (candidate if candidate.is_file() else target_csv)
    with metadata.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        rows, columns = list(reader), reader.fieldnames
    required = {"name", "point", "alt_folder", "condition", "exposure_s", "iso",
                "gimbal_pitch_deg", "speed_h_mps", "pred_motion_blur_px",
                "motion_dir_vs_image_up_deg"}
    if not columns or not required.issubset(columns) or not rows:
        raise ValueError(f"Metadata must be nonempty and contain {sorted(required)}")
    names = [row["name"] for row in rows]
    if len({name.casefold() for name in names}) != len(names):
        raise ValueError("Duplicate image names in metadata; cannot flatten safely")
    mapping = []
    for row in rows:
        name = row["name"]
        if Path(name).name != name or "/" in name or "\\" in name or Path(name).suffix.lower() not in {".jpg", ".jpeg"}:
            raise ValueError(f"Invalid JPG basename: {name}")
        for field in ("exposure_s", "iso", "gimbal_pitch_deg", "speed_h_mps", "pred_motion_blur_px"):
            row[field] = number(row, field)
        row["motion_dir_vs_image_up_deg"] = number(row, "motion_dir_vs_image_up_deg", required=False)
        relative = Path(row.get("path", name))
        if relative.is_absolute() or ".." in relative.parts or relative.name != name:
            raise ValueError(f"{name}: unsafe or inconsistent metadata path: {relative}")
        sources = [args.source / relative, args.source / name]
        source = next((path for path in sources if path.is_file()), None)
        if source is None and (flat / name).is_file():
            source = flat / name
        if source is None:
            raise FileNotFoundError(f"{name}: expected {args.source / relative} (or {flat / name})")
        destination = flat / name
        if os.path.lexists(destination) and (not destination.is_file() or not destination.samefile(source)):
            raise FileExistsError(f"Refusing to replace existing input: {destination}")
        row["source_path"] = str(source.resolve())
        mapping.append((source.resolve(), destination))
    if target_csv.exists() and metadata.resolve() != target_csv.resolve():
        if target_csv.read_bytes() != metadata.read_bytes():
            raise FileExistsError(f"Existing {target_csv} differs from supplied CSV; move the old CSV aside explicitly")
    flat.mkdir(parents=True, exist_ok=True)
    args.models.mkdir(parents=True, exist_ok=True)
    (Path(os.environ["TORCH_HOME"]) / "hub/checkpoints").mkdir(parents=True, exist_ok=True)
    for source, destination in mapping:
        if not destination.exists():
            destination.symlink_to(os.path.relpath(source, destination.parent))
    if metadata.resolve() != target_csv.resolve() and not target_csv.exists():
        shutil.copy2(metadata, target_csv)
    rows.sort(key=lambda row: row["name"])
    print(f"Prepared {len(rows)} JPG mappings: {flat}", flush=True)
    return rows, target_csv


def gt_eligible(row):
    return (row["gimbal_pitch_deg"] <= -85 and row["pred_motion_blur_px"] < 1
            and row["iso"] <= 800 and (
                row["exposure_s"] <= 1 / 100 or
                (row["condition"] == "靜止照片" and row["exposure_s"] <= 1 / 40 and row["speed_h_mps"] < 0.3)))


def initialize_runtime(seed):
    # Import only after --prepare has returned; no framework is needed to organize data.
    global np, cv2, torch, Image, ImageDraw, read_image, load_model, upscale
    global psnr, ssim, PerceptualMetric, blur, sk_wiener
    import numpy as np
    import cv2
    import torch
    from PIL import Image, ImageDraw
    from skimage.restoration import wiener as sk_wiener
    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "evaluation")]
    from drone_sr.image_io import read_image
    from drone_sr.inference import load_model, upscale
    from metrics import psnr, ssim
    from perceptual import PerceptualMetric
    import blur_metrics as blur
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: this benchmark never falls back to CPU inference")
    cache = Path(torch.hub.get_dir()) / "checkpoints/alexnet-owt-7be5be79.pth"
    if not cache.is_file():
        raise FileNotFoundError(f"LPIPS AlexNet must be prepared on the server at {cache}; no download was attempted")

    def no_download(*args, **kwargs):
        raise RuntimeError("Automatic weight downloads are disabled; prepare the server cache manually")
    torch.hub.download_url_to_file = no_download
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(max(1, int(os.environ["OMP_NUM_THREADS"])))
    cv2.setNumThreads(1)
    torch.ones(1, device="cuda").sum().item()
    free, total = torch.cuda.mem_get_info()
    print(f"CUDA: {torch.cuda.get_device_name(0)}; torch {torch.__version__}; CUDA {torch.version.cuda}", flush=True)
    print(f"VRAM free/total: {free / 2**30:.2f}/{total / 2**30:.2f} GiB", flush=True)
    return {"device": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
            "vram_free_bytes": free, "vram_total_bytes": total,
            "versions": {"python": sys.version, "torch": torch.__version__,
                         "numpy": np.__version__, "opencv": cv2.__version__}}


def rng_for(seed, *labels):
    digest = hashlib.sha256((str(seed) + "|" + "|".join(map(str, labels))).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def as_rgb(tensor):
    return tensor.detach().cpu().squeeze(0).clamp(0, 1).mul(255).round().to(torch.uint8).permute(1, 2, 0).numpy()


def metric_tensor(rgb):
    # Same RGB [0,255] BCHW convention as evaluation/metrics.py.
    return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).unsqueeze(0).float()


def load_rgb(path):
    with Image.open(path) as image:
        return np.array(image.convert("RGB"))


def save_rgb(path, rgb):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path)


def linear(rgb):
    value = rgb.astype(np.float64) / 255.0
    return np.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055) ** 2.4)


def srgb(value):
    value = np.clip(value, 0, 1)
    return np.where(value <= 0.0031308, 12.92 * value, 1.055 * value ** (1 / 2.4) - 0.055)


def rasterize(points):
    # Subpixel bilinear splatting retains the trajectory's time-weighted centroid.
    points = points - points.mean(axis=0)
    radius = int(np.ceil(np.max(np.abs(points)))) + 1
    kernel = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=np.float64)
    coords = points + radius
    base = np.floor(coords).astype(int)
    fraction = coords - base
    for dy, dx in ((0, 0), (0, 1), (1, 0), (1, 1)):
        weight = (fraction[:, 0] if dx else 1 - fraction[:, 0]) * (fraction[:, 1] if dy else 1 - fraction[:, 1])
        np.add.at(kernel, (base[:, 1] + dy, base[:, 0] + dx), weight)
    return kernel / kernel.sum()


def line_kernel(length, angle):
    if length <= 0:
        return np.ones((1, 1), dtype=np.float64)
    offsets = np.linspace(-length / 2, length / 2, max(33, int(np.ceil(length * 32)) + 1))
    theta = np.deg2rad(angle)
    # Angle is relative to image up: x right, y down.
    return rasterize(np.column_stack((offsets * np.sin(theta), -offsets * np.cos(theta))))


def trajectory_kernel(length, rng):
    # Inertial random walk inspired by Boracchi & Foi / DeblurGAN.
    # Gaussian accelerations + occasional reversals, then normalize max XY span.
    # https://github.com/KupynOrest/DeblurGAN/blob/master/motion_blur/generate_trajectory.py
    points = np.zeros((1024, 2), dtype=np.float64)
    theta = rng.uniform(0, 2 * np.pi)
    velocity = np.array([np.cos(theta), np.sin(theta)])
    for index in range(1, len(points)):
        acceleration = rng.normal(0, 0.12, 2) - 0.001 * points[index - 1]
        if rng.random() < 0.015:
            acceleration -= 1.8 * velocity
        velocity = 0.92 * velocity + acceleration
        velocity /= max(float(np.linalg.norm(velocity)), 1e-12)
        points[index] = points[index - 1] + velocity
    span = float(np.ptp(points, axis=0).max())
    if span <= 0:
        raise ValueError("Degenerate trajectory")
    return rasterize(points * (length / span))


def degrade(context, kernel, rng, condition):
    if condition == "C0":
        # C0's explicit no-harm definition takes precedence over glint compensation.
        encoded = context[PAD:-PAD, PAD:-PAD].astype(np.float64)
    else:
        light = linear(context)
        light[np.max(context, axis=2) >= 250] *= 4
        # filter2D is correlation: flip the PSF to perform true convolution.
        blurred = cv2.filter2D(light, -1, kernel[::-1, ::-1], borderType=cv2.BORDER_REFLECT)
        encoded = srgb(blurred[PAD:-PAD, PAD:-PAD]) * 255
    pixels = np.rint(np.clip(encoded + rng.normal(0, 1, encoded.shape), 0, 255)).astype(np.uint8)
    ok, jpeg = cv2.imencode(".jpg", cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return cv2.cvtColor(cv2.imdecode(jpeg, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def wiener_rgb(rgb, kernel, balance):
    # Reflect extension reduces FFT periodic-edge ringing; use the SAME PSF as synthesis.
    padding = max(32, max(kernel.shape) // 2 + 1)
    light = np.pad(linear(rgb), ((padding, padding), (padding, padding), (0, 0)), mode="reflect")
    restored = np.empty_like(light)
    for channel in range(3):
        restored[..., channel] = sk_wiener(light[..., channel], kernel, balance, clip=False)
    return np.rint(srgb(restored[padding:-padding, padding:-padding]) * 255).astype(np.uint8)


def nr_metrics(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    values, errors = {}, []
    for name, function in (
        ("laplacian_variance", blur.laplacian_variance),
        ("tenengrad", blur.tenengrad), ("cpbd", blur.cpbd_score),
        ("crete_roffet_blur", blur.crete_roffet_blur),
        ("cpbd_min_dir", lambda image: min(blur.cpbd_score(image), blur.cpbd_score(image.T))),
    ):
        try:
            value = float(function(gray))
            values[name] = value if math.isfinite(value) else None
            if values[name] is None:
                errors.append(f"{name}: non-finite/unmeasurable")
        except Exception as exc:
            values[name] = None
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    return values, "; ".join(errors)


def nr_change(before, after, log_energy=False):
    changes = {}
    for name in NR:
        left, right = before.get(name), after.get(name)
        if left is None or right is None:
            value = None
        elif log_energy and name in NR[:2]:
            value = math.log2(right / left) if left > 0 and right > 0 else None
        else:
            value = right - left
        changes[name] = value
    return changes


def group_for(length):
    return "[0,2)" if length < 2 else "[2,10)" if length < 10 else "[10,30)" if length < 30 else "[30,inf)"


def write_csv(path, rows, fields=FIELDS):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def json_save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def failure(failures, stage, error, source="", method=""):
    record = {"stage": stage, "source": source, "method": method, "error": str(error)}
    failures.append(record)
    print(f"FAIL [{stage}] {method} {source}: {error}", flush=True)


def create_pairs(out, photos, k, conditions, seed, angles, real, failures):
    records = []
    for row in photos:
        try:
            image = as_rgb(read_image(Path(row["source_path"])))
            height, width = image.shape[:2]
            margin = 0 if real else PAD
            if min(height, width) < SIZE + 2 * margin:
                raise ValueError(f"Image too small for {SIZE}px crop + {margin}px context: {width}x{height}")
            rng = rng_for(seed, "real-crops" if real else "synth-crops", row["name"])
            for crop_index in range(k):
                x = int(rng.integers(margin, width - SIZE - margin + 1))
                y = int(rng.integers(margin, height - SIZE - margin + 1))
                block = f"{Path(row['name']).stem}_{crop_index:02d}"
                gt = image[y:y + SIZE, x:x + SIZE].copy()
                common = {"block_id": block, "source": row["name"], "source_path": row["source_path"],
                          "point": row["point"], "alt_folder": row["alt_folder"], "iso": row["iso"],
                          "x": x, "y": y, "glint_frac": float((gt.max(axis=2) >= 250).mean())}
                if real:
                    path = out / "pairs/real" / f"{block}.png"
                    save_rgb(path, gt)
                    nr, errors = nr_metrics(gt)
                    length, angle = row["pred_motion_blur_px"], row["motion_dir_vs_image_up_deg"]
                    record = dict(common, condition="real", L=length, angle_deg=angle,
                                  group=group_for(length), input_path=str(path), input_nr=nr,
                                  nr_errors=errors, kernel_path=None)
                    if length >= 2 and angle is not None:
                        kernel_path = out / "kernels" / f"real_{block}.npy"
                        np.save(kernel_path, line_kernel(length, angle))
                        record["kernel_path"] = str(kernel_path)
                    records.append(record)
                    continue
                context = image[y - PAD:y + SIZE + PAD, x - PAD:x + SIZE + PAD]
                gt_path = out / "pairs/gt" / f"{block}.png"
                save_rgb(gt_path, gt)
                gt_nr, gt_errors = nr_metrics(gt[BORDER:-BORDER, BORDER:-BORDER])
                for condition in conditions:
                    krng = rng_for(seed, "kernel", block, condition)
                    if condition == "C0":
                        length, angle, kernel = 0, None, np.ones((1, 1), dtype=np.float64)
                    else:
                        kind, length_str = condition.split("-")
                        length = int(length_str)
                        angle = float(krng.choice(angles) * krng.choice([-1, 1])) if kind == "LIN" else None
                        kernel = line_kernel(length, angle) if kind == "LIN" else trajectory_kernel(length, krng)
                    kernel_path = out / "kernels" / f"{block}_{condition}.npy"
                    np.save(kernel_path, kernel)
                    degraded = degrade(context, kernel, rng_for(seed, "noise", block, condition), condition)
                    path = out / "pairs" / condition / f"{block}.png"
                    save_rgb(path, degraded)
                    nr, errors = nr_metrics(degraded[BORDER:-BORDER, BORDER:-BORDER])
                    records.append(dict(common, condition=condition, L=length, angle_deg=angle,
                                        group="", input_path=str(path), gt_path=str(gt_path),
                                        kernel_path=str(kernel_path), gt_nr=gt_nr, input_nr=nr,
                                        nr_errors="; ".join(filter(None, [gt_errors, errors]))))
        except Exception as exc:
            failure(failures, "real_pairs" if real else "synth_pairs", exc, row["name"])
    return records


def tune_wiener(records, conditions):
    balances = {}
    for condition in conditions:
        calibration = [record for record in records if record["condition"] == condition][:10]
        scores = []
        for balance in BALANCES:
            values = []
            for record in calibration:
                rgb = wiener_rgb(load_rgb(record["input_path"]), np.load(record["kernel_path"]), balance)
                gt = load_rgb(record["gt_path"])
                values.append(psnr(metric_tensor(rgb[BORDER:-BORDER, BORDER:-BORDER]),
                                   metric_tensor(gt[BORDER:-BORDER, BORDER:-BORDER])))
            scores.append(float(np.mean(values)))
        balances[condition] = BALANCES[int(np.argmax(scores))]
        print(f"Wiener {condition}: balance={balances[condition]} ({len(calibration)} calibration crops)", flush=True)
    return balances


def process(rgb, method, descriptor, record, balances):
    if method == "identity":
        return rgb.copy(), None
    if method == "unsharp":
        floating = rgb.astype(np.float64)
        smooth = cv2.GaussianBlur(floating, (0, 0), 2, borderType=cv2.BORDER_REFLECT)
        return np.rint(np.clip(floating + (floating - smooth), 0, 255)).astype(np.uint8), None
    if method == "wiener":
        condition = record["condition"]
        if condition == "real":
            nearest = min(LENGTHS, key=lambda length: abs(math.log2(record["L"] / length)))
            condition = f"LIN-{nearest}"
        balance = balances[condition]
        return wiener_rgb(rgb, np.load(record["kernel_path"]), balance), balance
    output = upscale(metric_tensor(rgb).div_(255), descriptor)
    if tuple(output.shape) != (1, 3, SIZE, SIZE) or not torch.isfinite(output).all():
        raise ValueError("Model output must be finite RGB 512x512")
    return as_rgb(output), None


def measure_output(record, rgb, method, seconds, balance, perceptual, real):
    row = {key: record.get(key) for key in FIELDS if key in record}
    row.update(method=method, status="ok", error="", seconds=seconds, balance=balance)
    started = time.perf_counter()
    measured = rgb if real else rgb[BORDER:-BORDER, BORDER:-BORDER]
    values, errors = (record["input_nr"], "") if method == "identity" else nr_metrics(measured)
    row["nr_errors"] = "; ".join(filter(None, [record.get("nr_errors", ""), errors]))
    for name in NR:
        row[f"input_{name}"], row[f"output_{name}"] = record["input_nr"][name], values[name]
    for name, value in nr_change(record["input_nr"], values).items():
        row[f"delta_input_{name}"] = value
    if not real:
        gt = load_rgb(record["gt_path"])[BORDER:-BORDER, BORDER:-BORDER]
        reference, output = metric_tensor(gt), metric_tensor(measured)
        row.update(psnr=psnr(reference, output), ssim=ssim(reference, output), lpips=perceptual(reference, output))
        if not all(math.isfinite(row[name]) for name in ("psnr", "ssim", "lpips")):
            raise ValueError("Non-finite full-reference metric")
        for name in NR:
            row[f"gt_{name}"] = record["gt_nr"][name]
        for name, value in nr_change(record["gt_nr"], values, log_energy=True).items():
            row[f"delta_gt_{name}"] = value
    row["metric_seconds"] = time.perf_counter() - started
    return row


def sample_keys(records, real):
    groups = defaultdict(list)
    for record in records:
        labels = [record["group"]] if real else [record["condition"]]
        if real and record["point"] == "第六點" and record["iso"] >= 1600:
            labels.append("point6_iso1600")
        for label in labels:
            if len(groups[label]) < 2:
                groups[label].append((record["block_id"], record["condition"]))
    return groups


def method_slug(method):
    return hashlib.sha256(method.encode()).hexdigest()[:12]


def save_panel(path, panels):
    width, height = SIZE * len(panels), SIZE + 30
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, rgb) in enumerate(panels):
        canvas.paste(Image.fromarray(rgb), (index * SIZE, 30))
        # Files can contain non-ASCII checkpoint names; default font may not.
        draw.text((index * SIZE + 4, 7), label.encode("ascii", "replace").decode(), fill="black")
    canvas.save(path)


def run_method(out, method, descriptor, records, balances, perceptual, real, keep, failures):
    rows = []
    for index, record in enumerate(records):
        row = {key: record.get(key) for key in FIELDS if key in record}
        row.update(method=method)
        if real and method == "wiener" and not record["kernel_path"]:
            row.update(status="skipped", error="predicted blur < 2" if record["L"] < 2 else "missing motion direction")
            rows.append(row)
            continue
        try:
            rgb = load_rgb(record["input_path"])
            if descriptor is not None:
                torch.cuda.synchronize()
            started = time.perf_counter()
            output, balance = process(rgb, method, descriptor, record, balances)
            if descriptor is not None:
                torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            row = measure_output(record, output, method, seconds, balance, perceptual, real)
            if (record["block_id"], record["condition"]) in keep:
                sample = out / ("real_samples" if real else "samples")
                output_path = sample / "outputs" / method_slug(method) / f"{record['block_id']}_{record['condition']}.png"
                save_rgb(output_path, output)
                if real:
                    save_panel(sample / f"{record['block_id']}_{method_slug(method)}.png",
                               [("input", rgb), (method, output)])
        except Exception as exc:
            row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            failure(failures, "real" if real else "synth", row["error"], record["source"], method)
            gc.collect()
            torch.cuda.empty_cache()
        rows.append(row)
        if descriptor is not None and row["status"] == "failed":
            # A checkpoint that cannot finish Part A is ineligible. Do not retry
            # thousands of identical-shape inputs after an OOM/runtime failure.
            rows.extend(failed_rows(records[index + 1:], method,
                                    "Checkpoint aborted after: " + row["error"]))
            break
        if (index + 1) % 100 == 0:
            print(f"{'B' if real else 'A'} {method}: {index + 1}/{len(records)}", flush=True)
    return rows


def failed_rows(records, method, error):
    return [dict({key: record.get(key) for key in FIELDS if key in record},
                 method=method, status="failed", error=error) for record in records]


def make_montages(out, records, methods, samples, real):
    lookup = {(record["block_id"], record["condition"]): record for record in records}
    sample_dir = out / ("real_samples" if real else "samples")
    for group_index, (group, keys) in enumerate(samples.items()):
        for sample_index, key in enumerate(keys):
            record = lookup[key]
            panels = [] if real else [("GT", load_rgb(record["gt_path"]))]
            panels.append(("input", load_rgb(record["input_path"])))
            for method in methods:
                path = sample_dir / "outputs" / method_slug(method) / f"{record['block_id']}_{record['condition']}.png"
                if path.is_file():
                    panels.append((method, load_rgb(path)))
            filename = f"group{group_index}_{sample_index}_{record['block_id']}_{record['condition']}.png"
            save_panel(sample_dir / filename, panels)
    json_save(sample_dir / "index.json", {
        group: [{"block_id": key[0], "condition": key[1]} for key in keys]
        for group, keys in samples.items()
    })


def write_summary(out, synth_rows, real_rows, settings, model_info, failures, wiener_balances):
    """Write the complete benchmark report; only complete Part A runs can select."""
    from collections import defaultdict

    metrics = ("psnr", "ssim", "lpips")
    full_conditions = ["C0"] + [
        f"{kind}-{length}" for kind in ("LIN", "TRAJ") for length in (4, 8, 16, 32, 64)
    ]
    rank_conditions = [
        f"{kind}-{length}" for kind in ("LIN", "TRAJ") for length in (4, 8, 16, 32)
    ]
    gate_conditions = [
        f"{kind}-{length}" for kind in ("LIN", "TRAJ") for length in (8, 16, 32)
    ]
    present = {str(r.get("condition")) for r in synth_rows}
    conditions = [c for c in full_conditions if c in present or c in settings.get("conditions", [])]
    methods = sorted(
        {str(r["method"]) for r in synth_rows + real_rows} | set(model_info),
        key=lambda m: ({"identity": 0, "unsharp": 1, "wiener": 2}.get(m, 3), m),
    )
    models = [m for m in methods if m.startswith("model/")]
    expected = int(settings.get("synth_blocks", 0))
    seed = int(settings.get("seed", 923))
    lines = []

    def num(value):
        try:
            value = float(value)
            return value if math.isfinite(value) else None
        except (TypeError, ValueError):
            return None

    def fmt(value, digits=4):
        value = num(value)
        return "—" if value is None else f"{value:.{digits}f}"

    def esc(value):
        return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")

    def table(headers, rows):
        lines.append("| " + " | ".join(map(esc, headers)) + " |")
        lines.append("| " + " | ".join("---" for _ in headers) + " |")
        lines.extend("| " + " | ".join(map(esc, row)) + " |" for row in rows)
        lines.append("")

    def median(values):
        values = [v for x in values if (v := num(x)) is not None]
        return float(np.median(values)) if values else None

    def cluster_stat(pairs, key):
        # Resample whole source photos, retaining all their blocks together.
        grouped = defaultdict(list)
        for source_name, value in pairs:
            value = num(value)
            if value is not None:
                grouped[str(source_name)].append(value)
        if not grouped:
            return None
        ordered = sorted(grouped)
        sums = np.asarray([sum(grouped[s]) for s in ordered], dtype=np.float64)
        counts = np.asarray([len(grouped[s]) for s in ordered], dtype=np.float64)
        local_seed = int.from_bytes(
            hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()[:8], "little"
        )
        rng = np.random.default_rng(local_seed)
        indices = rng.integers(0, len(ordered), size=(2000, len(ordered)))
        means = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
        low, high = np.quantile(means, (0.025, 0.975))
        return (float(sums.sum() / counts.sum()), float(low), float(high),
                int(counts.sum()), len(ordered))

    def show_stat(stat, digits=4):
        return "—" if stat is None else (
            f"{stat[0]:.{digits}f} [{stat[1]:.{digits}f}, {stat[2]:.{digits}f}]"
        )

    grouped_synth = defaultdict(list)
    lookup = {}
    duplicates = set()
    for row in synth_rows:
        method, condition = str(row["method"]), str(row["condition"])
        grouped_synth[(method, condition)].append(row)
        key = (method, condition, str(row["block_id"]))
        if key in lookup:
            duplicates.add((method, condition))
        lookup[key] = row

    def valid(row, metric):
        return row is not None and row.get("status") == "ok" and num(row.get(metric)) is not None

    def pairs(method, condition, metric="psnr", reference="identity"):
        result = []
        for row in grouped_synth[(method, condition)]:
            ref = lookup.get((reference, condition, str(row["block_id"])))
            if valid(row, metric) and valid(ref, metric):
                result.append((row, float(row[metric]), float(ref[metric])))
        return result

    gain_cache = {}

    def gain(method, condition):
        key = (method, condition)
        if key not in gain_cache:
            gain_cache[key] = cluster_stat(
                [(r["source"], output - original)
                 for r, output, original in pairs(method, condition)],
                f"gain:{method}:{condition}",
            )
        return gain_cache[key]

    def win(method, condition, metric, reference):
        paired = pairs(method, condition, metric, reference)
        wins = sum(a < b if metric == "lpips" else a > b for _, a, b in paired)
        return wins, len(paired)

    def win_text(method, condition, metric, reference):
        wins, total = win(method, condition, metric, reference)
        return "— (0)" if not total else f"{wins / total:.1%} ({wins}/{total})"

    def complete(method, requested):
        if expected <= 0:
            return False
        for condition in requested:
            rows = grouped_synth[(method, condition)]
            ref_rows = grouped_synth[("identity", condition)]
            keys = {str(r["block_id"]) for r in rows}
            ref_keys = {str(r["block_id"]) for r in ref_rows}
            if (len(rows) != expected or len(keys) != expected
                    or len(ref_rows) != expected or len(ref_keys) != expected
                    or keys != ref_keys or (method, condition) in duplicates):
                return False
            if any(not all(valid(row, metric) for metric in metrics) for row in rows + ref_rows):
                return False
        return True

    rank_complete = {m: complete(m, rank_conditions) for m in methods}
    a_complete = {m: complete(m, full_conditions) for m in methods}
    rank_methods = [m for m in methods if rank_complete[m]]
    psnr_ranks, lpips_ranks = defaultdict(list), defaultdict(list)

    def average_ranks(values, reverse):
        ordered = sorted(values, key=lambda item: (-item[1] if reverse else item[1], item[0]))
        result = {}
        i = 0
        while i < len(ordered):
            j = i + 1
            while j < len(ordered) and ordered[j][1] == ordered[i][1]:
                j += 1
            rank = ((i + 1) + j) / 2
            for method, _ in ordered[i:j]:
                result[method] = rank
            i = j
        return result

    for condition in rank_conditions:
        values = [(m, gain(m, condition)[0]) for m in rank_methods]
        for method, rank in average_ranks(values, True).items():
            psnr_ranks[method].append(rank)
        values = [
            (m, float(np.mean([r["lpips"] for r in grouped_synth[(m, condition)]])))
            for m in rank_methods
        ]
        for method, rank in average_ranks(values, False).items():
            lpips_ranks[method].append(rank)

    ranking = {}
    for method in rank_methods:
        # Each source is first averaged across blocks within each of the eight
        # conditions, then equally across conditions, before photo bootstrap.
        per_source = defaultdict(lambda: defaultdict(list))
        for condition in rank_conditions:
            for row, output, original in pairs(method, condition):
                per_source[str(row["source"])][condition].append(output - original)
        photo_gains = [
            (source_name, float(np.mean([np.mean(by_condition[c]) for c in rank_conditions])))
            for source_name, by_condition in per_source.items()
            if all(c in by_condition for c in rank_conditions)
        ]
        stat = cluster_stat(photo_gains, f"ranking:{method}")
        times = [num(r.get("seconds")) for c in rank_conditions
                 for r in grouped_synth[(method, c)]]
        times = [t for t in times if t is not None and t >= 0]
        ranking[method] = {
            "gain": stat,
            "psnr_rank": float(np.mean(psnr_ranks[method])),
            "lpips_rank": float(np.mean(lpips_ranks[method])),
            "seconds": float(np.mean(times)) if times else None,
        }
    ranked = sorted(
        (m for m in ranking if ranking[m]["gain"] is not None),
        key=lambda m: (-ranking[m]["gain"][0], ranking[m]["psnr_rank"], m),
    )
    gate = {}
    eligible = []
    for method in models:
        useful = all(
            (lambda wt: wt[1] == expected and wt[1] > 0 and wt[0] / wt[1] >= 0.9)(
                win(method, condition, "psnr", "identity")
            )
            for condition in gate_conditions
        )
        c0 = gain(method, "C0")
        harmless = c0 is not None and c0[3] == expected and c0[0] >= -0.3
        smoke_failed = str(model_info.get(method, {}).get("smoke_status", "")).lower() in {
            "failed", "error", "skipped"
        }
        passes = a_complete[method] and useful and harmless and not smoke_failed
        gate[method] = (useful, harmless, passes, smoke_failed)
        if passes and method in ranking:
            eligible.append(method)
    eligible.sort(key=lambda m: (-ranking[m]["gain"][0], ranking[m]["psnr_rank"], m))
    selected = None
    selection_reason = ""
    overlap = False
    if settings.get("mode") == "full" and eligible:
        selected = eligible[0]
        if len(eligible) > 1:
            first, second = eligible[:2]
            a, b = ranking[first], ranking[second]
            overlap = max(a["gain"][1], b["gain"][1]) <= min(a["gain"][2], b["gain"][2])
            if overlap and a["seconds"] is not None and b["seconds"] is not None:
                selected = min((first, second), key=lambda m: (ranking[m]["seconds"], eligible.index(m)))
                selection_reason = (
                    f"前兩名 {first} 與 {second} 的 PSNR 增益 95% CI 重疊，"
                    f"依八個排名條件的平均每塊耗時選擇較快者 {selected}。"
                )
            elif overlap:
                selection_reason = "前兩名增益 CI 重疊，但缺少有效耗時；無法執行速度決勝，不選模型。"
                selected = None
        if selected is not None and not selection_reason:
            selection_reason = "選擇通過完整性、有用與不傷害門檻後，平均 PSNR 增益最高的模型。"
    lpips_candidates = [m for m in models if a_complete[m] and m in ranking]
    lpips_best = min(
        lpips_candidates,
        key=lambda m: (ranking[m]["lpips_rank"], -ranking[m]["gain"][0], m),
        default=None,
    )

    lines += ["# Deblur 模型選擇實驗", "", "## 自動結論", ""]
    if settings.get("mode") != "full":
        lines += ["**本次為冒煙測試，不選模型，也不把局部結果當成正式排名。**",
                  "需完成全部 11 個條件後才可套用決策規則。", ""]
    elif selected is not None:
        lines += [f"**選中：{esc(selected)}**", "", selection_reason, ""]
    elif not eligible:
        lines += ["**不建議使用 deblur**：本次沒有模型同時通過完整性、有用與不傷害門檻。",
                  "缺失或失敗結果不可視為通過；這不表示未完整評測的 checkpoint 已被證明無效。", ""]
    else:
        lines += ["**尚無可自動選定的模型。** " + selection_reason, ""]
    if lpips_best is not None:
        lines += [
            f"完整模型中的 LPIPS 平均名次第一：**{esc(lpips_best)}**"
            + ("（與選中模型不同，單獨列出；不改變 PSNR 決策）。" if selected and lpips_best != selected
               else "。"),
            "",
        ]
    lines += ["Part A 決定模型；Part B 僅標記一致性，不參與排名或自動否決。", "",
              "## 設定與 checkpoint", ""]
    table(["設定", "值"], [(key, json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
                           if isinstance(value, (dict, list, tuple)) else value)
                          for key, value in sorted(settings.items())])
    table(["方法", "SHA-256（前 12 碼）", "spandrel 架構", "狀態", "冒煙測試", "錯誤"],
          [(m, str(model_info.get(m, {}).get("sha256", ""))[:12] or "—",
            model_info.get(m, {}).get("architecture") or "—",
            model_info.get(m, {}).get("status") or "—",
            model_info.get(m, {}).get("smoke_status") or "—",
            model_info.get(m, {}).get("error") or "—") for m in models])
    lines += [
        "固定 seed；模型 FP32、512×512 直接推論；全參考分數去除四邊各 32 px。"
        "勝率採嚴格勝出，平手不計勝。95% CI 為 2000 次來源照片 cluster bootstrap 的"
        "第 2.5／97.5 百分位，不把同張照片的裁塊視為獨立樣本。",
        "",
        "規格歧義採用 C0『只加雜訊和 JPEG』的定義：C0 跳過高光乘 4；其餘模糊條件"
        "在線性空間把 GT 最大通道 ≥250 的像素乘 4，再模糊、截斷、轉回 sRGB、加雜訊和 JPEG。",
        "",
        "## Part A 主表", "",
        "數值為平均 [95% CI]；n 與來源照片數依 PSNR／SSIM／LPIPS 順序列出。"
        "缺值不填零，失敗方法的局部結果僅供診斷。",
        "",
    ]
    main_rows = []
    for method in methods:
        for condition in conditions:
            stats = []
            for metric in metrics:
                stats.append(cluster_stat(
                    [(r["source"], r.get(metric)) for r in grouped_synth[(method, condition)]
                     if valid(r, metric)],
                    f"main:{method}:{condition}:{metric}",
                ))
            main_rows.append((method, condition, *(show_stat(s) for s in stats),
                              "/".join(str(s[3] if s else 0) for s in stats),
                              "/".join(str(s[4] if s else 0) for s in stats)))
    table(["方法", "條件", "PSNR (dB)", "SSIM", "LPIPS", "有效塊 n", "來源照片 n"], main_rows)
    lines += ["## Part A 勝率", "",
              "逐塊配對比較；括號為勝出數／可配對數。LPIPS 越低越好，其餘越高越好。", ""]
    table(["方法", "條件", "對照", "PSNR 勝率", "SSIM 勝率", "LPIPS 勝率"],
          [(m, c, reference, *(win_text(m, c, metric, reference) for metric in metrics))
           for m in methods for c in conditions for reference in ("identity", "wiener")])
    lines += ["## Part A 排名與門檻", "",
              "排名條件為 LIN、TRAJ 的 L=4、8、16、32。每條件先按 PSNR 增益排序，"
              "LPIPS 另按平均值排序，並列取平均名次；平均名次包含完整的參考方法。"
              "主排序為八條件平均 PSNR 增益，次序為平均 PSNR 名次。",
              "排名增益先在每個來源照片內聚合八條件，再以來源照片 bootstrap；"
              "耗時是這八條件的平均每塊方法處理時間。完整性要求所有 11 條件、"
              "每條件全部預期裁塊及三個全參考指標均成功；診斷指標缺值不當作模型選擇門檻。", ""]
    table(["排名", "方法", "PSNR 增益 (dB) [95% CI]", "PSNR 平均名次",
           "LPIPS 平均名次", "秒／塊", "11 條件完整"],
          [(i, m, show_stat(ranking[m]["gain"]), fmt(ranking[m]["psnr_rank"]),
            fmt(ranking[m]["lpips_rank"]), fmt(ranking[m]["seconds"]),
            "是" if a_complete[m] else "否") for i, m in enumerate(ranked, 1)]
          + [("—", m, "—", "—", "—", "—", "否") for m in methods if m not in ranked])
    gate_rows = []
    for m in models:
        useful, harmless, passes, smoke_failed = gate[m]
        gate_rows.append((m, "通過" if a_complete[m] else "未完整",
                          "失敗" if smoke_failed else "無已記錄失敗",
                          *(win_text(m, c, "psnr", "identity") for c in gate_conditions),
                          show_stat(gain(m, "C0")), "是" if useful else "否",
                          "是" if harmless else "否", "是" if passes else "否"))
    table(["模型", "完整性", "冒煙", *gate_conditions, "C0 增益 (dB)",
           "六條件皆 ≥90%", "C0 ≥−0.3 dB", "具選擇資格"], gate_rows)

    lines += ["## Part A 分組 PSNR 增益", "",
              "次要診斷表，使用八個排名條件；各塊先跨條件平均，再列組內平均，"
              "缺失模型的部分結果不參與正式選擇。glint 三分位邊界來自唯一 GT 裁塊，"
              "邊界相同時允許空組，避免把相同反光比例任意分開。", ""]
    block_tags = {}
    for row in synth_rows:
        block_tags.setdefault(str(row["block_id"]), row)
    glints = [num(r.get("glint_frac")) for r in block_tags.values()]
    glints = [v for v in glints if v is not None]
    cuts = np.quantile(glints, (1 / 3, 2 / 3)) if glints else None
    if cuts is not None:
        lines += [f"glint_frac 邊界：q1={cuts[0]:.6f}，q2={cuts[1]:.6f}。", ""]

    def glint_group(row):
        value = num(row.get("glint_frac"))
        if value is None or cuts is None:
            return "缺失"
        return "低（≤q1）" if value <= cuts[0] else ("中（q1,q2]）" if value <= cuts[1] else "高（>q2）")

    subgroup_rows = []
    for method in methods:
        blocks = defaultdict(list)
        tags = {}
        for condition in rank_conditions:
            for row, output, original in pairs(method, condition):
                key = str(row["block_id"])
                blocks[key].append(output - original)
                tags[key] = row
        for dimension in ("alt_folder", "glint 三分位"):
            groups = defaultdict(list)
            for key, values in blocks.items():
                label = tags[key].get("alt_folder", "缺失") if dimension == "alt_folder" else glint_group(tags[key])
                groups[str(label)].append((tags[key]["source"], float(np.mean(values)), len(values)))
            for label, values in sorted(groups.items()):
                subgroup_rows.append((method, dimension, label,
                                      fmt(np.mean([v[1] for v in values])), len(values),
                                      len({v[0] for v in values}),
                                      sum(v[2] == 8 for v in values)))
    table(["方法", "分組依據", "組別", "平均 PSNR 增益 (dB)", "有值塊", "來源照片", "完整八條件塊"],
          subgroup_rows)
    lines += ["## Part A 過度銳化診斷", "",
              "各欄為輸出−GT 的中位數；Laplacian／Tenengrad 使用 log2(輸出／GT)，"
              "其餘使用差值。除 Crété 外，正值代表比 GT 更清晰；Crété 負值代表更清晰。"
              "這只提示過度銳化或捏造細節的可能，不能單憑無參考分數確認真實細節。", ""]
    table(["方法", "條件", *NR, "有效 n（依指標順序）"],
          [(m, c, *(fmt(median(r.get("delta_gt_" + metric)
                             for r in grouped_synth[(m, c)] if r.get("status") == "ok"))
                    for metric in NR),
            "/".join(str(sum(r.get("status") == "ok" and num(r.get("delta_gt_" + metric)) is not None
                             for r in grouped_synth[(m, c)])) for metric in NR))
           for m in methods for c in conditions])
    lines += ["## Part A 有效範圍", "",
              "每顆模型、每種核，列出 PSNR 配對增益 95% CI 下界仍 >0 的最大已評測 L。"
              "只納入該條件完整結果，不要求較小 L 連續通過；『—』表示無條件達標。", ""]
    table(["模型", "核", "最大 L", "該條件增益 (dB) [95% CI]"],
          [(m, kind, (lengths[-1] if lengths else "—"),
            show_stat(gain(m, f"{kind}-{lengths[-1]}")) if lengths else "—")
           for m in models for kind in ("LIN", "TRAJ")
           for lengths in [[length for length in (4, 8, 16, 32, 64)
                            if complete(m, [f"{kind}-{length}"])
                            and gain(m, f"{kind}-{length}") is not None
                            and gain(m, f"{kind}-{length}")[1] > 0]]])

    real_groups = ["[0,2)", "[2,10)", "[10,30)", "[30,inf)"]
    real_lookup = defaultdict(list)
    for row in real_rows:
        real_lookup[(str(row["method"]), str(row.get("group")))].append(row)
        if str(row.get("point", "")).strip() in {"第六點", "6", "6.0"} and (num(row.get("iso")) or 0) >= 1600:
            real_lookup[(str(row["method"]), "第六點／ISO≥1600")].append(row)
    real_methods = [m for m in methods if m != "unsharp"]
    lines += ["## Part B 真實影像一致性", "",
              "下表是各無參考指標處理後−處理前的中位數，全部使用原始差值。"
              "沒有真實對齊 GT，這些值不能證明真實地物細節被恢復。", "",
              "Wiener 僅處理預測模糊 ≥2 px；方向缺失時跳過，不補成 0°。"
              "真實資料沒有 GT，balance 不在 Part B 調參：取合成 LIN-L 中 log2 尺度"
              "最接近的 L∈{4,8,16,32,64} 的 balance；真實 PSF 長度仍使用遙測值，不截斷到 64。", ""]
    real_table = []
    for group in real_groups + ["第六點／ISO≥1600"]:
        for method in real_methods:
            rows = real_lookup[(method, group)]
            good = [r for r in rows if r.get("status") == "ok"]
            real_table.append((group, method, len(good), len({r["source"] for r in good}),
                               sum(r.get("status") == "failed" for r in rows),
                               sum(r.get("status") == "skipped" for r in rows),
                               *(fmt(median(r.get("delta_input_" + metric) for r in good)) for metric in NR),
                               "/".join(str(sum(num(r.get("delta_input_" + metric)) is not None for r in good))
                                        for metric in NR)))
    table(["組別", "方法", "成功塊", "成功來源照片", "失敗塊", "跳過塊", *NR, "有效 n（依指標順序）"],
          real_table)
    lines += ["### 選中模型的兩項標記", ""]
    if selected is None:
        lines += ["尚未選中模型，兩項標記不適用。", ""]
    else:
        clear_models = {}
        clear_reference = {str(r["block_id"]) for r in real_lookup[("identity", "[0,2)")]}
        for method in models:
            rows = real_lookup[(method, "[0,2)")]
            good = [r for r in rows if r.get("status") == "ok"
                    and num(r.get("delta_input_crete_roffet_blur")) is not None]
            if (clear_reference and len(rows) == len(clear_reference)
                    and len(good) == len(clear_reference)
                    and {str(r["block_id"]) for r in good} == clear_reference):
                clear_models[method] = median(abs(float(r["delta_input_crete_roffet_blur"])) for r in good)
        selected_clear = clear_models.get(selected)
        if selected_clear is None:
            lines += ["1. 清晰組 [0,2) 的 median(|ΔCrété|)：資料不完整，無法判定是否最大。"]
        else:
            maximum = max(clear_models.values())
            flag = selected_clear == maximum
            lines += [f"1. 清晰組 [0,2) 的 median(|ΔCrété|)={selected_clear:.6f}；"
                      f"在 {len(clear_models)}/{len(models)} 顆清晰組資料完整模型中"
                      f"{'為最大（含並列；注意可能改動本來清楚的照片）' if flag else '不是最大'}。"]
        blur_rows = [r for r in real_lookup[(selected, "[10,30)")]
                     if r.get("status") == "ok"]
        blur_delta = median(r.get("delta_input_crete_roffet_blur") for r in blur_rows)
        blur_n = sum(num(r.get("delta_input_crete_roffet_blur")) is not None for r in blur_rows)
        if blur_delta is None:
            lines += ["2. [10,30) 的 ΔCrété：無有效資料，無法判定。", ""]
        else:
            lines += [f"2. [10,30) 的 median(ΔCrété)={blur_delta:.6f}（有效 {blur_n} 塊），"
                      f"{'<0，這個指標顯示變清晰' if blur_delta < 0 else '未 <0，這個指標未顯示變清晰'}。", ""]
        lines += ["上述標記只供一致性與目視檢查，未自動否決 Part A 的選擇。", ""]

    lines += ["## Wiener balance", ""]
    table(["合成條件", "所選 balance"], [(c, wiener_balances.get(c, "—")) for c in conditions])
    lines += ["各條件使用前 10 塊（不足則使用可用塊）在指定候選 balance 中選 PSNR 最高者；"
              "Part A 使用真實合成核；前 10 塊也保留在主表，故 Wiener 的評分含調參樣本，有偏樂觀的限制。", "",
              "## 失敗、跳過與缺值", ""]
    status_counts = defaultdict(int)
    for stage, rows in (("Part A", synth_rows), ("Part B", real_rows)):
        for row in rows:
            status_counts[(stage, str(row["method"]), str(row.get("status", "missing")))] += 1
    table(["階段", "方法", "狀態", "塊列數"],
          [(*key, count) for key, count in sorted(status_counts.items())])
    error_counts = defaultdict(int)
    for stage, rows in (("Part A", synth_rows), ("Part B", real_rows)):
        for row in rows:
            for field in ("error", "nr_errors"):
                error = row.get(field)
                if error:
                    error_counts[(stage, str(row["method"]), str(error))] += 1
    table(["階段", "方法", "逐列錯誤／缺值原因", "次數"],
          [(*key, count) for key, count in sorted(error_counts.items())])
    if failures:
        table(["階段", "來源", "方法", "錯誤"],
              [(f.get("stage", "—"), f.get("source", "—"), f.get("method", "—"),
                f.get("error", "—")) for f in failures])
    lines += [
        "CPBD 無可用邊緣而回傳 0 時保留 0；Crété 等指標非有限值記為缺值，"
        "表內顯示有效數量，不把缺值當成改善。若來源照片數只有 1，bootstrap CI"
        "會退化為單點，不能解讀為估計沒有不確定性。",
        "",
        "## 限制與檢視位置",
        "",
        "- 合成模糊在整塊影像上均勻，不包含水面本身運動；GT 海況與光線種類有限，大多來自第一點。",
        "- 評測是在 512×512 上直接推論，未評估整張 12 MP 圖片切塊後的接縫；部署時須另行處理。",
        "- Part A 的 Wiener 拿到真實核，設定對它有利，不要求模型打贏 Wiener。",
        "- 來源照片 cluster bootstrap 處理同張裁塊的相關性，但不代表對未涵蓋海況的泛化保證。",
        "- 無參考分數只供診斷；生成或銳化紋理不能直接當成真實地物細節。",
        "- 主表失敗方法可能只有部分樣本；正式選擇排除 Part A 不完整模型。Part B 缺值與失敗僅揭露。",
        "- samples/ 為各合成條件 GT｜輸入｜方法比較；real_samples/ 為真實各組輸入｜方法比較，應搭配目視。",
        "- results_synth.csv、results_real.csv 保留逐塊數據；kernels/ 保存固定 seed 的合成核。",
        "",
    ]
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return selected




def run_phase(args, photos, metadata, checkpoints, runtime, smoke, prior_info=None):
    mode = "smoke" if smoke else "full"
    # Microseconds and mkdir(exist_ok=False) prevent replacing any previous run.
    out = args.runs_root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    out.mkdir(parents=True, exist_ok=False)
    for folder in ("kernels", "samples", "real_samples", "pairs"):
        (out / folder).mkdir()
    print(f"START {mode}: {out}", flush=True)
    failures, synth_rows, real_rows = [], [], []
    model_info = {name: dict(info) for name, info in checkpoints.items()}
    if prior_info:
        for name in model_info:
            model_info[name]["smoke_status"] = prior_info[name].get("status", "failed")
    gt_photos = [row for row in photos if gt_eligible(row)]
    real_photos = [row for row in photos if row["gimbal_pitch_deg"] <= -85]
    if smoke:
        gt_photos, real_photos = gt_photos[:2], []
    conditions, k = (["C0", "LIN-8", "TRAJ-8"], 1) if smoke else (CONDITIONS, args.k)
    settings = dict(runtime, mode=mode, seed=args.seed, k=k, real_k=0 if smoke else 4,
                    conditions=conditions, gt_photos=len(gt_photos), real_photos=len(real_photos),
                    synth_blocks=len(gt_photos) * k, real_blocks=len(real_photos) * 4,
                    crop_size=SIZE, pad=PAD, metric_border=BORDER, bootstrap=2000,
                    noise_sigma_8bit=1.0, jpeg_quality=95, glint_threshold=250, glint_multiplier=4,
                    wiener_candidates=list(BALANCES), wiener_calibration_blocks=10,
                    gt_filter="pitch<=-85, blur<1, iso<=800, (exposure<=1/100 OR stationary & exposure<=1/40 & speed<0.3)",
                    lin_angles="abs(csv motion direction where speed>0.5), random sign",
                    trajectory="1024-step inertial random walk, maximum XY span L, centroid-centered bilinear PSF",
                    nr_region="Part A central 448x448; Part B full 512x512",
                    timing="synchronized per-crop processing including transfers/quantization; excludes disk IO, metrics, model load, warmup",
                    metadata_sha256=hashlib.sha256(metadata.read_bytes()).hexdigest(),
                    source=str(args.source.resolve()), models=str(args.models.resolve()),
                    method_ids={method_slug(name): name for name in ["identity", "unsharp", "wiener"] + list(model_info)})
    json_save(out / "settings.json", settings)
    if not gt_photos:
        raise ValueError("No GT photos match the specified filters")
    angles = [abs(row["motion_dir_vs_image_up_deg"]) for row in photos
              if row["speed_h_mps"] > 0.5 and row["motion_dir_vs_image_up_deg"] is not None]
    if not angles:
        raise ValueError("No measured motion directions for synthetic LIN sampling")
    synth = create_pairs(out, gt_photos, k, conditions, args.seed, angles, False, failures)
    real = create_pairs(out, real_photos, 4, [], args.seed, angles, True, failures)
    json_save(out / "pairs/manifest.json", {"synth": synth, "real": real})
    json_save(out / "failures.json", failures)
    balances = {}
    try:
        if len(synth) != settings["synth_blocks"] * len(conditions) or len(real) != settings["real_blocks"]:
            raise RuntimeError("Incomplete crop preparation; see failures.json. No model selection from a partial dataset")
        balances = tune_wiener(synth, conditions)
        json_save(out / "wiener_balances.json", balances)
        samples_a, samples_b = sample_keys(synth, False), sample_keys(real, True)
        keep_a = {key for keys in samples_a.values() for key in keys}
        keep_b = {key for keys in samples_b.values() for key in keys}
        perceptual = PerceptualMetric(device="cuda")
        for method in ["identity", "unsharp", "wiener"] + list(model_info):
            descriptor = None
            warmup = None
            try:
                if method in model_info:
                    info = model_info[method]
                    if info.get("smoke_status") == "failed":
                        raise RuntimeError("Excluded after checkpoint smoke failure; see smoke run")
                    descriptor = load_model(Path(info["path"]), role="deblur")
                    info["architecture"] = getattr(descriptor.architecture, "name", str(descriptor.architecture))
                    if descriptor.dtype != torch.float32 or descriptor.device.type != "cuda":
                        raise RuntimeError("Expected FP32 CUDA descriptor")
                    # One unmeasured warmup separates startup from per-crop speed.
                    warmup = upscale(read_image(Path(synth[0]["input_path"])), descriptor)
                    if tuple(warmup.shape) != (1, 3, SIZE, SIZE) or not torch.isfinite(warmup).all():
                        raise ValueError("Invalid warmup output")
                    del warmup
                    torch.cuda.synchronize()
                print(f"METHOD {method}", flush=True)
                a_rows = run_method(out, method, descriptor, synth, balances, perceptual, False, keep_a, failures)
                synth_rows.extend(a_rows)
                b_rows = []
                if real and method != "unsharp":
                    if descriptor is not None and any(row["status"] != "ok" for row in a_rows):
                        b_rows = failed_rows(real, method, "Skipped after Part A checkpoint failure")
                    else:
                        b_rows = run_method(out, method, descriptor, real, balances, perceptual, True, keep_b, failures)
                    real_rows.extend(b_rows)
                if method in model_info:
                    model_info[method]["status"] = "ok" if all(row["status"] == "ok" for row in a_rows) else "failed"
                    model_info[method]["real_status"] = "ok" if all(row["status"] == "ok" for row in b_rows) else "failed"
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                failure(failures, "method", error, method=method)
                synth_rows.extend(failed_rows(synth, method, error))
                if real and method != "unsharp":
                    real_rows.extend(failed_rows(real, method, error))
                if method in model_info:
                    model_info[method].update(status="failed", error=error)
            finally:
                warmup = None
                descriptor = None
                gc.collect()
                torch.cuda.empty_cache()
                write_csv(out / "results_synth.csv", synth_rows)
                write_csv(out / "results_real.csv", real_rows)
                json_save(out / "models.json", model_info)
                json_save(out / "failures.json", failures)
        del perceptual
        gc.collect()
        torch.cuda.empty_cache()
        methods = ["identity", "unsharp", "wiener"] + list(model_info)
        make_montages(out, synth, methods, samples_a, False)
        make_montages(out, real, [method for method in methods if method != "unsharp"], samples_b, True)
    except Exception as exc:
        failure(failures, "phase", f"{type(exc).__name__}: {exc}")
    finally:
        write_csv(out / "results_synth.csv", synth_rows)
        write_csv(out / "results_real.csv", real_rows)
        json_save(out / "models.json", model_info)
        json_save(out / "failures.json", failures)
        write_summary(out, synth_rows, real_rows, settings, model_info, failures, balances)
    print(f"FINISH {mode}: {out / 'summary.md'}", flush=True)
    return model_info, failures


def main():
    args = parse_args()
    photos, metadata = prepare_data(args)
    print(f"GT: {sum(gt_eligible(row) for row in photos)}; nadir: {sum(row['gimbal_pitch_deg'] <= -85 for row in photos)}", flush=True)
    if args.prepare:
        print(f"Place deblur checkpoints at {args.models}; AlexNet cache at {os.environ['TORCH_HOME']}/hub/checkpoints/")
        return 0
    paths = sorted(path for path in args.models.rglob("*")
                   if path.is_file() and path.suffix.lower() in {".pth", ".pt", ".ckpt", ".safetensors"})
    if not paths:
        raise FileNotFoundError(f"No deblur checkpoints in {args.models}; prepare them on the server first")
    runtime = initialize_runtime(args.seed)
    checkpoints = {}
    for path in paths:
        with path.open("rb") as stream:
            sha = hashlib.file_digest(stream, "sha256").hexdigest()
        checkpoints["model/" + path.relative_to(args.models).as_posix()] = {
            "path": str(path.resolve()), "sha256": sha, "architecture": "not loaded", "status": "pending"}
    print(f"Checkpoints: {len(checkpoints)}; no weights will be downloaded", flush=True)
    smoke_info, smoke_failures = run_phase(args, photos, metadata, checkpoints, runtime, smoke=True)
    if args.smoke:
        return 1 if smoke_failures else 0
    if any(record["stage"] == "phase" or record["method"] in {"identity", "unsharp", "wiener"} for record in smoke_failures) or not any(
            info["status"] == "ok" for info in smoke_info.values()):
        print("Full run stopped: smoke did not validate any usable checkpoint or the evaluation pipeline", file=sys.stderr)
        return 1
    _, full_failures = run_phase(args, photos, metadata, checkpoints, runtime, smoke=False, prior_info=smoke_info)
    return 1 if smoke_failures or full_failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
PY
