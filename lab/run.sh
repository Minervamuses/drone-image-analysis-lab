#!/usr/bin/env bash
# Point 6: six speed/height groups, independent SR and original-image Deblur.
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
import math
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.environ["DEBLUR_REPO_ROOT"])
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "evaluation")]
from metric_defs import BLUR_METRICS

SOURCE = ROOT / "923海上正攝_lab_extract/from_video/第六點"
MODELS = ROOT / "models"
RUNS = ROOT / "evaluation/runs/point6"
CHECKPOINT_SUFFIXES = {".pth", ".pt", ".ckpt", ".safetensors"}
DEBLUR_NAMES = (
    "Uformer_B_GoPro.pth",
    "fftformer_GoPro.pth",
    "model_deblurring.pth",
    "motion_deblurring.pth",
)
FIELDS = (
    "height_m", "speed_ms", "image", "experiment", "kind", "model",
    "source_path", "original_path", "input_path", "reference_path", "output_path",
    "width", "height", "psnr", "ssim", "lpips", *BLUR_METRICS,
    *(name + "_valid" for name in BLUR_METRICS),
    "status", "invalid_metrics", "error",
)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="bash lab/run.sh",
        description="Point 6: first 20 frames per height/speed, all SR then four independent Deblur models.",
        epilog=(
            "Source: 923海上正攝_lab_extract/from_video/第六點/{20M,90M}/速度{5,8,15}ms. "
            "Recursively sort relative video/frame paths; take the first N per speed folder, "
            "not N per video. Skip 60M and still photos. "
            "SR: every checkpoint directly in models/, RGB 4x, shared bicubic LR and cropped HR. "
            "Deblur: Uformer, FFTformer, MPRNet and Restormer in models/deblur/, original size; "
            "NAFNet is excluded. Existing CUDA, dependencies and LPIPS AlexNet cache are required. "
            "No automatic downloads or CPU inference. "
            "Each run creates evaluation/runs/point6/<UTC time>/ with copied input/, sr_hr/, "
            "sr_lr/, output/ and one results.csv; no summaries or interpretation."
        ),
    )
    parser.add_argument("--limit", type=int, default=20, help="Frames PER height/speed group (1..20; default 20)")
    parser.add_argument("--dry-run", action="store_true", help="Check and list selected paths only; no writes, imports of ML packages or inference")
    args = parser.parse_args()
    if not 1 <= args.limit <= 20:
        parser.error("--limit must be between 1 and 20 per group")
    return args


def select_groups(limit):
    groups = []
    for height in (20, 90):
        for speed in (5, 8, 15):
            folder = SOURCE / f"{height}M" / f"速度{speed}ms"
            # Lab names are zero-padded: 0145/frames/0145_0000004.png.
            images = sorted(
                (path for path in folder.rglob("*")
                 if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}),
                key=lambda path: path.relative_to(folder).as_posix(),
            )
            if len(images) < limit:
                raise ValueError(f"{folder}: need {limit} frames, found {len(images)}")
            groups.append((height, speed, images[:limit]))
    return groups


def select_models():
    # The lab stores SR weights at this level; never include deblur/ or torch-cache/.
    sr = sorted(path for path in MODELS.iterdir()
                if path.is_file() and path.suffix.lower() in CHECKPOINT_SUFFIXES) if MODELS.is_dir() else []
    if not sr:
        raise ValueError(f"no SR checkpoints directly in {MODELS}")
    deblur = [MODELS / "deblur" / name for name in DEBLUR_NAMES]
    missing = [str(path) for path in deblur if not path.is_file()]
    if missing:
        raise ValueError("missing required Deblur checkpoints: " + ", ".join(missing))
    return sr, deblur


def load_runtime():
    global torch, read_image, unique_output_path, write_png, load_model, upscale
    global measure_tensor, failed_measurements, psnr, ssim, PerceptualMetric
    global to_pil_image, mod_crop, downscale, save_png
    import cv2
    import torch
    import lpips
    from torchvision.transforms.functional import to_pil_image
    from drone_sr.image_io import read_image, unique_output_path, write_png
    from drone_sr.inference import load_model, upscale
    from blur_metrics import measure_tensor, failed_measurements
    from degradation import mod_crop, downscale, save_png
    from metrics import psnr, ssim
    from perceptual import PerceptualMetric

    def no_download(*args, **kwargs):
        raise RuntimeError("Automatic weight downloads are disabled; prepare the existing lab cache")

    torch.hub.download_url_to_file = no_download
    cache = Path(torch.hub.get_dir()) / "checkpoints/alexnet-owt-7be5be79.pth"
    calibration = Path(lpips.__file__).resolve().parent / "weights/v0.1/alex.pth"
    for path in (cache, calibration):
        if not path.is_file():
            raise ValueError(f"required existing LPIPS weights missing: {path}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; CPU inference is disabled")
    torch.ones(1, device="cuda:0").sum().item()
    free, total = torch.cuda.mem_get_info()
    cv2.setNumThreads(1)
    print(f"torch: {torch.__version__}; CUDA: {torch.version.cuda}; GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}; VRAM free/total bytes: {free}/{total}")
    print(f"PSNR/SSIM: CPU float64; LPIPS: cuda:0; cache: {cache}")


def prepare_group(group, run):
    height, speed, sources = group
    records = []
    for source in sources:
        relative = source.relative_to(SOURCE)
        original = run / "input" / relative
        record = dict(height_m=height, speed_ms=speed, image=relative.as_posix(),
                      source_path=str(source), original_path=str(original), error="", sr_error="")
        try:
            original.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, original)
        except Exception as exc:
            record["error"] = f"input copy: {type(exc).__name__}: {exc}"
        decoded = image = cropped = None
        try:
            if not record["error"]:
                decoded = read_image(original)
                image = to_pil_image(decoded.squeeze(0).mul(255).round().to(torch.uint8), mode="RGB")
                cropped, _ = mod_crop(image)
                if min(cropped.size) < 4:
                    raise ValueError("image is too small for 4x bicubic downsampling")
                hr = unique_output_path((run / "sr_hr" / relative).with_suffix(".png"))
                lr = run / "sr_lr" / relative.parent / hr.name
                save_png(cropped, hr)
                save_png(downscale(cropped), lr)
                record.update(hr_path=str(hr), lr_path=str(lr))
        except Exception as exc:
            # An SR preparation error does not prevent independent Deblur of the copied original.
            record["sr_error"] = f"SR preparation: {type(exc).__name__}: {exc}"
        finally:
            del decoded, image, cropped
        records.append(record)
    return records


def result_row(record, experiment, kind, model=""):
    row = {name: record[name] for name in ("height_m", "speed_ms", "image", "source_path", "original_path")}
    row.update(experiment=experiment, kind=kind, model=model, status="failed", error="",
               input_path=record.get("lr_path", "") if experiment == "sr" else record["original_path"],
               reference_path=record.get("hr_path", "") if experiment == "sr" else "",
               output_path="")
    return row


def measure_sharpness(row):
    image = None
    try:
        image = read_image(Path(row["output_path"] or row["input_path"]))
        row.update(width=image.shape[-1], height=image.shape[-2])
        measurements = measure_tensor(image)
    except Exception as exc:
        measurements = failed_measurements(f"{type(exc).__name__}: {exc}")
    finally:
        del image
    invalid = [name for name in BLUR_METRICS if not measurements[name]["valid"]]
    for name in BLUR_METRICS:
        row[name] = measurements[name]["value"]
        row[name + "_valid"] = int(measurements[name]["valid"])
    row.update(
        status="failed" if any(value["status"] == "failed" for value in measurements.values())
        else ("partial" if invalid else "ok"),
        invalid_metrics=";".join(invalid),
        error="; ".join(f"{name}: {measurements[name]['status']}: {measurements[name]['reason']}" for name in invalid),
    )


def infer_model(role, checkpoint, records, run):
    model = checkpoint.relative_to(ROOT).as_posix()
    descriptor, load_error = None, ""
    rows = []
    try:
        descriptor = load_model(checkpoint, role=role)
        expected_scale = 4 if role == "sr" else 1
        if descriptor.scale != expected_scale:
            raise ValueError(f"{role} requires scale={expected_scale}, got {descriptor.scale}")
        if descriptor.device.type != "cuda":
            raise RuntimeError("CUDA inference is required; CPU fallback is disabled")
        print(f"Model: {model}; architecture: {descriptor.architecture.name}; device: {descriptor.device}")
    except Exception as exc:
        load_error = f"model load: {type(exc).__name__}: {exc}"
    try:
        for index, record in enumerate(records, 1):
            row = result_row(record, role, "model_output", model)
            row["error"] = "; ".join(filter(None, (
                record["error"], record["sr_error"] if role == "sr" else "", load_error,
            )))
            destination = unique_output_path((run / "output" / role / checkpoint.name / record["image"]).with_suffix(".png"))
            row["output_path"] = str(destination)
            image = result = None
            try:
                if not row["error"]:
                    image = read_image(Path(row["input_path"]))
                    result = upscale(image, descriptor)
                    expected = (*image.shape[:-2], image.shape[-2] * expected_scale, image.shape[-1] * expected_scale)
                    if tuple(result.shape) != expected:
                        raise ValueError(f"Expected {expected}, got {tuple(result.shape)}")
                    write_png(result, destination, Path(row["input_path"]))
                    row.update(status="saved", width=result.shape[-1], height=result.shape[-2])
            except Exception as exc:
                row["error"] = f"inference/save: {type(exc).__name__}: {exc}"
            finally:
                del image, result
                if row["status"] == "failed":
                    torch.cuda.empty_cache()
            rows.append(row)
            print(f"{role} {checkpoint.name} {index}/{len(records)}: {row['status']} {row['error']}", flush=True)
    finally:
        # Restoration weights are released before loading LPIPS for this batch.
        del descriptor
        gc.collect()
        torch.cuda.empty_cache()
    return rows


def score_sr(rows, emit):
    perceptual, lpips_error = None, ""
    if any(row["status"] == "saved" for row in rows):
        try:
            perceptual = PerceptualMetric(device="cuda:0")
        except Exception as exc:
            lpips_error = f"LPIPS load: {type(exc).__name__}: {exc}"
    try:
        for row in rows:
            if row["status"] != "saved":
                emit(row)
                continue
            truth = restored = None
            errors = []
            try:
                # Score the saved PNG against the shared cropped HR, in integer RGB levels.
                truth = read_image(Path(row["reference_path"])).mul_(255).round_()
                restored = read_image(Path(row["output_path"])).mul_(255).round_()
                for name, metric in (("psnr", psnr), ("ssim", ssim), ("lpips", perceptual)):
                    try:
                        if metric is None:
                            raise RuntimeError(lpips_error)
                        value = float(metric(truth, restored))
                        if not math.isfinite(value) and not (name == "psnr" and value == math.inf):
                            raise ValueError("non-finite metric result")
                        row[name] = value
                    except Exception as exc:
                        errors.append(f"{name}: {type(exc).__name__}: {exc}")
            except Exception as exc:
                errors.append(f"metric input: {type(exc).__name__}: {exc}")
            finally:
                del truth, restored
                if errors:
                    torch.cuda.empty_cache()
            row.update(status="failed" if errors else "ok", error="; ".join(errors))
            emit(row)
    finally:
        del perceptual
        gc.collect()
        torch.cuda.empty_cache()


def run_experiment(groups, sr_models, deblur_models, run):
    failures = 0
    with (run / "results.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        stream.flush()

        def emit(row):
            nonlocal failures
            writer.writerow(row)
            stream.flush()
            failures += row["status"] == "failed"
            if row["status"] != "ok":
                print(f"Metrics {row['experiment']} {row['model']} {row['image']}: {row['status']} {row['error']}", flush=True)

        for group in groups:
            print(f"Group: {group[0]}M / {group[1]}ms; frames: {len(group[2])}", flush=True)
            records = prepare_group(group, run)
            for checkpoint in sr_models:
                score_sr(infer_model("sr", checkpoint, records, run), emit)
            # Baseline is measured once per original; never use SR or synthesized blur as input.
            for record in records:
                row = result_row(record, "deblur", "original")
                if record["error"]:
                    row.update(error=record["error"], invalid_metrics=";".join(BLUR_METRICS))
                else:
                    measure_sharpness(row)
                emit(row)
            for checkpoint in deblur_models:
                for row in infer_model("deblur", checkpoint, records, run):
                    if row["status"] == "saved":
                        measure_sharpness(row)
                    else:
                        row["invalid_metrics"] = ";".join(BLUR_METRICS)
                    emit(row)
    return 1 if failures else 0


def main():
    args = parse_args()
    try:
        groups = select_groups(args.limit)
        sr_models, deblur_models = select_models()
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Source: {SOURCE}; SR: {len(sr_models)}; Deblur: {len(deblur_models)}")
    for height, speed, sources in groups:
        print(f"{height}M/{speed}ms: {len(sources)} frames; {sources[0].relative_to(SOURCE)} .. {sources[-1].relative_to(SOURCE)}")
    for role, checkpoints in (("sr", sr_models), ("deblur", deblur_models)):
        for checkpoint in checkpoints:
            print(f"{role}: {checkpoint.relative_to(ROOT)}")
    if args.dry_run:
        return 0
    try:
        load_runtime()
    except Exception as exc:
        print(f"error: runtime preflight: {type(exc).__name__}: {exc}; no installation/download attempted", file=sys.stderr)
        return 2
    started = time.perf_counter()
    run = RUNS / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run.mkdir(parents=True, exist_ok=False)
    print(f"Run: {run}; input: {run / 'input'}; output: {run / 'output'}", flush=True)
    status = run_experiment(groups, sr_models, deblur_models, run)
    print(f"Results: {run / 'results.csv'}; exit status: {status}; elapsed: {time.perf_counter() - started:.1f}s", flush=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
PY
