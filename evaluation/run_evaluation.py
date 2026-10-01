#!/usr/bin/env python3
"""Evaluate selected local checkpoints against the same source sample.

    .venv/bin/python evaluation/run_evaluation.py [--model NAME | --all] [--limit N]

Every artefact lands in a new evaluation/runs/<timestamp>/ directory. Nothing is
ever written into an existing run, and nothing outside evaluation/runs/ is
written at all - input/, output/, models/ and src/ are read-only to this tool.

The default limit is deliberately small. Running the whole dataset is not
authorized by the plan and needs the user's agreement; see evaluation/PLANS.md.
"""

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from perceptual import PerceptualMetric, fix_cudnn_determinism  # noqa: E402
from report import describe_environment, render_failure_report, render_report, write_report  # noqa: E402
from runner import DEFAULT_LIMIT, release_device_memory, run_batch, select_sources  # noqa: E402
from runs import RUNS_ROOT, allocate_run_directory  # noqa: E402
from sources import discover_sources  # noqa: E402
from summary import summarise  # noqa: E402

MODELS_ROOT = PROJECT_ROOT / "models"
CHECKPOINT_SUFFIXES = {".pth", ".pt", ".ckpt", ".safetensors"}


def _parse(argv):
    parser = argparse.ArgumentParser(description="Compare the SR line against a bicubic baseline.")
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "input", help="source image directory")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help=f"how many images (default {DEFAULT_LIMIT})")
    parser.add_argument("--seed", type=int, default=None, help="sample randomly with this seed instead of taking the first N")
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT, help="where run directories are created")
    models = parser.add_mutually_exclusive_group()
    models.add_argument("--model", default="model.pth", help="checkpoint filename in models/ (default: model.pth)")
    models.add_argument("--all", action="store_true", help="evaluate every checkpoint in models/ on the same sample")
    arguments = parser.parse_args(argv)
    if arguments.limit <= 0:
        parser.error("--limit must be positive")
    return arguments


def select_checkpoints(model: str, all_models: bool) -> list[Path]:
    if not MODELS_ROOT.is_dir():
        raise ValueError(f"Model directory not found: {MODELS_ROOT}")
    if all_models:
        candidates = sorted(
            (path for path in MODELS_ROOT.iterdir() if path.is_file() and path.suffix.lower() in CHECKPOINT_SUFFIXES),
            key=lambda path: (path.name.casefold(), path.name),
        )
        # model.pth may be an alias for a named checkpoint in this folder.
        checkpoints = []
        seen = set()
        for path in candidates:
            target = path.resolve()
            if target not in seen:
                checkpoints.append(path)
                seen.add(target)
        if not checkpoints:
            raise ValueError(f"No checkpoints found in {MODELS_ROOT}")
        return checkpoints
    if not model or Path(model).name != model or model in {".", ".."}:
        raise ValueError("--model must be a checkpoint filename inside models/")
    checkpoint = MODELS_ROOT / model
    if not checkpoint.is_file():
        raise ValueError(f"SR model not found: {checkpoint}")
    return [checkpoint]


def _evaluate_checkpoint(arguments, selected, discovered, checkpoint, run_dir) -> int:
    started = datetime.now(timezone.utc).astimezone()
    # The SR line picks its own device through the pipeline and
    # LPIPS follows it, so a run never mixes devices for the parts that depend
    # on one. PSNR and SSIM stay on CPU float64 by construction.
    from sr_line import SuperResolutionLine

    sr_line = SuperResolutionLine(checkpoint)
    perceptual = PerceptualMetric(device=sr_line.device)
    print(f"device        : {sr_line.device} (SR line and LPIPS; PSNR/SSIM always CPU float64)")
    print()

    def progress(index, total, name):
        print(f"[{index}/{total}] {name}", flush=True)

    clock = time.perf_counter()
    results, failures = run_batch(selected, run_dir, sr_line, perceptual, progress=progress)
    elapsed = time.perf_counter() - clock

    summary = summarise(results, failures)
    environment = describe_environment(
        started=started,
        run_dir=run_dir,
        project_root=PROJECT_ROOT,
        source_directory=arguments.input,
        sampling="sequential" if arguments.seed is None else f"random (seed {arguments.seed})",
        limit=arguments.limit,
        discovered=discovered,
        selected=len(selected),
        sr_line=sr_line,
        lpips_device=str(sr_line.device),
    )
    report_path = run_dir / "report.md"
    write_report(report_path, render_report(environment, results, failures, summary))

    print()
    print(f"measured      : {summary.included} image(s) on both lines; {summary.failed} excluded")
    for metric in summary.metrics:
        if metric.winner == "n/a":
            print(f"  {metric.name:6s} no comparison available")
        else:
            print(f"  {metric.name:6s} SR {metric.sr_mean:.6f} | bicubic {metric.bicubic_mean:.6f} -> {metric.winner}")
    print(f"elapsed       : {elapsed:.1f} s ({elapsed / max(len(selected), 1):.1f} s per image)")
    print(f"report        : {report_path}")
    if summary.included == 0:
        print("error: nothing was measured on both lines", file=sys.stderr)
        return 1
    return 0


def main(argv=None) -> int:
    arguments = _parse(argv)
    fix_cudnn_determinism()

    if not arguments.input.is_dir():
        print(f"error: not a directory: {arguments.input}", file=sys.stderr)
        return 2
    try:
        checkpoints = select_checkpoints(arguments.model, arguments.all)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    discovered = len(discover_sources(arguments.input))
    selected = select_sources(arguments.input, limit=arguments.limit, seed=arguments.seed)
    if not selected:
        print(f"error: no .png/.jpg/.jpeg sources in {arguments.input}", file=sys.stderr)
        return 2
    print(f"sources       : {len(selected)} of {discovered} discovered in {arguments.input}")

    status = 0
    for checkpoint in checkpoints:
        run_dir = allocate_run_directory(arguments.runs_root)
        print(f"checkpoint    : {checkpoint.name}", flush=True)
        print(f"run directory : {run_dir}", flush=True)
        try:
            status = max(status, _evaluate_checkpoint(arguments, selected, discovered, checkpoint, run_dir))
        except Exception as error:
            reason = f"{type(error).__name__}: {error}"
            report_path = run_dir / "report.md"
            write_report(report_path, render_failure_report(checkpoint, len(selected), reason))
            print(f"error: {checkpoint.name}: {reason}", file=sys.stderr)
            print(f"report        : {report_path}")
            status = 1
        # The helper's model and LPIPS references are gone before loading the next checkpoint.
        release_device_memory()
    print(f"checkpoints attempted: {', '.join(checkpoint.name for checkpoint in checkpoints)}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
