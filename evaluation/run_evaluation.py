#!/usr/bin/env python3
"""Evaluate one selected SR/deblur order, or explicitly opt into legacy SR."""

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from report import describe_environment, render_failure_report, render_report, write_report, write_mode_reports, describe_mode_environment  # noqa: E402
from runner import DEFAULT_LIMIT, release_device_memory, run_batch, select_sources, run_ordered_batch  # noqa: E402
from runs import RUNS_ROOT, allocate_run_directory  # noqa: E402
from sources import discover_sources  # noqa: E402
from summary import summarise  # noqa: E402

MODELS_ROOT = PROJECT_ROOT / "models" / "sr"
DEBLUR_MODELS_ROOT = PROJECT_ROOT / "models" / "deblur"
CHECKPOINT_SUFFIXES = {".pth", ".pt", ".ckpt", ".safetensors"}


def _parse(argv):
    parser = argparse.ArgumentParser(
        description="Evaluate one ordered --sr/--deblur path; --legacy-sr selects synthetic SR evaluation.",
        allow_abbrev=False,
    )
    parser.add_argument("--sr", dest="stages", action="append_const", const="sr")
    parser.add_argument("--deblur", dest="stages", action="append_const", const="deblur")
    parser.add_argument("--sr-model", type=Path, help="one SR checkpoint; required with --sr")
    parser.add_argument("--deblur-model", type=Path, help="one deblur checkpoint; otherwise scan models/deblur/")
    parser.add_argument("--legacy-sr", action="store_true", help="old bicubic degradation plus PSNR/SSIM/LPIPS")
    parser.add_argument("--input", type=Path, help="source directory (new modes: evaluation/data/input/)")
    parser.add_argument("--limit", type=int, help="sample size (new modes: 1; legacy: 5)")
    parser.add_argument("--seed", type=int, default=None, help="sample randomly with this seed")
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT, help="directory for fresh runs")
    models = parser.add_mutually_exclusive_group()
    models.add_argument("--model", help="legacy only: checkpoint filename in models/sr/")
    models.add_argument("--all", action="store_true", help="legacy only: all checkpoints in models/sr/")
    arguments = parser.parse_args(argv)
    if arguments.legacy_sr:
        if arguments.stages or arguments.sr_model or arguments.deblur_model:
            parser.error("--legacy-sr cannot be combined with new mode flags/checkpoints")
        arguments.model = arguments.model or "model.pth"
    else:
        if not arguments.stages:
            parser.error("at least one mode flag is required: --sr or --deblur")
        if len(arguments.stages) != len(set(arguments.stages)):
            parser.error("mode flags must not be repeated")
        if arguments.model or arguments.all:
            parser.error("--model/--all require --legacy-sr")
        if "sr" in arguments.stages and arguments.sr_model is None:
            parser.error("--sr requires --sr-model")
        if arguments.sr_model is not None and "sr" not in arguments.stages:
            parser.error("--sr-model requires --sr")
        if arguments.deblur_model is not None and "deblur" not in arguments.stages:
            parser.error("--deblur-model requires --deblur")
    if arguments.limit is None:
        arguments.limit = DEFAULT_LIMIT if arguments.legacy_sr else 1
    if arguments.limit <= 0:
        parser.error("--limit must be positive")
    if arguments.input is None:
        arguments.input = PROJECT_ROOT / ("input" if arguments.legacy_sr else "evaluation/data/input")
    for name in ("input", "runs_root", "sr_model", "deblur_model"):
        path = getattr(arguments, name)
        if path is not None and not path.is_absolute():
            setattr(arguments, name, PROJECT_ROOT / path)
    return arguments


def select_mode_checkpoints(arguments) -> list[dict[str, Path]]:
    sr = arguments.sr_model
    if "sr" in arguments.stages and not sr.is_file():
        raise ValueError(f"SR model not found: {sr}")
    if "deblur" not in arguments.stages:
        return [{"sr": sr}]
    if arguments.deblur_model is not None:
        if not arguments.deblur_model.is_file():
            raise ValueError(f"deblur model not found: {arguments.deblur_model}")
        candidates = [arguments.deblur_model]
    else:
        if not DEBLUR_MODELS_ROOT.is_dir():
            raise ValueError(f"Model directory not found: {DEBLUR_MODELS_ROOT}")
        candidates = sorted(
            (path for path in DEBLUR_MODELS_ROOT.iterdir()
             if path.is_file() and path.suffix.lower() in CHECKPOINT_SUFFIXES),
            key=lambda path: (path.name.casefold(), path.name),
        )
    selected, seen = [], set()
    for candidate in candidates:
        target = candidate.resolve()
        if target not in seen:
            selected.append({"deblur": candidate, **({"sr": sr} if "sr" in arguments.stages else {})})
            seen.add(target)
    if not selected:
        raise ValueError(f"No checkpoints found in {DEBLUR_MODELS_ROOT}")
    return selected


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
        raise ValueError("--model must be a checkpoint filename inside models/sr/")
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
    from perceptual import PerceptualMetric

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


def _legacy_main(arguments) -> int:
    from perceptual import fix_cudnn_determinism

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


def main(argv=None) -> int:
    arguments = _parse(argv)
    if arguments.legacy_sr:
        return _legacy_main(arguments)
    try:
        if not arguments.input.is_dir():
            raise ValueError(f"not a directory: {arguments.input}")
        checkpoints = select_mode_checkpoints(arguments)
        discovered = len(discover_sources(arguments.input))
        selected = select_sources(arguments.input, limit=arguments.limit, seed=arguments.seed)
        if not selected:
            raise ValueError(f"no .png/.jpg/.jpeg sources in {arguments.input}")
        run_dir = allocate_run_directory(arguments.runs_root, subdirectories=())
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(f"mode          : {' -> '.join(arguments.stages)}")
    print(f"sources       : {len(selected)} of {discovered} discovered in {arguments.input}")
    print(f"combinations  : {len(checkpoints)}")
    for combination in checkpoints:
        print("checkpoints   : " + " -> ".join(str(combination[role]) for role in arguments.stages))
    print(f"run directory : {run_dir}", flush=True)
    environment = describe_mode_environment(arguments, discovered, selected)
    baselines, combinations = {}, []
    status = 0
    for combination in checkpoints:
        result = run_ordered_batch(selected, run_dir, arguments.stages, combination, baselines)
        combinations.append(result)
        failures = sum(row["status"] == "failed" for row in result["rows"])
        print(f"{result['id']}: success={len(result['rows']) - failures}, failed={failures}", flush=True)
        metric_errors = sum(
            record.get("status") == "failed"
            for row in result["rows"] if row["status"] == "success"
            for measurements in (row.get("before", {}), row.get("after", {}))
            for record in measurements.values()
        )
        if result.get("error") or failures or metric_errors:
            status = 1
        if metric_errors:
            print(f"metric errors : {metric_errors}; successful PNGs retained", flush=True)
        # Persist every completed combination, including failures.
        write_mode_reports(run_dir, environment, combinations)
        release_device_memory()
    print(f"report        : {run_dir / 'report.md'}")
    print(f"per image     : {run_dir / 'per_image.md'}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
