"""Markdown evaluation data: checkpoint, run parameters, scores, and failures."""

import hashlib
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from summary import HIGHER_IS_BETTER, decide_winner

REQUIRED_HEADER_FIELDS = (
    "執行時間",
    "run 目錄",
    "git HEAD",
    "worktree 狀態",
    "原圖來源資料夾",
    "探索規則",
    "取樣方式",
    "取樣上限",
    "探索到的張數",
    "實際處理張數",
    "降採樣與放大演算法",
    "倍率",
    "Pillow 版本",
    "mod-crop",
    "checkpoint",
    "模型檔",
    "模型 SHA-256",
    "模型架構",
    "模型 scale",
    "SR tile 設定",
    "SR 線 device",
    "PSNR／SSIM device",
    "LPIPS device",
    "cudnn.benchmark",
    "色彩空間與 data_range",
    "SSIM 參數",
    "SSIM 邊界與變異數",
    "LPIPS 套件版本",
    "LPIPS net",
    "LPIPS 線性層權重 SHA-256",
    "LPIPS backbone 來源",
    "LPIPS backbone SHA-256",
)


@dataclass(frozen=True)
class RunEnvironment:
    started: datetime
    run_dir: Path
    git_head: str
    worktree_clean: bool
    source_directory: Path
    discovery_rule: str
    sampling: str
    limit: int
    discovered: int
    selected: int
    pillow_version: str
    model_target: str
    model_sha256: str
    architecture: str
    model_scale: int
    tile_size: int
    sr_device: str
    lpips_device: str
    cudnn_benchmark: bool
    lpips_version: str
    lpips_net: str
    lpips_linear_sha256: str
    lpips_backbone_url: str
    lpips_backbone_sha256: str


def _cell(value) -> str:
    return str(value).replace("|", "&#124;").replace("\r", " ").replace("\n", " ")


def _number(value: float | None, digits: int) -> str:
    if value is None:
        return "n/a"
    if value == float("inf"):
        return "inf"
    if value == float("-inf"):
        return "-inf"
    return f"{value:.{digits}f}"


_DIGITS = {"PSNR": 4, "SSIM": 6, "LPIPS": 6}


def _size(pair: tuple[int, int]) -> str:
    return f"{pair[0]}×{pair[1]}"


def _header_rows(environment: RunEnvironment, sr_device: str) -> dict[str, str]:
    return {
        "執行時間": environment.started.strftime("%Y-%m-%d %H:%M:%S %z (%Z)"),
        "run 目錄": str(environment.run_dir),
        "git HEAD": environment.git_head,
        "worktree 狀態": "clean" if environment.worktree_clean else "dirty",
        "原圖來源資料夾": str(environment.source_directory),
        "探索規則": environment.discovery_rule,
        "取樣方式": environment.sampling,
        "取樣上限": str(environment.limit),
        "探索到的張數": str(environment.discovered),
        "實際處理張數": str(environment.selected),
        "降採樣與放大演算法": "Pillow Image.Resampling.BICUBIC",
        "倍率": f"{environment.model_scale}×",
        "Pillow 版本": environment.pillow_version,
        "mod-crop": "right/bottom; multiple=4",
        "checkpoint": Path(environment.model_target).name,
        "模型檔": environment.model_target,
        "模型 SHA-256": environment.model_sha256,
        "模型架構": environment.architecture,
        "模型 scale": str(environment.model_scale),
        "SR tile 設定": f"core={environment.tile_size}; halo=32",
        "SR 線 device": sr_device,
        "PSNR／SSIM device": "CPU; float64",
        "LPIPS device": environment.lpips_device,
        "cudnn.benchmark": str(environment.cudnn_benchmark),
        "色彩空間與 data_range": "RGB; 8-bit; data_range=255",
        "SSIM 參數": "Gaussian 11×11; σ=1.5; K1=0.01; K2=0.03; channel_mean",
        "SSIM 邊界與變異數": "valid; Gaussian-weighted population variance",
        "LPIPS 套件版本": environment.lpips_version,
        "LPIPS net": environment.lpips_net,
        "LPIPS 線性層權重 SHA-256": environment.lpips_linear_sha256,
        "LPIPS backbone 來源": environment.lpips_backbone_url,
        "LPIPS backbone SHA-256": environment.lpips_backbone_sha256,
    }


def _per_image_table(results) -> list[str]:
    lines = [
        "| 檔名 | 原始尺寸 | 真值（裁切後） | LR 尺寸 "
        "| SR PSNR | bicubic PSNR | PSNR 勝方 "
        "| SR SSIM | bicubic SSIM | SSIM 勝方 "
        "| SR LPIPS | bicubic LPIPS | LPIPS 勝方 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for item in results:
        cells = [_cell(item.source_name), _size(item.original), _size(item.cropped), _size(item.low)]
        for name in ("PSNR", "SSIM", "LPIPS"):
            sr_value = getattr(item.sr, name.lower())
            bicubic_value = getattr(item.bicubic, name.lower())
            winner, _ = decide_winner(name, sr_value, bicubic_value)
            cells += [_number(sr_value, _DIGITS[name]), _number(bicubic_value, _DIGITS[name]), winner]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _average_table(summary) -> list[str]:
    lines = [
        "| 納入張數 | 排除張數 |",
        "|---|---|",
        f"| {summary.included} | {summary.failed} |",
        "",
        "| 指標 | 方向 | SR 平均 | bicubic 平均 | 勝方 | 差距 | 納入張數 | 因 inf 排除 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for metric in summary.metrics:
        direction = "↑" if HIGHER_IS_BETTER[metric.name] else "↓"
        digits = _DIGITS[metric.name]
        lines.append(
            f"| {metric.name} | {direction} | {_number(metric.sr_mean, digits)} "
            f"| {_number(metric.bicubic_mean, digits)} | {metric.winner} "
            f"| {_number(metric.margin, digits)} | {metric.counted} | {metric.excluded_infinite} |"
        )
    return lines


def _failure_table(failures) -> list[str]:
    lines = ["| 檔名 | 環節 | 原因 |", "|---|---|---|"]
    lines += [f"| {_cell(f.source_name)} | {_cell(f.stage)} | {_cell(f.reason)} |" for f in failures]
    return lines


def render_report(environment: RunEnvironment, results, failures, summary) -> str:
    sections = [
        "# 評估數據",
        "",
        "## 執行資料",
        "",
        "| 項目 | 值 |",
        "|---|---|",
    ]
    rows = _header_rows(environment, environment.sr_device)
    missing = [field for field in REQUIRED_HEADER_FIELDS if not rows.get(field)]
    if missing:
        raise ValueError(f"Report header is missing required fields: {missing}")
    sections += [f"| {field} | {_cell(rows[field])} |" for field in REQUIRED_HEADER_FIELDS]
    sections += ["", "## 逐張成績", ""] + _per_image_table(results)
    sections += ["", "## 平均", ""] + _average_table(summary)
    sections += ["", "## 失敗與排除", ""] + _failure_table(failures) + [""]
    return "\n".join(sections)


def render_failure_report(model_path: Path, selected: int, error: str) -> str:
    """Record a checkpoint-level failure when no complete result is available."""
    try:
        model_sha256 = _sha256(model_path)
    except OSError:
        model_sha256 = "n/a"
    return "\n".join([
        "# 評估數據",
        "",
        "| 項目 | 值 |",
        "|---|---|",
        f"| checkpoint | {_cell(model_path.name)} |",
        f"| 模型檔 | {_cell(model_path)} |",
        f"| 模型 SHA-256 | {model_sha256} |",
        f"| 選取張數 | {selected} |",
        "| 狀態 | failed |",
        f"| 原因 | {_cell(error)} |",
        "",
    ])


def write_report(destination: Path, text: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(project_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=project_root, capture_output=True, text=True, check=True
    ).stdout.strip()


def describe_environment(
    *,
    started: datetime,
    run_dir: Path,
    project_root: Path,
    source_directory: Path,
    sampling: str,
    limit: int,
    discovered: int,
    selected: int,
    sr_line,
    lpips_device: str,
) -> RunEnvironment:
    """Read the actual state of everything the report records."""
    import lpips
    import torch
    from importlib.metadata import version
    from PIL import Image as PILImage
    from torchvision.models import AlexNet_Weights

    from drone_sr.tiling import TILE_SIZE

    status = _git(project_root, "status", "--porcelain")
    tracked_dirty = [line for line in status.splitlines() if not line.startswith("??")]

    backbone_url = AlexNet_Weights.IMAGENET1K_V1.url
    cached = Path(torch.hub.get_dir()) / "checkpoints" / backbone_url.rsplit("/", 1)[-1]
    backbone_sha = _sha256(cached) if cached.is_file() else "n/a"

    return RunEnvironment(
        started=started,
        run_dir=run_dir,
        git_head=_git(project_root, "rev-parse", "HEAD"),
        worktree_clean=not tracked_dirty,
        source_directory=source_directory,
        discovery_rule="direct children; .png/.jpg/.jpeg; case-insensitive",
        sampling=sampling,
        limit=limit,
        discovered=discovered,
        selected=selected,
        pillow_version=PILImage.__version__ if hasattr(PILImage, "__version__") else version("pillow"),
        model_target=str(sr_line.model_path),
        model_sha256=_sha256(sr_line.model_path),
        architecture=str(sr_line.architecture),
        model_scale=sr_line.scale,
        tile_size=TILE_SIZE,
        sr_device=str(sr_line.device),
        lpips_device=lpips_device,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        lpips_version=version("lpips"),
        lpips_net="alex",
        lpips_linear_sha256=_sha256(Path(lpips.__file__).parent / "weights" / "v0.1" / "alex.pth"),
        lpips_backbone_url=backbone_url,
        lpips_backbone_sha256=backbone_sha,
    )


def describe_mode_environment(arguments, discovered, selected) -> dict:
    """Collect run facts without importing LPIPS or initializing a model."""
    import os
    import platform
    from datetime import timezone
    from importlib.metadata import PackageNotFoundError, version

    import torch

    root = Path(__file__).resolve().parents[1]
    packages = {}
    for name in ("torch", "torchvision", "spandrel", "spandrel_extra_arches",
                 "Pillow", "numpy", "scipy", "opencv-python-headless", "scikit-image"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "not installed"
    facts = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": _git(root, "rev-parse", "HEAD"),
        "worktree": _git(root, "status", "--short"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "mode": " -> ".join(arguments.stages),
        "input_directory": str(arguments.input),
        "sampling": "sequential" if arguments.seed is None else f"seed={arguments.seed}",
        "limit": arguments.limit,
        "discovered": discovered,
        "selected": [str(path.resolve()) for path in selected],
        "cuda_runtime": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not set"),
        "precision": "float32; RGB 8-bit PNG; no intermediate quantization",
        "tiling": "SR 512 core / 32 halo; x1 respects descriptor tiling recommendation",
    }
    for name in ("memory.max", "memory.current"):
        path = Path("/sys/fs/cgroup") / name
        if path.exists():
            facts["cgroup_" + name] = path.read_text().strip()
    return facts


def _mode_link(path, run_dir: Path, label: str) -> str:
    import os
    from urllib.parse import quote

    if path is None:
        return "N/A"
    return f"[{_cell(label)}](<{quote(os.path.relpath(path, run_dir), safe='/')}>)"


def write_mode_reports(run_dir: Path, environment: dict, combinations: list[dict]) -> None:
    """Both views consume the same unrounded records; Phase 02 metrics are pending."""
    import json

    main = ["# SR / deblur evaluation", "", "[逐張資料](per_image.md)", "",
            "指標待量測；本階段只驗證處理路徑，不代表去模糊效果。", "",
            "## 執行環境", "", "| 項目 | 值 |", "|---|---|"]
    for key, value in environment.items():
        text = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
        main.append(f"| {_cell(key)} | {_cell(text)} |")
    detail = ["# 逐張資料", "", "[主報告](report.md)", ""]
    for combo in combinations:
        rows = combo["rows"]
        success = sum(row["status"] == "success" for row in rows)
        title = combo["id"]
        main += ["", f"## {_cell(title)}", "",
                 f"模式：{' → '.join(combo['order'])}；成功 {success} / {len(rows)}；失敗 {len(rows) - success}。",
                 f"耗時：{combo.get('elapsed_seconds', 'N/A')} seconds",
                 f"模型錯誤：{_cell(combo.get('error') or '無')}", "",
                 "| role | checkpoint | SHA-256 | architecture | scale | device |",
                 "|---|---|---|---|---|---|"]
        for model in combo.get("models", []):
            main.append("| " + " | ".join(_cell(model.get(key, "N/A")) for key in
                        ("role", "path", "sha256", "architecture", "scale", "device")) + " |")
        main += ["", "樣本目視入口："]
        for index, row in enumerate(rows, 1):
            anchor = f"{title}-{index}"
            main.append(f"- [{_cell(Path(row['input']).name)}](per_image.md#{anchor})：{row['status']}")
            detail += [f'<a id="{anchor}"></a>', f"## {_cell(title)} / {_cell(Path(row['input']).name)}", "",
                       f"順序：{' → '.join(combo['order'])}",
                       f"input: {_mode_link(row['input'], run_dir, Path(row['input']).name)} / {row.get('input_size')}",
                       f"output: {_mode_link(row.get('output'), run_dir, 'PNG')} / {row.get('output_size')}",
                       f"status: {row['status']}; stage: {_cell(row.get('failure_stage'))}; reason: {_cell(row.get('reason'))}",
                       "", "| 指標 | 前 | 後 | 變化 | 有效性／原因 |",
                       "|---|---|---|---|---|"]
            for metric in ("laplacian_variance", "tenengrad", "cpbd", "crete_roffet_blur"):
                detail.append(f"| {metric} | N/A | N/A | N/A | 待量測 |")
            detail.append("")
    write_report(run_dir / "report.md", "\n".join(main) + "\n")
    write_report(run_dir / "per_image.md", "\n".join(detail) + "\n")
