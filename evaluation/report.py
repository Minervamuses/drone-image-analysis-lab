"""Markdown evaluation data: checkpoint, run parameters, scores, and failures."""

import hashlib
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from summary import (BLUR_METRICS, HIGHER_IS_BETTER, decide_winner, mode_change,
                     summarise_mode, summarise_mode_comparisons)

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
        "| SR LPIPS | bicubic LPIPS | LPIPS 勝方 | 輸出檔名（hr/lr/bicubic/sr） |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for item in results:
        cells = [_cell(item.source_name), _size(item.original), _size(item.cropped), _size(item.low)]
        for name in ("PSNR", "SSIM", "LPIPS"):
            sr_value = getattr(item.sr, name.lower())
            bicubic_value = getattr(item.bicubic, name.lower())
            winner, _ = decide_winner(name, sr_value, bicubic_value)
            cells += [_number(sr_value, _DIGITS[name]), _number(bicubic_value, _DIGITS[name]), winner]
        cells.append(_cell(item.output_name or f"{Path(item.source_name).stem}.png"))
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
    if torch.cuda.is_available():
        facts["gpu_name"] = torch.cuda.get_device_name(0)
        free, total = torch.cuda.mem_get_info(0)
        facts["gpu_memory_free_bytes"] = free
        facts["gpu_memory_total_bytes"] = total
    memory = Path("/proc/meminfo")
    if memory.exists():
        facts["host_memory"] = [line for line in memory.read_text().splitlines()
                                if line.startswith(("MemTotal:", "MemAvailable:"))]
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


def _mode_value(value) -> str:
    import json

    if value is None:
        return "N/A"
    if isinstance(value, (dict, list, tuple)):
        return _cell(json.dumps(value, ensure_ascii=False, default=str))
    return _cell(value)


def _mode_number(value) -> str:
    import math

    if value is None:
        return "N/A"
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return "N/A (non-finite)"
    return f"{value:.10g}"


def _mode_record(row: dict, field: str, name: str) -> dict:
    return row.get(field, {}).get(name) or {
        "value": None, "valid": False, "status": "unavailable",
        "reason": "指標記錄缺失（待量測或未產生）", "debug": {},
    }


def _mode_validity(record: dict) -> str:
    return _cell(f"valid={bool(record.get('valid'))}; {record.get('status', 'unavailable')}; "
                 f"{record.get('reason') or '無'}")


def _mode_models(models) -> list[str]:
    fields = ("role", "path", "sha256", "architecture", "scale", "device", "tiling", "size_requirements")
    lines = ["| role | checkpoint | SHA-256 | architecture | scale | device | tiling | size_requirements |",
             "|---|---|---|---|---|---|---|---|"]
    lines += ["| " + " | ".join(_mode_value(model.get(key)) for key in fields) + " |" for model in models]
    return lines


def _mode_summary_table(summary: dict) -> list[str]:
    lines = ["| 指標 | 逐張變化 | 朝清晰方向 | 變化中位數 | 朝清晰比例 | 朝清晰／平手／反向 | 有效／總數 |",
             "|---|---|---|---|---|---|---|"]
    for metric in summary["metrics"].values():
        direction = "> 1" if metric["kind"] == "ratio" else ("> 0" if metric["higher_is_better"] else "< 0")
        ratio = metric["improvement_proportion"]
        proportion = "N/A" if ratio is None else f"{metric['improved']}/{metric['valid']} ({ratio:.1%})"
        lines.append(f"| {metric['name']} | {metric['kind']} | {direction} | {_mode_number(metric['median_change'])} "
                     f"| {proportion} | {metric['improved']}／{metric['tied']}／{metric['reversed']} "
                     f"| {metric['valid']}/{metric['total']} |")
    lines += ["", "| 指標 | 排除／不可量測原因 | 張數 |", "|---|---|---|"]
    excluded = False
    for metric in summary["metrics"].values():
        for reason, count in metric["exclusions"].items():
            excluded = True
            lines.append(f"| {metric['name']} | {_cell(reason)} | {count} |")
    if not excluded:
        lines.append("| 全部 | 無 | 0 |")
    lines += ["", f"CPBD 處理後可量邊緣消失：{summary['metrics']['cpbd']['cpbd_edges_disappeared']} 張。"]
    return lines


def _mode_comparisons(comparisons: dict) -> list[str]:
    lines = ["", "## 同指標共同有效樣本對照", "",
             "覆蓋率分母是該比較所有模型選取輸入的聯集；僅使用列出的共同有效圖片，不跨指標合成總分或宣稱最佳模型。"]
    for name in comparisons["global_unavailable"]:
        lines += ["", f"全體模型 / {name}：共同有效樣本為空，N/A；不產生全體排名。"]
    for label, key in (("全體模型", "global"), ("配對模型", "pairwise")):
        for comparison in comparisons[key]:
            lines += ["", f"### {label} / {comparison['metric']}", "",
                      f"模型：{_cell(', '.join(comparison['models']))}",
                      f"共同有效：{comparison['valid']}/{comparison['total']}；覆蓋率 {comparison['coverage']:.1%}。", "",
                      "| 模型組合 | 共同樣本的逐張變化中位數 |", "|---|---|"]
            for identifier, value in comparison["median_changes"].items():
                lines.append(f"| {_cell(identifier)} | {_mode_number(value)} |")
            lines += ["", "共同樣本完整來源："]
            lines += [f"- {_cell(source)}" for source in comparison["inputs"]]
    if not comparisons["global"] and not comparisons["pairwise"] and not comparisons["global_unavailable"]:
        lines += ["", "N/A：少於兩個 deblur 模型組合；SR／combine 不作跨尺寸變化比較。"]
    return lines


def write_mode_reports(run_dir: Path, environment: dict, combinations: list[dict]) -> None:
    """Both views consume the same unrounded records; formatting is render-only."""
    main = ["# SR / deblur evaluation", "", "[逐張資料](per_image.md)", "",
            "四項指標各自描述清晰度相關變化，不等於去模糊成功率或真實細節恢復。", "",
            "## 執行環境", "", "| 項目 | 值 |", "|---|---|"]
    for key, value in environment.items():
        main.append(f"| {_cell(key)} | {_mode_value(value)} |")
    main += ["", "## 量測與解讀限制", "",
             "量測使用同一解碼／EXIF／RGB 8-bit 規則，轉為灰階 0–255；後值來自實際交付 PNG，沒有為計分 resize。",
             "Laplacian 使用 ksize=1；Tenengrad 使用 3×3 Sobel 梯度平方的平均；CPBD 保留邊緣 debug；Crété h_size=9。",
             "先算每張後÷前（Laplacian／Tenengrad）或後−前（CPBD／Crété），再取中位數；平手留在有效分母。",
             "比值前值為 0、CPBD 任一方無可量邊緣、Crété 非有限或單一指標失敗均排除該項，沒有有效分母即 N/A。",
             "SR／combine 僅列原始前後分數，變化及其摘要標示「跨尺寸不適用」。中位數不代表對稱抵消後的總改善量。",
             "雜訊與過銳化可能提高梯度分數；假紋理不能當成真實地物細節。低紋理可能無可量邊緣或使指標不可量測。",
             "請從下方樣本入口目視檢查清晰度、雜訊、色偏與 tile 邊界；此報告不提供未經驗證的畫質勝負結論。"]
    detail = ["# 逐張資料", "", "[主報告](report.md)", ""]
    for combo in combinations:
        rows = combo["rows"]
        summary = summarise_mode(combo)
        title = combo["id"]
        main += ["", f"## {_cell(title)}", "",
                 f"模式：{' → '.join(combo['order'])}；成功 {summary['success']} / {summary['total']}；失敗 {summary['failed']}。",
                 f"耗時：{_mode_number(combo.get('elapsed_seconds'))} seconds",
                 f"資源用量：{_mode_value(combo.get('resources'))}",
                 f"模型錯誤：{_cell(combo.get('error') or '無')}", ""]
        main += _mode_models(combo.get("models", []))
        main += [""] + _mode_summary_table(summary)
        main += ["", "### 處理失敗", "", "| input | 階段 | 原因 |", "|---|---|---|"]
        for row in rows:
            if row.get("status") != "success":
                main.append(f"| {_cell(row['input'])} | {_mode_value(row.get('failure_stage'))} | {_mode_value(row.get('reason'))} |")
        if not summary["failed"]:
            main.append("| 無 | — | — |")
        main += ["", "樣本目視入口："]
        detail += [f"## {_cell(title)}", ""] + _mode_models(combo.get("models", [])) + [""]
        for index, row in enumerate(rows, 1):
            anchor = f"{title}-{index}"
            main.append(f"- [{_cell(Path(row['input']).name)}](per_image.md#{anchor})：{row['status']}")
            detail += [f'<a id="{anchor}"></a>', f"### {_cell(title)} / {_cell(Path(row['input']).name)}", "",
                       f"順序：{' → '.join(combo['order'])}；模型身分見本組 checkpoint／SHA-256 表。",
                       f"input: {_mode_link(row['input'], run_dir, row['input'])} / 尺寸 {_mode_value(row.get('input_size'))}",
                       f"output: {_mode_link(row.get('output'), run_dir, row.get('output') or 'PNG')} / 尺寸 {_mode_value(row.get('output_size'))}",
                       f"status: {row['status']}; stage: {_mode_value(row.get('failure_stage'))}; reason: {_mode_value(row.get('reason'))}",
                       f"耗時：{_mode_number(row.get('elapsed_seconds'))} seconds", "",
                       "| 指標 | 前 | 前有效性／原因 | 後 | 後有效性／原因 | 變化 | 變化有效性／原因 | debug（前／後／變化） |",
                       "|---|---|---|---|---|---|---|---|"]
            for metric in BLUR_METRICS:
                before = _mode_record(row, "before", metric)
                after = _mode_record(row, "after", metric)
                change = mode_change(row, metric, combo["order"])
                kind = "ratio" if metric in ("laplacian_variance", "tenengrad") else "delta"
                debug = {"before": before.get("debug", {}), "after": after.get("debug", {}), "change": change.get("debug", {})}
                detail.append(f"| {metric} | {_mode_number(before.get('value'))} | {_mode_validity(before)} "
                              f"| {_mode_number(after.get('value'))} | {_mode_validity(after)} "
                              f"| {kind}: {_mode_number(change.get('value'))} | {_mode_validity(change)} | {_mode_value(debug)} |")
            detail.append("")
    main += _mode_comparisons(summarise_mode_comparisons(combinations))
    write_report(run_dir / "report.md", "\n".join(main) + "\n")
    write_report(run_dir / "per_image.md", "\n".join(detail) + "\n")
