"""Averaging, under the inclusion rules GOALS.md fixes.

Two rules, both of them load-bearing and neither of them adjustable to make a
result look better:

- An image is averaged only if BOTH lines measured it. If either line failed,
  the image counts for neither, so the two averages always cover the same set.
- An infinite PSNR (identical images, MSE 0) drops that image from the PSNR
  average only. Its SSIM and LPIPS still count, and the report says how many
  images were dropped and why.
"""

import math
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from statistics import median

from metric_defs import BLUR_METRICS, RATIO_METRICS

METRIC_NAMES = ("PSNR", "SSIM", "LPIPS")
# LPIPS is a distance: lower is more similar. The other two are fidelity scores.
HIGHER_IS_BETTER = {"PSNR": True, "SSIM": True, "LPIPS": False}


@dataclass(frozen=True)
class MetricSummary:
    name: str
    higher_is_better: bool
    sr_mean: float | None
    bicubic_mean: float | None
    counted: int
    excluded_infinite: int
    winner: str
    margin: float | None


@dataclass(frozen=True)
class Summary:
    included: int
    failed: int
    metrics: tuple[MetricSummary, ...]


def _field(scores, name: str) -> float:
    return getattr(scores, name.lower())


def decide_winner(name: str, sr_value: float | None, bicubic_value: float | None):
    """(winner, margin) for one metric. Used per image and for the averages."""
    if sr_value is None or bicubic_value is None:
        return "n/a", None
    if sr_value == bicubic_value:
        return "tie", 0.0
    sr_ahead = sr_value > bicubic_value if HIGHER_IS_BETTER[name] else sr_value < bicubic_value
    return ("SR" if sr_ahead else "bicubic"), abs(sr_value - bicubic_value)


def _summarise_metric(name: str, results) -> MetricSummary:
    usable = results
    excluded = 0
    if name == "PSNR":
        usable = [r for r in results if math.isfinite(_field(r.sr, name)) and math.isfinite(_field(r.bicubic, name))]
        excluded = len(results) - len(usable)

    if usable:
        sr_mean = sum(_field(r.sr, name) for r in usable) / len(usable)
        bicubic_mean = sum(_field(r.bicubic, name) for r in usable) / len(usable)
    else:
        sr_mean = bicubic_mean = None

    winner, margin = decide_winner(name, sr_mean, bicubic_mean)
    return MetricSummary(
        name=name,
        higher_is_better=HIGHER_IS_BETTER[name],
        sr_mean=sr_mean,
        bicubic_mean=bicubic_mean,
        counted=len(usable),
        excluded_infinite=excluded,
        winner=winner,
        margin=margin,
    )


def summarise(results, failures) -> Summary:
    return Summary(
        included=len(results),
        failed=len(failures),
        metrics=tuple(_summarise_metric(name, results) for name in METRIC_NAMES),
    )


# These summaries deliberately remain separate from the legacy reference scores.
def mode_change(row: dict, name: str, order) -> dict:
    """Read one recorded change without inventing missing measurements."""
    if list(order) != ["deblur"]:
        return {"value": None, "valid": False, "status": "unmeasurable",
                "reason": "跨尺寸不適用", "debug": {}}
    if row.get("status") != "success":
        return {"value": None, "valid": False, "status": "failed",
                "reason": f"處理失敗 ({row.get('failure_stage') or 'unknown'}): {row.get('reason') or '未提供原因'}",
                "debug": {}}
    record = row.get("changes", {}).get(name)
    if record is None:
        return {"value": None, "valid": False, "status": "unavailable",
                "reason": "指標記錄缺失（待量測或未產生）", "debug": {}}
    value = record.get("value")
    if record.get("valid") and (not isinstance(value, (int, float)) or not math.isfinite(value)):
        return {**record, "value": None, "valid": False, "status": "failed", "reason": "變化值非有限數值"}
    return record


def _cpbd_edges_disappeared(row: dict) -> bool:
    before = row.get("before", {}).get("cpbd", {}).get("debug", {})
    after = row.get("after", {}).get("cpbd", {}).get("debug", {})
    keys = ("cpbd_edge_count", "cpbd_valid_blocks")
    return (row.get("status") == "success"
            and all(isinstance(before.get(key), (int, float)) and before[key] > 0 for key in keys)
            and any(after.get(key) == 0 for key in keys))


def summarise_mode(combo: dict) -> dict:
    """Summarise per-image changes, independently for each no-reference metric."""
    rows = combo["rows"]
    metrics = {}
    for name in BLUR_METRICS:
        changes = [(row, mode_change(row, name, combo["order"])) for row in rows]
        valid = [(row, record["value"]) for row, record in changes if record.get("valid")]
        values = [value for _, value in valid]
        kind = "ratio" if name in RATIO_METRICS else "delta"
        neutral = 1.0 if kind == "ratio" else 0.0
        higher = name != "crete_roffet_blur"
        improved = sum(value > neutral if higher else value < neutral for value in values)
        tied = sum(value == neutral for value in values)
        metrics[name] = {
            "name": name, "kind": kind, "higher_is_better": higher,
            "median_change": median(values) if values else None,
            "improved": improved, "tied": tied, "reversed": len(values) - improved - tied,
            "valid": len(values), "total": len(rows),
            "improvement_proportion": improved / len(values) if values else None,
            "exclusions": dict(Counter(record.get("reason") or record.get("status") or "指標無效"
                                       for _, record in changes if not record.get("valid"))),
            "valid_inputs": [row["input"] for row, _ in valid],
            "cpbd_edges_disappeared": sum(_cpbd_edges_disappeared(row) for row in rows) if name == "cpbd" else 0,
        }
    success = sum(row.get("status") == "success" for row in rows)
    return {"id": combo["id"], "total": len(rows), "success": success,
            "failed": len(rows) - success, "metrics": metrics}


def _common_comparison(combos, name: str) -> dict | None:
    by_model = {}
    total_inputs = set()
    for combo in combos:
        total_inputs.update(row["input"] for row in combo["rows"])
        values = {}
        for row in combo["rows"]:
            record = mode_change(row, name, combo["order"])
            if record.get("valid"):
                values[row["input"]] = record["value"]
        by_model[combo["id"]] = values
    common = set.intersection(*(set(values) for values in by_model.values()))
    if not common:
        return None
    return {
        "metric": name, "models": list(by_model), "inputs": sorted(common),
        "valid": len(common), "total": len(total_inputs), "coverage": len(common) / len(total_inputs),
        "median_changes": {identifier: median(values[source] for source in common)
                           for identifier, values in by_model.items()},
    }


def summarise_mode_comparisons(combos: list[dict]) -> dict:
    """Common sets are specific to each metric and each model comparison."""
    deblur = [combo for combo in combos if list(combo["order"]) == ["deblur"]]
    result = {"global": [], "pairwise": [], "global_unavailable": []}
    if len(deblur) < 2:
        return result
    for name in BLUR_METRICS:
        common = _common_comparison(deblur, name)
        if common is None:
            result["global_unavailable"].append(name)
        else:
            result["global"].append(common)
        for pair in combinations(deblur, 2):
            common = _common_comparison(pair, name)
            if common is not None:
                result["pairwise"].append(common)
    return result
