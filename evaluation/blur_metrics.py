"""The four no-reference blur / sharpness metrics, each on a 2-D grayscale image (0-255 scale).

    laplacian_variance   higher = sharper    [0, inf)
    tenengrad            higher = sharper    [0, inf)
    cpbd_score           higher = sharper    [0, 1]
    crete_roffet_blur    higher = BLURRIER   [0, 1]

Vendored from the user-supplied metrics tool (2026-10-02). The algorithm
constants and kernels are retained; the tensor wrapper uses RGB, and missing
CPBD edges / non-finite Crete results are explicitly unmeasurable.
Source metrics.py SHA-256:
ccdc3be71793d2098225978150e5310f7079b5530ae0ce8894d6bfdbe1096a37
Source preprocessing.py SHA-256:
d7c643a61519d372edadabe97e75acce8c3607cd92a7cfc5ac6ada900a4cfbbc
"""
from __future__ import annotations

import math

import cv2
import numpy as np
import torch
from skimage.feature import canny
from skimage.measure import blur_effect


# The CPBD code below (cpbd_score and its helpers) is derived from the CPBD
# reference software, whose licence (also in LICENSE-CPBD.txt) requires this notice:
#
# Copyright (c) 2009-2010 Arizona Board of Regents.  All Rights Reserved.
#  Contact: Lina Karam (karam@asu.edu) and Niranjan Narvekar (nnarveka@asu.edu)
#  Image, Video, and Usabilty (IVU) Lab, http://ivulab.asu.edu , Arizona State University
#  This copyright statement may not be removed from any file containing it or from modifications to these files.
#  This copyright notice must also be included in any file or product that is derived from the source files.
#
#  Redistribution and use of this code in source and binary forms,  with or without modification, are permitted provided that the
#  following conditions are met:
#  - Redistribution's of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#  - Redistribution's in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer
# in the documentation and/or other materials provided with the distribution.
#  - The Image, Video, and Usability Laboratory (IVU Lab, http://ivulab.asu.edu) is acknowledged in any publication that
#  reports research results using this code, copies of this code, or modifications of this code.
#  The code and our papers are to be cited in the bibliography as:
#
# N. D. Narvekar and L. J. Karam, "CPBD Sharpness Metric Software", http://ivulab.asu.edu/Quality/CPBD
#
# N. D. Narvekar and L. J. Karam, "A No-Reference Image Blur Metric Based on the Cumulative
# Probability of Blur Detection (CPBD)," accepted and to appear in the IEEE Transactions on Image Processing,  2011.
#
# N. D. Narvekar and L. J. Karam, "An Improved No-Reference Sharpness Metric Based on the Probability of Blur Detection," International Workshop on Video Processing and Quality Metrics for Consumer Electronics (VPQM), January 2010, http://www.vpqm.org (pdf)
#
# N. D. Narvekar and L. J. Karam, "A No Reference Perceptual Quality Metric based on Cumulative Probability of Blur Detection," First International Workshop on the Quality of Multimedia Experience (QoMEX), pp. 87-91, July 2009.
#
#  DISCLAIMER:
#  This software is provided by the copyright holders and contributors "as is" and any express or implied warranties, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed. In no event shall the Arizona Board of Regents, Arizona State University, IVU Lab members, authors or contributors be liable for any direct, indirect, incidental, special, exemplary, or consequential damages (including, but not limited to, procurement of substitute
# goods or services; loss of use, data, or profits; or business interruption) however caused and on any theory of liability, whether in contract, strict liability, or tort (including negligence or otherwise) arising in any way out of the use of this software, even if advised of the possibility of such damage.

# CPBD (Narvekar & Karam 2011), parameters of the reference implementation.
CPBD_BLOCK = 64
CPBD_EDGE_BLOCK_RATIO = 0.002  # a block is an edge block above 0.2 % Canny pixels
CPBD_BETA = 3.6
CPBD_LOW_CONTRAST = 50  # block contrast <= 50 -> w_JNB = 5 px, otherwise 3 px
CPBD_MAX_MARGIN = 100  # each side of an edge is followed for at most 101 px
# P_blur is pooled in 101 buckets of round(100 * P_blur); buckets 0..63 are the
# edges whose blur stays below P_JNB = 1 - exp(-1) ~ 0.632.
CPBD_HIST_CUTOFF = 64

CRETE_ROFFET_KERNEL = 9  # length of the re-blurring mean filter


def laplacian_variance(gray: np.ndarray) -> float:
    """Variance of the Laplacian (OpenCV ``ksize=1`` kernel, CV_64F response).

    Parameters
    ----------
    gray
        2-D grayscale image on the 0-255 scale (uint8 or float).

    Returns
    -------
    float
        >= 0, unbounded. Higher values indicate a sharper image. Noise and
        texture raise it too, so use it as a fast filter, not on its own.
    """
    return float(cv2.Laplacian(_check_gray(gray), cv2.CV_64F, ksize=1).var())


def _sobel_energy(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return gx * gx + gy * gy


def tenengrad(gray: np.ndarray) -> float:
    """Tenengrad: mean of Gx^2 + Gy^2 over the image (3x3 Sobel, CV_64F).

    The mean rather than the classic sum, so the score does not grow with the
    pixel count; the sum is ``tenengrad * width * height`` (the debug column
    ``tenengrad_sum``).

    Parameters
    ----------
    gray
        2-D grayscale image on the 0-255 scale (uint8 or float).

    Returns
    -------
    float
        >= 0, unbounded. Higher values indicate a sharper image (stronger
        gradients). Low-texture scenes such as calm sea score low even in focus.
    """
    return float(np.mean(_sobel_energy(_check_gray(gray))))


def cpbd_score(gray: np.ndarray) -> float:
    """CPBD, the Cumulative Probability of Blur Detection (Narvekar & Karam 2011).

    64 x 64 blocks with more than 0.2 % Canny edge pixels are edge blocks. In
    them, every edge pixel's Marziliano width w is compared with the block's
    just-noticeable width w_JNB (5 px at contrast <= 50, else 3 px):
    P_blur = 1 - exp(-(w / w_JNB)^3.6). CPBD is the share of edges with
    P_blur <= P_JNB ~ 0.63, from the reference's 101-bucket histogram.

    Parameters
    ----------
    gray
        2-D grayscale image on the 0-255 scale (uint8 or float).

    Returns
    -------
    float
        In [0, 1]. Higher values indicate a sharper image. 0.0 when there are
        no measurable edges (also for an image smaller than one block), which
        may mean a textureless scene rather than blur; the debug columns
        ``cpbd_edge_count`` and ``cpbd_valid_blocks`` tell the two apart.
    """
    return _cpbd(_check_gray(gray))[0]


def _cpbd(gray: np.ndarray) -> tuple[float, dict]:
    """CPBD score plus its edge statistics, for a float64 gray image."""
    stats = {"cpbd_edge_count": 0, "cpbd_valid_blocks": 0, "cpbd_mean_edge_width": math.nan}
    n_rows, n_cols = gray.shape[0] // CPBD_BLOCK, gray.shape[1] // CPBD_BLOCK
    if n_rows == 0 or n_cols == 0:
        return 0.0, stats

    def blocks(a: np.ndarray) -> np.ndarray:  # (row block, y, col block, x); the remainder is dropped
        return a[: n_rows * CPBD_BLOCK, : n_cols * CPBD_BLOCK].reshape(n_rows, CPBD_BLOCK, n_cols, CPBD_BLOCK)

    # Canny (skimage defaults, as the reference) only picks the edge blocks;
    # the widths are measured on the thinned Sobel edges.
    edge_blocks = (
        np.count_nonzero(blocks(canny(gray)), axis=(1, 3)) > CPBD_BLOCK * CPBD_BLOCK * CPBD_EDGE_BLOCK_RATIO
    )
    gray_blocks = blocks(gray)
    contrast = (gray_blocks.max(axis=(1, 3)) - gray_blocks.min(axis=(1, 3))).astype(np.int64)  # truncated, as the reference
    w_jnb = np.where(contrast <= CPBD_LOW_CONTRAST, 5.0, 3.0)

    width_blocks = blocks(_marziliano_edge_widths(gray, _sobel_vertical_edges(gray)))
    counted = (width_blocks > 0) & edge_blocks[:, None, :, None]
    widths = width_blocks[counted]
    stats["cpbd_valid_blocks"] = int(edge_blocks.sum())
    stats["cpbd_edge_count"] = int(widths.size)
    if widths.size == 0:
        return 0.0, stats
    stats["cpbd_mean_edge_width"] = float(widths.mean())

    jnb = np.broadcast_to(w_jnb[:, None, :, None], width_blocks.shape)[counted]
    p_blur = 1.0 - np.exp(-((widths / jnb) ** CPBD_BETA))
    buckets = np.rint(p_blur * 100.0).astype(np.int64)  # rint rounds half to even, like Python's round()
    hist = np.bincount(buckets, minlength=101) / widths.size
    return float(hist[:CPBD_HIST_CUTOFF].sum()), stats


def _sobel_vertical_edges(gray: np.ndarray) -> np.ndarray:
    """Edges whose widths CPBD measures: Octave's ``edge(I, "sobel", "vertical")``, as in the reference.

    Squared horizontal Sobel response (kernel / 8), zeroed at or below
    2 * sqrt(mean), then thinned to pixels that beat both horizontal or both
    vertical neighbours.
    """
    strength = (cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT) / 8.0) ** 2
    strength[strength <= 2.0 * np.sqrt(strength.mean())] = 0.0
    padded = np.pad(strength, 1)  # zero border, like the reference
    centre = padded[1:-1, 1:-1]
    horizontal_peak = (centre > padded[1:-1, :-2]) & (centre > padded[1:-1, 2:])
    vertical_peak = (centre > padded[:-2, 1:-1]) & (centre > padded[2:, 1:-1])
    return horizontal_peak | vertical_peak


def _marziliano_edge_widths(gray: np.ndarray, edge_map: np.ndarray) -> np.ndarray:
    """Marziliano edge width (px) at every edge pixel, 0 elsewhere; same H x W as ``gray``.

    As in the CPBD reference, only edges whose gradient direction rounds to 0 or
    180 degrees are measured, i.e. along the row, which is the edge normal. From
    the edge pixel the profile is followed left and right for as long as the
    intensity keeps strictly falling / rising with the edge's own slope, so to
    the nearest local extremum on each side; each side counts 1 + its run
    (run capped at 100). Edge pixels on the image border are skipped.

    Two quirks of the reference are not copied: a pixel with gx == 0 is taken
    at its true angle (+-90, so skipped) rather than as 0, and an image whose
    gradient angles are all exactly 0 still gets widths.
    """
    gray = _check_gray(gray)
    if edge_map.shape != gray.shape:
        raise ValueError(f"edge_map shape {edge_map.shape} differs from image shape {gray.shape}")
    widths = np.zeros(gray.shape)
    rows, cols = np.nonzero(edge_map[1:-1, 1:-1])
    rows += 1
    cols += 1
    if rows.size == 0:
        return widths

    gx = gray[rows, cols + 1] - gray[rows, cols - 1]  # central differences, like np.gradient
    gy = gray[rows + 1, cols] - gray[rows - 1, cols]
    angle = 45.0 * np.round(np.degrees(np.arctan2(gy, gx)) / 45.0)

    step = np.diff(gray, axis=1)  # step[:, k] = gray[:, k + 1] - gray[:, k]
    for measured, slope in ((angle == 0, step > 0), (np.abs(angle) == 180, step < 0)):
        r, c = rows[measured], cols[measured]
        # Padded by one column per side, so run index k sits at k + 1 and the
        # steps beyond the image read 0.
        run_left = np.pad(_run_lengths(slope), ((0, 0), (1, 1)))  # same-slope steps ending at k
        run_right = np.pad(_run_lengths(slope[:, ::-1])[:, ::-1], ((0, 0), (1, 1)))  # starting at k
        left = np.minimum(run_left[r, c - 1], CPBD_MAX_MARGIN) + 1  # steps c-2, c-3, ...
        right = np.minimum(run_right[r, c + 2], CPBD_MAX_MARGIN) + 1  # steps c+1, c+2, ...
        widths[r, c] = left + right
    return widths


def _run_lengths(mask: np.ndarray) -> np.ndarray:
    """Per pixel, how many consecutive True values along the row end there."""
    total = np.cumsum(mask, axis=1, dtype=np.int32)
    return total - np.maximum.accumulate(np.where(mask, 0, total), axis=1)


def crete_roffet_blur(gray: np.ndarray) -> float:
    """Crété-Roffet perceptual blur metric (Crété-Roffet et al. 2007), via scikit-image.

    ``skimage.measure.blur_effect`` with a 9-pixel re-blurring filter: per axis,
    the image is re-blurred with a 9-pixel mean filter, and the metric is the
    share of the Sobel response that re-blurring does not remove,
    B = (M1 - M2) / M1, over the image minus a 2-pixel border. scikit-image
    compares Sobel responses, where the paper uses neighbour differences, so
    the values run somewhat higher than the paper's formula would give.

    Parameters
    ----------
    gray
        2-D grayscale image on the 0-255 scale (uint8 or float).

    Returns
    -------
    float
        max(B_ver, B_hor) in [0, 1]. Higher values indicate more blur. Large
        smooth areas (sky, calm sea) and one-directional texture also push it
        up. Non-finite results (for example, an empty inner area) remain
        non-finite so the measurement wrapper can mark them unmeasurable.
    """
    with np.errstate(divide="ignore", invalid="ignore"):  # 0 / 0 on such images
        score = float(blur_effect(_check_gray(gray), h_size=CRETE_ROFFET_KERNEL))
    return float(np.clip(score, 0.0, 1.0)) if math.isfinite(score) else score


MIN_SIDE = 3


def _check_size(shape: tuple[int, ...]) -> None:
    if shape[0] < MIN_SIDE or shape[1] < MIN_SIDE:
        raise ValueError(f"image is {shape[1]}x{shape[0]} px; at least {MIN_SIDE}x{MIN_SIDE} is needed")


def _check_gray(gray: np.ndarray) -> np.ndarray:
    """float64 version of a 2-D grayscale image on the 0-255 scale (no copy if already float64)."""
    if not isinstance(gray, np.ndarray):
        raise TypeError(f"expected a numpy.ndarray, got {type(gray).__name__}")
    if gray.ndim != 2:
        raise ValueError(f"expected a 2-D grayscale image, got shape {gray.shape}; convert it with to_gray()")
    _check_size(gray.shape)
    if gray.dtype.kind == "f" and not np.isfinite(gray).all():
        raise ValueError("image contains NaN or Inf")
    return gray.astype(np.float64, copy=False)



BLUR_METRICS = ("laplacian_variance", "tenengrad", "cpbd", "crete_roffet_blur")


def _measurement(value, *, status="valid", reason=None, debug=None) -> dict:
    return {
        "value": value,
        "valid": status == "valid",
        "status": status,
        "reason": reason,
        "debug": {} if debug is None else debug,
    }


def failed_measurements(reason: str) -> dict:
    """Keep every metric slot when an image cannot be measured."""
    return {name: _measurement(None, status="failed", reason=reason) for name in BLUR_METRICS}


def _tensor_gray(tensor: torch.Tensor) -> np.ndarray:
    """Apply the delivered PNG's quantization, then convert RGB to gray 0..255."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Expected an RGB torch.Tensor")
    if tensor.ndim != 4 or tuple(tensor.shape[:2]) != (1, 3):
        raise ValueError(f"Expected RGB BCHW input, got {tuple(tensor.shape)}")
    if not torch.isfinite(tensor).all():
        raise ValueError("Image contains NaN or Inf")
    pixels = tensor.detach().cpu().squeeze(0).clamp(0, 1).mul(255).round().to(torch.uint8)
    rgb = pixels.permute(1, 2, 0).contiguous().numpy()
    _check_size(rgb.shape)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float64)


def measure_tensor(tensor: torch.Tensor) -> dict:
    """Measure one decoded RGB image, isolating each metric's failure."""
    try:
        gray = _tensor_gray(tensor)
    except Exception as exc:
        return failed_measurements(f"preprocessing: {type(exc).__name__}: {exc}")

    measurements = {}
    for name in BLUR_METRICS:
        try:
            debug = {}
            if name == "laplacian_variance":
                value = laplacian_variance(gray)
            elif name == "tenengrad":
                value = tenengrad(gray)
            elif name == "cpbd":
                value, debug = _cpbd(gray)
                # JSON has no NaN; retain missing edge-width statistics as null.
                debug = {key: None if isinstance(item, float) and not math.isfinite(item) else item
                         for key, item in debug.items()}
                if debug["cpbd_edge_count"] == 0:
                    measurements[name] = _measurement(
                        value, status="unmeasurable", reason="CPBD 無可量測邊緣", debug=debug
                    )
                    continue
            else:
                value = crete_roffet_blur(gray)
            if not math.isfinite(value):
                measurements[name] = _measurement(
                    None, status="unmeasurable", reason="指標結果非有限值", debug=debug
                )
            else:
                measurements[name] = _measurement(float(value), debug=debug)
        except Exception as exc:
            measurements[name] = _measurement(
                None, status="failed", reason=f"{type(exc).__name__}: {exc}"
            )
    return measurements


def compare_measurements(before: dict, after: dict, *, applicable: bool = True) -> dict:
    """Compute each image's ratio or delta without substituting a zero baseline."""
    changes = {}
    for name in BLUR_METRICS:
        kind = "ratio" if name in ("laplacian_variance", "tenengrad") else "delta"
        change = {"value": None, "valid": False, "status": "unmeasurable",
                  "reason": None, "kind": kind}
        changes[name] = change
        if not applicable:
            change["reason"] = "跨尺寸不適用"
            continue
        original, output = before[name], after[name]
        reasons = []
        for label, measurement in (("before", original), ("after", output)):
            if not measurement["valid"]:
                reasons.append(f"{label}: {measurement.get('reason') or '指標無效'}")
            elif measurement["value"] is None or not math.isfinite(measurement["value"]):
                reasons.append(f"{label}: 指標結果非有限值")
            elif name == "cpbd" and measurement.get("debug", {}).get("cpbd_edge_count", 0) <= 0:
                reasons.append(f"{label}: CPBD 無可量測邊緣")
        if reasons:
            change["reason"] = "; ".join(reasons)
            if original["status"] == "failed" or output["status"] == "failed":
                change["status"] = "failed"
            continue
        if kind == "ratio" and original["value"] == 0:
            change["reason"] = "前值為 0，無法計算比值"
            continue
        value = output["value"] / original["value"] if kind == "ratio" else output["value"] - original["value"]
        if not math.isfinite(value):
            change["reason"] = "變化結果非有限值"
            continue
        change.update(value=float(value), valid=True, status="valid")
    return changes
