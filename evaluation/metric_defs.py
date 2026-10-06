"""Shared no-reference blur metric names and change types, without runtime dependencies."""

BLUR_METRICS = (
    "laplacian_variance",
    "tenengrad",
    "cpbd",
    "crete_roffet_blur",
)

RATIO_METRICS = frozenset({"laplacian_variance", "tenengrad"})
