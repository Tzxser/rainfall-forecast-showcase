"""Deterministic competition output quantisation."""

from __future__ import annotations

import numpy as np


OUTSIDE_VALUE = np.uint16(65535)
MAX_MM = 200.0


def quantize_mm(
    precipitation_mm: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Quantise millimetres using decimal half-up rounding after clipping to [0, 200]."""

    value = np.asarray(precipitation_mm)
    if not np.issubdtype(value.dtype, np.number):
        raise TypeError("precipitation must be numeric")
    value = value.astype(np.float64, copy=False)
    if not np.all(np.isfinite(value)):
        raise ValueError("precipitation contains NaN or Inf")
    if valid_mask is None:
        valid = np.ones(value.shape, dtype=bool)
    else:
        try:
            valid = np.broadcast_to(np.asarray(valid_mask, dtype=bool), value.shape)
        except ValueError as exc:
            raise ValueError("valid_mask is not broadcastable to precipitation") from exc
    clipped = np.clip(value, 0.0, MAX_MM)
    quantized = np.floor(clipped * 10.0 + 0.5).astype(np.uint16)
    return np.where(valid, quantized, OUTSIDE_VALUE).astype(np.uint16, copy=False)


def dequantize_uint16(
    values: np.ndarray,
    outside_as_nan: bool = True,
) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype != np.uint16:
        raise TypeError("values must have dtype uint16")
    invalid_land = (array > 2000) & (array != OUTSIDE_VALUE)
    if np.any(invalid_land):
        raise ValueError("uint16 values contain reserved codes other than 65535")
    result = array.astype(np.float32) / 10.0
    if outside_as_nan:
        result[array == OUTSIDE_VALUE] = np.nan
    return result
