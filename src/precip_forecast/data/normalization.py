"""Mask-aware normalisation with serialisable statistics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NormalizationStats:
    mean: float
    std: float
    count: int


def fit_stats(values: np.ndarray, valid_mask: np.ndarray | None = None) -> NormalizationStats:
    array = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(array)
    if valid_mask is not None:
        valid &= np.broadcast_to(np.asarray(valid_mask, dtype=bool), array.shape)
    selected = array[valid]
    if selected.size == 0:
        raise ValueError("cannot fit normalisation without valid values")
    std = float(selected.std())
    return NormalizationStats(float(selected.mean()), max(std, 1e-6), int(selected.size))


def normalize(values: np.ndarray, stats: NormalizationStats, fill: float = 0.0) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    result = (array - stats.mean) / stats.std
    return np.where(np.isfinite(result), result, fill).astype(np.float32)
