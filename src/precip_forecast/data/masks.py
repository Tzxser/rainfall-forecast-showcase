"""Mask-aware utilities."""

from __future__ import annotations

import numpy as np


def combine_valid_masks(*masks: np.ndarray) -> np.ndarray:
    if not masks:
        raise ValueError("at least one mask is required")
    result = np.asarray(masks[0], dtype=bool).copy()
    for mask in masks[1:]:
        result &= np.asarray(mask, dtype=bool)
    return result


def apply_missing_mask(values: np.ndarray, valid_mask: np.ndarray, fill: float = 0.0) -> np.ndarray:
    array = np.asarray(values)
    mask = np.broadcast_to(np.asarray(valid_mask, dtype=bool), array.shape)
    return np.where(mask & np.isfinite(array), array, fill)
