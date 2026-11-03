"""Independent Task50 competition-style metric facade.

This module deliberately exposes raw contingency counts and quantized scores.
It never labels the local proxy as the official leaderboard score because the
cross-team Vmax constants are unavailable.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from precip_forecast.evaluation.full_field import (
    FullFieldAccumulator,
    default_scoring_configs,
    quantize_evaluation_pair,
)


def competition_metrics(
    prediction_mm: np.ndarray,
    target_mm: np.ndarray,
    valid_mask: np.ndarray,
) -> dict[str, Any]:
    """Score 4/5/6-hour fields after official tenth-mm quantization."""

    prediction = np.asarray(prediction_mm)
    target = np.asarray(target_mm)
    valid = np.asarray(valid_mask, dtype=bool)
    if prediction.shape != target.shape or prediction.shape != valid.shape:
        raise ValueError("prediction, target and valid mask shapes differ")
    if prediction.ndim not in {3, 4} or prediction.shape[-3] != 3:
        raise ValueError("competition fields must have lead shape (...,3,H,W)")
    quantized_prediction, quantized_target = quantize_evaluation_pair(
        prediction, target, valid
    )
    accumulator = FullFieldAccumulator(representation="uint16_tenths_mm")
    accumulator.update(quantized_prediction, quantized_target, valid)
    result = accumulator.result(
        default_scoring_configs()["main_exclusive"],
        config_name="main_exclusive",
    )
    result["official_final_score"] = None
    result["quantized_before_scoring"] = True
    return result


__all__ = ["competition_metrics"]
