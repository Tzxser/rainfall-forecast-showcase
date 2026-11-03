"""Deterministic no-learning baselines for real precipitation histories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


LEAD_HOURS = (4, 5, 6)


@dataclass(frozen=True)
class BaselinePrediction:
    """A finite prediction paired with an explicit availability mask.

    Invalid input locations are represented by ``valid_mask=False`` rather than
    by silently treating their precipitation as zero.  ``values_mm`` remains
    finite everywhere so it can safely pass through deterministic quantisation.
    """

    baseline_id: str
    name: str
    values_mm: np.ndarray
    valid_mask: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.values_mm)
        valid = np.asarray(self.valid_mask, dtype=bool)
        if values.shape != valid.shape:
            raise ValueError("values_mm and valid_mask shapes differ")
        if values.ndim not in {3, 4} or values.shape[-3] != 3:
            raise ValueError("baseline output must have shape (3,H,W) or (N,3,H,W)")
        if not np.issubdtype(values.dtype, np.floating):
            raise TypeError("values_mm must use a floating dtype")
        if not np.all(np.isfinite(values)):
            raise ValueError("baseline values_mm must be finite everywhere")
        if np.any(values < 0.0) or np.any(values > 200.0):
            raise ValueError("baseline values_mm must be within [0, 200]")
        object.__setattr__(self, "values_mm", values.astype(np.float32, copy=False))
        object.__setattr__(self, "valid_mask", valid)


def _normalise_history(
    history_mm: np.ndarray,
    history_valid: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, bool]:
    history = np.asanyarray(history_mm)
    unbatched = history.ndim == 3
    if unbatched:
        history = history[None, ...]
    if history.ndim != 4 or history.shape[1] < 6:
        raise ValueError("history_mm must have shape (T,H,W) or (N,T,H,W), with T>=6")
    if not np.issubdtype(history.dtype, np.number):
        raise TypeError("history_mm must be numeric")

    masked = np.ma.asarray(history).astype(np.float32)
    masked_values = np.asarray(masked.filled(np.nan), dtype=np.float32)
    valid = ~np.ma.getmaskarray(masked) & np.isfinite(masked_values)
    valid &= masked_values >= 0.0
    if history_valid is not None:
        supplied = np.asarray(history_valid, dtype=bool)
        if unbatched and supplied.ndim == 3:
            supplied = supplied[None, ...]
        try:
            valid &= np.broadcast_to(supplied, history.shape)
        except ValueError as exc:
            raise ValueError("history_valid is not broadcastable to history_mm") from exc
    clean = np.where(valid, np.clip(masked_values, 0.0, 200.0), 0.0).astype(
        np.float32, copy=False
    )
    return clean, valid, unbatched


def _restore_batch(
    values: np.ndarray,
    valid: np.ndarray,
    unbatched: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if unbatched:
        return values[0], valid[0]
    return values, valid


def zero_prediction(
    spatial_valid: np.ndarray,
    *,
    batched: bool = False,
) -> BaselinePrediction:
    """B00: zero precipitation on an explicit full-field domain."""

    domain = np.asarray(spatial_valid, dtype=bool)
    if batched:
        if domain.ndim == 2:
            raise ValueError("batched zero baseline requires (N,H,W) spatial_valid")
        if domain.ndim != 3:
            raise ValueError("spatial_valid must have shape (N,H,W)")
        valid = np.repeat(domain[:, None], 3, axis=1)
    else:
        if domain.ndim != 2:
            raise ValueError("spatial_valid must have shape (H,W)")
        valid = np.repeat(domain[None], 3, axis=0)
    return BaselinePrediction("B00", "Zero", np.zeros(valid.shape, dtype=np.float32), valid)


def persistence_prediction(
    history_mm: np.ndarray,
    history_valid: np.ndarray | None = None,
) -> BaselinePrediction:
    """B01: copy PRE at t-1 to leads t+4, t+5 and t+6."""

    history, valid, unbatched = _normalise_history(history_mm, history_valid)
    values = np.repeat(history[:, -1:, :, :], 3, axis=1)
    output_valid = np.repeat(valid[:, -1:, :, :], 3, axis=1)
    values, output_valid = _restore_batch(values, output_valid, unbatched)
    return BaselinePrediction("B01", "Persistence", values, output_valid)


def history_mean_prediction(
    history_mm: np.ndarray,
    *,
    window_hours: int,
    history_valid: np.ndarray | None = None,
) -> BaselinePrediction:
    """B02: strict-valid mean of the last three or six hourly PRE fields."""

    if window_hours not in {3, 6}:
        raise ValueError("window_hours must be 3 or 6")
    history, valid, unbatched = _normalise_history(history_mm, history_valid)
    selected = history[:, -window_hours:]
    selected_valid = valid[:, -window_hours:]
    output_valid_2d = np.all(selected_valid, axis=1)
    mean = selected.mean(axis=1, dtype=np.float32)
    mean = np.where(output_valid_2d, mean, 0.0).astype(np.float32, copy=False)
    values = np.repeat(mean[:, None], 3, axis=1)
    output_valid = np.repeat(output_valid_2d[:, None], 3, axis=1)
    values, output_valid = _restore_batch(values, output_valid, unbatched)
    return BaselinePrediction(
        f"B02_mean{window_hours}",
        f"History mean ({window_hours}h)",
        values,
        output_valid,
    )


def bounded_log_trend_prediction(
    history_mm: np.ndarray,
    history_valid: np.ndarray | None = None,
    *,
    window_hours: int = 6,
    max_abs_log1p_slope_per_hour: float = 0.35,
) -> BaselinePrediction:
    """B03: bounded linear extrapolation in ``log1p(mm)`` space.

    Inputs end at t-1, so lead targets t+4/t+5/t+6 are respectively 5, 6
    and 7 hours beyond the final observed field.  Every fitted history point
    must be valid; missing values are never replaced and declared available.
    """

    if window_hours < 3 or window_hours > 6:
        raise ValueError("window_hours must be in [3, 6]")
    if not np.isfinite(max_abs_log1p_slope_per_hour) or max_abs_log1p_slope_per_hour <= 0:
        raise ValueError("max_abs_log1p_slope_per_hour must be finite and positive")
    history, valid, unbatched = _normalise_history(history_mm, history_valid)
    selected = history[:, -window_hours:]
    selected_valid = valid[:, -window_hours:]
    output_valid_2d = np.all(selected_valid, axis=1)

    x = np.arange(-(window_hours - 1), 1, dtype=np.float32)
    centered = x - x.mean(dtype=np.float32)
    denominator = float(np.sum(centered * centered))
    log_values = np.log1p(selected)
    slope = np.sum(log_values * centered.reshape(1, -1, 1, 1), axis=1) / denominator
    slope = np.clip(
        slope,
        -max_abs_log1p_slope_per_hour,
        max_abs_log1p_slope_per_hour,
    )
    intercept_at_mean = log_values.mean(axis=1, dtype=np.float32)
    future_offsets_from_last = np.asarray(
        [lead_hour + 1 for lead_hour in LEAD_HOURS], dtype=np.float32
    )
    future_x = future_offsets_from_last - float(x.mean())
    predicted_log = (
        intercept_at_mean[:, None]
        + slope[:, None] * future_x.reshape(1, 3, 1, 1)
    )
    predicted_log = np.clip(predicted_log, 0.0, np.log1p(200.0))
    values = np.expm1(predicted_log).astype(np.float32, copy=False)
    output_valid = np.repeat(output_valid_2d[:, None], 3, axis=1)
    values = np.where(output_valid, values, 0.0).astype(np.float32, copy=False)
    values, output_valid = _restore_batch(values, output_valid, unbatched)
    return BaselinePrediction("B03", "Bounded log1p trend", values, output_valid)


def build_observation_baselines(
    history_mm: np.ndarray,
    history_valid: np.ndarray | None,
    spatial_valid: np.ndarray,
    *,
    max_abs_log1p_slope_per_hour: float = 0.35,
) -> Iterator[BaselinePrediction]:
    """Yield B00, B01, both B02 variants and B03 without retaining all outputs."""

    history = np.asanyarray(history_mm)
    batched = history.ndim == 4
    zero_domain = np.asarray(spatial_valid, dtype=bool)
    if batched and zero_domain.ndim == 2:
        zero_domain = np.broadcast_to(zero_domain, (history.shape[0], *zero_domain.shape))
    yield zero_prediction(zero_domain, batched=batched)
    yield persistence_prediction(history_mm, history_valid)
    yield history_mean_prediction(history_mm, window_hours=3, history_valid=history_valid)
    yield history_mean_prediction(history_mm, window_hours=6, history_valid=history_valid)
    yield bounded_log_trend_prediction(
        history_mm,
        history_valid,
        window_hours=6,
        max_abs_log1p_slope_per_hour=max_abs_log1p_slope_per_hour,
    )
