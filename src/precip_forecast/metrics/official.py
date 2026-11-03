"""Configurable implementation of the currently documented competition metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class MetricConfig:
    event_definition: str = "exclusive_bins"
    aggregation: str = "global"
    empty_event: str = "skip_renormalize"
    bias_clip: bool = True
    component_rounding: str = "documented_3dp"
    lead_weights: tuple[float, float, float] = (0.4, 0.3, 0.3)
    thresholds_mm: tuple[float, float, float, float] = (0.1, 5.0, 20.0, 50.0)

    def validate(self) -> None:
        if self.event_definition not in {"exclusive_bins", "cumulative_thresholds"}:
            raise ValueError(f"unsupported event_definition: {self.event_definition}")
        if self.aggregation not in {"global", "samplewise_mean"}:
            raise ValueError(f"unsupported aggregation: {self.aggregation}")
        if self.empty_event not in {"skip_renormalize", "zero_keep_weight"}:
            raise ValueError(f"unsupported empty_event: {self.empty_event}")
        if self.component_rounding not in {"documented_3dp", "full_precision"}:
            raise ValueError(f"unsupported component_rounding: {self.component_rounding}")
        if len(self.lead_weights) != 3 or not np.isclose(sum(self.lead_weights), 1.0):
            raise ValueError("lead_weights must contain three values summing to 1")
        if len(self.thresholds_mm) != 4:
            raise ValueError("exactly four thresholds are required")
        if not all(
            np.isfinite(lower) and np.isfinite(upper) and lower < upper
            for lower, upper in zip(self.thresholds_mm, self.thresholds_mm[1:])
        ):
            raise ValueError("thresholds_mm must be finite and strictly increasing")


def _as_batch(array: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(array)
    if value.ndim == 3:
        value = value[None, ...]
    if value.ndim != 4 or value.shape[1] != 3:
        raise ValueError(f"{name} must have shape (3,H,W) or (N,3,H,W)")
    return value


def _to_mm(array: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
    value = _as_batch(array, name)
    if value.dtype == np.uint16:
        missing = value == np.uint16(65535)
        mm = value.astype(np.float64) / 10.0
        mm[missing] = np.nan
        return mm, missing
    if not np.issubdtype(value.dtype, np.number):
        raise TypeError(f"{name} must be numeric")
    return value.astype(np.float64, copy=False), np.zeros(value.shape, dtype=bool)


def _normalise_mask(mask: np.ndarray | None, shape: tuple[int, ...]) -> np.ndarray:
    """Return a boolean mask broadcast to ``(N, 3, H, W)``.

    Accepted common forms include ``(H,W)``, ``(3,H,W)``, ``(N,H,W)``,
    ``(N,1,H,W)`` and ``(N,3,H,W)``.
    """

    if mask is None:
        return np.ones(shape, dtype=bool)
    value = np.asarray(mask, dtype=bool)
    if value.shape == (shape[0], shape[2], shape[3]):
        value = value[:, None, :, :]
    try:
        return np.broadcast_to(value, shape).copy()
    except ValueError as exc:
        raise ValueError(f"mask shape {value.shape} is not broadcastable to {shape}") from exc


def _event(array: np.ndarray, category: int, config: MetricConfig) -> np.ndarray:
    lower = config.thresholds_mm[category]
    if config.event_definition == "cumulative_thresholds" or category == 3:
        return array >= lower
    upper = config.thresholds_mm[category + 1]
    return (array >= lower) & (array < upper)


def _counts(pred_event: np.ndarray, true_event: np.ndarray, valid: np.ndarray) -> dict[str, int]:
    pred = pred_event & valid
    truth = true_event & valid
    return {
        "H": int(np.count_nonzero(pred & truth)),
        "F": int(np.count_nonzero(pred & ~truth & valid)),
        "M": int(np.count_nonzero(~pred & truth & valid)),
        "Z": int(np.count_nonzero(~pred & ~truth & valid)),
    }


def _metrics(counts: Mapping[str, int], bias_clip: bool) -> tuple[float, float, float]:
    h, f, m, z = (counts[key] for key in ("H", "F", "M", "Z"))
    total = h + f + m + z
    acc = (h + z) / total if total else np.nan
    ts_denom = h + f + m
    ts = h / ts_denom if ts_denom else np.nan

    observed = h + m
    forecast = h + f
    if observed == 0 and forecast == 0:
        bias = np.nan
    elif observed == 0 or forecast == 0:
        # The official document explicitly assigns zero to pure false-alarm
        # and pure-miss cases. Only both-empty is an undefined contribution.
        bias = 0.0
    else:
        bias = 1.0 - abs(forecast / observed - 1.0)
        if bias_clip:
            bias = float(np.clip(bias, 0.0, 1.0))
    return float(acc), float(ts), float(bias)


def _mean_defined(values: Sequence[float], mode: str) -> float:
    array = np.asarray(values, dtype=np.float64)
    if mode == "zero_keep_weight":
        return float(np.nan_to_num(array, nan=0.0).mean())
    defined = np.isfinite(array)
    return float(array[defined].mean()) if np.any(defined) else np.nan


def _lead_weighted(values: Sequence[float], config: MetricConfig) -> float:
    array = np.asarray(values, dtype=np.float64)
    weights = np.asarray(config.lead_weights, dtype=np.float64)
    defined = np.isfinite(array)
    if config.empty_event == "zero_keep_weight":
        return float(np.dot(np.nan_to_num(array, nan=0.0), weights))
    if not np.any(defined):
        return np.nan
    selected = weights[defined]
    return float(np.dot(array[defined], selected / selected.sum()))


def _aggregate_category(
    pred_event: np.ndarray,
    true_event: np.ndarray,
    valid: np.ndarray,
    config: MetricConfig,
) -> dict[str, Any]:
    total = _counts(pred_event, true_event, valid)
    if config.aggregation == "global":
        acc, ts, bias = _metrics(total, config.bias_clip)
    else:
        per_sample = [
            _metrics(_counts(pred_event[i], true_event[i], valid[i]), config.bias_clip)
            for i in range(pred_event.shape[0])
        ]
        acc = _mean_defined([item[0] for item in per_sample], config.empty_event)
        ts = _mean_defined([item[1] for item in per_sample], config.empty_event)
        bias = _mean_defined([item[2] for item in per_sample], config.empty_event)
    return {
        **total,
        "ACC": acc,
        "TS": ts,
        "BIAS_SCORE": bias,
        "empty_ts": not np.isfinite(ts),
        "empty_bias": not np.isfinite(bias),
    }


def _round_half_up_3dp(value: float) -> float:
    if not np.isfinite(value):
        return float(value)
    return float(Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


def relative_score(
    aggregates: Mapping[str, Any],
    vmax: Mapping[str, float],
    component_rounding: str = "documented_3dp",
) -> dict[str, Any]:
    """Compute the documented nine-component relative score for supplied Vmax values.

    ``documented_3dp`` follows the published worked example: round every component
    to three decimal places before summing, then retain three decimals for the total.
    ``full_precision`` is retained for comparison with a future official script.
    """

    if component_rounding not in {"documented_3dp", "full_precision"}:
        raise ValueError(f"unsupported component_rounding: {component_rounding}")
    keys = ["ACC", "TS1", "TS2", "TS3", "TS4", "BIAS1", "BIAS2", "BIAS3", "BIAS4"]
    points = {
        "ACC": 10.0,
        "TS1": 15.0,
        "TS2": 15.0,
        "TS3": 15.0,
        "TS4": 15.0,
        "BIAS1": 7.5,
        "BIAS2": 7.5,
        "BIAS3": 7.5,
        "BIAS4": 7.5,
    }
    missing = [key for key in keys if key not in vmax]
    if missing:
        raise ValueError(f"missing Vmax entries: {missing}")
    raw_components: dict[str, float] = {}
    for key in keys:
        maximum = float(vmax[key])
        value = float(aggregates[key])
        if not np.isfinite(maximum) or maximum <= 0:
            raise ValueError(f"Vmax[{key}] must be finite and positive")
        raw_components[key] = points[key] * value / maximum

    if component_rounding == "documented_3dp":
        components = {key: _round_half_up_3dp(value) for key, value in raw_components.items()}
        total = _round_half_up_3dp(sum(components.values()))
    else:
        components = dict(raw_components)
        total = float(sum(components.values()))
    return {
        "components": components,
        "raw_components": raw_components,
        "total": total,
        "component_rounding": component_rounding,
    }


def score_from_weighted_metrics(
    acc: float,
    ts: Sequence[float],
    bias: Sequence[float],
    vmax: Mapping[str, float] | None = None,
    component_rounding: str = "documented_3dp",
) -> dict[str, Any]:
    """Score already lead-weighted metrics, including the official document example."""

    if len(ts) != 4 or len(bias) != 4:
        raise ValueError("ts and bias must each contain four values")
    aggregates = {"ACC": float(acc)}
    aggregates.update({f"TS{i + 1}": float(value) for i, value in enumerate(ts)})
    aggregates.update({f"BIAS{i + 1}": float(value) for i, value in enumerate(bias)})
    proxy_points = (
        10.0 * aggregates["ACC"]
        + 15.0 * sum(aggregates[f"TS{i}"] for i in range(1, 5))
        + 7.5 * sum(aggregates[f"BIAS{i}"] for i in range(1, 5))
    )
    result: dict[str, Any] = {
        **aggregates,
        "proxy_points": float(proxy_points),
        "official_relative_score": None,
    }
    if vmax is not None:
        result["official_relative_score"] = relative_score(
            aggregates,
            vmax,
            component_rounding=component_rounding,
        )
    return result


def evaluate_precipitation(
    prediction: np.ndarray,
    target: np.ndarray,
    valid_mask: np.ndarray | None = None,
    config: MetricConfig | None = None,
    vmax: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Evaluate precipitation with all documented ambiguity switches.

    Float inputs are interpreted as millimetres. ``uint16`` inputs are interpreted as
    tenths of millimetres, with 65535 reserved for invalid/outside-domain pixels.
    """

    config = config or MetricConfig()
    config.validate()
    pred, pred_missing = _to_mm(prediction, "prediction")
    truth, _ = _to_mm(target, "target")
    if pred.shape != truth.shape:
        raise ValueError(f"prediction and target shapes differ: {pred.shape} != {truth.shape}")
    region = _normalise_mask(valid_mask, pred.shape)
    valid = region & np.isfinite(truth)
    invalid_prediction = valid & (~np.isfinite(pred) | pred_missing | (pred < 0.0) | (pred > 200.0))
    if np.any(invalid_prediction):
        count = int(np.count_nonzero(invalid_prediction))
        raise ValueError(f"prediction contains {count} invalid values inside the valid region")

    leads: list[dict[str, Any]] = []
    acc_by_lead: list[float] = []
    ts_by_category: list[list[float]] = [[] for _ in range(4)]
    bias_by_category: list[list[float]] = [[] for _ in range(4)]
    for lead in range(3):
        lead_pred = pred[:, lead]
        lead_truth = truth[:, lead]
        lead_valid = valid[:, lead]
        rain_result = _aggregate_category(
            lead_pred >= config.thresholds_mm[0],
            lead_truth >= config.thresholds_mm[0],
            lead_valid,
            config,
        )
        rain_counts = {key: rain_result[key] for key in ("H", "F", "M", "Z")}
        rain_acc = rain_result["ACC"]
        acc_by_lead.append(rain_acc)
        categories = []
        for category in range(4):
            result = _aggregate_category(
                _event(lead_pred, category, config),
                _event(lead_truth, category, config),
                lead_valid,
                config,
            )
            categories.append(result)
            ts_by_category[category].append(result["TS"])
            bias_by_category[category].append(result["BIAS_SCORE"])
        leads.append(
            {
                "lead_hour": lead + 4,
                "valid_pixels": int(np.count_nonzero(lead_valid)),
                "ACC": rain_acc,
                "rain_counts": rain_counts,
                "categories": categories,
            }
        )

    weighted_acc = _lead_weighted(acc_by_lead, config)
    weighted_ts = [_lead_weighted(values, config) for values in ts_by_category]
    weighted_bias = [_lead_weighted(values, config) for values in bias_by_category]
    scoring = score_from_weighted_metrics(
        weighted_acc,
        weighted_ts,
        weighted_bias,
        vmax=vmax,
        component_rounding=config.component_rounding,
    )
    return {
        "config": asdict(config),
        "input_shape": list(pred.shape),
        "valid_pixels": int(np.count_nonzero(valid)),
        "leads": leads,
        "weighted": {
            "ACC": weighted_acc,
            "TS": weighted_ts,
            "BIAS_SCORE": weighted_bias,
        },
        "proxy_points": scoring["proxy_points"],
        "official_relative_score": scoring["official_relative_score"],
        "note": (
            "proxy_points is not the official final score"
            if vmax is None
            else "relative score computed only from the caller-supplied Vmax values"
        ),
    }
