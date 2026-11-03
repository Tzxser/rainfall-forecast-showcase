"""Memory-bounded full-field scoring for streamed real-data anchors.

The accumulator stores a 5x5 observed/forecast contingency table per lead.
That table is sufficient to derive both exclusive-bin and cumulative-threshold
scores without retaining full prediction fields or re-reading the data.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping, Sequence

import numpy as np

from precip_forecast.inference.quantize import OUTSIDE_VALUE, quantize_mm
from precip_forecast.metrics.official import MetricConfig, score_from_weighted_metrics


def default_scoring_configs() -> dict[str, MetricConfig]:
    """Return the V2 main configuration and two required sensitivities."""

    return {
        "main_exclusive": MetricConfig(
            event_definition="exclusive_bins",
            aggregation="global",
            empty_event="skip_renormalize",
            bias_clip=True,
            component_rounding="documented_3dp",
        ),
        "sensitivity_cumulative": MetricConfig(
            event_definition="cumulative_thresholds",
            aggregation="global",
            empty_event="skip_renormalize",
            bias_clip=True,
            component_rounding="documented_3dp",
        ),
        "sensitivity_no_bias_clip": MetricConfig(
            event_definition="exclusive_bins",
            aggregation="global",
            empty_event="skip_renormalize",
            bias_clip=False,
            component_rounding="documented_3dp",
        ),
    }


def _as_batch(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 3:
        array = array[None, ...]
    if array.ndim != 4 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (3,H,W) or (N,3,H,W)")
    return array


def _normalise_mask(mask: np.ndarray | None, shape: tuple[int, ...]) -> np.ndarray:
    if mask is None:
        return np.ones(shape, dtype=bool)
    value = np.asarray(mask, dtype=bool)
    if value.shape == (shape[0], shape[2], shape[3]):
        value = value[:, None]
    try:
        return np.broadcast_to(value, shape)
    except ValueError as exc:
        raise ValueError(f"valid_mask shape {value.shape} is not broadcastable to {shape}") from exc


def _to_mm(values: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
    array = _as_batch(values, name)
    if array.dtype == np.uint16:
        outside = array == OUTSIDE_VALUE
        invalid_reserved = (array > 2000) & ~outside
        if np.any(invalid_reserved):
            raise ValueError(f"{name} contains reserved uint16 values")
        mm = array.astype(np.float32) / 10.0
        mm[outside] = np.nan
        return mm, outside
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must be numeric")
    return array.astype(np.float32, copy=False), np.zeros(array.shape, dtype=bool)


def _score_counts(counts: Mapping[str, int], bias_clip: bool) -> tuple[float, float, float]:
    h, f, m, z = (int(counts[key]) for key in ("H", "F", "M", "Z"))
    total = h + f + m + z
    acc = (h + z) / total if total else np.nan
    denominator = h + f + m
    ts = h / denominator if denominator else np.nan
    observed = h + m
    forecast = h + f
    if observed == 0 and forecast == 0:
        bias = np.nan
    elif observed == 0 or forecast == 0:
        bias = 0.0
    else:
        bias = 1.0 - abs(forecast / observed - 1.0)
        if bias_clip:
            bias = float(np.clip(bias, 0.0, 1.0))
    return float(acc), float(ts), float(bias)


def _weighted(values: Sequence[float], config: MetricConfig) -> float:
    array = np.asarray(values, dtype=np.float64)
    weights = np.asarray(config.lead_weights, dtype=np.float64)
    defined = np.isfinite(array)
    if config.empty_event == "zero_keep_weight":
        return float(np.dot(np.nan_to_num(array, nan=0.0), weights))
    if not np.any(defined):
        return np.nan
    selected = weights[defined]
    return float(np.dot(array[defined], selected / selected.sum()))


def _binary_counts(
    contingency: np.ndarray,
    truth_event: np.ndarray,
    prediction_event: np.ndarray,
) -> dict[str, int]:
    truth = np.asarray(truth_event, dtype=bool)
    prediction = np.asarray(prediction_event, dtype=bool)
    return {
        "H": int(contingency[np.ix_(truth, prediction)].sum()),
        "F": int(contingency[np.ix_(~truth, prediction)].sum()),
        "M": int(contingency[np.ix_(truth, ~prediction)].sum()),
        "Z": int(contingency[np.ix_(~truth, ~prediction)].sum()),
    }


class FullFieldAccumulator:
    """Accumulate exact category contingencies from streamed full fields."""

    def __init__(
        self,
        *,
        thresholds_mm: Sequence[float] = (0.1, 5.0, 20.0, 50.0),
        representation: str = "float_mm",
    ) -> None:
        thresholds = np.asarray(tuple(thresholds_mm), dtype=np.float32)
        if thresholds.shape != (4,) or not np.all(np.isfinite(thresholds)):
            raise ValueError("thresholds_mm must contain four finite values")
        if not np.all(thresholds[1:] > thresholds[:-1]):
            raise ValueError("thresholds_mm must be strictly increasing")
        self.thresholds_mm = thresholds
        self.representation = str(representation)
        self.contingency = np.zeros((3, 5, 5), dtype=np.int64)
        self.sample_count = 0
        self.valid_pixels_by_lead = np.zeros(3, dtype=np.int64)

    def update(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
        valid_mask: np.ndarray | None = None,
    ) -> None:
        pred, pred_outside = _to_mm(prediction, "prediction")
        truth, _ = _to_mm(target, "target")
        if pred.shape != truth.shape:
            raise ValueError(f"prediction and target shapes differ: {pred.shape} != {truth.shape}")
        region = _normalise_mask(valid_mask, pred.shape)
        valid = region & np.isfinite(truth)
        invalid_prediction = valid & (
            ~np.isfinite(pred) | pred_outside | (pred < 0.0) | (pred > 200.0)
        )
        if np.any(invalid_prediction):
            raise ValueError(
                f"prediction contains {int(np.count_nonzero(invalid_prediction))} "
                "invalid values inside the evaluation region"
            )

        pred_class = np.searchsorted(self.thresholds_mm, pred, side="right")
        truth_class = np.searchsorted(self.thresholds_mm, truth, side="right")
        for lead in range(3):
            lead_valid = valid[:, lead]
            self.valid_pixels_by_lead[lead] += int(np.count_nonzero(lead_valid))
            encoded = (
                truth_class[:, lead][lead_valid].astype(np.int64) * 5
                + pred_class[:, lead][lead_valid].astype(np.int64)
            )
            self.contingency[lead] += np.bincount(encoded, minlength=25).reshape(5, 5)
        self.sample_count += int(pred.shape[0])

    def result(self, config: MetricConfig, *, config_name: str) -> dict[str, Any]:
        config.validate()
        if config.aggregation != "global":
            raise ValueError(
                "streamed contingency evaluation supports global aggregation only; "
                "samplewise metrics require a separate per-sample accumulator"
            )
        if not np.array_equal(
            self.thresholds_mm, np.asarray(config.thresholds_mm, dtype=np.float32)
        ):
            raise ValueError("accumulator and MetricConfig thresholds differ")

        leads: list[dict[str, Any]] = []
        acc_values: list[float] = []
        ts_values: list[list[float]] = [[] for _ in range(4)]
        bias_values: list[list[float]] = [[] for _ in range(4)]
        codes = np.arange(5)
        for lead in range(3):
            matrix = self.contingency[lead]
            rain_counts = _binary_counts(matrix, codes >= 1, codes >= 1)
            rain_acc, _, _ = _score_counts(rain_counts, config.bias_clip)
            acc_values.append(rain_acc)
            categories = []
            for category in range(4):
                if config.event_definition == "cumulative_thresholds":
                    event = codes >= category + 1
                else:
                    event = codes == category + 1
                counts = _binary_counts(matrix, event, event)
                _, ts, bias = _score_counts(counts, config.bias_clip)
                categories.append(
                    {
                        **counts,
                        "TS": ts,
                        "BIAS_SCORE": bias,
                        "empty_ts": not np.isfinite(ts),
                        "empty_bias": not np.isfinite(bias),
                    }
                )
                ts_values[category].append(ts)
                bias_values[category].append(bias)
            leads.append(
                {
                    "lead_hour": lead + 4,
                    "valid_pixels": int(self.valid_pixels_by_lead[lead]),
                    "ACC": rain_acc,
                    "rain_counts": rain_counts,
                    "categories": categories,
                }
            )

        weighted_acc = _weighted(acc_values, config)
        weighted_ts = [_weighted(values, config) for values in ts_values]
        weighted_bias = [_weighted(values, config) for values in bias_values]
        proxy = score_from_weighted_metrics(weighted_acc, weighted_ts, weighted_bias)
        return {
            "config_name": config_name,
            "config": asdict(config),
            "representation": self.representation,
            "sample_count": self.sample_count,
            "valid_pixels": int(self.valid_pixels_by_lead.sum()),
            "leads": leads,
            "weighted": {
                "ACC": weighted_acc,
                "TS": weighted_ts,
                "BIAS_SCORE": weighted_bias,
            },
            "proxy_points": proxy["proxy_points"],
            "official_relative_score": None,
            "note": "proxy_points is not the official final score; no Vmax was supplied",
        }


def quantize_evaluation_pair(
    prediction_mm: np.ndarray,
    target_mm: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Quantise prediction and target under the exact same explicit mask."""

    prediction = np.asarray(prediction_mm)
    target = np.asarray(target_mm)
    valid = np.asarray(valid_mask, dtype=bool)
    if prediction.shape != target.shape:
        raise ValueError("prediction_mm and target_mm must have identical shapes")
    if prediction.ndim == 4 and valid.shape == (
        prediction.shape[0],
        prediction.shape[2],
        prediction.shape[3],
    ):
        valid = valid[:, None]
    try:
        valid = np.broadcast_to(valid, prediction.shape)
    except ValueError as exc:
        raise ValueError("valid_mask is not broadcastable to prediction_mm") from exc
    if not np.all(np.isfinite(prediction)):
        raise ValueError("prediction_mm must be finite before quantisation")
    target_safe = np.where(np.isfinite(target), target, 0.0)
    return quantize_mm(prediction, valid), quantize_mm(target_safe, valid)


def evaluate_float_and_quantized(
    prediction_mm: np.ndarray,
    target_mm: np.ndarray,
    valid_mask: np.ndarray,
    *,
    configs: Mapping[str, MetricConfig] | None = None,
) -> dict[str, dict[str, Any]]:
    """Convenience interface for one in-memory full-field batch."""

    scoring_configs = dict(configs or default_scoring_configs())
    float_accumulator = FullFieldAccumulator(representation="float_mm")
    float_accumulator.update(prediction_mm, target_mm, valid_mask)
    quantized_prediction, quantized_target = quantize_evaluation_pair(
        prediction_mm, target_mm, valid_mask
    )
    quantized_accumulator = FullFieldAccumulator(representation="uint16_tenths_mm")
    quantized_accumulator.update(quantized_prediction, quantized_target, valid_mask)
    return {
        "float_mm": {
            name: float_accumulator.result(config, config_name=name)
            for name, config in scoring_configs.items()
        },
        "uint16_tenths_mm": {
            name: quantized_accumulator.result(config, config_name=name)
            for name, config in scoring_configs.items()
        },
    }
