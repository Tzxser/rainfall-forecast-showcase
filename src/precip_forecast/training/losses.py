"""Composable V3 precipitation losses with fail-closed class statistics.

The ordinal branch deliberately treats positive and negative examples
asymmetrically: fold-derived weights are applied only to positive events, while
easy negatives receive the stronger focal suppression.  This avoids the V2
failure mode where a rare-class multiplier also amplified millions of dry
pixels.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import Tensor
from torch.nn import functional as F


V3_POSITIVE_WEIGHT_CAPS = (8.0, 16.0, 32.0, 64.0)
V3_REGRESSION_BIN_WEIGHTS = (0.05, 1.0, 2.5, 6.0, 12.0)


def _at_least_float32(value: Tensor) -> Tensor:
    """Promote half precision for loss math without narrowing float64 callers."""

    return (
        value.float()
        if value.dtype in (torch.float16, torch.bfloat16)
        else value
    )


@dataclass(frozen=True)
class LossConfig:
    """Fixed Loss V3 hyperparameters.

    ``class_weights`` is retained as a compatibility spelling for V2 callers,
    but its semantics are now positive-only.  New code should pass
    ``positive_class_weights`` explicitly.  ``focal_gamma`` similarly remains
    a positive-gamma compatibility override; the negative gamma is always
    validated to be larger.
    """

    ordinal_weight: float = 0.35
    regression_weight: float = 0.20
    soft_ts_weight: float = 0.25
    area_weight: float = 0.15
    multiscale_weight: float = 0.05
    positive_focal_gamma: float = 0.5
    negative_focal_gamma: float = 3.0
    focal_gamma: float | None = None
    lead_weights: tuple[float, float, float] = (0.4, 0.3, 0.3)
    thresholds_mm: tuple[float, float, float, float] = (0.1, 5.0, 20.0, 50.0)
    threshold_weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    class_weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    positive_class_weights: tuple[float, float, float, float] | None = None
    cumulative_priors: tuple[float, float, float, float] | None = None
    regression_bin_weights: tuple[float, float, float, float, float] = (
        V3_REGRESSION_BIN_WEIGHTS
    )
    dry_false_rain_weight: float = 0.02
    huber_beta: float = 1.0
    epsilon: float = 1.0e-6

    def __post_init__(self) -> None:
        scalar_weights = (
            self.ordinal_weight,
            self.regression_weight,
            self.soft_ts_weight,
            self.area_weight,
            self.multiscale_weight,
        )
        if any(not math.isfinite(value) or value < 0 for value in scalar_weights):
            raise ValueError("loss component weights must be finite and non-negative")
        if not math.isclose(sum(scalar_weights), 1.0, rel_tol=0.0, abs_tol=1.0e-8):
            raise ValueError("Loss V3 component weights must sum to one")
        positive_gamma = self.effective_positive_focal_gamma
        if (
            not math.isfinite(positive_gamma)
            or positive_gamma < 0
            or not math.isfinite(self.negative_focal_gamma)
            or self.negative_focal_gamma <= positive_gamma
        ):
            raise ValueError("negative focal gamma must be finite and greater than positive gamma")
        if len(self.lead_weights) != 3 or len(self.thresholds_mm) != 4:
            raise ValueError("Loss V3 requires three leads and four thresholds")
        if len(self.threshold_weights) != 4 or len(self.effective_positive_weights) != 4:
            raise ValueError("ordinal weights must contain four values")
        if self.cumulative_priors is not None:
            if len(self.cumulative_priors) != 4 or any(
                not math.isfinite(value) or not 0.0 <= value <= 1.0
                for value in self.cumulative_priors
            ):
                raise ValueError("cumulative_priors must contain four probabilities")
            if any(
                right > left
                for left, right in zip(
                    self.cumulative_priors, self.cumulative_priors[1:]
                )
            ):
                raise ValueError("cumulative_priors must be non-increasing")
        if len(self.regression_bin_weights) != 5:
            raise ValueError("regression_bin_weights must contain five values")
        vectors = (
            self.lead_weights,
            self.threshold_weights,
            self.effective_positive_weights,
            self.regression_bin_weights,
        )
        if any(
            any(not math.isfinite(value) or value <= 0 for value in vector)
            for vector in vectors
        ):
            raise ValueError("lead, threshold, class, and regression weights must be positive")
        if tuple(self.thresholds_mm) != tuple(sorted(self.thresholds_mm)) or any(
            not math.isfinite(value) or value < 0 for value in self.thresholds_mm
        ):
            raise ValueError("thresholds must be finite, non-negative, and sorted")
        if not math.isfinite(self.dry_false_rain_weight) or self.dry_false_rain_weight < 0:
            raise ValueError("dry false-rain weight must be finite and non-negative")
        if not math.isfinite(self.huber_beta) or self.huber_beta <= 0:
            raise ValueError("huber_beta must be finite and positive")
        if not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")

    @property
    def effective_positive_focal_gamma(self) -> float:
        return (
            float(self.positive_focal_gamma)
            if self.focal_gamma is None
            else float(self.focal_gamma)
        )

    @property
    def effective_positive_weights(self) -> tuple[float, float, float, float]:
        return self.class_weights if self.positive_class_weights is None else self.positive_class_weights

    @property
    def expected_ordinal_normalizers(self) -> tuple[float, float, float, float]:
        """Fold-expected class multipliers, independent of patch composition."""

        if self.cumulative_priors is None:
            return (1.0, 1.0, 1.0, 1.0)
        return tuple(
            (1.0 - prior) + prior * positive_weight
            for prior, positive_weight in zip(
                self.cumulative_priors, self.effective_positive_weights
            )
        )


def derive_asymmetric_ordinal_statistics(
    positive_counts: Sequence[int],
    valid_count: int,
    *,
    smoothing_count: float = 100.0,
    caps: Sequence[float] = V3_POSITIVE_WEIGHT_CAPS,
) -> dict[str, object]:
    """Derive fold-train-only positive weights and cumulative priors.

    ``valid_count`` is the number of valid lead/pixel targets used for every
    cumulative threshold.  Validation data must never be supplied here.
    Absent classes remain explicit and are marked in the returned evidence;
    smoothing keeps their fixed formula finite instead of silently dropping
    them.
    """

    positives = tuple(int(value) for value in positive_counts)
    cap_values = tuple(float(value) for value in caps)
    if len(positives) != 4 or len(cap_values) != 4:
        raise ValueError("four cumulative positive counts and caps are required")
    if valid_count <= 0 or not math.isfinite(smoothing_count) or smoothing_count <= 0:
        raise ValueError("valid_count and smoothing_count must be positive")
    if any(value < 0 or value > valid_count for value in positives):
        raise ValueError("positive counts must be within valid_count")
    if any(right > left for left, right in zip(positives, positives[1:])):
        raise ValueError("cumulative positive counts must be non-increasing")
    if any(not math.isfinite(value) or value < 1.0 for value in cap_values):
        raise ValueError("positive-weight caps must be finite and at least one")

    negatives = tuple(valid_count - value for value in positives)
    weights = tuple(
        min(
            cap,
            math.sqrt(
                (negative + smoothing_count) / (positive + smoothing_count)
            ),
        )
        for positive, negative, cap in zip(positives, negatives, cap_values)
    )
    priors = tuple(
        (positive + smoothing_count) / (valid_count + 2.0 * smoothing_count)
        for positive in positives
    )
    return {
        "method": "fold_train_only_asymmetric_sqrt_v3",
        "positive_counts": positives,
        "negative_counts": negatives,
        "valid_count": int(valid_count),
        "smoothing_count": float(smoothing_count),
        "positive_weight_caps": cap_values,
        "positive_weights": weights,
        "cumulative_priors": priors,
        "absent_classes": tuple(value == 0 for value in positives),
        "validation_data_used": False,
    }


def cumulative_targets(target_mm: Tensor, thresholds: tuple[float, ...]) -> Tensor:
    return torch.stack(
        [(target_mm >= threshold) for threshold in thresholds], dim=2
    ).to(target_mm.dtype)


def asymmetric_ordinal_terms(
    logits: Tensor,
    target: Tensor,
    config: LossConfig,
) -> tuple[Tensor, Tensor]:
    """Return elementwise asymmetric focal loss and static normalisers.

    The focal modulation is evaluated from logits in log space.  Computing it
    as ``(1 - sigmoid(logits)) ** gamma`` is mathematically equivalent but is
    not numerically safe for ``gamma < 1``: once sigmoid rounds to exactly one,
    the forward value is finite while the derivative of ``0 ** gamma`` is
    infinite.  The resulting ``0 * inf`` produced NaN gradients in the real
    BF16 stability run even though its loss stayed finite.
    """

    if logits.shape != target.shape or logits.ndim != 5 or logits.shape[2] != 4:
        raise ValueError("ordinal logits and targets must have shape (B,3,4,H,W)")
    # Loss V3 is an explicit FP32 island for half-precision model outputs.  The
    # cast remains differentiable back to the original logits.  Disabling an
    # enclosing autocast context makes that precision boundary explicit.
    with torch.autocast(device_type=logits.device.type, enabled=False):
        work_logits = _at_least_float32(logits)
        work_target = target.to(dtype=work_logits.dtype)
        positive = work_target > 0.5
        positive_loss = F.softplus(-work_logits)
        negative_loss = F.softplus(work_logits)
        positive_gamma = config.effective_positive_focal_gamma
        positive_modulation = (
            torch.ones_like(work_logits)
            if positive_gamma == 0.0
            else torch.exp(positive_gamma * F.logsigmoid(-work_logits))
        )
        negative_modulation = torch.exp(
            config.negative_focal_gamma * F.logsigmoid(work_logits)
        )
        positive_weights = work_logits.new_tensor(
            config.effective_positive_weights
        ).view(1, 1, 4, 1, 1)
        loss = torch.where(
            positive,
            positive_weights * positive_modulation * positive_loss,
            negative_modulation * negative_loss,
        )
        # The normaliser is fixed from fold-training priors, not the current
        # event-centred patch labels.  Positive weights therefore remain
        # effective in rare-event patches while the expected whole-fold loss
        # scale is stable.
        normalizer = work_logits.new_tensor(
            config.expected_ordinal_normalizers
        ).view(1, 1, 4, 1, 1)
    return loss, normalizer


def ordinal_bce(
    logits: Tensor,
    target: Tensor,
    valid_mask: Tensor,
    config: LossConfig,
) -> Tensor:
    loss, expected_weight = asymmetric_ordinal_terms(logits, target, config)
    lead = loss.new_tensor(config.lead_weights).view(1, 3, 1, 1, 1)
    threshold = loss.new_tensor(config.threshold_weights).view(1, 1, 4, 1, 1)
    mask = valid_mask[:, :, None].to(loss.dtype)
    weighted_mask = lead * threshold * mask
    denominator = (expected_weight * weighted_mask).sum().clamp_min(config.epsilon)
    return (loss * weighted_mask).sum() / denominator


def soft_ts_loss(
    probabilities: Tensor,
    target: Tensor,
    valid_mask: Tensor,
    config: LossConfig,
) -> Tensor:
    mask = valid_mask[:, :, None].to(probabilities.dtype)
    probability = probabilities * mask
    truth = target * mask
    spatial_dims = (0, 3, 4)
    hit = (probability * truth).sum(dim=spatial_dims)
    false_alarm = (probability * (1.0 - truth)).sum(dim=spatial_dims)
    miss = ((1.0 - probability) * truth).sum(dim=spatial_dims)
    event_count = truth.sum(dim=spatial_dims)
    valid_count = mask.sum(dim=spatial_dims)
    event_loss = 1.0 - hit / (
        hit + false_alarm + miss + config.epsilon
    )
    # Explicit empty-truth semantics: the loss is mean false-alarm
    # probability, hence zero for a perfectly dry forecast and differentiable
    # toward zero for any non-zero forecast.  This avoids a constant loss of
    # one with zero gradient on absent high-threshold events.
    empty_truth_loss = false_alarm / valid_count.clamp_min(config.epsilon)
    per_item_loss = torch.where(event_count > 0, event_loss, empty_truth_loss)
    lead = probabilities.new_tensor(config.lead_weights).view(3, 1)
    threshold = probabilities.new_tensor(config.threshold_weights).view(1, 4)
    has_valid = (valid_mask.sum(dim=(0, 2, 3)) > 0).to(probabilities.dtype).view(
        3, 1
    )
    weights = lead * threshold * has_valid
    # An entirely invalid lead contributes neither loss nor gradient.  Writing
    # this as a weighted (1 - TS), rather than ``1 - weighted TS``, also makes
    # the all-invalid result exactly zero.
    return (per_item_loss * weights).sum() / weights.sum().clamp_min(config.epsilon)


def log_area_constraint(
    probabilities: Tensor,
    target: Tensor,
    valid_mask: Tensor,
    config: LossConfig,
) -> Tensor:
    """Smooth lead/threshold event-count agreement in log area space."""

    mask = valid_mask[:, :, None].to(probabilities.dtype)
    predicted_count = (probabilities * mask).sum(dim=(-2, -1))
    true_count = (target * mask).sum(dim=(-2, -1))
    difference = F.smooth_l1_loss(
        torch.log1p(predicted_count),
        torch.log1p(true_count),
        reduction="none",
        beta=config.huber_beta,
    )
    lead = probabilities.new_tensor(config.lead_weights).view(1, 3, 1)
    threshold = probabilities.new_tensor(config.threshold_weights).view(1, 1, 4)
    has_valid = (valid_mask.sum(dim=(-2, -1)) > 0)[:, :, None].to(probabilities.dtype)
    weights = lead * threshold * has_valid
    return (difference * weights).sum() / weights.sum().clamp_min(config.epsilon)


def area_constraint(
    probabilities: Tensor,
    target: Tensor,
    valid_mask: Tensor,
) -> Tensor:
    """Compatibility wrapper using the fixed V3 log-area constraint."""

    return log_area_constraint(probabilities, target, valid_mask, LossConfig())


def event_weighted_regression(
    prediction_log1p: Tensor,
    target_mm: Tensor,
    valid_mask: Tensor,
    config: LossConfig,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return weighted log1p Huber, dry penalty, and their combined term."""

    if prediction_log1p.shape != target_mm.shape or target_mm.shape != valid_mask.shape:
        raise ValueError("regression prediction, target, and mask shapes differ")
    valid = valid_mask.to(torch.bool)
    if bool(torch.any(valid & (~torch.isfinite(target_mm) | (target_mm < 0)))):
        raise ValueError("valid regression targets must be finite non-negative millimetres")
    clean_target = torch.where(valid, target_mm, torch.zeros_like(target_mm))
    target_log1p = torch.log1p(clean_target)
    huber = F.smooth_l1_loss(
        prediction_log1p,
        target_log1p,
        reduction="none",
        beta=config.huber_beta,
    )
    dry, light, moderate, heavy, extreme = config.regression_bin_weights
    intensity = torch.full_like(clean_target, dry)
    intensity = torch.where(clean_target >= 0.1, intensity.new_tensor(light), intensity)
    intensity = torch.where(clean_target >= 5.0, intensity.new_tensor(moderate), intensity)
    intensity = torch.where(clean_target >= 20.0, intensity.new_tensor(heavy), intensity)
    intensity = torch.where(clean_target >= 50.0, intensity.new_tensor(extreme), intensity)
    lead = clean_target.new_tensor(config.lead_weights).view(1, 3, 1, 1)
    mask = valid.to(clean_target.dtype)
    weights = intensity * lead * mask
    valid_pixel_normalizer = (lead * mask).sum().clamp_min(config.epsilon)
    event_huber = (huber * weights).sum() / valid_pixel_normalizer

    dry_mask = valid & (clean_target < 0.1)
    false_rain_excess = F.relu(
        prediction_log1p - prediction_log1p.new_tensor(math.log1p(0.1))
    )
    dry_weights = dry_mask.to(clean_target.dtype) * lead
    dry_penalty = (false_rain_excess * dry_weights).sum() / dry_weights.sum().clamp_min(
        config.epsilon
    )
    combined = event_huber + config.dry_false_rain_weight * dry_penalty
    return event_huber, dry_penalty, combined


def multiscale_consistency(
    regression: Tensor,
    target_log1p: Tensor,
    valid_mask: Tensor,
) -> Tensor:
    losses = []
    prediction = regression.squeeze(2)
    mask = valid_mask.to(dtype=prediction.dtype)
    for scale in (2, 4):
        if min(prediction.shape[-2:]) < scale:
            continue
        mask_pool = F.avg_pool2d(mask.flatten(0, 1), scale).unflatten(
            0, mask.shape[:2]
        )
        support = mask_pool > 0
        denominator = mask_pool.clamp_min(torch.finfo(prediction.dtype).eps)
        pred_pool = F.avg_pool2d(
            (prediction * mask).flatten(0, 1), scale
        ).unflatten(0, prediction.shape[:2]) / denominator
        target_pool = F.avg_pool2d(
            (target_log1p * mask).flatten(0, 1), scale
        ).unflatten(0, target_log1p.shape[:2]) / denominator
        difference = (pred_pool - target_pool).abs()
        losses.append(
            (difference * support).sum()
            / support.sum().clamp_min(1).to(difference.dtype)
        )
    if not losses:
        return prediction.sum() * 0.0
    return torch.stack(losses).mean()


def precipitation_loss(
    outputs: dict[str, Tensor],
    target_mm: Tensor,
    valid_mask: Tensor,
    config: LossConfig | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    config = config or LossConfig()
    if target_mm.ndim != 4 or target_mm.shape[1] != 3:
        raise ValueError("target_mm must have shape (B,3,H,W)")
    if valid_mask.shape != target_mm.shape:
        raise ValueError("target and valid-mask shapes differ")
    supplied_valid = valid_mask.to(torch.bool)
    if bool(
        torch.any(
            supplied_valid & (~torch.isfinite(target_mm) | (target_mm < 0.0))
        )
    ):
        raise ValueError("valid targets must be finite non-negative millimetres")
    valid = supplied_valid & torch.isfinite(target_mm)
    clean_target = torch.nan_to_num(target_mm, nan=0.0).clamp_min(0.0)
    ordinal_target = cumulative_targets(clean_target, config.thresholds_mm)
    expected_ordinal_shape = (
        target_mm.shape[0],
        3,
        4,
        target_mm.shape[-2],
        target_mm.shape[-1],
    )
    expected_regression_shape = (
        target_mm.shape[0],
        3,
        1,
        target_mm.shape[-2],
        target_mm.shape[-1],
    )
    if outputs.get("ordinal_logits") is None or tuple(
        outputs["ordinal_logits"].shape
    ) != expected_ordinal_shape:
        raise ValueError("ordinal_logits must have shape (B,3,4,H,W)")
    if outputs.get("regression_log1p") is None or tuple(
        outputs["regression_log1p"].shape
    ) != expected_regression_shape:
        raise ValueError("regression_log1p must have shape (B,3,1,H,W)")
    supplied_probabilities = outputs.get("ordinal_probabilities")
    if supplied_probabilities is not None and tuple(
        supplied_probabilities.shape
    ) != expected_ordinal_shape:
        raise ValueError("ordinal_probabilities must have shape (B,3,4,H,W)")
    # Logits are the authoritative Loss V3 source; model-supplied probabilities
    # are an inference convenience and are only shape-validated here.  Casting
    # a supplied BF16 probability after sigmoid would preserve values already
    # rounded to 0/1.  All non-integer powers, ratios, and spatial reductions
    # stay at least FP32 even when the model forward is under BF16 autocast.
    with torch.autocast(
        device_type=outputs["ordinal_logits"].device.type,
        enabled=False,
    ):
        loss_logits = _at_least_float32(outputs["ordinal_logits"])
        probabilities = torch.sigmoid(loss_logits)
        loss_target = _at_least_float32(clean_target)
        ordinal_target = _at_least_float32(ordinal_target)
        ordinal = ordinal_bce(loss_logits, ordinal_target, valid, config)
        loss_regression = _at_least_float32(outputs["regression_log1p"])
        prediction_log = loss_regression.squeeze(2)
        event_huber, dry_penalty, regression = event_weighted_regression(
            prediction_log, loss_target, valid, config
        )
        ts = soft_ts_loss(probabilities, ordinal_target, valid, config)
        area = log_area_constraint(probabilities, ordinal_target, valid, config)
        multiscale = multiscale_consistency(
            loss_regression, torch.log1p(loss_target), valid
        )
        total = (
            config.ordinal_weight * ordinal
            + config.regression_weight * regression
            + config.soft_ts_weight * ts
            + config.area_weight * area
            + config.multiscale_weight * multiscale
        )
    return total, {
        "loss": total.detach(),
        "ordinal_bce": ordinal.detach(),
        "ordinal_asymmetric_focal": ordinal.detach(),
        "regression_huber": regression.detach(),
        "event_weighted_log1p_huber": event_huber.detach(),
        "dry_false_rain": dry_penalty.detach(),
        "soft_ts": ts.detach(),
        "area": area.detach(),
        "log_area": area.detach(),
        "multiscale": multiscale.detach(),
    }
