"""Monotone cumulative-threshold parameterisation."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def monotone_logits(raw: Tensor) -> Tensor:
    """Convert arbitrary last-but-spatial channel logits to non-increasing logits."""

    if raw.ndim < 3 or raw.shape[-3] != 4:
        raise ValueError("raw tensor must have four threshold channels at axis -3")
    first = raw[..., :1, :, :]
    decrements = F.softplus(raw[..., 1:, :, :])
    return torch.cat([first, first - torch.cumsum(decrements, dim=-3)], dim=-3)


def cumulative_probabilities(logits: Tensor) -> Tensor:
    return torch.sigmoid(logits)


def raw_bias_from_cumulative_prior(
    probabilities: Tensor,
    *,
    epsilon: float = 1.0e-4,
) -> Tensor:
    """Invert :func:`monotone_logits` for four cumulative event priors.

    The returned four raw biases reproduce the clipped cumulative
    probabilities after the monotone parameterisation.  Priors must be
    non-increasing because every higher threshold is a subset of the previous
    event.
    """

    values = torch.as_tensor(probabilities, dtype=torch.float64)
    if values.shape != (4,):
        raise ValueError("cumulative ordinal prior must contain four values")
    if not bool(torch.all(torch.isfinite(values))):
        raise ValueError("cumulative ordinal prior contains NaN or Inf")
    if not 0.0 < epsilon < 0.5:
        raise ValueError("epsilon must be within (0, 0.5)")
    # Validate the un-clipped evidence with scalar bounds.  Clipping is only a
    # numerical logit guard for valid endpoint priors; it must never turn an
    # out-of-domain prior into apparently valid training evidence.
    minimum = float(torch.amin(values).item())
    maximum = float(torch.amax(values).item())
    if minimum < 0.0 or maximum > 1.0:
        raise ValueError("cumulative ordinal prior must be within [0, 1]")
    values = values.clamp(epsilon, 1.0 - epsilon)
    if bool(torch.any(values[1:] > values[:-1])):
        raise ValueError("cumulative ordinal prior must be non-increasing")
    logits = torch.logit(values)
    decrements = (logits[:-1] - logits[1:]).clamp_min(epsilon)
    # Stable inverse softplus: x + log(1-exp(-x)).
    raw_decrements = decrements + torch.log(-torch.expm1(-decrements))
    return torch.cat((logits[:1], raw_decrements)).to(dtype=torch.float32)


class OrdinalHead(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.projection = nn.Conv2d(channels, 4, kernel_size=1)

    def forward(self, features: Tensor) -> Tensor:
        return monotone_logits(self.projection(features))
