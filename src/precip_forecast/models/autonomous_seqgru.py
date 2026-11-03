"""A from-scratch Task65 ablation with causal spatial sequence aggregation.

Only ``seq_temporal[0]`` differs from :class:`Task65Sequence`: a six-step
ConvGRU replaces the flattened-frame 1x1 projection.  Its original GroupNorm,
SiLU, residual block, encoder, decoder, input whitelist and availability gates
are inherited unchanged.  There is no checkpoint loading, new input, loss,
count head or decoder in this module.

The caller must supply Task65's audited chronological slots, at offsets
(-66, -54, -42, -30, -18, -6) minutes.  Each stem feature occupies 32 adjacent
channels of the inherited B,192,H/4,W/4 tensor.  Per-pixel validity stays in
the original four-channel frame input; slot quality is NOT a new recurrent
mask.  Missing-sequence suppression remains the inherited seq_available and
radar-modality gate.  ``control=True`` repeats the latest usable raw frame
using Task65's existing repeat_latest implementation before the same stem
and ConvGRU, without changing the measured availability masks.

This is architecture source, not a claim of training, validation or promotion.
Any checkpoint used in a new experiment must have its own from-scratch
training identity; loading old CAL/HELD-exposed Task65 weights is not allowed.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from precip_forecast.models.task65_seq4 import (
    INPUT_KEYS,
    Task65Sequence,
    input_only,
    repeat_latest,
)


FRAME_OFFSETS_MINUTES = (-66, -54, -42, -30, -18, -6)
FRAMES = 6
FRAME_FEATURE_CHANNELS = 32
HIDDEN_CHANNELS = 64
MIXER_PARAMETER_COUNT = 166080
EXPECTED_MODEL_PARAMETER_COUNT = 13415877


class CausalSpatialConvGRU(nn.Module):
    """Fixed B,192,H,W -> B,64,H,W chronological six-frame ConvGRU.

    Both the reset/update gates and candidate state use 3x3 spatial
    convolutions.  Hidden state starts at zero on every forward; there is no
    cross-sample state, bidirectional pass, dropout or attention.  We use
    h_t = z_t * h_(t-1) + (1-z_t) * candidate_t.  These parameters are shared
    across all six timesteps.
    """

    def __init__(self) -> None:
        super().__init__()
        joined_channels = FRAME_FEATURE_CHANNELS + HIDDEN_CHANNELS
        self.gates = nn.Conv2d(
            joined_channels, 2 * HIDDEN_CHANNELS, 3, padding=1, bias=True
        )
        self.candidate = nn.Conv2d(
            joined_channels, HIDDEN_CHANNELS, 3, padding=1, bias=True
        )

    def forward(self, value: Tensor) -> Tensor:
        if value.ndim != 4 or value.shape[1] != FRAMES * FRAME_FEATURE_CHANNELS:
            raise ValueError("Expected B,192,H,W in six chronological 32-channel blocks")
        if not value.is_floating_point():
            raise TypeError("Sequence stem features must be floating point")
        batch, _, height, width = value.shape
        if batch < 1 or height < 1 or width < 1:
            raise ValueError("Sequence feature dimensions must be positive")
        frames = value.reshape(batch, FRAMES, FRAME_FEATURE_CHANNELS, height, width)
        hidden = value.new_zeros((batch, HIDDEN_CHANNELS, height, width))
        for time_index in range(FRAMES):
            frame = frames[:, time_index]
            reset, update = torch.sigmoid(
                self.gates(torch.cat((frame, hidden), dim=1))
            ).chunk(2, dim=1)
            proposal = torch.tanh(
                self.candidate(torch.cat((frame, reset * hidden), dim=1))
            )
            hidden = update * hidden + (1.0 - update) * proposal
        return hidden


class AutonomousSeqGRU(Task65Sequence):
    """Task65 with only its temporal projection replaced; initialize fresh."""

    def __init__(self, class_frequency=None, *, control: bool = False) -> None:
        super().__init__(class_frequency, control=control)
        projection = self.seq_temporal[0]
        if not (
            isinstance(projection, nn.Conv2d)
            and projection.in_channels == FRAMES * FRAME_FEATURE_CHANNELS
            and projection.out_channels == HIDDEN_CHANNELS
            and projection.kernel_size == (1, 1)
            and projection.bias is None
        ):
            raise RuntimeError("Inherited Task65 temporal projection identity changed")
        # Retain the exact inherited GN / activation / residual tail.  This
        # intentionally does not override forward or load an existing model.
        self.seq_temporal[0] = CausalSpatialConvGRU()


__all__ = [
    "AutonomousSeqGRU",
    "CausalSpatialConvGRU",
    "EXPECTED_MODEL_PARAMETER_COUNT",
    "FRAME_OFFSETS_MINUTES",
    "INPUT_KEYS",
    "MIXER_PARAMETER_COUNT",
    "input_only",
    "repeat_latest",
]
