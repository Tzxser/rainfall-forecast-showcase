"""Bounded spatial-rank corrector for the Task54 NWP-anchored decoder.

The network never predicts rain amount or event area.  It only changes the
per-threshold spatial ordering of the raw Regional-NWP prior.  Exact nested
areas are imposed later by the separately cross-fitted AreaModel decoder.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from precip_forecast.models.fusioncast53 import (
    bounded_monotonic_logits,
    prior_logits_from_mm,
)


def _groups(channels: int) -> int:
    for value in (32, 16, 8, 4, 2, 1):
        if channels % value == 0:
            return value
    return 1


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
    ) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.GroupNorm(_groups(output_channels), output_channels),
            nn.SiLU(inplace=True),
        )


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            ConvNormAct(channels, channels),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(channels), channels),
        )

    def forward(self, value: Tensor) -> Tensor:
        return F.silu(value + self.body(value), inplace=True)


class DecoderBlock(nn.Module):
    def __init__(self, input_channels: int, skip_channels: int, output_channels: int) -> None:
        super().__init__()
        self.project = ConvNormAct(input_channels + skip_channels, output_channels)
        self.refine = ResidualBlock(output_channels)

    def forward(self, value: Tensor, skip: Tensor) -> Tensor:
        value = F.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.refine(self.project(torch.cat((value, skip), dim=1)))


@dataclass(frozen=True)
class Task54RankConfig:
    observation_channels: int = 18
    radar_channels: int = 18
    nwp_channels: int = 84
    width: int = 256
    correction_bound: float = 2.0
    prior_temperatures: tuple[float, float, float, float] = (0.35, 0.45, 0.55, 0.70)

    def validate(self) -> None:
        if self.observation_channels != 18 or self.radar_channels != 18 or self.nwp_channels != 84:
            raise ValueError("Task54 rank feature contract differs")
        if self.width != 256:
            raise ValueError("Task54 rank width is frozen at 256")
        if self.correction_bound not in {1.0, 2.0, 3.0}:
            raise ValueError("Task54 rank correction bound must be 1, 2 or 3")
        if len(self.prior_temperatures) != 4 or any(value <= 0.0 for value in self.prior_temperatures):
            raise ValueError("Task54 rank prior temperatures are invalid")


class Task54RankCorrector(nn.Module):
    """A compact 1/8-scale multimodal encoder with a full-resolution rank head."""

    def __init__(self, config: Task54RankConfig = Task54RankConfig()) -> None:
        super().__init__()
        config.validate()
        self.config = config
        width = config.width

        self.observation_half = nn.Sequential(
            ConvNormAct(config.observation_channels, 64, stride=2),
            ResidualBlock(64),
        )
        self.observation_quarter = nn.Sequential(
            ConvNormAct(64, 128, stride=2),
            ResidualBlock(128),
        )
        self.observation_eighth = nn.Sequential(
            ConvNormAct(128, width, stride=2),
            ResidualBlock(width),
        )
        self.radar_encoder = nn.Sequential(
            ConvNormAct(config.radar_channels, width),
            ResidualBlock(width),
            ResidualBlock(width),
        )
        self.nwp_encoder = nn.Sequential(
            ConvNormAct(config.nwp_channels, width),
            ResidualBlock(width),
            ResidualBlock(width),
        )
        self.prior_encoder = nn.Sequential(
            ConvNormAct(2, width),
            ResidualBlock(width),
        )
        self.lead_embedding = nn.Embedding(3, width)
        self.fusion = nn.Sequential(
            ConvNormAct(width * 5, width, kernel_size=1),
            ResidualBlock(width),
            ResidualBlock(width),
        )
        self.decode_quarter = DecoderBlock(width, 128, 192)
        self.decode_half = DecoderBlock(192, 64, 128)
        self.decode_full = nn.Sequential(
            ConvNormAct(128, 64),
            ResidualBlock(64),
        )
        self.rank_delta_head = nn.Conv2d(64, 4, 1)

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        observation = batch["observation"]
        radar = batch["radar"]
        nwp = batch["nwp"]
        prior_mm = batch["nwp_prior_raw_mm"]
        prior_valid = batch["nwp_prior_valid"]
        if observation.ndim != 4 or radar.ndim != 4:
            raise ValueError("Task54 observation/Radar inputs must be BCHW")
        if nwp.ndim != 5 or nwp.shape[1] != 3:
            raise ValueError("Task54 NWP environment must be B,3,C,H,W")
        if prior_mm.shape != prior_valid.shape or prior_mm.shape[:2] != (observation.shape[0], 3):
            raise ValueError("Task54 raw NWP prior shapes differ")
        if prior_mm.shape[-2:] != observation.shape[-2:]:
            raise ValueError("Task54 prior and observation grids differ")
        if not torch.isfinite(prior_mm).all() or torch.any(prior_mm < 0.0):
            raise ValueError("Task54 raw NWP prior is invalid")

        batch_size = observation.shape[0]
        availability = batch.get("modality_available")
        if availability is None:
            availability = observation.new_ones((batch_size, 3))
        if availability.shape != (batch_size, 3) or not torch.isfinite(availability).all():
            raise ValueError("Task54 modality availability is invalid")
        availability = availability.to(observation.dtype).clamp(0.0, 1.0)

        observation_half = self.observation_half(observation)
        observation_quarter = self.observation_quarter(observation_half)
        observation_eighth = self.observation_eighth(observation_quarter)
        obs_available = availability[:, 0].view(batch_size, 1, 1, 1)
        observation_half = observation_half * obs_available
        observation_quarter = observation_quarter * obs_available
        observation_eighth = observation_eighth * obs_available

        radar_feature = self.radar_encoder(radar)
        radar_feature = radar_feature * availability[:, 1].view(batch_size, 1, 1, 1)
        nwp_flat = nwp.reshape(batch_size * 3, *nwp.shape[2:])
        nwp_feature = self.nwp_encoder(nwp_flat).reshape(
            batch_size, 3, self.config.width, *nwp_flat.shape[-2:]
        )
        nwp_feature = nwp_feature * availability[:, 2].view(batch_size, 1, 1, 1, 1)

        base_logits = prior_logits_from_mm(prior_mm, self.config.prior_temperatures)
        prior_log = torch.log1p(prior_mm.clamp(0.0, 200.0))
        embeddings = self.lead_embedding(torch.arange(3, device=observation.device))
        scores: list[Tensor] = []
        corrections: list[Tensor] = []
        for lead in range(3):
            low_shape = observation_eighth.shape[-2:]
            prior_low = F.adaptive_avg_pool2d(
                torch.stack(
                    (prior_log[:, lead], prior_valid[:, lead].to(prior_log.dtype)),
                    dim=1,
                ),
                low_shape,
            )
            prior_feature = self.prior_encoder(prior_low)
            lead_feature = embeddings[lead].view(1, -1, 1, 1).expand(
                batch_size, -1, *low_shape
            )
            fused = self.fusion(
                torch.cat(
                    (
                        observation_eighth,
                        radar_feature,
                        nwp_feature[:, lead],
                        prior_feature,
                        lead_feature,
                    ),
                    dim=1,
                )
            )
            value = self.decode_quarter(fused, observation_quarter)
            value = self.decode_half(value, observation_half)
            value = F.interpolate(
                value, size=observation.shape[-2:], mode="bilinear", align_corners=False
            )
            raw_delta = self.rank_delta_head(self.decode_full(value))
            corrected, actual = bounded_monotonic_logits(
                base_logits[:, lead : lead + 1],
                raw_delta.unsqueeze(1),
                self.config.correction_bound,
            )
            scores.append(corrected.squeeze(1))
            corrections.append(actual.squeeze(1))
        return {
            "base_rank_scores": base_logits,
            "rank_scores": torch.stack(scores, dim=1),
            "rank_correction": torch.stack(corrections, dim=1),
        }


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


__all__ = [
    "Task54RankConfig",
    "Task54RankCorrector",
    "parameter_count",
]
