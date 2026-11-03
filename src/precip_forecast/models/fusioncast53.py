"""FusionCast53: bounded conditional correction of a frozen NWP prior."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from precip_forecast.models.fusioncast52 import (
    ConvNormAct,
    DecoderStage,
    LowResolutionEncoder,
    ObservationEncoder,
    ResidualBlock,
)


THRESHOLDS_MM = (0.1, 5.0, 20.0, 50.0)


def prior_logits_from_mm(
    prior_mm: Tensor,
    temperatures: tuple[float, float, float, float],
) -> Tensor:
    if prior_mm.ndim != 4 or prior_mm.shape[1] != 3:
        raise ValueError("Task53 NWP prior must have shape (B,3,H,W)")
    if not torch.isfinite(prior_mm).all() or torch.any(prior_mm < 0.0):
        raise ValueError("Task53 NWP prior must be finite and nonnegative")
    temperature = prior_mm.new_tensor(temperatures).view(1, 1, 4, 1, 1)
    if torch.any(temperature <= 0.0):
        raise ValueError("Task53 prior temperatures must be positive")
    thresholds = prior_mm.new_tensor(THRESHOLDS_MM).view(1, 1, 4, 1, 1)
    return (torch.log1p(prior_mm).unsqueeze(2) - torch.log1p(thresholds)) / temperature


def bounded_monotonic_logits(base: Tensor, raw_delta: Tensor, bound: float) -> tuple[Tensor, Tensor]:
    if base.shape != raw_delta.shape or base.shape[2] != 4:
        raise ValueError("Task53 ordinal base and delta shapes differ")
    if bound not in {1.0, 2.0, 3.0}:
        raise ValueError("Task53 ordinal correction bound must be one of 1,2,3")
    proposed = base + float(bound) * torch.tanh(raw_delta)
    corrected: list[Tensor] = []
    for threshold in range(4):
        lower = base[:, :, threshold] - float(bound)
        upper = base[:, :, threshold] + float(bound)
        if corrected:
            upper = torch.minimum(upper, corrected[-1])
        value = torch.maximum(lower, torch.minimum(proposed[:, :, threshold], upper))
        corrected.append(value)
    logits = torch.stack(corrected, dim=2)
    return logits, logits - base


@dataclass(frozen=True)
class FusionCast53Config:
    observation_channels: int = 18
    radar_channels: int = 18
    nwp_channels: int = 84
    use_observation: bool = True
    use_radar: bool = True
    use_nwp_environment: bool = True
    ordinal_correction_bound: float = 2.0
    intensity_log_correction_bound: float = 0.7
    prior_temperatures: tuple[float, float, float, float] = (0.35, 0.45, 0.55, 0.70)

    def validate(self) -> None:
        if self.ordinal_correction_bound not in {1.0, 2.0, 3.0}:
            raise ValueError("Task53 correction bound must be one of 1,2,3")
        if not 0.0 < self.intensity_log_correction_bound <= 1.5:
            raise ValueError("Task53 intensity correction bound is invalid")
        if len(self.prior_temperatures) != 4 or any(value <= 0.0 for value in self.prior_temperatures):
            raise ValueError("Task53 prior temperatures are invalid")


class IndependentCorrectionGate(nn.Module):
    """One sigmoid evidence gate; never participates in modal softmax competition."""

    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(320 * 3 + 1, 96, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(96, 1, 1),
        )

    def forward(self, prior: Tensor, evidence: Tensor, lead: Tensor, available: Tensor) -> Tensor:
        availability = available.view(available.shape[0], 1, 1, 1).expand(
            -1, -1, *prior.shape[-2:]
        )
        return torch.sigmoid(self.body(torch.cat((prior, evidence, lead, availability), dim=1))) * availability


class FusionCast53(nn.Module):
    """NWP-preserving 4/5/6-hour probabilistic correction model."""

    def __init__(self, config: FusionCast53Config = FusionCast53Config()) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.observation_encoder = ObservationEncoder(config.observation_channels)
        self.radar_encoder = LowResolutionEncoder(config.radar_channels, width=128)
        self.nwp_encoder = LowResolutionEncoder(config.nwp_channels, width=128)
        self.prior_encoder = nn.Sequential(
            ConvNormAct(2, 128),
            nn.Conv2d(128, 320, 1, bias=False),
            nn.GroupNorm(16, 320),
            nn.SiLU(inplace=True),
        )
        self.observation_delta = nn.Sequential(
            nn.Conv2d(320, 320, 1, bias=False), nn.GroupNorm(16, 320), nn.SiLU(inplace=True)
        )
        self.radar_delta = nn.Sequential(
            nn.Conv2d(320, 320, 1, bias=False), nn.GroupNorm(16, 320), nn.SiLU(inplace=True)
        )
        self.observation_gate = IndependentCorrectionGate()
        self.radar_gate = IndependentCorrectionGate()
        self.lead_embeddings = nn.Embedding(3, 320)
        self.fused_body = ResidualBlock(320)
        self.decode8 = DecoderStage(320, 256, 256)
        self.decode4 = DecoderStage(256, 160, 160)
        self.decode2 = DecoderStage(160, 96, 96)
        self.decode1 = DecoderStage(96, 48, 48)
        self.ordinal_delta_head = nn.Conv2d(48, 4, 1)
        self.regression_delta_head = nn.Conv2d(48, 1, 1)
        self.area_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(320, 128),
            nn.SiLU(inplace=True),
            nn.Linear(128, 4),
        )

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        observation = batch["observation"]
        radar = batch["radar"]
        nwp = batch["nwp"]
        prior_mm = batch["nwp_prior_mm"]
        prior_valid = batch["nwp_prior_valid"]
        if observation.ndim != 4 or radar.ndim != 4 or nwp.ndim != 5 or nwp.shape[1] != 3:
            raise ValueError("FusionCast53 input shapes are invalid")
        if prior_mm.shape != prior_valid.shape or prior_mm.shape[:2] != (observation.shape[0], 3):
            raise ValueError("FusionCast53 NWP prior shapes are invalid")
        if prior_mm.shape[-2:] != observation.shape[-2:]:
            raise ValueError("FusionCast53 prior and observation grids differ")
        batch_size = observation.shape[0]
        availability = batch.get("modality_available")
        if availability is None:
            availability = observation.new_ones((batch_size, 3))
        if availability.shape != (batch_size, 3) or not torch.isfinite(availability).all():
            raise ValueError("FusionCast53 modality availability is invalid")
        availability = availability.to(observation.dtype).clamp(0.0, 1.0)
        enabled = availability.new_tensor((
            float(self.config.use_observation),
            float(self.config.use_radar),
            float(self.config.use_nwp_environment),
        )).view(1, 3)
        availability = availability * enabled

        skips = self.observation_encoder(observation)
        obs_available = availability[:, 0].view(batch_size, 1, 1, 1)
        skips = [value * obs_available.to(value.dtype) for value in skips]
        _, radar16 = self.radar_encoder(radar)
        radar_available = availability[:, 1].view(batch_size, 1, 1, 1)
        radar16 = radar16 * radar_available.to(radar16.dtype)
        nwp_flat = nwp.reshape(batch_size * 3, *nwp.shape[2:])
        _, nwp16_flat = self.nwp_encoder(nwp_flat)
        nwp16 = nwp16_flat.reshape(batch_size, 3, 320, *nwp16_flat.shape[-2:])
        nwp16 = nwp16 * availability[:, 2].view(batch_size, 1, 1, 1, 1).to(nwp16.dtype)

        base_logits = prior_logits_from_mm(prior_mm, self.config.prior_temperatures)
        prior_log = torch.log1p(prior_mm.clamp(0.0, 200.0))
        outputs_logits: list[Tensor] = []
        outputs_correction: list[Tensor] = []
        outputs_regression: list[Tensor] = []
        outputs_area: list[Tensor] = []
        outputs_obs_gate: list[Tensor] = []
        outputs_radar_gate: list[Tensor] = []
        embeddings = self.lead_embeddings(torch.arange(3, device=observation.device))
        for lead in range(3):
            spatial_size = skips[-1].shape[-2:]
            lead_map = embeddings[lead].view(1, 320, 1, 1).expand(batch_size, -1, *spatial_size)
            prior_low = F.adaptive_avg_pool2d(
                torch.stack((prior_log[:, lead], prior_valid[:, lead].to(prior_log.dtype)), dim=1),
                spatial_size,
            )
            prior_feature = self.prior_encoder(prior_low) + nwp16[:, lead] + lead_map
            observation_evidence = self.observation_delta(skips[-1])
            radar_evidence = self.radar_delta(radar16)
            obs_gate = self.observation_gate(
                prior_feature, observation_evidence, lead_map, availability[:, 0]
            )
            radar_gate = self.radar_gate(
                prior_feature, radar_evidence, lead_map, availability[:, 1]
            )
            fused = self.fused_body(
                prior_feature + obs_gate * observation_evidence + radar_gate * radar_evidence
            )
            outputs_area.append(self.area_head(fused))
            value = self.decode8(fused, skips[3])
            value = self.decode4(value, skips[2])
            value = self.decode2(value, skips[1])
            value = self.decode1(value, skips[0])
            raw_delta = self.ordinal_delta_head(value)
            corrected, actual_delta = bounded_monotonic_logits(
                base_logits[:, lead:lead + 1],
                raw_delta.unsqueeze(1),
                self.config.ordinal_correction_bound,
            )
            regression = prior_log[:, lead] + self.config.intensity_log_correction_bound * torch.tanh(
                self.regression_delta_head(value).squeeze(1)
            )
            outputs_logits.append(corrected.squeeze(1))
            outputs_correction.append(actual_delta.squeeze(1))
            outputs_regression.append(regression.clamp(0.0, math.log1p(200.0)))
            outputs_obs_gate.append(obs_gate)
            outputs_radar_gate.append(radar_gate)
        return {
            "base_ordinal_logits": base_logits,
            "ordinal_logits": torch.stack(outputs_logits, dim=1),
            "ordinal_correction": torch.stack(outputs_correction, dim=1),
            "regression_log1p": torch.stack(outputs_regression, dim=1),
            "area_log_counts": torch.stack(outputs_area, dim=1),
            "observation_correction_gate": torch.stack(outputs_obs_gate, dim=1),
            "radar_correction_gate": torch.stack(outputs_radar_gate, dim=1),
        }


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


__all__ = [
    "FusionCast53",
    "FusionCast53Config",
    "bounded_monotonic_logits",
    "parameter_count",
    "prior_logits_from_mm",
]
