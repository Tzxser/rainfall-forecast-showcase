"""FusionCast52: lead-aware observation, Radar and Regional-NWP fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _groups(channels: int) -> int:
    for value in (16, 8, 4, 2, 1):
        if channels % value == 0:
            return value
    return 1


class ConvNormAct(nn.Sequential):
    def __init__(self, source: int, target: int, *, stride: int = 1) -> None:
        super().__init__(
            nn.Conv2d(source, target, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(_groups(target), target),
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

    def forward(self, inputs: Tensor) -> Tensor:
        return F.silu(inputs + self.body(inputs), inplace=True)


class EncoderStage(nn.Module):
    def __init__(self, source: int, target: int, *, downsample: bool) -> None:
        super().__init__()
        self.project = ConvNormAct(source, target, stride=2 if downsample else 1)
        self.blocks = nn.Sequential(ResidualBlock(target), ResidualBlock(target))

    def forward(self, inputs: Tensor) -> Tensor:
        return self.blocks(self.project(inputs))


class ObservationEncoder(nn.Module):
    def __init__(self, channels: int = 18) -> None:
        super().__init__()
        widths = (48, 96, 160, 256, 320)
        self.stages = nn.ModuleList()
        source = channels
        for index, target in enumerate(widths):
            self.stages.append(EncoderStage(source, target, downsample=index > 0))
            source = target

    def forward(self, inputs: Tensor) -> list[Tensor]:
        outputs: list[Tensor] = []
        value = inputs
        for stage in self.stages:
            value = stage(value)
            outputs.append(value)
        return outputs


class LowResolutionEncoder(nn.Module):
    """Encode a 1/8-scale modality at 1/8 and 1/16 scales."""

    def __init__(self, channels: int, width: int = 128) -> None:
        super().__init__()
        self.at_eighth = nn.Sequential(
            ConvNormAct(channels, width),
            ResidualBlock(width),
            ResidualBlock(width),
        )
        self.at_sixteenth = nn.Sequential(
            ConvNormAct(width, 320, stride=2),
            ResidualBlock(320),
        )

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        eighth = self.at_eighth(inputs)
        return eighth, self.at_sixteenth(eighth)


class DecoderStage(nn.Module):
    def __init__(self, source: int, skip: int, target: int) -> None:
        super().__init__()
        self.fuse = nn.Sequential(
            ConvNormAct(source + skip, target),
            ResidualBlock(target),
        )

    def forward(self, inputs: Tensor, skip: Tensor) -> Tensor:
        inputs = F.interpolate(inputs, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat((inputs, skip), dim=1))


def monotonic_ordinal_logits(raw: Tensor) -> Tensor:
    """Map five unconstrained fields to four non-increasing logits."""

    if raw.shape[-3] != 5:
        raise ValueError("ordinal raw head must have five channels")
    base = raw[..., :1, :, :]
    decrements = F.softplus(raw[..., 1:, :, :])
    return base - torch.cumsum(decrements, dim=-3)


@dataclass(frozen=True)
class FusionCast52Config:
    observation_channels: int = 18
    radar_channels: int = 18
    nwp_channels: int = 84
    use_observation: bool = True
    use_radar: bool = True
    use_nwp: bool = True


class FusionCast52(nn.Module):
    """A 15M–30M class-probability-first FusionCast baseline.

    Observation inputs are full resolution. Radar and each target-lead NWP
    tensor are 1/8 resolution.  The decoder is shared, but each lead has its
    own embedding and spatial modality gate.
    """

    def __init__(self, config: FusionCast52Config = FusionCast52Config()) -> None:
        super().__init__()
        self.config = config
        self.observation_encoder = ObservationEncoder(config.observation_channels)
        self.radar_encoder = LowResolutionEncoder(config.radar_channels, width=128)
        self.nwp_encoder = LowResolutionEncoder(config.nwp_channels, width=128)
        self.lead_embeddings = nn.Embedding(3, 320)
        self.gate = nn.Sequential(
            # The three explicit availability planes let the fusion gate
            # distinguish a physically quiet encoded field from a modality
            # that is absent or intentionally dropped during training.
            nn.Conv2d(320 * 4 + 3, 128, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, 3, 1),
        )
        self.cross = nn.Sequential(
            ConvNormAct(320 * 3, 320),
            ResidualBlock(320),
        )
        self.decode8 = DecoderStage(320, 256, 256)
        self.decode4 = DecoderStage(256, 160, 160)
        self.decode2 = DecoderStage(160, 96, 96)
        self.decode1 = DecoderStage(96, 48, 48)
        self.ordinal_head = nn.Conv2d(48, 5, 1)
        self.regression_head = nn.Conv2d(48, 1, 1)
        self.area_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(320, 128),
            nn.SiLU(inplace=True),
            nn.Linear(128, 4),
        )
        self.aux_lead_embeddings = nn.Embedding(3, 160)
        self.aux_body = nn.Sequential(ConvNormAct(160, 128), ResidualBlock(128))
        self.aux_ordinal_head = nn.Conv2d(128, 5, 1)
        self.aux_regression_head = nn.Conv2d(128, 1, 1)

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        observation = batch["observation"]
        radar = batch["radar"]
        nwp = batch["nwp"]
        if observation.ndim != 4 or radar.ndim != 4 or nwp.ndim != 5 or nwp.shape[1] != 3:
            raise ValueError("FusionCast52 input shapes are invalid")
        batch_size = observation.shape[0]
        availability = batch.get("modality_available")
        if availability is None:
            availability = observation.new_ones((batch_size, 3))
        if availability.shape != (batch_size, 3):
            raise ValueError("FusionCast52 modality_available must have shape (B,3)")
        if not torch.isfinite(availability).all():
            raise ValueError("FusionCast52 modality_available contains NaN/Inf")
        enabled = availability.new_tensor(
            (
                float(self.config.use_observation),
                float(self.config.use_radar),
                float(self.config.use_nwp),
            )
        ).view(1, 3)
        availability = availability.to(observation.dtype).clamp(0.0, 1.0) * enabled

        skips = self.observation_encoder(observation)
        observation_available = availability[:, 0].view(batch_size, 1, 1, 1)
        # Mask the encoded path, not only the raw input.  This prevents an
        # all-zero modality-dropout sample from backpropagating through a deep
        # chain of zero-variance normalization layers.
        skips = [value * observation_available.to(value.dtype) for value in skips]
        _, radar16 = self.radar_encoder(radar)
        radar_available = availability[:, 1].view(batch_size, 1, 1, 1)
        radar16 = radar16 * radar_available.to(radar16.dtype)
        nwp_flat = nwp.reshape(batch_size * 3, *nwp.shape[2:])
        _, nwp16_flat = self.nwp_encoder(nwp_flat)
        nwp16 = nwp16_flat.reshape(batch_size, 3, 320, *nwp16_flat.shape[-2:])
        nwp_available = availability[:, 2].view(batch_size, 1, 1, 1, 1)
        nwp16 = nwp16 * nwp_available.to(nwp16.dtype)

        decoded: list[Tensor] = []
        ordinal: list[Tensor] = []
        regression: list[Tensor] = []
        area: list[Tensor] = []
        gates: list[Tensor] = []
        lead_ids = torch.arange(3, device=observation.device)
        embeddings = self.lead_embeddings(lead_ids)
        for lead in range(3):
            lead_map = embeddings[lead].view(1, 320, 1, 1).expand(batch_size, -1, *skips[-1].shape[-2:])
            sources = (skips[-1], radar16, nwp16[:, lead])
            availability_map = availability.to(skips[-1].dtype).view(batch_size, 3, 1, 1).expand(
                -1, -1, *skips[-1].shape[-2:]
            )
            weights = torch.softmax(
                self.gate(torch.cat((*sources, lead_map, availability_map), dim=1)),
                dim=1,
            )
            fused = sum(source * weights[:, index : index + 1] for index, source in enumerate(sources))
            fused = fused + self.cross(torch.cat(sources, dim=1)) + lead_map
            area.append(self.area_head(fused))
            value = self.decode8(fused, skips[3])
            value = self.decode4(value, skips[2])
            value = self.decode2(value, skips[1])
            value = self.decode1(value, skips[0])
            decoded.append(value)
            ordinal.append(monotonic_ordinal_logits(self.ordinal_head(value)))
            regression.append(F.softplus(self.regression_head(value)))
            gates.append(weights)

        aux_ordinal: list[Tensor] = []
        aux_regression: list[Tensor] = []
        aux_base = skips[2]
        aux_embeddings = self.aux_lead_embeddings(lead_ids)
        for lead in range(3):
            value = aux_base + aux_embeddings[lead].view(1, 160, 1, 1)
            value = self.aux_body(value)
            raw_ordinal = self.aux_ordinal_head(value)
            raw_regression = self.aux_regression_head(value)
            target_size = observation.shape[-2:]
            raw_ordinal = F.interpolate(raw_ordinal, size=target_size, mode="bilinear", align_corners=False)
            raw_regression = F.interpolate(raw_regression, size=target_size, mode="bilinear", align_corners=False)
            aux_ordinal.append(monotonic_ordinal_logits(raw_ordinal))
            aux_regression.append(F.softplus(raw_regression))

        return {
            "ordinal_logits": torch.stack(ordinal, dim=1),
            "regression_log1p": torch.stack(regression, dim=1).squeeze(2),
            "area_logits": torch.stack(area, dim=1),
            "aux_ordinal_logits": torch.stack(aux_ordinal, dim=1),
            "aux_regression_log1p": torch.stack(aux_regression, dim=1).squeeze(2),
            "lead_gates": torch.stack(gates, dim=1),
        }


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


__all__ = [
    "FusionCast52",
    "FusionCast52Config",
    "monotonic_ordinal_logits",
    "parameter_count",
]
