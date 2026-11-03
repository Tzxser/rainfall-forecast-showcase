"""A 2M-10M parameter multi-resolution fusion model for interface validation."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from precip_forecast.data.synthetic import MODALITY_ORDER
from .ordinal import monotone_logits


DEFAULT_CHANNELS = {
    "precip_history": 1,
    "radar_3d": 2,
    "convective_initiation": 1,
    "regional_nwp": 10,
    "satellite": 5,
    "global_nwp": 8,
}


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        groups = min(8, out_channels)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.block(value)


class TemporalEncoder(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.encoder = ConvBlock(input_channels, output_channels)

    def forward(self, value: Tensor) -> Tensor:
        if value.ndim != 5:
            raise ValueError("modality tensor must have shape (B,T,C,H,W)")
        finite = torch.isfinite(value)
        numerator = torch.where(finite, value, torch.zeros_like(value)).sum(dim=1)
        denominator = finite.sum(dim=1).clamp_min(1)
        pooled = numerator / denominator
        return self.encoder(pooled)


class FusionLite(nn.Module):
    """Shared backbone with lead embeddings, monotone ordinal and regression heads."""

    def __init__(
        self,
        channel_map: Mapping[str, int] | None = None,
        modality_width: int = 48,
        width: int = 96,
    ) -> None:
        super().__init__()
        channels = dict(DEFAULT_CHANNELS if channel_map is None else channel_map)
        if tuple(channels) != MODALITY_ORDER:
            missing = [name for name in MODALITY_ORDER if name not in channels]
            if missing:
                raise ValueError(f"channel map missing modalities: {missing}")
        self.channel_map = channels
        self.encoders = nn.ModuleDict(
            {
                name: TemporalEncoder(channels[name], modality_width)
                for name in MODALITY_ORDER
            }
        )
        self.fuse = ConvBlock(modality_width * len(MODALITY_ORDER), width)
        self.down1 = ConvBlock(width, width * 2, stride=2)
        self.down2 = ConvBlock(width * 2, width * 4, stride=2)
        self.up1 = ConvBlock(width * 6, width * 2)
        self.up2 = ConvBlock(width * 3, width)
        self.lead_embedding = nn.Embedding(3, width)
        self.ordinal_projection = nn.Conv2d(width, 4, 1)
        self.regression_projection = nn.Conv2d(width, 1, 1)

    def forward(
        self,
        inputs: Mapping[str, Tensor],
        modality_available: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if any(name not in inputs for name in MODALITY_ORDER):
            raise ValueError("all six modality tensors are required")
        reference = inputs["precip_history"]
        target_size = reference.shape[-2:]
        batch = reference.shape[0]
        features = []
        for index, name in enumerate(MODALITY_ORDER):
            value = inputs[name]
            if value.shape[2] != self.channel_map[name]:
                raise ValueError(f"{name} channel count differs from configured contract")
            encoded = self.encoders[name](value)
            encoded = F.interpolate(encoded, size=target_size, mode="bilinear", align_corners=False)
            if modality_available is not None:
                if modality_available.shape != (batch, len(MODALITY_ORDER)):
                    raise ValueError("modality_available must have shape (B,6)")
                gate = modality_available[:, index].to(encoded.dtype).view(batch, 1, 1, 1)
                encoded = encoded * gate
            features.append(encoded)
        level0 = self.fuse(torch.cat(features, dim=1))
        level1 = self.down1(level0)
        level2 = self.down2(level1)
        decoded1 = F.interpolate(level2, size=level1.shape[-2:], mode="bilinear", align_corners=False)
        decoded1 = self.up1(torch.cat([decoded1, level1], dim=1))
        decoded0 = F.interpolate(decoded1, size=level0.shape[-2:], mode="bilinear", align_corners=False)
        decoded0 = self.up2(torch.cat([decoded0, level0], dim=1))

        lead_ids = torch.arange(3, device=decoded0.device)
        embedding = self.lead_embedding(lead_ids).view(1, 3, -1, 1, 1)
        lead_features = decoded0[:, None] + embedding
        flattened = lead_features.reshape(batch * 3, decoded0.shape[1], *target_size)
        raw_ordinal = self.ordinal_projection(flattened).reshape(batch, 3, 4, *target_size)
        ordinal_logits = monotone_logits(raw_ordinal)
        regression = F.softplus(self.regression_projection(flattened)).reshape(
            batch, 3, 1, *target_size
        )
        return {
            "ordinal_logits": ordinal_logits,
            "ordinal_probabilities": torch.sigmoid(ordinal_logits),
            "regression_log1p": regression,
        }


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
