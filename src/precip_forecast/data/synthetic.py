"""Small multi-resolution synthetic fixtures for remote engineering validation only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


MODALITY_ORDER = (
    "precip_history",
    "radar_3d",
    "convective_initiation",
    "regional_nwp",
    "satellite",
    "global_nwp",
)


@dataclass
class SyntheticBatch:
    inputs: dict[str, np.ndarray]
    target_mm: np.ndarray
    valid_mask: np.ndarray
    modality_available: np.ndarray
    sample_ids: list[str]

    def validate(self) -> None:
        batch, leads, height, width = self.target_mm.shape
        if leads != 3:
            raise ValueError("target must contain three leads")
        if self.valid_mask.shape != (batch, 1, height, width):
            raise ValueError("valid_mask shape mismatch")
        if self.modality_available.shape != (batch, len(MODALITY_ORDER)):
            raise ValueError("modality_available shape mismatch")
        if len(self.sample_ids) != batch or len(set(self.sample_ids)) != batch:
            raise ValueError("sample ids must be unique and match batch")
        for name in MODALITY_ORDER:
            if name not in self.inputs or self.inputs[name].shape[0] != batch:
                raise ValueError(f"missing or invalid modality: {name}")


def _gaussian(
    height: int,
    width: int,
    center_y: float,
    center_x: float,
    scale: float,
    amplitude: float,
) -> np.ndarray:
    y, x = np.mgrid[:height, :width]
    distance = ((y - center_y) / scale) ** 2 + ((x - center_x) / scale) ** 2
    return amplitude * np.exp(-0.5 * distance)


def _resize_mean(values: np.ndarray, height: int, width: int) -> np.ndarray:
    yi = np.linspace(0, values.shape[-2] - 1, height).round().astype(int)
    xi = np.linspace(0, values.shape[-1] - 1, width).round().astype(int)
    return values[..., yi[:, None], xi[None, :]]


def generate_synthetic_batch(
    batch_size: int = 2,
    height: int = 64,
    width: int = 96,
    seed: int = 2026,
    inject_missing: bool = True,
) -> SyntheticBatch:
    if batch_size <= 0 or height < 16 or width < 16:
        raise ValueError("batch_size must be positive and spatial dimensions at least 16")
    rng = np.random.default_rng(seed)
    histories = []
    targets = []
    for sample in range(batch_size):
        frames = []
        base_y = height * (0.35 + 0.08 * sample)
        base_x = width * (0.25 + 0.05 * sample)
        for step in range(-6, 7):
            intensity = 12.0 + 1.5 * step
            primary = _gaussian(
                height,
                width,
                base_y + 0.7 * step,
                base_x + 1.4 * step,
                max(3.0, height / 10),
                max(0.0, intensity),
            )
            newborn = _gaussian(
                height,
                width,
                height * 0.68,
                width * (0.75 - 0.01 * step),
                max(2.0, height / 14),
                max(0.0, (step - 1) * 3.0),
            )
            noise = rng.normal(0.0, 0.08, size=(height, width))
            frames.append(np.maximum(primary + newborn + noise, 0.0).astype(np.float32))
        histories.append(np.stack(frames[:6]))
        targets.append(np.stack([frames[9], frames[10], frames[11]]))
    history = np.stack(histories)[:, :, None]
    target = np.stack(targets)
    batch = target.shape[0]

    radar_base = np.repeat(history[:, -1:], 30, axis=1)
    radar = np.repeat(radar_base, 2, axis=2)
    radar += rng.normal(0.0, 0.15, radar.shape).astype(np.float32)
    ci = _resize_mean(np.repeat(history[:, -1:], 36, axis=1), height // 2, width // 2)
    regional = _resize_mean(np.repeat(target[:, :1, None], 4, axis=1), height // 3, width // 3)
    regional = np.repeat(regional, 10, axis=2)
    satellite = _resize_mean(np.repeat(history[:, -1:], 6, axis=1), height // 4, width // 4)
    satellite = np.repeat(satellite, 5, axis=2)
    global_nwp = _resize_mean(np.repeat(target[:, :1, None], 3, axis=1), height // 8, width // 8)
    global_nwp = np.repeat(global_nwp, 8, axis=2)
    valid = np.ones((batch, 1, height, width), dtype=bool)
    valid[:, :, : max(1, height // 12), :] = False
    valid[:, :, :, : max(1, width // 16)] = False
    available = np.ones((batch, len(MODALITY_ORDER)), dtype=bool)
    inputs = {
        "precip_history": history.astype(np.float32),
        "radar_3d": radar.astype(np.float32),
        "convective_initiation": ci.astype(np.float32),
        "regional_nwp": regional.astype(np.float32),
        "satellite": satellite.astype(np.float32),
        "global_nwp": global_nwp.astype(np.float32),
    }
    if inject_missing:
        inputs["radar_3d"][0, 3] = np.nan
        inputs["satellite"][0, :, :, 0, 0] = np.nan
        if batch > 1:
            available[1, MODALITY_ORDER.index("convective_initiation")] = False
            inputs["convective_initiation"][1] = np.nan
    result = SyntheticBatch(
        inputs=inputs,
        target_mm=target.astype(np.float32),
        valid_mask=valid,
        modality_available=available,
        sample_ids=[f"synthetic_{index:04d}" for index in range(batch)],
    )
    result.validate()
    return result
