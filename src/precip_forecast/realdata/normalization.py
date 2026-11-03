"""Leakage-safe fold normalization for real precipitation caches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np


FOLD_PROCESSES: Mapping[str, Mapping[str, tuple[str, ...]]] = {
    "A": {"train": ("P0001",), "validation": ("P0002",)},
    "B": {"train": ("P0002",), "validation": ("P0001",)},
}


@dataclass
class RunningMoments:
    """Mergeable population moments over explicitly valid values."""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, values: Any, valid_mask: Any) -> None:
        array = np.asarray(values, dtype=np.float64)
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != array.shape:
            raise ValueError("normalization validity mask shape mismatch")
        if np.any(valid & ~np.isfinite(array)):
            raise ValueError("valid normalization values contain NaN or Inf")
        selected = array[valid]
        if selected.size == 0:
            return
        batch_count = int(selected.size)
        batch_mean = float(selected.mean(dtype=np.float64))
        centered = selected - batch_mean
        batch_m2 = float(np.dot(centered, centered))
        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            return
        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.m2 += batch_m2 + delta * delta * self.count * batch_count / total
        self.mean += delta * batch_count / total
        self.count = total

    def as_dict(self) -> dict[str, float | int]:
        if self.count <= 0:
            raise ValueError("cannot finalize normalization without valid training values")
        variance = max(self.m2 / self.count, 0.0)
        return {
            "count": self.count,
            "mean": self.mean,
            "std": max(float(np.sqrt(variance)), 1.0e-6),
        }


def compute_fold_normalization(
    hours: Iterable[tuple[str, np.ndarray, np.ndarray]],
    *,
    folds: Mapping[str, Mapping[str, tuple[str, ...]]] = FOLD_PROCESSES,
) -> dict[str, Any]:
    """Fit log-precipitation statistics using each fold's train process only."""

    materialized = list(hours)
    known_processes = {
        process
        for split in folds.values()
        for role in ("train", "validation")
        for process in split[role]
    }
    unexpected = sorted({str(process) for process, _, _ in materialized} - known_processes)
    if unexpected:
        raise ValueError(f"normalization input has unknown processes: {unexpected}")
    result: dict[str, Any] = {"schema_version": 1, "feature": "log1p_mm", "folds": {}}
    for fold, split in folds.items():
        train_processes = tuple(split["train"])
        validation_processes = tuple(split["validation"])
        overlap = set(train_processes) & set(validation_processes)
        if overlap:
            raise ValueError(f"fold {fold} process leakage: {sorted(overlap)}")
        moments = RunningMoments()
        used_hours = 0
        for process, values, valid_mask in materialized:
            if process in train_processes:
                moments.update(values, valid_mask)
                used_hours += 1
        stats = moments.as_dict()
        result["folds"][fold] = {
            "train_processes": list(train_processes),
            "validation_processes_excluded": list(validation_processes),
            "source_hour_count": used_hours,
            "statistics": stats,
        }
    return result


def normalize_for_model(
    values: Any,
    valid_mask: Any,
    statistics: Mapping[str, float | int],
) -> np.ndarray:
    """Normalize valid values and explicitly fill only invalid model entries."""

    array = np.asarray(values, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)
    if array.shape != valid.shape:
        raise ValueError("normalization validity mask shape mismatch")
    mean = float(statistics["mean"])
    std = float(statistics["std"])
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0.0:
        raise ValueError("invalid normalization statistics")
    if np.any(valid & ~np.isfinite(array)):
        raise ValueError("valid input contains NaN or Inf")
    output = np.zeros(array.shape, dtype=np.float32)
    output[valid] = (array[valid] - mean) / std
    if not np.all(np.isfinite(output)):
        raise ValueError("normalized model input contains NaN or Inf")
    return output
