"""Compact, causal Radar Lite cache for Task51 real-data training.

The official NetCDF files remain read-only.  Only the three hourly files
``t-3, t-2, t-1`` are used; their ten six-minute frames therefore span
``[-180, -6]`` minutes and cannot cross the business anchor ``t``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import warnings
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from precip_forecast.realdata.pseudotime import PseudoTime, add_hours, extract_time_token, hours_between


RADAR_SCHEMA_VERSION = "task51-radar-lite-v1"
RADAR_TIME_GATE_VERSION = "task51-radar-causal-window-v1"
SOURCE_HOUR_OFFSETS = (-3, -2, -1)
WITHIN_FILE_MINUTES = tuple(float(value) for value in range(0, 60, 6))
FEATURE_NAMES = (
    "latest_zmax",
    "recent_60m_zmax",
    "recent_180m_zmax",
    "mean_180m_zmax",
    "latest_minus_early_zmax",
    "latest_midlevel_zmax",
    "recent_180m_midlevel_zmax",
    "wet_fraction_ge20dbz",
)
VALUE_CLIPS = (
    (-20.0, 70.0), (-20.0, 70.0), (-20.0, 70.0), (-20.0, 70.0),
    (-90.0, 90.0), (-20.0, 70.0), (-20.0, 70.0), (0.0, 1.0),
)
LOW_GRID_STRIDE = 8
LOW_GRID_SHAPE = (64, 96)
MODEL_INPUT_CHANNELS = len(FEATURE_NAMES) * 2 + 2


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def radar_source_tokens(anchor_token: str) -> tuple[str, str, str]:
    return tuple(add_hours(anchor_token, offset) for offset in SOURCE_HOUR_OFFSETS)  # type: ignore[return-value]


def resulting_offsets_minutes() -> tuple[float, ...]:
    return tuple(
        hour * 60.0 + minute
        for hour in SOURCE_HOUR_OFFSETS
        for minute in WITHIN_FILE_MINUTES
    )


def _nearest_indices(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = np.asarray(source, np.float64)
    target = np.asarray(target, np.float64)
    if source.ndim != 1 or target.ndim != 1 or source.size < target.size:
        raise ValueError("Radar coordinate arrays are invalid")
    indices = np.argmin(np.abs(source[:, None] - target[None, :]), axis=0)
    if len(set(int(value) for value in indices)) != target.size:
        raise ValueError("Radar low-grid coordinate mapping is not one-to-one")
    if float(np.max(np.abs(source[indices] - target))) > 0.006:
        raise ValueError("Radar coordinates do not cover the official low grid")
    return indices.astype(np.int64)


def summarize_hour(args: tuple[str, np.ndarray, np.ndarray]) -> tuple[str, dict[str, Any]]:
    """Read one official file into two compact ten-frame composites."""

    path_text, target_latitude, target_longitude = args
    path = Path(path_text)
    from netCDF4 import Dataset, num2date

    token = extract_time_token(path)
    with Dataset(path, "r") as dataset:
        required = {"REF", "time", "level", "latitude", "longitude"}
        if not required.issubset(dataset.variables):
            raise ValueError(f"Radar schema mismatch: {path}")
        ref = dataset.variables["REF"]
        if tuple(ref.dimensions) != ("time", "level", "latitude", "longitude"):
            raise ValueError(f"Radar REF dimensions changed: {path}")
        if str(getattr(ref, "units", "")) != "dBZ":
            raise ValueError(f"Radar REF units changed: {path}")
        time = dataset.variables["time"]
        decoded = num2date(
            np.asarray(time[:], np.float64),
            units=str(getattr(time, "units", "")),
            calendar=str(getattr(time, "calendar", "standard")),
            only_use_cftime_datetimes=True,
        )
        offsets = tuple(float((value - decoded[0]).total_seconds() / 60.0) for value in decoded)
        if offsets != WITHIN_FILE_MINUTES:
            raise ValueError(f"Radar within-file time offsets changed: {path}")
        expected = PseudoTime.parse(token)
        first = decoded[0]
        if (first.year, first.month, first.day, first.hour, first.minute) != (
            expected.year, expected.month, expected.day, expected.hour, 0
        ):
            raise ValueError(f"Radar filename/internal time mismatch: {path}")
        latitude = np.asarray(dataset.variables["latitude"][:], np.float64)
        longitude = np.asarray(dataset.variables["longitude"][:], np.float64)
        y_index = _nearest_indices(latitude, target_latitude)
        x_index = _nearest_indices(longitude, target_longitude)
        raw = ref[:, :, y_index, x_index]
        values = np.asarray(np.ma.asarray(raw).astype(np.float32).filled(np.nan), np.float32)
        values[(~np.isfinite(values)) | (values <= -900.0)] = np.nan
        values = np.clip(values, -20.0, 70.0)
        levels = np.asarray(dataset.variables["level"][:], np.float32)
        middle = (levels >= 3.0) & (levels <= 7.0)
        if not np.any(middle):
            raise ValueError("Radar levels do not contain the audited 3..7 km band")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            zmax = np.nanmax(values, axis=1)
            mid = np.nanmax(values[:, middle], axis=1)
        z_valid = np.isfinite(zmax)
        mid_valid = np.isfinite(mid)
        zmax = np.where(z_valid, zmax, 0.0).astype(np.float32)
        mid = np.where(mid_valid, mid, 0.0).astype(np.float32)
    return token, {
        "zmax": zmax,
        "z_valid": z_valid,
        "mid": mid,
        "mid_valid": mid_valid,
        "path": str(path),
        "size_bytes": path.stat().st_size,
    }


def _masked_last(values: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if values.shape[0] == 0:
        return np.zeros(values.shape[1:], np.float32), np.zeros(values.shape[1:], bool)
    output = np.zeros(values.shape[1:], np.float32)
    output_valid = np.zeros(values.shape[1:], bool)
    for index in range(values.shape[0] - 1, -1, -1):
        take = valid[index] & ~output_valid
        output[take] = values[index][take]
        output_valid[take] = True
    return output, output_valid


def _masked_max(values: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if values.shape[0] == 0:
        return np.zeros(values.shape[1:], np.float32), np.zeros(values.shape[1:], bool)
    selected = np.where(valid, values, -np.inf)
    output = np.max(selected, axis=0)
    output_valid = np.any(valid, axis=0)
    output[~output_valid] = 0.0
    return output.astype(np.float32), output_valid


def _masked_mean(values: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if values.shape[0] == 0:
        return np.zeros(values.shape[1:], np.float32), np.zeros(values.shape[1:], bool)
    count = np.sum(valid, axis=0)
    output = np.zeros(values.shape[1:], np.float32)
    np.divide(np.sum(np.where(valid, values, 0.0), axis=0), count, out=output, where=count > 0)
    return output, count > 0


def aggregate_anchor(
    anchor_token: str,
    hourly: Mapping[str, Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, float, bool, int]:
    z_values: list[np.ndarray] = []
    z_masks: list[np.ndarray] = []
    mid_values: list[np.ndarray] = []
    mid_masks: list[np.ndarray] = []
    offsets: list[float] = []
    missing = 0
    for token in radar_source_tokens(anchor_token):
        record = hourly.get(token)
        if record is None:
            missing += 1
            continue
        base = hours_between(token, anchor_token) * 60.0
        for index, within in enumerate(WITHIN_FILE_MINUTES):
            relative = base + within
            if relative > -6.0 + 1e-6:
                raise ValueError("Task51 Radar cache attempted to use a future frame")
            z_values.append(np.asarray(record["zmax"])[index])
            z_masks.append(np.asarray(record["z_valid"])[index])
            mid_values.append(np.asarray(record["mid"])[index])
            mid_masks.append(np.asarray(record["mid_valid"])[index])
            offsets.append(relative)
    if not z_values:
        empty_values = np.zeros((len(FEATURE_NAMES), *LOW_GRID_SHAPE), np.float32)
        return empty_values, np.zeros_like(empty_values, bool), 3.0, False, missing
    z = np.stack(z_values); zv = np.stack(z_masks)
    mid = np.stack(mid_values); midv = np.stack(mid_masks)
    offset = np.asarray(offsets, np.float32)
    latest, latest_valid = _masked_last(z, zv)
    recent = offset >= -60.0
    recent_max, recent_valid = _masked_max(z[recent], zv[recent])
    all_max, all_max_valid = _masked_max(z, zv)
    all_mean, all_mean_valid = _masked_mean(z, zv)
    early = offset <= -120.0
    early_mean, early_valid = _masked_mean(z[early], zv[early])
    change_valid = latest_valid & early_valid
    change = np.where(change_valid, latest - early_mean, 0.0).astype(np.float32)
    latest_mid, latest_mid_valid = _masked_last(mid, midv)
    all_mid, all_mid_valid = _masked_max(mid, midv)
    wet_count = np.sum(zv & (z >= 20.0), axis=0)
    valid_count = np.sum(zv, axis=0)
    wet = np.zeros(LOW_GRID_SHAPE, np.float32)
    np.divide(wet_count, valid_count, out=wet, where=valid_count > 0)
    wet_valid = valid_count > 0
    values = np.stack((latest, recent_max, all_max, all_mean, change, latest_mid, all_mid, wet))
    masks = np.stack((latest_valid, recent_valid, all_max_valid, all_mean_valid, change_valid, latest_mid_valid, all_mid_valid, wet_valid))
    for channel, (lower, upper) in enumerate(VALUE_CLIPS):
        values[channel] = np.clip(values[channel], lower, upper)
    latest_offset = float(max(offsets))
    age_hours = min(3.0, max(0.0, -latest_offset / 60.0))
    return values.astype(np.float32), masks, age_hours, True, missing


def build_rows(
    anchor_manifest: str | Path,
    split_manifest: str | Path,
    precipitation_cache: Any,
) -> list[dict[str, Any]]:
    split = json.loads(Path(split_manifest).read_text(encoding="utf-8"))
    train_ids = set(split["train_sample_ids"]); dev_ids = set(split["dev_sample_ids"])
    guard_ids = set(split.get("guard_sample_ids", []))
    with Path(anchor_manifest).open(encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    result: list[dict[str, Any]] = []
    for row in rows:
        sample_id = row["sample_id"]
        role = "train" if sample_id in train_ids else "dev" if sample_id in dev_ids else None
        if sample_id in guard_ids:
            continue
        if role is None:
            raise ValueError(f"Task51 Radar anchor is outside split: {sample_id}")
        result.append({"sample_id": sample_id, "process": row["process"], "anchor_token": row["anchor_token"], "role": role, "official_split": "train"})
    for number in range(1, 613):
        sample_id = f"V{number:04d}"
        tokens = precipitation_cache.tokens.get(("validation", sample_id), [])
        if not tokens:
            raise ValueError(f"validation precipitation anchor missing: {sample_id}")
        anchor = PseudoTime.parse(tokens[-1]).add_hours(1).token()
        result.append({"sample_id": sample_id, "process": sample_id, "anchor_token": anchor, "role": "validation", "official_split": "validation"})
    if len(result) != len(rows) - len(guard_ids) + 612:
        raise ValueError("Task51 Radar row construction lost samples")
    return result


def _radar_directory(data_root: Path, row: Mapping[str, Any]) -> Path:
    if row["official_split"] == "train":
        return data_root / "TRAIN_DATA" / str(row["process"]) / "RADA_REF"
    return data_root / "VAL_DATA" / str(row["sample_id"]) / "RADA_REF"


def _normalization(values: np.ndarray, masks: np.ndarray, train_indices: Sequence[int]) -> dict[str, Any]:
    sums = np.zeros(len(FEATURE_NAMES), np.float64)
    squares = np.zeros(len(FEATURE_NAMES), np.float64)
    counts = np.zeros(len(FEATURE_NAMES), np.int64)
    for start in range(0, len(train_indices), 64):
        indices = np.asarray(train_indices[start : start + 64], np.int64)
        block = np.asarray(values[indices], np.float32)
        valid = np.asarray(masks[indices], bool)
        for channel in range(len(FEATURE_NAMES)):
            selected = block[:, channel][valid[:, channel]]
            if selected.size:
                sums[channel] += float(np.sum(selected, dtype=np.float64))
                squares[channel] += float(np.sum(selected.astype(np.float64) ** 2, dtype=np.float64))
                counts[channel] += int(selected.size)
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    variances = np.divide(squares, counts, out=np.zeros_like(squares), where=counts > 0) - means**2
    std = np.sqrt(np.maximum(variances, 0.0)); std[(std < 1e-3) | (counts == 0)] = 1.0
    return {
        "schema_version": 1,
        "source_split": "train",
        "validation_data_used": False,
        "feature_names": list(FEATURE_NAMES),
        "mean": means.tolist(),
        "std": std.tolist(),
        "valid_counts": counts.tolist(),
        "clip_bounds": [list(pair) for pair in VALUE_CLIPS],
    }


def build_cache(
    *,
    data_root: str | Path,
    cache_root: str | Path,
    rows: Sequence[Mapping[str, Any]],
    target_latitude: np.ndarray,
    target_longitude: np.ndarray,
    real_data_epoch: str,
    radar_gate_path: str | Path,
    workers: int,
) -> dict[str, Any]:
    data_root = Path(data_root).resolve(strict=True)
    cache_root = Path(cache_root).resolve()
    try:
        cache_root.relative_to(data_root)
    except ValueError:
        pass
    else:
        raise ValueError("Task51 Radar cache cannot be written inside official data")
    gate_path = Path(radar_gate_path).resolve(strict=True)
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("status") != "PASS" or not gate.get("radar_model_authorized"):
        raise ValueError("Task51 Radar cache requires a passed causal gate")
    required = gate.get("requirements", {})
    if not required.get("resulting_frame_offsets_are_minus_180_to_minus_6_minutes"):
        raise ValueError("Radar gate lacks the Task51 causal cache-window proof")
    low_latitude = np.asarray(target_latitude)[::LOW_GRID_STRIDE]
    low_longitude = np.asarray(target_longitude)[::LOW_GRID_STRIDE]
    if low_latitude.shape != (64,) or low_longitude.shape != (96,):
        raise ValueError("official grid does not reduce to Task51 64x96")
    stage = cache_root.with_name(cache_root.name + ".partial")
    if cache_root.exists():
        manifest = json.loads((cache_root / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("real_data_epoch") != real_data_epoch or manifest.get("radar_gate_sha256") != sha256_file(gate_path):
            raise ValueError("existing Task51 Radar cache identity differs")
        return manifest
    stage.mkdir(parents=True, exist_ok=True)
    count = len(rows)
    shape = (count, len(FEATURE_NAMES), *LOW_GRID_SHAPE)
    values_path = stage / "values_float16.npy"; masks_path = stage / "valid_uint8.npy"
    if values_path.exists():
        values = np.lib.format.open_memmap(values_path, mode="r+")
        masks = np.lib.format.open_memmap(masks_path, mode="r+")
        ages = np.lib.format.open_memmap(stage / "age_hours_float16.npy", mode="r+")
        available = np.lib.format.open_memmap(stage / "source_available_uint8.npy", mode="r+")
        state = json.loads((stage / "build_state.json").read_text(encoding="utf-8"))
        completed = set(state.get("completed_processes", []))
    else:
        values = np.lib.format.open_memmap(values_path, mode="w+", dtype=np.float16, shape=shape)
        masks = np.lib.format.open_memmap(masks_path, mode="w+", dtype=np.uint8, shape=shape)
        ages = np.lib.format.open_memmap(stage / "age_hours_float16.npy", mode="w+", dtype=np.float16, shape=(count,))
        available = np.lib.format.open_memmap(stage / "source_available_uint8.npy", mode="w+", dtype=np.uint8, shape=(count,))
        completed: set[str] = set()
    grouped: dict[str, list[tuple[int, Mapping[str, Any]]]] = {}
    for index, row in enumerate(rows):
        grouped.setdefault(f"{row['official_split']}:{row['process']}", []).append((index, row))
    records: dict[int, dict[str, Any]] = {}
    existing_index = stage / "index.csv"
    if existing_index.exists():
        with existing_index.open(encoding="utf-8", newline="") as handle:
            records = {int(row["index"]): dict(row) for row in csv.DictReader(handle)}
    source_files_read = 0; source_bytes_read = 0
    pool = ProcessPoolExecutor(max_workers=workers)
    for group, items in sorted(grouped.items()):
        if group in completed:
            continue
        directory = _radar_directory(data_root, items[0][1])
        inventory = {
            extract_time_token(path): path
            for path in sorted(directory.glob("*.NC")) + sorted(directory.glob("*.nc"))
            if path.is_file()
        } if directory.is_dir() else {}
        needed = sorted({token for _, row in items for token in radar_source_tokens(str(row["anchor_token"]))})
        tasks = [(str(inventory[token]), low_latitude, low_longitude) for token in needed if token in inventory]
        hourly: dict[str, Mapping[str, Any]] = {}
        if tasks:
            for token, summary in pool.map(summarize_hour, tasks, chunksize=1):
                hourly[token] = summary
                source_files_read += 1; source_bytes_read += int(summary["size_bytes"])
        for index, row in items:
            feature, valid, age, source_available, missing = aggregate_anchor(str(row["anchor_token"]), hourly)
            values[index] = feature.astype(np.float16)
            masks[index] = valid.astype(np.uint8)
            ages[index] = np.float16(age); available[index] = np.uint8(source_available)
            records[index] = {**{key: str(row[key]) for key in ("sample_id", "process", "anchor_token", "role", "official_split")}, "index": str(index), "missing_source_hour_count": str(missing)}
        values.flush(); masks.flush(); ages.flush(); available.flush()
        completed.add(group)
        (stage / "build_state.json").write_text(json.dumps({"completed_processes": sorted(completed), "total_processes": len(grouped)}, indent=2) + "\n", encoding="utf-8")
        with existing_index.open("w", encoding="utf-8", newline="") as handle:
            fieldnames = ("index", "sample_id", "process", "anchor_token", "role", "official_split", "missing_source_hour_count")
            writer = csv.DictWriter(handle, fieldnames=fieldnames); writer.writeheader()
            for index in sorted(records): writer.writerow(records[index])
    pool.shutdown(wait=True)
    if len(records) != count:
        raise ValueError("Task51 Radar cache index is incomplete")
    train_indices = [index for index, row in enumerate(rows) if row["role"] == "train"]
    normalization = _normalization(values, masks, train_indices)
    (stage / "normalization.json").write_text(json.dumps(normalization, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    np.save(stage / "latitude.npy", low_latitude, allow_pickle=False)
    np.save(stage / "longitude.npy", low_longitude, allow_pickle=False)
    payload_names = ("values_float16.npy", "valid_uint8.npy", "age_hours_float16.npy", "source_available_uint8.npy", "index.csv", "normalization.json", "latitude.npy", "longitude.npy")
    payloads = [{"path": name, "size_bytes": (stage / name).stat().st_size, "sha256": sha256_file(stage / name)} for name in payload_names]
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "real_data_epoch": real_data_epoch,
        "radar_schema_version": RADAR_SCHEMA_VERSION,
        "radar_time_gate_version": RADAR_TIME_GATE_VERSION,
        "radar_gate_sha256": sha256_file(gate_path),
        "source_hour_offsets": list(SOURCE_HOUR_OFFSETS),
        "resulting_offsets_minutes": list(resulting_offsets_minutes()),
        "future_frames_used": False,
        "feature_names": list(FEATURE_NAMES),
        "model_input_channels": MODEL_INPUT_CHANNELS,
        "grid_shape": list(LOW_GRID_SHAPE),
        "row_count": count,
        "train_row_count": sum(row["role"] == "train" for row in rows),
        "dev_row_count": sum(row["role"] == "dev" for row in rows),
        "validation_row_count": sum(row["role"] == "validation" for row in rows),
        "source_files_read_this_invocation": source_files_read,
        "source_bytes_read_this_invocation": source_bytes_read,
        "payloads": payloads,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (stage / "COMPLETED").write_text(sha256_file(stage / "manifest.json") + "\n", encoding="ascii")
    os.rename(stage, cache_root)
    return manifest


class Task51RadarCache:
    def __init__(self, root: str | Path, *, expected_epoch: str) -> None:
        self.root = Path(root).resolve(strict=True)
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("status") != "complete" or self.manifest.get("real_data_epoch") != expected_epoch:
            raise ValueError("Task51 Radar cache identity mismatch")
        self.values = np.load(self.root / "values_float16.npy", mmap_mode="r")
        self.masks = np.load(self.root / "valid_uint8.npy", mmap_mode="r")
        self.ages = np.load(self.root / "age_hours_float16.npy", mmap_mode="r")
        self.available = np.load(self.root / "source_available_uint8.npy", mmap_mode="r")
        self.normalization = json.loads((self.root / "normalization.json").read_text(encoding="utf-8"))
        with (self.root / "index.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.lookup = {row["sample_id"]: int(row["index"]) for row in rows}
        if len(self.lookup) != int(self.manifest["row_count"]):
            raise ValueError("Task51 Radar cache index count mismatch")
        self.mean = np.asarray(self.normalization["mean"], np.float32)
        self.std = np.asarray(self.normalization["std"], np.float32)

    def get(self, sample_id: str) -> tuple[np.ndarray, float]:
        index = self.lookup[str(sample_id)]
        values = np.asarray(self.values[index], np.float32)
        masks = np.asarray(self.masks[index], bool)
        normalized = np.zeros_like(values, np.float32)
        normalized[masks] = np.clip(((values - self.mean[:, None, None]) / self.std[:, None, None])[masks], -8.0, 8.0)
        age = np.full(LOW_GRID_SHAPE, min(float(self.ages[index]) / 3.0, 1.0), np.float32)
        available = float(self.available[index])
        availability = np.full(LOW_GRID_SHAPE, available, np.float32)
        result = np.concatenate((normalized, masks.astype(np.float32), age[None], availability[None]), axis=0)
        if result.shape != (MODEL_INPUT_CHANNELS, *LOW_GRID_SHAPE) or not np.all(np.isfinite(result)):
            raise ValueError("Task51 Radar model tensor is invalid")
        return np.array(result, copy=True, order="C"), available


class Task51RadarDataset:
    """Add low-resolution Radar Lite and explicit availability to Task50 data."""

    def __init__(self, base: Any, radar_cache: Task51RadarCache) -> None:
        self.base = base; self.radar_cache = radar_cache

    @property
    def rows(self) -> Any:
        return self.base.rows

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.base[index]
        radar, available = self.radar_cache.get(str(sample["sample_id"]))
        modality = sample.get("modality_available")
        if modality is not None:
            radar_flag = float(modality[1].item() if hasattr(modality[1], "item") else modality[1])
            available *= radar_flag
        if available <= 0.0:
            radar[...] = 0.0
        try:
            import torch

            sample["radar"] = torch.from_numpy(radar)
            sample["radar_available"] = torch.tensor(available, dtype=torch.float32)
        except ImportError:
            sample["radar"] = radar; sample["radar_available"] = np.float32(available)
        return sample


__all__ = [
    "FEATURE_NAMES", "LOW_GRID_SHAPE", "MODEL_INPUT_CHANNELS", "RADAR_SCHEMA_VERSION",
    "RADAR_TIME_GATE_VERSION", "SOURCE_HOUR_OFFSETS", "Task51RadarCache",
    "Task51RadarDataset", "aggregate_anchor", "build_cache", "build_rows",
    "radar_source_tokens", "resulting_offsets_minutes", "summarize_hour",
]
