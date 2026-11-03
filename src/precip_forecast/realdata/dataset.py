"""Anchor-driven, mask-explicit datasets backed by the completed cache."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from precip_forecast.cache.manifest import SHA256_RE
from precip_forecast.cache.reader import PrecipitationCache
from precip_forecast.realdata.anchors import read_anchor_index
from precip_forecast.realdata.normalization import FOLD_PROCESSES, normalize_for_model
from precip_forecast.realdata.pseudotime import add_hours, extract_time_token


def _as_writable_c_contiguous(values: np.ndarray) -> np.ndarray:
    """Return an array that can be shared safely with ``torch.from_numpy``."""

    array = np.asarray(values)
    if array.flags.writeable and array.flags.c_contiguous:
        return array
    return np.array(array, copy=True, order="C", subok=False)


def ordered_anchor_sequence_sha256(sample_ids: Sequence[str]) -> str:
    """Hash an ordered, unique anchor sequence without reading sample payloads."""

    ordered = tuple(sample_ids)
    if not ordered or any(
        not isinstance(value, str) or not value or value != value.strip()
        for value in ordered
    ):
        raise ValueError("ordered anchor sequence must contain non-empty strings")
    if len(set(ordered)) != len(ordered):
        raise ValueError("ordered anchor sequence contains duplicate sample_id values")
    payload = {
        "schema_version": 1,
        "ordered_sample_ids": list(ordered),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _select_anchors(
    rows: Sequence[Mapping[str, Any]], *, fold: str, split: str
) -> list[Mapping[str, Any]]:
    if fold not in FOLD_PROCESSES:
        raise ValueError(f"unsupported fold: {fold!r}")
    if split in {"train", "validation"}:
        field = f"fold_{fold.lower()}_role"
        selected = [row for row in rows if str(row.get(field)) == split]
        expected = set(FOLD_PROCESSES[fold][split])
        actual = {str(row["process"]) for row in selected}
        if actual != expected:
            raise ValueError(
                f"anchor split evidence differs from fixed {fold}/{split}: {sorted(actual)}"
            )
    elif split in {"overfit_debug", "io_benchmark"}:
        selected = [row for row in rows if bool(row.get(split, False))]
    elif split == "all":
        selected = list(rows)
    else:
        raise ValueError("split must be train, validation, overfit_debug, io_benchmark, or all")
    if not selected:
        raise ValueError(f"anchor selection is empty for fold={fold}, split={split}")
    return selected


class CachedPrecipitationDataset:
    """Six historical PRE frames and three target leads on the fallback canvas.

    Continuous invalid values are filled with zero only after normalization;
    the returned validity channels and target mask make that operation explicit.
    """

    def __init__(
        self,
        *,
        anchor_index: str | Path | Sequence[Mapping[str, Any]],
        cache_root: str | Path | None = None,
        cache: PrecipitationCache | None = None,
        fold: str,
        split: str,
        as_torch: bool = True,
        verify_cache: str = "all",
        report_worker_rss: bool = False,
        data_epoch_id: str | None = None,
        source_manifest_sha256: str | None = None,
    ) -> None:
        if (cache is None) == (cache_root is None):
            raise ValueError("provide exactly one of cache or cache_root")
        self.cache = cache or PrecipitationCache(cache_root, verify=verify_cache)  # type: ignore[arg-type]
        if self.cache.verification_mode != "all":
            raise ValueError("training dataset requires a fully SHA256-verified cache")
        self.fold = str(fold).upper()
        if isinstance(anchor_index, (str, Path)):
            rows = read_anchor_index(Path(anchor_index), eligible_only=True)
        else:
            rows = [dict(row) for row in anchor_index if bool(row.get("eligible", False))]
        self.anchors = _select_anchors(rows, fold=self.fold, split=split)
        self.split = str(split)
        self.dataset_role = f"full_{self.split}_fold_base"
        self.ordered_sample_ids = tuple(row["sample_id"] for row in self.anchors)
        self.anchor_sequence_sha256 = ordered_anchor_sequence_sha256(
            self.ordered_sample_ids
        )
        if (data_epoch_id is None) != (source_manifest_sha256 is None):
            raise ValueError(
                "data_epoch_id and source_manifest_sha256 must be supplied together"
            )
        if data_epoch_id is not None and (
            not SHA256_RE.fullmatch(str(data_epoch_id).lower())
            or not SHA256_RE.fullmatch(str(source_manifest_sha256).lower())
        ):
            raise ValueError("dataset data/source identity must contain full SHA256 values")
        self.data_epoch_id = (
            None if data_epoch_id is None else str(data_epoch_id).lower()
        )
        self.source_manifest_sha256 = (
            None
            if source_manifest_sha256 is None
            else str(source_manifest_sha256).lower()
        )
        try:
            fold_normalization = self.cache.normalization["folds"][self.fold]
            self.statistics = fold_normalization["statistics"]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"cache lacks normalization for fold {self.fold}") from exc
        expected_train = list(FOLD_PROCESSES[self.fold]["train"])
        expected_validation = list(FOLD_PROCESSES[self.fold]["validation"])
        if fold_normalization.get("train_processes") != expected_train or fold_normalization.get(
            "validation_processes_excluded"
        ) != expected_validation:
            raise ValueError("cache normalization process evidence violates fixed folds")
        self.as_torch = bool(as_torch)
        self.report_worker_rss = bool(report_worker_rss)

    def __len__(self) -> int:
        return len(self.anchors)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.anchors[index]
        process = str(row["process"])
        input_paths = list(row.get("pre_input_paths", ()))
        target_paths = list(row.get("target_paths", ()))
        if len(input_paths) != 6 or len(target_paths) != 3:
            raise ValueError(f"anchor windows are incomplete: {row.get('sample_id')}")
        anchor_token = str(row["anchor_token"])
        expected_inputs = [add_hours(anchor_token, offset) for offset in range(-6, 0)]
        expected_targets = [add_hours(anchor_token, offset) for offset in (4, 5, 6)]
        if [extract_time_token(path) for path in input_paths] != expected_inputs:
            raise ValueError(f"PRE input window violates t-6..t-1: {row.get('sample_id')}")
        if [extract_time_token(path) for path in target_paths] != expected_targets:
            raise ValueError(f"PRE target window violates t+4/t+5/t+6: {row.get('sample_id')}")
        if list(row.get("target_leads", ())) != [4, 5, 6]:
            raise ValueError(f"target lead evidence is not [4,5,6]: {row.get('sample_id')}")
        inputs = [self.cache.get_hour_from_source(process, path) for path in input_paths]
        targets = [self.cache.get_hour_from_source(process, path) for path in target_paths]
        domain = np.asarray(self.cache.domain_mask, dtype=bool)
        input_source_valid = np.stack(
            [np.asarray(hour.valid_mask, dtype=bool) for hour in inputs]
        )
        input_valid = input_source_valid & domain[None, ...]
        input_rain = np.stack([np.asarray(hour.rain_mask, dtype=bool) for hour in inputs])
        input_rain &= input_valid
        input_log = np.stack([np.asarray(hour.log1p_mm, dtype=np.float32) for hour in inputs])
        last_precipitation_mm = np.zeros(domain.shape, dtype=np.float32)
        last_precipitation_log1p = np.zeros(domain.shape, dtype=np.float32)
        last_precipitation_mm[input_valid[-1]] = np.asarray(
            inputs[-1].raw_mm, dtype=np.float32
        )[input_valid[-1]]
        last_precipitation_log1p[input_valid[-1]] = input_log[-1][input_valid[-1]]
        normalized = normalize_for_model(input_log, input_valid, self.statistics)
        target_source_valid = np.stack(
            [np.asarray(hour.valid_mask, dtype=bool) for hour in targets]
        )
        target_valid = target_source_valid & domain[None, ...]
        target_raw = np.stack([np.asarray(hour.raw_mm, dtype=np.float32) for hour in targets])
        target_mm = np.zeros(target_raw.shape, dtype=np.float32)
        target_mm[target_valid] = target_raw[target_valid]
        model_input = np.concatenate(
            [
                normalized,
                input_rain.astype(np.float32),
                input_valid.astype(np.float32),
            ],
            axis=0,
        )
        if not np.all(np.isfinite(model_input)) or not np.all(np.isfinite(target_mm)):
            raise ValueError("NaN or Inf reached a precipitation model tensor")
        result: dict[str, Any] = {
            "inputs": model_input,
            "precip_log1p_normalized": normalized,
            "rain_mask": input_rain,
            "input_valid_mask": input_valid,
            "input_source_valid_mask": input_source_valid,
            "last_precipitation_mm": last_precipitation_mm[None, ...],
            "last_precipitation_log1p": last_precipitation_log1p[None, ...],
            "last_precipitation_valid_mask": input_valid[-1][None, ...],
            "domain_mask": domain,
            "target_mm": target_mm,
            "target_valid_mask": target_valid,
            "target_source_valid_mask": target_source_valid,
            "sample_id": str(row["sample_id"]),
            "process": process,
            "anchor_token": anchor_token,
            "target_leads": np.asarray([4, 5, 6], dtype=np.int64),
            "cache_content_sha256": self.cache.cache_content_sha256,
        }
        if self.as_torch:
            import torch

            for key in (
                "inputs",
                "precip_log1p_normalized",
                "rain_mask",
                "input_valid_mask",
                "input_source_valid_mask",
                "last_precipitation_mm",
                "last_precipitation_log1p",
                "last_precipitation_valid_mask",
                "domain_mask",
                "target_mm",
                "target_valid_mask",
                "target_source_valid_mask",
                "target_leads",
            ):
                result[key] = torch.from_numpy(_as_writable_c_contiguous(result[key]))
        if self.report_worker_rss:
            import resource

            result["worker_peak_rss_bytes"] = int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            ) * 1024
        return result
