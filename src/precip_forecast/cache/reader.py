"""Completed-only memory-mapped reader for precipitation event caches."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from precip_forecast.cache.manifest import (
    SHA256_RE,
    content_identifier,
    resolve_payload,
    safe_relative_path,
    sha256_file,
)
from precip_forecast.realdata.pseudotime import extract_time_token


@dataclass(frozen=True)
class CachedPrecipitationHour:
    raw_mm: np.ndarray
    log1p_mm: np.ndarray
    rain_mask: np.ndarray
    valid_mask: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    process: str
    token: str
    source_relative_path: str


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_sha256s(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if "  " not in raw:
            raise ValueError(f"invalid SHA256SUMS line {number}")
        digest, relative = raw.split("  ", 1)
        digest = digest.lower()
        relative = safe_relative_path(relative).as_posix()
        if not SHA256_RE.fullmatch(digest) or relative in result:
            raise ValueError(f"invalid SHA256SUMS line {number}")
        result[relative] = digest
    if not result:
        raise ValueError("SHA256SUMS is empty")
    return result


class PrecipitationCache:
    """Read a checksummed cache only after its completion marker validates."""

    def __init__(self, cache_root: str | Path, *, verify: str = "metadata") -> None:
        if verify not in {"none", "metadata", "all"}:
            raise ValueError("verify must be none, metadata, or all")
        self.verification_mode = verify
        root_argument = Path(cache_root)
        if not root_argument.is_absolute():
            raise ValueError("cache_root must be an explicit absolute path")
        self.container_root = root_argument.resolve()
        namespaced = self.container_root / "precipitation_v1"
        if not (self.container_root / "manifest.json").is_file() and namespaced.is_dir():
            self.root = namespaced.resolve()
        else:
            self.root = self.container_root
        completed_path = self.root / "COMPLETED"
        manifest_path = self.root / "manifest.json"
        sums_path = self.root / "SHA256SUMS.txt"
        for required in (completed_path, manifest_path, sums_path):
            if not required.is_file():
                raise ValueError(f"cache is incomplete; missing {required.name}")
        self.completed = _load_json(completed_path)
        self.manifest = _load_json(manifest_path)
        if self.manifest.get("status") != "complete":
            raise ValueError("cache manifest is not marked complete")
        if self.manifest.get("modality") != "precipitation":
            raise ValueError("cache is not precipitation-only")
        manifest_sha = sha256_file(manifest_path)
        if self.completed.get("manifest_sha256") != manifest_sha:
            raise ValueError("COMPLETED does not match manifest.json")
        if self.completed.get("sha256s_sha256") != sha256_file(sums_path):
            raise ValueError("COMPLETED does not match SHA256SUMS.txt")
        if self.completed.get("cache_content_sha256") != self.manifest.get(
            "cache_content_sha256"
        ):
            raise ValueError("cache identity differs between marker and manifest")

        records = self.manifest.get("payloads")
        if not isinstance(records, list) or not records:
            raise ValueError("cache manifest has no payload records")
        self.payload_records: dict[str, Mapping[str, Any]] = {}
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("invalid cache payload record")
            relative = safe_relative_path(str(record.get("path", ""))).as_posix()
            digest = str(record.get("sha256", "")).lower()
            if relative in self.payload_records or not SHA256_RE.fullmatch(digest):
                raise ValueError("duplicate or invalid cache payload record")
            self.payload_records[relative] = record
        sums = _read_sha256s(sums_path)
        expected_sums = {
            relative: str(record["sha256"]).lower()
            for relative, record in self.payload_records.items()
        }
        if sums != expected_sums:
            raise ValueError("SHA256SUMS and manifest payload sets differ")
        required_payloads = {
            "source_fingerprint.json",
            "preprocess_config.json",
            "data_fingerprint.json",
            "normalization.json",
            "hours.json",
            "latitude.npy",
            "longitude.npy",
            "domain_mask.npy",
        }
        if not required_payloads.issubset(self.payload_records):
            raise ValueError("cache is missing required metadata/coordinate payloads")

        declared_metadata_hashes = {
            "preprocess_config.json": self.manifest.get("preprocess_config_sha256"),
            "source_fingerprint.json": self.manifest.get("source_fingerprint_sha256"),
            "data_fingerprint.json": self.manifest.get("data_fingerprint_sha256"),
            "normalization.json": self.manifest.get("normalization_sha256"),
        }
        for relative, declared in declared_metadata_hashes.items():
            if relative not in expected_sums or declared != expected_sums[relative]:
                raise ValueError(f"manifest metadata hash mismatch: {relative}")
        recomputed_identity = content_identifier(
            preprocess_config_sha256=str(declared_metadata_hashes["preprocess_config.json"]),
            source_fingerprint_sha256=str(declared_metadata_hashes["source_fingerprint.json"]),
            data_fingerprint_sha256=str(declared_metadata_hashes["data_fingerprint.json"]),
            normalization_sha256=str(declared_metadata_hashes["normalization.json"]),
            payload_records=records,
        )
        if recomputed_identity != self.manifest.get("cache_content_sha256"):
            raise ValueError("cache content identity cannot be reproduced from payloads")

        metadata_names = {
            "source_fingerprint.json",
            "preprocess_config.json",
            "data_fingerprint.json",
            "normalization.json",
            "hours.json",
        }
        if verify != "none":
            verify_names = set(self.payload_records) if verify == "all" else metadata_names
            if not metadata_names.issubset(self.payload_records):
                raise ValueError("cache is missing required metadata payloads")
            for relative in sorted(verify_names):
                path = resolve_payload(self.root, relative)
                record = self.payload_records[relative]
                if not path.is_file() or path.stat().st_size != int(record["size_bytes"]):
                    raise ValueError(f"cache payload size mismatch: {relative}")
                if sha256_file(path) != str(record["sha256"]):
                    raise ValueError(f"cache payload hash mismatch: {relative}")

        self.preprocess_config = _load_json(self.root / "preprocess_config.json")
        self.source_fingerprint = _load_json(self.root / "source_fingerprint.json")
        self.data_fingerprint = _load_json(self.root / "data_fingerprint.json")
        self.normalization = _load_json(self.root / "normalization.json")
        if int(self.source_fingerprint.get("verified_failed", -1)) != 0:
            raise ValueError("cached source fingerprint records failed verification")
        if str(self.source_fingerprint.get("declared_local_commit", "")).lower() != str(
            self.manifest.get("source_commit", "")
        ).lower():
            raise ValueError("cached source commit differs from manifest")
        if self.preprocess_config.get("modality") != "precipitation":
            raise ValueError("cached preprocess config is not precipitation-only")
        expected_validity_semantics = (
            "cached valid_mask is source validity; model validity is "
            "source_valid_mask AND mapped_domain_mask"
        )
        if (
            self.preprocess_config.get("valid_mask_semantics")
            != expected_validity_semantics
            or self.manifest.get("valid_mask_semantics") != expected_validity_semantics
        ):
            raise ValueError("cache validity-mask semantics are not explicit")
        if self.preprocess_config.get("cache_namespace") == "precipitation_v1":
            configured_domain = self.preprocess_config.get("domain_mask")
            fingerprinted_domain = self.data_fingerprint.get("domain_mask_source")
            if (
                not isinstance(configured_domain, dict)
                or configured_domain != fingerprinted_domain
                or configured_domain.get("relative_path") != "guangxi_mask.nc"
                or not SHA256_RE.fullmatch(str(configured_domain.get("sha256", "")))
            ):
                raise ValueError("production cache lacks audited guangxi_mask.nc evidence")
        hours_document = _load_json(self.root / "hours.json")
        processes = hours_document.get("processes")
        if not isinstance(processes, dict) or set(processes) != {"P0001", "P0002"}:
            raise ValueError("cache hour index must contain exactly P0001 and P0002")

        self.latitude = np.load(
            resolve_payload(self.root, "latitude.npy"), mmap_mode="r", allow_pickle=False
        )
        self.longitude = np.load(
            resolve_payload(self.root, "longitude.npy"), mmap_mode="r", allow_pickle=False
        )
        self.domain_mask = np.load(
            resolve_payload(self.root, "domain_mask.npy"), mmap_mode="r", allow_pickle=False
        )
        canvas_shape = tuple(int(value) for value in self.manifest.get("canvas_shape", ()))
        if len(canvas_shape) != 2 or min(canvas_shape) <= 0:
            raise ValueError("cache manifest has an invalid canvas shape")
        if self.latitude.shape != (canvas_shape[0],) or self.longitude.shape != (
            canvas_shape[1],
        ):
            raise ValueError("cached coordinates differ from canvas shape")
        if (
            self.domain_mask.shape != canvas_shape
            or self.domain_mask.dtype != np.bool_
            or not np.any(self.domain_mask)
        ):
            raise ValueError("cached domain mask is empty, non-boolean, or mis-shaped")
        coordinate_hashes = self.manifest.get("coordinate_hashes")
        expected_coordinate_hashes = {
            "latitude_sha256": self.payload_records["latitude.npy"]["sha256"],
            "longitude_sha256": self.payload_records["longitude.npy"]["sha256"],
            "domain_mask_sha256": self.payload_records["domain_mask.npy"]["sha256"],
        }
        if coordinate_hashes != expected_coordinate_hashes:
            raise ValueError("manifest coordinate/domain hashes differ from payloads")
        if (
            self.normalization.get("validity_rule")
            != "source_valid_mask AND mapped_domain_mask"
            or self.normalization.get("domain_mask_sha256")
            != expected_coordinate_hashes["domain_mask_sha256"]
        ):
            raise ValueError("normalization does not prove use of the mapped domain mask")

        array_records = self.manifest.get("arrays")
        if not isinstance(array_records, list) or len(array_records) != 8:
            raise ValueError("precipitation cache must contain four arrays per process")
        self.arrays: dict[str, dict[str, np.ndarray]] = {"P0001": {}, "P0002": {}}
        for record in array_records:
            process = str(record.get("process", ""))
            kind = str(record.get("kind", ""))
            if process not in self.arrays or kind in self.arrays[process]:
                raise ValueError("invalid or duplicate process array")
            relative = str(record["path"])
            if relative not in self.payload_records:
                raise ValueError("array is absent from payload records")
            array = np.load(
                resolve_payload(self.root, relative), mmap_mode="r", allow_pickle=False
            )
            if list(array.shape) != list(record.get("shape", [])):
                raise ValueError(f"cached array shape mismatch: {relative}")
            if array.dtype.str != str(record.get("dtype")):
                raise ValueError(f"cached array dtype mismatch: {relative}")
            self.arrays[process][kind] = array
        expected_kinds = {"raw_mm", "log1p_mm", "rain_mask", "valid_mask"}
        if any(set(values) != expected_kinds for values in self.arrays.values()):
            raise ValueError("cache process arrays are incomplete")

        self.hours: dict[str, dict[str, Mapping[str, Any]]] = {}
        for process, rows in processes.items():
            if not isinstance(rows, list) or not rows:
                raise ValueError(f"empty cache hour index for {process}")
            lookup: dict[str, Mapping[str, Any]] = {}
            for row in rows:
                token = str(row.get("token", ""))
                index = int(row.get("index", -1))
                if token in lookup or index != len(lookup):
                    raise ValueError(f"non-contiguous or duplicate hour index for {process}")
                lookup[token] = row
            length = next(iter(self.arrays[process].values())).shape[0]
            if length != len(lookup):
                raise ValueError(f"array/hour index length mismatch for {process}")
            self.hours[process] = lookup
        fingerprint_files = self.data_fingerprint.get("files")
        if not isinstance(fingerprint_files, list):
            raise ValueError("cache data fingerprint has no source files")
        fingerprint_lookup: dict[tuple[str, str], Mapping[str, Any]] = {}
        for row in fingerprint_files:
            if not isinstance(row, dict):
                raise ValueError("invalid data fingerprint row")
            key = (str(row.get("process", "")), str(row.get("token", "")))
            if key in fingerprint_lookup:
                raise ValueError("duplicate data fingerprint process/hour")
            fingerprint_lookup[key] = row
        expected_keys = {
            (process, token) for process, lookup in self.hours.items() for token in lookup
        }
        if (
            set(fingerprint_lookup) != expected_keys
            or self.data_fingerprint.get("unique_precipitation_hour_count") != len(expected_keys)
            or self.manifest.get("unique_hour_count") != len(expected_keys)
        ):
            raise ValueError("data fingerprint and hour index scopes differ")
        for process, token in expected_keys:
            hour_row = self.hours[process][token]
            fingerprint_row = fingerprint_lookup[(process, token)]
            if (
                hour_row.get("source_relative_path") != fingerprint_row.get("relative_path")
                or hour_row.get("source_sha256") != fingerprint_row.get("sha256")
            ):
                raise ValueError(f"source evidence mismatch for {process}/{token}")
        coordinate_consistency = self.manifest.get("coordinate_consistency")
        if (
            not isinstance(coordinate_consistency, dict)
            or coordinate_consistency
            != self.data_fingerprint.get("coordinate_consistency")
        ):
            raise ValueError("manifest/data-fingerprint coordinate evidence differs")
        tolerance = float(coordinate_consistency.get("coordinate_tolerance_degrees", -1.0))
        if (
            not np.isfinite(tolerance)
            or tolerance < 0.0
            or tolerance
            != float(self.preprocess_config.get("coordinate_tolerance_degrees", np.nan))
            or coordinate_consistency.get("comparison")
            != "shape equality and numpy.allclose(rtol=0, atol=tolerance)"
        ):
            raise ValueError("cache coordinate tolerance evidence is invalid")
        coordinate_fields = (
            "latitude_exact_to_canonical",
            "longitude_exact_to_canonical",
            "coordinates_exact_to_canonical",
            "max_abs_latitude_drift_degrees",
            "max_abs_longitude_drift_degrees",
            "latitude_exact_to_process_canonical",
            "longitude_exact_to_process_canonical",
            "coordinates_exact_to_process_canonical",
            "max_abs_latitude_drift_to_process_canonical_degrees",
            "max_abs_longitude_drift_to_process_canonical_degrees",
            "coordinates_within_tolerance",
        )
        drift_fields = tuple(
            field for field in coordinate_fields if "drift" in field
        )
        for process, token in expected_keys:
            hour_row = self.hours[process][token]
            fingerprint_row = fingerprint_lookup[(process, token)]
            if any(hour_row.get(field) != fingerprint_row.get(field) for field in coordinate_fields):
                raise ValueError(f"per-file coordinate evidence mismatch for {process}/{token}")
            if fingerprint_row.get("coordinates_within_tolerance") is not True:
                raise ValueError(f"coordinate tolerance failed for {process}/{token}")
            for field in drift_fields:
                drift = float(fingerprint_row.get(field, np.nan))
                if not np.isfinite(drift) or drift < 0.0 or drift > tolerance:
                    raise ValueError(f"invalid coordinate drift for {process}/{token}: {field}")
        ordered_keys = sorted(expected_keys)
        first_process, first_token = ordered_keys[0]
        first_row = fingerprint_lookup[(first_process, first_token)]
        expected_canonical = {
            "process": first_process,
            "token": first_token,
            "relative_path": first_row["relative_path"],
        }
        process_coordinate_summary: dict[str, Any] = {}
        for process in sorted(self.hours):
            rows = [fingerprint_lookup[(process, token)] for token in self.hours[process]]
            process_coordinate_summary[process] = {
                "file_count": len(rows),
                "exact_to_global_canonical_count": sum(
                    bool(row["coordinates_exact_to_canonical"]) for row in rows
                ),
                "exact_to_process_canonical_count": sum(
                    bool(row["coordinates_exact_to_process_canonical"]) for row in rows
                ),
                "within_tolerance_count": sum(
                    bool(row["coordinates_within_tolerance"]) for row in rows
                ),
                "max_abs_latitude_drift_degrees": max(
                    float(row["max_abs_latitude_drift_degrees"]) for row in rows
                ),
                "max_abs_longitude_drift_degrees": max(
                    float(row["max_abs_longitude_drift_degrees"]) for row in rows
                ),
                "max_abs_latitude_drift_to_process_canonical_degrees": max(
                    float(row["max_abs_latitude_drift_to_process_canonical_degrees"])
                    for row in rows
                ),
                "max_abs_longitude_drift_to_process_canonical_degrees": max(
                    float(row["max_abs_longitude_drift_to_process_canonical_degrees"])
                    for row in rows
                ),
            }
        recomputed_coordinate_summary = {
            "schema_version": 1,
            "canonical_source": expected_canonical,
            "canonical_coordinate_policy": "first precipitation source in sorted process/token order",
            "comparison": "shape equality and numpy.allclose(rtol=0, atol=tolerance)",
            "coordinate_tolerance_degrees": tolerance,
            "file_count": len(expected_keys),
            "exact_to_global_canonical_count": sum(
                bool(row["coordinates_exact_to_canonical"])
                for row in fingerprint_lookup.values()
            ),
            "within_tolerance_count": sum(
                bool(row["coordinates_within_tolerance"])
                for row in fingerprint_lookup.values()
            ),
            "max_abs_latitude_drift_degrees": max(
                float(row["max_abs_latitude_drift_degrees"])
                for row in fingerprint_lookup.values()
            ),
            "max_abs_longitude_drift_degrees": max(
                float(row["max_abs_longitude_drift_degrees"])
                for row in fingerprint_lookup.values()
            ),
            "per_process": process_coordinate_summary,
        }
        if coordinate_consistency != recomputed_coordinate_summary:
            raise ValueError("cache coordinate consistency summary is not reproducible")
        folds = self.normalization.get("folds")
        expected_folds = {
            "A": ("P0001", "P0002"),
            "B": ("P0002", "P0001"),
        }
        if not isinstance(folds, dict) or set(folds) != set(expected_folds):
            raise ValueError("cache normalization must contain exactly folds A and B")
        for fold, (train_process, validation_process) in expected_folds.items():
            value = folds[fold]
            statistics = value.get("statistics") if isinstance(value, dict) else None
            if (
                not isinstance(value, dict)
                or value.get("train_processes") != [train_process]
                or value.get("validation_processes_excluded") != [validation_process]
                or value.get("source_hour_count") != len(self.hours[train_process])
                or not isinstance(statistics, dict)
                or int(statistics.get("count", 0)) <= 0
                or not np.isfinite(float(statistics.get("mean", np.nan)))
                or not np.isfinite(float(statistics.get("std", np.nan)))
                or float(statistics.get("std", 0.0)) <= 0.0
            ):
                raise ValueError(f"cache normalization evidence is invalid for fold {fold}")

    @property
    def cache_content_sha256(self) -> str:
        return str(self.manifest["cache_content_sha256"])

    @property
    def size_bytes(self) -> int:
        return sum(int(record["size_bytes"]) for record in self.payload_records.values())

    def get_hour(self, process: str, token: str) -> CachedPrecipitationHour:
        try:
            row = self.hours[process][token]
        except KeyError as exc:
            raise KeyError(f"precipitation hour is absent from cache: {process}/{token}") from exc
        index = int(row["index"])
        arrays = self.arrays[process]
        return CachedPrecipitationHour(
            raw_mm=arrays["raw_mm"][index],
            log1p_mm=arrays["log1p_mm"][index],
            rain_mask=arrays["rain_mask"][index],
            valid_mask=arrays["valid_mask"][index],
            latitude=self.latitude,
            longitude=self.longitude,
            process=process,
            token=token,
            source_relative_path=str(row["source_relative_path"]),
        )

    def get_hour_from_source(self, process: str, source_path: str | Path) -> CachedPrecipitationHour:
        token = extract_time_token(source_path)
        result = self.get_hour(process, token)
        relative = Path(str(source_path))
        parts = relative.parts
        if "TRAIN_DATA" in parts:
            relative = Path(*parts[parts.index("TRAIN_DATA") :])
            if relative.as_posix() != result.source_relative_path:
                raise ValueError("anchor source path and cache source evidence differ")
        return result

    def verify_hour_invariants(self, process: str, token: str) -> None:
        hour = self.get_hour(process, token)
        valid = np.asarray(hour.valid_mask, dtype=bool)
        rain = np.asarray(hour.rain_mask, dtype=bool)
        if np.any(np.isfinite(hour.raw_mm[~valid])):
            raise ValueError("invalid cached raw precipitation is not NaN")
        if np.any(np.isfinite(hour.log1p_mm[~valid])):
            raise ValueError("invalid cached log precipitation is not NaN")
        if np.any(~np.isfinite(hour.raw_mm[valid])) or np.any(hour.raw_mm[valid] < 0):
            raise ValueError("valid cached precipitation is corrupt")
        if np.any(rain & ~valid):
            raise ValueError("cached rain mask includes invalid pixels")
