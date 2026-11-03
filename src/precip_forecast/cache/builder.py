"""Transactional, time-indexed precipitation cache construction."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from precip_forecast.cache.manifest import (
    atomic_write_bytes,
    atomic_write_json,
    canonical_json_bytes,
    content_identifier,
    file_record,
    fsync_directory,
    render_sha256s,
    sha256_bytes,
    sha256_file,
    validate_cache_root,
)
from precip_forecast.features.precipitation import (
    FALLBACK_CANVAS_SHAPE,
    PrecipitationHour,
    read_precipitation_domain_mask,
    read_precipitation_file,
)
from precip_forecast.realdata.normalization import compute_fold_normalization
from precip_forecast.realdata.pseudotime import add_hours, extract_time_token


DEFAULT_CACHE_LIMIT_BYTES = 80 * 1024**3
DEFAULT_MINIMUM_FREE_BYTES = 300 * 1024**3
FULL_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")
ARRAY_KINDS: Mapping[str, np.dtype[Any]] = {
    "raw_mm": np.dtype("float32"),
    "log1p_mm": np.dtype("float32"),
    "rain_mask": np.dtype("bool"),
    "valid_mask": np.dtype("bool"),
}


def _is_within(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def collect_precipitation_sources(
    anchors: Sequence[Mapping[str, Any]], data_root: str | Path
) -> dict[str, list[dict[str, str]]]:
    """Deduplicate PRE inputs and targets by process/hour with path evidence."""

    root = Path(data_root).resolve()
    if not anchors:
        raise ValueError("cannot cache an empty anchor index")
    found: dict[tuple[str, str], Path] = {}
    for row in anchors:
        if not bool(row.get("eligible", False)):
            raise ValueError("cache builder accepts eligible anchor rows only")
        process = str(row.get("process", ""))
        if process not in {"P0001", "P0002"}:
            raise ValueError(f"unsupported sample process: {process!r}")
        anchor_token = str(row.get("anchor_token", ""))
        if list(row.get("target_leads", ())) != [4, 5, 6]:
            raise ValueError(f"anchor {row.get('sample_id')} has unsafe target leads")
        field_offsets = {
            "pre_input_paths": tuple(range(-6, 0)),
            "target_paths": (4, 5, 6),
        }
        paths: list[str] = []
        for field, offsets in field_offsets.items():
            values = row.get(field)
            if not isinstance(values, list) or len(values) != len(offsets):
                raise ValueError(f"anchor {row.get('sample_id')} has an incomplete {field}")
            actual_tokens = [extract_time_token(value) for value in values]
            expected_tokens = [add_hours(anchor_token, offset) for offset in offsets]
            if actual_tokens != expected_tokens:
                raise ValueError(f"anchor {row.get('sample_id')} has an unsafe {field} window")
            paths.extend(str(value) for value in values)
        for value in paths:
            path = Path(value).resolve()
            expected_parent = (root / "TRAIN_DATA" / process / "SURF_PRE").resolve()
            if (
                not path.is_file()
                or path.suffix.lower() not in {".nc", ".nc4"}
                or path.parent != expected_parent
                or not _is_within(root, path)
            ):
                raise ValueError(f"anchor PRE path violates the read-only data contract: {path}")
            token = extract_time_token(path)
            key = (process, token)
            prior = found.get(key)
            if prior is not None and prior != path:
                raise ValueError(f"two source files claim {process}/{token}")
            found[key] = path
    grouped: dict[str, list[dict[str, str]]] = {}
    for (process, token), path in sorted(found.items()):
        grouped.setdefault(process, []).append(
            {
                "token": token,
                "source_path": str(path),
                "source_relative_path": path.relative_to(root).as_posix(),
            }
        )
    if set(grouped) != {"P0001", "P0002"}:
        raise ValueError("both P0001 and P0002 are required for fold-safe sample caching")
    return grouped


def estimate_precipitation_cache_bytes(
    unique_hour_count: int,
    shape: tuple[int, int] = FALLBACK_CANVAS_SHAPE,
) -> dict[str, int]:
    if unique_hour_count <= 0 or len(shape) != 2 or min(shape) <= 0:
        raise ValueError("cache estimate requires positive hours and a 2-D shape")
    points = int(unique_hour_count) * int(shape[0]) * int(shape[1])
    array_bytes = points * sum(dtype.itemsize for dtype in ARRAY_KINDS.values())
    # NPY headers, coordinates, JSON, and conservative filesystem overhead.
    overhead = max(16 * 1024**2, int(array_bytes * 0.02))
    return {
        "unique_hour_count": int(unique_hour_count),
        "grid_points_per_hour": int(shape[0]) * int(shape[1]),
        "array_bytes": array_bytes,
        "overhead_bytes": overhead,
        "estimated_total_bytes": array_bytes + overhead,
    }


def _load_source_fingerprint(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("source fingerprint must be a JSON object")
    if int(value.get("verified_failed", -1)) != 0 or int(value.get("file_count", 0)) <= 0:
        raise ValueError("source fingerprint is not a successful non-empty verification")
    commit = str(value.get("declared_local_commit", ""))
    if not FULL_COMMIT_RE.fullmatch(commit):
        raise ValueError("source fingerprint lacks a full 40-character commit")
    return value


def _atomic_save_array(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        with temporary.open("wb") as handle:
            np.save(handle, values, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _open_memmap(path: Path, *, dtype: np.dtype[Any], shape: tuple[int, ...]) -> np.memmap:
    return np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=dtype,
        shape=shape,
        fortran_order=False,
    )


def _flush_memmap(value: np.memmap, path: Path) -> None:
    value.flush()
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_existing_cache(
    cache_root: Path,
    *,
    expected_config_sha256: str,
    expected_source_sha256: str,
) -> dict[str, Any]:
    from precip_forecast.cache.reader import PrecipitationCache

    completed = cache_root / "COMPLETED"
    manifest_path = cache_root / "manifest.json"
    if not completed.is_file() or not manifest_path.is_file():
        raise FileExistsError(f"cache target exists but is incomplete: {cache_root}")
    manifest = PrecipitationCache(cache_root, verify="all").manifest
    if manifest.get("preprocess_config_sha256") != expected_config_sha256:
        raise FileExistsError("completed cache was built with a different preprocess config")
    if manifest.get("source_fingerprint_sha256") != expected_source_sha256:
        raise FileExistsError("completed cache was built from a different source fingerprint")
    return manifest


def build_precipitation_cache(
    *,
    anchors: Sequence[Mapping[str, Any]],
    data_root: str | Path,
    cache_root: str | Path,
    source_fingerprint_path: str | Path,
    preprocess_config: Mapping[str, Any],
    cache_limit_bytes: int = DEFAULT_CACHE_LIMIT_BYTES,
    minimum_free_bytes: int = DEFAULT_MINIMUM_FREE_BYTES,
    required_cache_root: str | Path | None = None,
    cache_namespace: str | None = None,
    canvas_shape: tuple[int, int] = FALLBACK_CANVAS_SHAPE,
    domain_mask_path: str | Path | None = None,
    domain_mask_array: np.ndarray | None = None,
    coordinate_tolerance_degrees: float = 2.5e-5,
    hour_loader: Callable[[str | Path], PrecipitationHour] = read_precipitation_file,
    reuse_completed: bool = True,
) -> dict[str, Any]:
    """Build the entire precipitation cache in a sibling staging directory.

    The read-only source is never opened for writing.  A target becomes
    readable only after every payload is flushed, checksummed, and atomically
    renamed together with a ``COMPLETED`` marker.
    """

    cache_container, data = validate_cache_root(
        cache_root, data_root, required_root=required_cache_root
    )
    if cache_namespace is not None:
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", cache_namespace):
            raise ValueError("cache_namespace must be a simple lowercase identifier")
        cache = cache_container / cache_namespace
    else:
        cache = cache_container
    if not data.is_dir():
        raise ValueError(f"read-only data root does not exist: {data}")
    if cache_limit_bytes <= 0 or minimum_free_bytes < 0:
        raise ValueError("cache limits must be non-negative and cache limit positive")
    if not np.isfinite(coordinate_tolerance_degrees) or coordinate_tolerance_degrees < 0.0:
        raise ValueError("coordinate tolerance must be finite and non-negative")
    if str(preprocess_config.get("modality", "")) != "precipitation":
        raise ValueError("initial cache implementation is precipitation-only")
    canvas_shape = tuple(int(value) for value in canvas_shape)
    if len(canvas_shape) != 2 or min(canvas_shape) <= 0:
        raise ValueError("canvas_shape must contain two positive dimensions")
    if tuple(preprocess_config.get("canvas_shape", ())) != canvas_shape:
        raise ValueError("preprocess canvas_shape and builder canvas_shape differ")
    if required_cache_root is not None and canvas_shape != FALLBACK_CANVAS_SHAPE:
        raise ValueError("approved sample cache requires the audited PRE 601x901 canvas")
    if (domain_mask_path is None) == (domain_mask_array is None):
        raise ValueError("provide exactly one of domain_mask_path or domain_mask_array")
    mask_source: Path | None = None
    mask_source_evidence: dict[str, Any] | None = None
    if domain_mask_path is not None:
        mask_source = Path(domain_mask_path).resolve()
        expected_mask = (data / "guangxi_mask.nc").resolve()
        if mask_source != expected_mask or not mask_source.is_file():
            raise ValueError(f"domain mask must be the audited source {expected_mask}")
        mask_stat_before = mask_source.stat()
        mask_digest = sha256_file(mask_source)
        mask_stat_after = mask_source.stat()
        if (mask_stat_before.st_size, mask_stat_before.st_mtime_ns) != (
            mask_stat_after.st_size,
            mask_stat_after.st_mtime_ns,
        ):
            raise RuntimeError("domain mask changed while hashing")
        mask_source_evidence = {
            "relative_path": mask_source.relative_to(data).as_posix(),
            "size_bytes": mask_stat_after.st_size,
            "mtime_ns": mask_stat_after.st_mtime_ns,
            "sha256": mask_digest,
        }
    elif required_cache_root is not None:
        raise ValueError("approved sample cache requires the audited guangxi_mask.nc")

    sources = collect_precipitation_sources(anchors, data)
    hour_count = sum(len(items) for items in sources.values())
    estimate = estimate_precipitation_cache_bytes(hour_count, canvas_shape)
    if estimate["estimated_total_bytes"] > cache_limit_bytes:
        raise ValueError("estimated precipitation cache exceeds configured cache limit")

    source_fingerprint_file = Path(source_fingerprint_path)
    source_fingerprint = _load_source_fingerprint(source_fingerprint_file)
    source_bytes = source_fingerprint_file.read_bytes()
    source_sha = sha256_bytes(source_bytes)
    resolved_config = dict(preprocess_config)
    resolved_config.update(
        {
            "schema_version": 1,
            "modality": "precipitation",
            "data_root": str(data),
            "canvas_shape": list(canvas_shape),
            "cache_layout": "deduplicated_process_hour_arrays_v1",
            "cache_namespace": cache_namespace,
            "invalid_continuous_storage": "NaN_with_separate_valid_mask",
            "valid_mask_semantics": (
                "cached valid_mask is source validity; model validity is "
                "source_valid_mask AND mapped_domain_mask"
            ),
            "rain_threshold_mm": 0.1,
            "domain_mask": mask_source_evidence or "explicit_test_array",
            "coordinate_tolerance_degrees": coordinate_tolerance_degrees,
        }
    )
    config_bytes = canonical_json_bytes(resolved_config)
    config_sha = sha256_bytes(config_bytes)

    if cache.exists():
        if reuse_completed:
            return _validate_existing_cache(
                cache,
                expected_config_sha256=config_sha,
                expected_source_sha256=source_sha,
            )
        raise FileExistsError(f"cache target already exists: {cache}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    free_before = shutil.disk_usage(cache.parent).free
    if free_before < minimum_free_bytes:
        raise OSError(
            f"free space {free_before} is below required floor {minimum_free_bytes}"
        )

    stage = Path(
        tempfile.mkdtemp(prefix=f".{cache.name}.build.", dir=str(cache.parent))
    ).resolve()
    if stage.parent != cache.parent or not stage.name.startswith(f".{cache.name}.build."):
        raise RuntimeError("temporary cache directory failed path validation")
    started = datetime.now(timezone.utc)
    try:
        atomic_write_bytes(stage / "source_fingerprint.json", source_bytes)
        atomic_write_bytes(stage / "preprocess_config.json", config_bytes)
        coordinate_hashes: dict[str, str] = {}
        data_fingerprint_rows: list[dict[str, Any]] = []
        hours_manifest: dict[str, list[dict[str, Any]]] = {}
        array_metadata: list[dict[str, Any]] = []
        normalization_inputs: list[tuple[str, np.ndarray, np.ndarray]] = []
        reference_latitude: np.ndarray | None = None
        reference_longitude: np.ndarray | None = None
        canonical_coordinate_source: dict[str, str] | None = None
        process_coordinate_references: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        mapped_domain_mask: np.ndarray | None = None
        domain_mapping_evidence: dict[str, Any] | None = None

        for process, items in sorted(sources.items()):
            shape = (len(items), *canvas_shape)
            arrays: dict[str, np.memmap] = {}
            array_paths: dict[str, Path] = {}
            for kind, dtype in ARRAY_KINDS.items():
                path = stage / f"{process}_{kind}.npy"
                arrays[kind] = _open_memmap(path, dtype=dtype, shape=shape)
                array_paths[kind] = path
            process_hours: list[dict[str, Any]] = []
            for index, item in enumerate(items):
                source_path = Path(item["source_path"])
                stat_before = source_path.stat()
                decoded = hour_loader(source_path)
                stat_after = source_path.stat()
                if (stat_before.st_size, stat_before.st_mtime_ns) != (
                    stat_after.st_size,
                    stat_after.st_mtime_ns,
                ):
                    raise RuntimeError(f"source changed while caching: {source_path}")
                if decoded.raw_mm.shape != canvas_shape:
                    raise ValueError(f"unexpected PRE canvas for {source_path}")
                if reference_latitude is None:
                    reference_latitude = np.asarray(decoded.latitude, dtype=np.float64).copy()
                    reference_longitude = np.asarray(decoded.longitude, dtype=np.float64).copy()
                    canonical_coordinate_source = {
                        "process": process,
                        "token": item["token"],
                        "relative_path": item["source_relative_path"],
                    }
                    if mask_source is not None:
                        mapped_domain_mask, domain_mapping_evidence = read_precipitation_domain_mask(
                            mask_source,
                            target_latitude=reference_latitude,
                            target_longitude=reference_longitude,
                            coordinate_tolerance=coordinate_tolerance_degrees,
                        )
                    else:
                        mapped_domain_mask = np.asarray(domain_mask_array, dtype=bool)
                        if mapped_domain_mask.shape != canvas_shape or not np.any(
                            mapped_domain_mask
                        ):
                            raise ValueError("explicit domain mask is empty or has the wrong shape")
                        domain_mapping_evidence = {
                            "source": "explicit_test_array",
                            "mapped_shape": list(mapped_domain_mask.shape),
                            "mapped_true_count": int(np.count_nonzero(mapped_domain_mask)),
                        }
                decoded_latitude = np.asarray(decoded.latitude, dtype=np.float64)
                decoded_longitude = np.asarray(decoded.longitude, dtype=np.float64)
                if (
                    decoded_latitude.shape != reference_latitude.shape
                    or decoded_longitude.shape != reference_longitude.shape
                ):
                    raise ValueError(f"PRE coordinate shape changed across hours: {source_path}")
                if process not in process_coordinate_references:
                    process_coordinate_references[process] = (
                        decoded_latitude.copy(),
                        decoded_longitude.copy(),
                    )
                process_latitude, process_longitude = process_coordinate_references[process]
                latitude_drift = float(
                    np.max(np.abs(decoded_latitude - reference_latitude), initial=0.0)
                )
                longitude_drift = float(
                    np.max(np.abs(decoded_longitude - reference_longitude), initial=0.0)
                )
                process_latitude_drift = float(
                    np.max(np.abs(decoded_latitude - process_latitude), initial=0.0)
                )
                process_longitude_drift = float(
                    np.max(np.abs(decoded_longitude - process_longitude), initial=0.0)
                )
                latitude_exact = bool(np.array_equal(reference_latitude, decoded_latitude))
                longitude_exact = bool(np.array_equal(reference_longitude, decoded_longitude))
                process_latitude_exact = bool(
                    np.array_equal(process_latitude, decoded_latitude)
                )
                process_longitude_exact = bool(
                    np.array_equal(process_longitude, decoded_longitude)
                )
                coordinates_within_tolerance = bool(
                    np.allclose(
                        reference_latitude,
                        decoded_latitude,
                        rtol=0.0,
                        atol=coordinate_tolerance_degrees,
                    )
                    and np.allclose(
                        reference_longitude,
                        decoded_longitude,
                        rtol=0.0,
                        atol=coordinate_tolerance_degrees,
                    )
                )
                if not coordinates_within_tolerance:
                    raise ValueError(
                        "PRE coordinate drift exceeds tolerance "
                        f"{coordinate_tolerance_degrees:.9g}: {source_path}; "
                        f"latitude={latitude_drift:.9g}, longitude={longitude_drift:.9g}"
                    )
                coordinate_evidence = {
                    "latitude_exact_to_canonical": latitude_exact,
                    "longitude_exact_to_canonical": longitude_exact,
                    "coordinates_exact_to_canonical": latitude_exact and longitude_exact,
                    "max_abs_latitude_drift_degrees": latitude_drift,
                    "max_abs_longitude_drift_degrees": longitude_drift,
                    "latitude_exact_to_process_canonical": process_latitude_exact,
                    "longitude_exact_to_process_canonical": process_longitude_exact,
                    "coordinates_exact_to_process_canonical": (
                        process_latitude_exact and process_longitude_exact
                    ),
                    "max_abs_latitude_drift_to_process_canonical_degrees": (
                        process_latitude_drift
                    ),
                    "max_abs_longitude_drift_to_process_canonical_degrees": (
                        process_longitude_drift
                    ),
                    "coordinates_within_tolerance": coordinates_within_tolerance,
                }
                arrays["raw_mm"][index] = decoded.raw_mm
                arrays["log1p_mm"][index] = decoded.log1p_mm
                arrays["rain_mask"][index] = decoded.rain_mask
                arrays["valid_mask"][index] = decoded.valid_mask
                normalization_inputs.append(
                    (
                        process,
                        arrays["log1p_mm"][index],
                        np.asarray(arrays["valid_mask"][index], dtype=bool)
                        & mapped_domain_mask,
                    )
                )
                source_digest = sha256_file(source_path)
                stat_final = source_path.stat()
                if (stat_after.st_size, stat_after.st_mtime_ns) != (
                    stat_final.st_size,
                    stat_final.st_mtime_ns,
                ):
                    raise RuntimeError(f"source changed while hashing: {source_path}")
                data_fingerprint_rows.append(
                    {
                        "process": process,
                        "token": item["token"],
                        "relative_path": item["source_relative_path"],
                        "size_bytes": stat_final.st_size,
                        "mtime_ns": stat_final.st_mtime_ns,
                        "sha256": source_digest,
                        **coordinate_evidence,
                    }
                )
                process_hours.append(
                    {
                        "index": index,
                        "token": item["token"],
                        "source_relative_path": item["source_relative_path"],
                        "source_sha256": source_digest,
                        "valid_count": int(np.count_nonzero(decoded.valid_mask)),
                        "rain_count": int(np.count_nonzero(decoded.rain_mask)),
                        **coordinate_evidence,
                    }
                )
            hours_manifest[process] = process_hours
            for kind, array in arrays.items():
                _flush_memmap(array, array_paths[kind])
                del array
                record = file_record(stage, array_paths[kind])
                record.update(
                    {
                        "kind": kind,
                        "process": process,
                        "dtype": ARRAY_KINDS[kind].str,
                        "shape": list(shape),
                    }
                )
                array_metadata.append(record)

        if (
            reference_latitude is None
            or reference_longitude is None
            or canonical_coordinate_source is None
            or mapped_domain_mask is None
            or domain_mapping_evidence is None
        ):
            raise RuntimeError("cache build did not establish PRE coordinates and domain mask")
        if mask_source is not None:
            final_mask_stat = mask_source.stat()
            if (
                final_mask_stat.st_size != mask_source_evidence["size_bytes"]
                or final_mask_stat.st_mtime_ns != mask_source_evidence["mtime_ns"]
            ):
                raise RuntimeError("domain mask changed while cache was built")
        _atomic_save_array(stage / "latitude.npy", reference_latitude)
        _atomic_save_array(stage / "longitude.npy", reference_longitude)
        _atomic_save_array(stage / "domain_mask.npy", mapped_domain_mask)
        coordinate_hashes = {
            "latitude_sha256": sha256_file(stage / "latitude.npy"),
            "longitude_sha256": sha256_file(stage / "longitude.npy"),
            "domain_mask_sha256": sha256_file(stage / "domain_mask.npy"),
        }
        coordinate_process_summary: dict[str, Any] = {}
        for process in sorted(hours_manifest):
            rows = hours_manifest[process]
            coordinate_process_summary[process] = {
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
        coordinate_consistency = {
            "schema_version": 1,
            "canonical_source": canonical_coordinate_source,
            "canonical_coordinate_policy": "first precipitation source in sorted process/token order",
            "comparison": "shape equality and numpy.allclose(rtol=0, atol=tolerance)",
            "coordinate_tolerance_degrees": coordinate_tolerance_degrees,
            "file_count": len(data_fingerprint_rows),
            "exact_to_global_canonical_count": sum(
                bool(row["coordinates_exact_to_canonical"])
                for row in data_fingerprint_rows
            ),
            "within_tolerance_count": sum(
                bool(row["coordinates_within_tolerance"])
                for row in data_fingerprint_rows
            ),
            "max_abs_latitude_drift_degrees": max(
                float(row["max_abs_latitude_drift_degrees"])
                for row in data_fingerprint_rows
            ),
            "max_abs_longitude_drift_degrees": max(
                float(row["max_abs_longitude_drift_degrees"])
                for row in data_fingerprint_rows
            ),
            "per_process": coordinate_process_summary,
        }
        atomic_write_json(stage / "hours.json", {"schema_version": 1, "processes": hours_manifest})
        data_fingerprint = {
            "schema_version": 1,
            "data_root": str(data),
            "data_access": "read_only",
            "unique_precipitation_hour_count": hour_count,
            "files": sorted(data_fingerprint_rows, key=lambda row: (row["process"], row["token"])),
            "coordinate_consistency": coordinate_consistency,
            "domain_mask_source": mask_source_evidence or {
                "source": "explicit_test_array",
                "sha256": coordinate_hashes["domain_mask_sha256"],
            },
        }
        atomic_write_json(stage / "data_fingerprint.json", data_fingerprint)
        normalization = compute_fold_normalization(normalization_inputs)
        normalization["validity_rule"] = "source_valid_mask AND mapped_domain_mask"
        normalization["domain_mask_sha256"] = coordinate_hashes["domain_mask_sha256"]
        atomic_write_json(stage / "normalization.json", normalization)

        metadata_paths = [
            stage / "source_fingerprint.json",
            stage / "preprocess_config.json",
            stage / "data_fingerprint.json",
            stage / "normalization.json",
            stage / "hours.json",
            stage / "latitude.npy",
            stage / "longitude.npy",
            stage / "domain_mask.npy",
        ]
        payload_records = array_metadata + [file_record(stage, path) for path in metadata_paths]
        by_path = {str(record["path"]): record for record in payload_records}
        data_sha = by_path["data_fingerprint.json"]["sha256"]
        normalization_sha = by_path["normalization.json"]["sha256"]
        cache_id = content_identifier(
            preprocess_config_sha256=config_sha,
            source_fingerprint_sha256=source_sha,
            data_fingerprint_sha256=data_sha,
            normalization_sha256=normalization_sha,
            payload_records=payload_records,
        )
        free_after_payloads = shutil.disk_usage(stage).free
        if free_after_payloads < minimum_free_bytes:
            raise OSError(
                f"free space after cache payloads {free_after_payloads} is below "
                f"required floor {minimum_free_bytes}"
            )
        elapsed = (datetime.now(timezone.utc) - started).total_seconds()
        manifest = {
            "schema_version": 1,
            "cache_content_sha256": cache_id,
            "status": "complete",
            "modality": "precipitation",
            "layout": "deduplicated_process_hour_arrays_v1",
            "canvas_shape": list(canvas_shape),
            "processes": hours_manifest,
            "unique_hour_count": hour_count,
            "source_commit": source_fingerprint["declared_local_commit"].lower(),
            "source_fingerprint_sha256": source_sha,
            "data_fingerprint_sha256": data_sha,
            "preprocess_config_sha256": config_sha,
            "normalization_sha256": normalization_sha,
            "coordinate_hashes": coordinate_hashes,
            "coordinate_consistency": coordinate_consistency,
            "domain_mask": domain_mapping_evidence,
            "valid_mask_semantics": resolved_config["valid_mask_semantics"],
            "arrays": array_metadata,
            "payloads": payload_records,
            "estimated_total_bytes": estimate["estimated_total_bytes"],
            "actual_payload_bytes": sum(int(record["size_bytes"]) for record in payload_records),
            "free_bytes_before": free_before,
            "free_bytes_after_payloads": free_after_payloads,
            "minimum_free_bytes": minimum_free_bytes,
            "build_elapsed_seconds": elapsed,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        if manifest["actual_payload_bytes"] > cache_limit_bytes:
            raise ValueError("built precipitation cache exceeds configured cache limit")
        atomic_write_json(stage / "manifest.json", manifest)
        manifest_sha = sha256_file(stage / "manifest.json")
        sums_text = render_sha256s(payload_records)
        atomic_write_bytes(stage / "SHA256SUMS.txt", sums_text.encode("utf-8"))
        marker = {
            "schema_version": 1,
            "cache_content_sha256": cache_id,
            "manifest_sha256": manifest_sha,
            "sha256s_sha256": sha256_file(stage / "SHA256SUMS.txt"),
        }
        atomic_write_json(stage / "COMPLETED", marker)
        fsync_directory(stage)
        if cache.exists():
            raise FileExistsError(f"cache target appeared during build: {cache}")
        os.rename(stage, cache)
        fsync_directory(cache.parent)
        return manifest
    except BaseException:
        if stage.exists() and stage.parent == cache.parent and stage.name.startswith(
            f".{cache.name}.build."
        ):
            shutil.rmtree(stage)
        raise
