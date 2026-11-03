"""Strict precipitation feature decoding for the released sample.

Missing observations and physical zero rainfall are deliberately different:
cached continuous fields contain ``NaN`` at invalid pixels, while every field
is accompanied by an explicit boolean validity mask.  Filling invalid values
for a neural-network input is a later, mask-aware dataset operation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


PRECIPITATION_VARIABLE = "PRE_HOR"
RAIN_THRESHOLD_MM = 0.1
FALLBACK_CANVAS_SHAPE = (601, 901)
_ACCEPTED_MM_UNITS = {"mm", "millimeter", "millimeters", "millimetre", "millimetres"}


@dataclass(frozen=True)
class PrecipitationHour:
    """Decoded precipitation fields and the source-coordinate evidence."""

    raw_mm: np.ndarray
    log1p_mm: np.ndarray
    rain_mask: np.ndarray
    valid_mask: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    source_path: str
    variable_name: str = PRECIPITATION_VARIABLE

    def __post_init__(self) -> None:
        shape = self.raw_mm.shape
        if len(shape) != 2:
            raise ValueError(f"precipitation hour must be two-dimensional, got {shape}")
        for name in ("log1p_mm", "rain_mask", "valid_mask"):
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} shape differs from raw_mm")
        if self.latitude.shape != (shape[0],) or self.longitude.shape != (shape[1],):
            raise ValueError("coordinate vectors do not match precipitation shape")
        if self.raw_mm.dtype != np.float32 or self.log1p_mm.dtype != np.float32:
            raise TypeError("continuous precipitation features must be float32")
        if self.rain_mask.dtype != np.bool_ or self.valid_mask.dtype != np.bool_:
            raise TypeError("precipitation masks must be boolean")
        if np.any(np.isfinite(self.raw_mm[~self.valid_mask])):
            raise ValueError("invalid raw precipitation must remain NaN")
        if np.any(np.isfinite(self.log1p_mm[~self.valid_mask])):
            raise ValueError("invalid log precipitation must remain NaN")
        if np.any(~np.isfinite(self.raw_mm[self.valid_mask])):
            raise ValueError("valid raw precipitation contains NaN or Inf")
        if np.any(self.raw_mm[self.valid_mask] < 0.0):
            raise ValueError("valid precipitation cannot be negative")
        if np.any(self.rain_mask & ~self.valid_mask):
            raise ValueError("rain mask marks an invalid pixel as rainy")


def _squeeze_singleton_leading_dimensions(values: Any) -> np.ma.MaskedArray:
    result = np.ma.asarray(values)
    while result.ndim > 2 and result.shape[0] == 1:
        result = result[0]
    if result.ndim != 2:
        raise ValueError(
            "PRE_HOR must be a 2-D field with optional singleton leading dimensions; "
            f"got {result.shape}"
        )
    return result


def prepare_precipitation(
    values: Any,
    *,
    explicit_valid_mask: Any | None = None,
    rain_threshold_mm: float = RAIN_THRESHOLD_MM,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decode one precipitation field without conflating missing and zero.

    Masked, non-finite, and negative values are invalid.  Negative values are
    not clipped because doing so would silently turn corrupt/missing values
    into dry observations.
    """

    if not np.isfinite(rain_threshold_mm) or rain_threshold_mm < 0.0:
        raise ValueError("rain_threshold_mm must be finite and non-negative")
    masked = _squeeze_singleton_leading_dimensions(values).astype(np.float64)
    data = np.asarray(masked.filled(np.nan), dtype=np.float64)
    valid = ~np.ma.getmaskarray(masked)
    valid &= np.isfinite(data)
    valid &= data >= 0.0
    if explicit_valid_mask is not None:
        supplied = np.asarray(explicit_valid_mask, dtype=bool)
        if supplied.shape != data.shape:
            raise ValueError("explicit_valid_mask shape differs from precipitation")
        valid &= supplied

    raw_mm = np.full(data.shape, np.nan, dtype=np.float32)
    raw_mm[valid] = data[valid].astype(np.float32, copy=False)
    log1p_mm = np.full(data.shape, np.nan, dtype=np.float32)
    log1p_mm[valid] = np.log1p(data[valid]).astype(np.float32, copy=False)
    rain_mask = np.zeros(data.shape, dtype=bool)
    rain_mask[valid] = data[valid] >= rain_threshold_mm
    return raw_mm, log1p_mm, rain_mask, valid.astype(bool, copy=False)


def _coordinate_variable(dataset: Any, kind: str) -> Any:
    candidates = {
        "latitude": ("latitude", "lat", "LAT", "Latitude"),
        "longitude": ("longitude", "lon", "LON", "Longitude"),
    }[kind]
    for name in candidates:
        if name in dataset.variables:
            variable = dataset.variables[name]
            if getattr(variable, "ndim", len(variable.shape)) == 1:
                return variable
    standard_name = kind
    found = [
        variable
        for variable in dataset.variables.values()
        if str(getattr(variable, "standard_name", "")).strip().lower() == standard_name
        and len(variable.shape) == 1
    ]
    if len(found) != 1:
        raise ValueError(f"could not identify a unique 1-D {kind} coordinate")
    return found[0]


def _validate_coordinate(values: Any, *, name: str, expected_size: int) -> np.ndarray:
    coordinate = np.asarray(values, dtype=np.float64)
    if expected_size < 2:
        raise ValueError(f"{name} coordinate must contain at least two points")
    if coordinate.shape != (expected_size,):
        raise ValueError(f"{name} coordinate shape {coordinate.shape} != {(expected_size,)}")
    if not np.all(np.isfinite(coordinate)):
        raise ValueError(f"{name} coordinate contains NaN or Inf")
    differences = np.diff(coordinate)
    if not (np.all(differences > 0.0) or np.all(differences < 0.0)):
        raise ValueError(f"{name} coordinate is not strictly monotonic")
    coordinate.setflags(write=False)
    return coordinate


def _nearest_coordinate_indices(
    source: np.ndarray, target: np.ndarray, *, tolerance: float
) -> tuple[np.ndarray, float]:
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("coordinate tolerance must be finite and non-negative")
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if (
        source.ndim != 1
        or target.ndim != 1
        or source.size < 2
        or target.size < 1
        or not np.all(np.isfinite(source))
        or not np.all(np.isfinite(target))
    ):
        raise ValueError("coordinate mapping requires finite non-empty 1-D vectors")
    order = np.argsort(source)
    sorted_source = source[order]
    if np.any(np.diff(sorted_source) <= 0.0):
        raise ValueError("source coordinate must be strictly monotonic")
    insertion = np.searchsorted(sorted_source, target)
    upper = np.clip(insertion, 0, sorted_source.size - 1)
    lower = np.clip(insertion - 1, 0, sorted_source.size - 1)
    choose_upper = np.abs(sorted_source[upper] - target) < np.abs(
        sorted_source[lower] - target
    )
    selected_sorted = np.where(choose_upper, upper, lower)
    errors = np.abs(sorted_source[selected_sorted] - target)
    maximum = float(errors.max(initial=0.0))
    if maximum > tolerance:
        raise ValueError(
            f"domain mask coordinate error {maximum:.9g} exceeds tolerance {tolerance:.9g}"
        )
    return order[selected_sorted], maximum


def read_precipitation_domain_mask(
    path: str | Path,
    *,
    target_latitude: Any,
    target_longitude: Any,
    coordinate_tolerance: float = 2.5e-5,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Map the official mask to PRE coordinates without shape-based resizing."""

    from netCDF4 import Dataset

    source_path = Path(path).resolve()
    if not source_path.is_file() or source_path.suffix.lower() not in {".nc", ".nc4"}:
        raise ValueError(f"domain mask is not a NetCDF file: {source_path}")
    target_lat_array = np.asarray(target_latitude)
    target_lon_array = np.asarray(target_longitude)
    if target_lat_array.ndim != 1 or target_lon_array.ndim != 1:
        raise ValueError("target PRE coordinates must be one-dimensional")
    target_lat = _validate_coordinate(
        target_lat_array,
        name="target latitude",
        expected_size=target_lat_array.size,
    )
    target_lon = _validate_coordinate(
        target_lon_array,
        name="target longitude",
        expected_size=target_lon_array.size,
    )
    with Dataset(str(source_path), mode="r") as dataset:
        latitude_variable = _coordinate_variable(dataset, "latitude")
        longitude_variable = _coordinate_variable(dataset, "longitude")
        source_lat = _validate_coordinate(
            latitude_variable[:],
            name="mask latitude",
            expected_size=latitude_variable.shape[0],
        )
        source_lon = _validate_coordinate(
            longitude_variable[:],
            name="mask longitude",
            expected_size=longitude_variable.shape[0],
        )
        candidates = [
            (name, variable)
            for name, variable in dataset.variables.items()
            if tuple(variable.shape) == (source_lat.size, source_lon.size)
        ]
        if "mask" in dataset.variables and tuple(dataset.variables["mask"].shape) == (
            source_lat.size,
            source_lon.size,
        ):
            mask_name, variable = "mask", dataset.variables["mask"]
        elif len(candidates) == 1:
            mask_name, variable = candidates[0]
        else:
            raise ValueError(
                f"could not identify a unique 2-D domain mask; candidates={[name for name, _ in candidates]}"
            )
        raw = np.ma.asarray(variable[:])
        data = np.asarray(raw.filled(0), dtype=np.float64)
        valid = ~np.ma.getmaskarray(raw) & np.isfinite(data)
        observed = set(np.unique(data[valid]).tolist())
        if not observed.issubset({0.0, 1.0}):
            raise ValueError(f"unconfirmed domain mask values: {sorted(observed)[:20]}")
        source_mask = valid & (data == 1.0)
    lat_indices, lat_error = _nearest_coordinate_indices(
        source_lat, target_lat, tolerance=coordinate_tolerance
    )
    lon_indices, lon_error = _nearest_coordinate_indices(
        source_lon, target_lon, tolerance=coordinate_tolerance
    )
    mapped = np.asarray(source_mask[np.ix_(lat_indices, lon_indices)], dtype=bool)
    if mapped.shape != (target_lat.size, target_lon.size) or not np.any(mapped):
        raise ValueError("mapped PRE domain mask is empty or has the wrong shape")
    return mapped, {
        "source_path": str(source_path),
        "source_variable": mask_name,
        "source_shape": [int(source_lat.size), int(source_lon.size)],
        "mapped_shape": list(mapped.shape),
        "mapped_true_count": int(np.count_nonzero(mapped)),
        "coordinate_tolerance_degrees": coordinate_tolerance,
        "maximum_latitude_error_degrees": lat_error,
        "maximum_longitude_error_degrees": lon_error,
        "mapping": "nearest coordinate with explicit tolerance; no shape resize",
    }


def read_precipitation_file(
    path: str | Path,
    *,
    expected_shape: tuple[int, int] | None = FALLBACK_CANVAS_SHAPE,
    require_mm_units: bool = True,
) -> PrecipitationHour:
    """Read one audited ``PRE_HOR`` NetCDF file.

    The import is local so metadata-only tools and unit tests can import this
    module without requiring netCDF4 on the local backup computer.
    """

    from netCDF4 import Dataset

    source = Path(path).resolve()
    if not source.is_file() or source.suffix.lower() not in {".nc", ".nc4"}:
        raise ValueError(f"precipitation source is not a NetCDF file: {source}")
    with Dataset(str(source), mode="r") as dataset:
        if PRECIPITATION_VARIABLE not in dataset.variables:
            raise ValueError(f"{PRECIPITATION_VARIABLE} is absent: {source}")
        variable = dataset.variables[PRECIPITATION_VARIABLE]
        units = str(getattr(variable, "units", "")).strip().lower()
        if require_mm_units and units not in _ACCEPTED_MM_UNITS:
            raise ValueError(f"unconfirmed precipitation units {units!r}: {source}")
        raw = _squeeze_singleton_leading_dimensions(variable[:])
        if expected_shape is not None and raw.shape != tuple(expected_shape):
            raise ValueError(
                f"precipitation shape {raw.shape} differs from audited canvas {expected_shape}: {source}"
            )
        latitude = _validate_coordinate(
            _coordinate_variable(dataset, "latitude")[:],
            name="latitude",
            expected_size=raw.shape[0],
        )
        longitude = _validate_coordinate(
            _coordinate_variable(dataset, "longitude")[:],
            name="longitude",
            expected_size=raw.shape[1],
        )
        raw_mm, log1p_mm, rain_mask, valid_mask = prepare_precipitation(raw)
    return PrecipitationHour(
        raw_mm=raw_mm,
        log1p_mm=log1p_mm,
        rain_mask=rain_mask,
        valid_mask=valid_mask,
        latitude=latitude,
        longitude=longitude,
        source_path=str(source),
    )
