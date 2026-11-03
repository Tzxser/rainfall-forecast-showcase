"""Bounded-memory Radar Lite feature construction.

The competition radar files contain ten six-minute volumes per file and each
volume has 17 vertical levels.  This module deliberately reduces one spatial
tile and one time volume at a time; it has no API that concatenates the four
hourly files into a ``40 x 17 x H x W`` raw tensor.

Missing data and conservative no-positive-echo values are separate throughout:

* masked, non-finite, and declared fill values are invalid;
* finite negative dBZ values remain metadata-valid candidates, but their signal
  contribution is conservatively floored at 0 dBZ until official semantics are
  documented;
* ``below_zero_fraction`` records those candidates;
* every value field has an explicit validity mask, so a finite output zero is
  never silently treated as observed data.

The reducer is array-only.  File selection, availability-time checks, NetCDF
chunk reads, and cache writes belong to the real-data adapter/cache layers.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from precip_forecast.realdata.grids import GridSpec
from precip_forecast.realdata.target_mapping import GridMap


RADAR_LITE_FEATURE_VERSION = "radar-lite-v1"
DEFAULT_FILL_VALUES = (-999.0,)
DEFAULT_WINDOWS_MINUTES = (30, 60, 180)
REFLECTIVITY_FEATURES = (
    "zmax",
    "low_max",
    "mid_max",
    "high_max",
)
ECHO_TOP_FEATURES = ("etop18", "etop35")
INSTANTANEOUS_FEATURES = (
    *REFLECTIVITY_FEATURES,
    *ECHO_TOP_FEATURES,
    "valid_fraction",
    "below_zero_fraction",
)


def _summary_feature_names() -> tuple[str, ...]:
    names = [f"latest_{name}" for name in INSTANTANEOUS_FEATURES]
    for window in DEFAULT_WINDOWS_MINUTES:
        prefix = f"w{window:03d}"
        names.extend(f"{prefix}_{name}_max" for name in INSTANTANEOUS_FEATURES)
        names.append(f"{prefix}_zmax_mean")
        names.append(f"{prefix}_time_valid_fraction")
    names.extend(
        (
            "w060_zmax_trend_per_hour",
            "w060_etop35_trend_per_hour",
        )
    )
    return tuple(names)


RADAR_LITE_SUMMARY_FEATURES = _summary_feature_names()


def radar_lite_feature_spec() -> dict[str, Any]:
    """Return a fresh canonical description used to version cached features."""

    return {
        "schema": "precip-radar-lite-feature-spec-v1",
        "feature_version": RADAR_LITE_FEATURE_VERSION,
        "vertical_reduction": {
            "zmax": "maximum over all metadata-valid levels",
            "low_max_height_km": "height <= 3",
            "mid_max_height_km": "3 <= height <= 7",
            "high_max_height_km": "height > 7",
            "echo_top_thresholds_dbz": [18.0, 35.0],
            "echo_top_unit": "km",
        },
        "missingness": {
            "masked_nonfinite_and_declared_fill_are_invalid": True,
            "default_declared_fill_values": list(DEFAULT_FILL_VALUES),
            "invalid_output_value": 0.0,
            "explicit_validity_mask": True,
        },
        "negative_dbz": {
            "official_semantics_confirmed": False,
            "metadata_valid_finite_candidate": True,
            "conservative_feature_floor_dbz": 0.0,
            "additional_feature": "below_zero_fraction",
        },
        "temporal_summary": {
            "offset_unit": "minutes relative to anchor",
            "future_offsets_rejected": True,
            "windows_minutes": list(DEFAULT_WINDOWS_MINUTES),
            "window_interval": "closed [-window, 0]",
            "window_max_features": list(INSTANTANEOUS_FEATURES),
            "window_mean_features": ["zmax"],
            "trend_60min_features": ["zmax", "etop35"],
            "trend_unit": "feature units per hour",
            "window_validity": "valid zmax frames / available mapped frames",
            "summary_feature_names": list(RADAR_LITE_SUMMARY_FEATURES),
        },
        "spatial_mapping": {
            "coordinate_derived": True,
            "continuous_method": "bilinear",
            "missing_renormalization": True,
            "outside_mask_explicit": True,
            "nominal_initial_resolution_degrees": 0.02,
            "shape_ratio_inference_forbidden": True,
        },
    }


def radar_lite_feature_hash() -> str:
    """Return the deterministic SHA256 of :func:`radar_lite_feature_spec`."""

    payload = json.dumps(
        radar_lite_feature_spec(),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


RADAR_LITE_FEATURE_HASH = radar_lite_feature_hash()


def _finite_float32(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional spatial array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite output values")
    result = np.ascontiguousarray(array)
    result.setflags(write=False)
    return result


def _boolean_2d(values: np.ndarray, *, name: str, shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(values, dtype=bool)
    if array.shape != shape:
        raise ValueError(f"{name} has shape {array.shape}; expected {shape}")
    result = np.ascontiguousarray(array)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class RadarInstantaneousFeatures:
    """Finite per-volume fields plus a mask for each field."""

    values: Mapping[str, np.ndarray]
    valid_masks: Mapping[str, np.ndarray]
    grid_hash: str | None = None

    def __post_init__(self) -> None:
        value_keys = tuple(self.values)
        mask_keys = tuple(self.valid_masks)
        if set(value_keys) != set(INSTANTANEOUS_FEATURES):
            raise ValueError(
                "instantaneous value fields must be exactly "
                f"{INSTANTANEOUS_FEATURES}; got {value_keys}"
            )
        if set(mask_keys) != set(INSTANTANEOUS_FEATURES):
            raise ValueError("every instantaneous value requires a validity mask")

        normalized_values: dict[str, np.ndarray] = {}
        shape: tuple[int, int] | None = None
        for name in INSTANTANEOUS_FEATURES:
            array = _finite_float32(self.values[name], name=name)
            if shape is None:
                shape = tuple(array.shape)
            elif array.shape != shape:
                raise ValueError("instantaneous fields do not share one spatial shape")
            normalized_values[name] = array
        assert shape is not None
        normalized_masks = {
            name: _boolean_2d(self.valid_masks[name], name=f"{name}_valid", shape=shape)
            for name in INSTANTANEOUS_FEATURES
        }
        for fraction_name in ("valid_fraction", "below_zero_fraction"):
            fraction = normalized_values[fraction_name]
            if bool(np.any((fraction < 0.0) | (fraction > 1.0))):
                raise ValueError(f"{fraction_name} must be within [0, 1]")
        if self.grid_hash is not None and (
            not isinstance(self.grid_hash, str) or len(self.grid_hash) != 64
        ):
            raise ValueError("grid_hash must be a 64-character hash or None")
        object.__setattr__(self, "values", MappingProxyType(normalized_values))
        object.__setattr__(self, "valid_masks", MappingProxyType(normalized_masks))

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.values["zmax"].shape)

    @property
    def missing_column_mask(self) -> np.ndarray:
        return np.logical_not(self.valid_masks["zmax"])

    @property
    def no_positive_echo_mask(self) -> np.ndarray:
        """Metadata-valid columns with no reflectivity above the 0 dBZ floor."""

        return self.valid_masks["zmax"] & (self.values["zmax"] <= 0.0)


@dataclass(frozen=True)
class RadarLiteSummary:
    """Deterministically ordered temporal Radar Lite channels."""

    values: Mapping[str, np.ndarray]
    valid_masks: Mapping[str, np.ndarray]
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if tuple(self.values) != RADAR_LITE_SUMMARY_FEATURES:
            raise ValueError("Radar Lite summary fields are missing or out of order")
        if tuple(self.valid_masks) != RADAR_LITE_SUMMARY_FEATURES:
            raise ValueError("Radar Lite summary masks are missing or out of order")
        normalized_values: dict[str, np.ndarray] = {}
        shape: tuple[int, int] | None = None
        for name in RADAR_LITE_SUMMARY_FEATURES:
            array = _finite_float32(self.values[name], name=name)
            if shape is None:
                shape = tuple(array.shape)
            elif array.shape != shape:
                raise ValueError("Radar Lite summary fields have inconsistent shapes")
            normalized_values[name] = array
        assert shape is not None
        normalized_masks = {
            name: _boolean_2d(self.valid_masks[name], name=f"{name}_valid", shape=shape)
            for name in RADAR_LITE_SUMMARY_FEATURES
        }
        if self.metadata.get("feature_hash") != RADAR_LITE_FEATURE_HASH:
            raise ValueError("Radar Lite summary feature hash is inconsistent")
        object.__setattr__(self, "values", MappingProxyType(normalized_values))
        object.__setattr__(self, "valid_masks", MappingProxyType(normalized_masks))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def feature_names(self) -> tuple[str, ...]:
        return RADAR_LITE_SUMMARY_FEATURES

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.values[RADAR_LITE_SUMMARY_FEATURES[0]].shape)

    def as_array(self) -> np.ndarray:
        """Stack only reduced summary channels, never raw radar volumes."""

        return np.stack([self.values[name] for name in self.feature_names], axis=0)

    def masks_as_array(self) -> np.ndarray:
        return np.stack([self.valid_masks[name] for name in self.feature_names], axis=0)


class RadarVolumeReducer:
    """Incrementally reduce vertical levels for one time and one spatial tile.

    The object retains only two-dimensional accumulators.  A NetCDF reader may
    call :meth:`update_level` for each decoded level without ever materialising
    a full four-hour tensor.
    """

    def __init__(
        self,
        spatial_shape: tuple[int, int],
        *,
        fill_values: Iterable[float] = DEFAULT_FILL_VALUES,
        grid_hash: str | None = None,
    ) -> None:
        if len(spatial_shape) != 2 or min(spatial_shape) <= 0:
            raise ValueError("spatial_shape must contain two positive dimensions")
        self.spatial_shape = (int(spatial_shape[0]), int(spatial_shape[1]))
        normalized_fill: list[float] = []
        for value in fill_values:
            numeric = float(value)
            if math.isnan(numeric):
                continue
            normalized_fill.append(numeric)
        self.fill_values = tuple(normalized_fill)
        self.grid_hash = grid_hash
        self._maxima = {
            name: np.zeros(self.spatial_shape, dtype=np.float32)
            for name in REFLECTIVITY_FEATURES
        }
        self._maxima_valid = {
            name: np.zeros(self.spatial_shape, dtype=bool)
            for name in REFLECTIVITY_FEATURES
        }
        self._echo_tops = {
            "etop18": np.zeros(self.spatial_shape, dtype=np.float32),
            "etop35": np.zeros(self.spatial_shape, dtype=np.float32),
        }
        self._valid_count = np.zeros(self.spatial_shape, dtype=np.uint16)
        self._below_zero_count = np.zeros(self.spatial_shape, dtype=np.uint16)
        self._level_count = 0
        self._band_level_counts = {"low_max": 0, "mid_max": 0, "high_max": 0}
        self._finalized = False

    @staticmethod
    def _bands_for_height(height_km: float) -> tuple[str, ...]:
        # The taskbook states both ``low <= 3 km`` and ``mid 3--7 km``.
        # Therefore an exactly 3 km CAPPI contributes to both summaries rather
        # than silently resolving the documented overlap in one direction.
        bands: list[str] = []
        if height_km <= 3.0:
            bands.append("low_max")
        if 3.0 <= height_km <= 7.0:
            bands.append("mid_max")
        if height_km > 7.0:
            bands.append("high_max")
        return tuple(bands)

    def update_level(self, reflectivity_dbz: np.ndarray, height_km: float) -> None:
        if self._finalized:
            raise RuntimeError("cannot update a finalized RadarVolumeReducer")
        height = float(height_km)
        if not math.isfinite(height) or height < 0:
            raise ValueError("height_km must be finite and non-negative")
        masked = np.ma.asarray(reflectivity_dbz)
        if masked.ndim != 2 or tuple(masked.shape) != self.spatial_shape:
            raise ValueError(
                f"level shape must be {self.spatial_shape}; got {tuple(masked.shape)}"
            )
        raw = np.asarray(masked.data, dtype=np.float32)
        valid = np.logical_not(np.ma.getmaskarray(masked))
        np.logical_and(valid, np.isfinite(raw), out=valid)
        for fill_value in self.fill_values:
            np.logical_and(valid, raw != fill_value, out=valid)

        conservative = np.maximum(raw, 0.0)
        conservative[~valid] = 0.0
        np.maximum(
            self._maxima["zmax"],
            conservative,
            out=self._maxima["zmax"],
            where=valid,
        )
        np.logical_or(self._maxima_valid["zmax"], valid, out=self._maxima_valid["zmax"])

        for band in self._bands_for_height(height):
            self._band_level_counts[band] += 1
            np.maximum(
                self._maxima[band], conservative, out=self._maxima[band], where=valid
            )
            np.logical_or(self._maxima_valid[band], valid, out=self._maxima_valid[band])

        for name, threshold in (("etop18", 18.0), ("etop35", 35.0)):
            hit = valid & (raw >= threshold)
            np.maximum(
                self._echo_tops[name],
                np.float32(height),
                out=self._echo_tops[name],
                where=hit,
            )
        self._valid_count += valid.astype(np.uint16, copy=False)
        self._below_zero_count += (valid & (raw < 0.0)).astype(np.uint16, copy=False)
        self._level_count += 1

    def finalize(self) -> RadarInstantaneousFeatures:
        if self._finalized:
            raise RuntimeError("RadarVolumeReducer.finalize may only be called once")
        if self._level_count == 0:
            raise ValueError("at least one vertical level is required")
        missing_bands = [
            name for name, count in self._band_level_counts.items() if count == 0
        ]
        if missing_bands:
            raise ValueError(f"height coordinates do not cover bands: {missing_bands}")
        self._finalized = True
        column_valid = self._valid_count > 0
        valid_fraction = self._valid_count.astype(np.float32) / np.float32(
            self._level_count
        )
        below_zero_fraction = np.zeros(self.spatial_shape, dtype=np.float32)
        np.divide(
            self._below_zero_count,
            self._valid_count,
            out=below_zero_fraction,
            where=column_valid,
        )
        values: dict[str, np.ndarray] = {
            **self._maxima,
            **self._echo_tops,
            "valid_fraction": valid_fraction,
            "below_zero_fraction": below_zero_fraction,
        }
        valid_masks: dict[str, np.ndarray] = {
            **self._maxima_valid,
            "etop18": column_valid,
            "etop35": column_valid,
            # A zero valid fraction is meaningful metadata, not a missing
            # feature.  Primary echo fields still remain invalid there.
            "valid_fraction": np.ones(self.spatial_shape, dtype=bool),
            "below_zero_fraction": column_valid,
        }
        return RadarInstantaneousFeatures(values, valid_masks, self.grid_hash)


def reduce_reflectivity_volume(
    reflectivity_dbz: np.ndarray,
    heights_km: Sequence[float],
    *,
    fill_values: Iterable[float] = DEFAULT_FILL_VALUES,
    grid_hash: str | None = None,
) -> RadarInstantaneousFeatures:
    """Reduce one ``(level, latitude, longitude)`` time volume."""

    reflectivity = np.ma.asarray(reflectivity_dbz)
    heights = np.asarray(heights_km, dtype=np.float64)
    if reflectivity.ndim != 3:
        raise ValueError("reflectivity volume must have shape (level, latitude, longitude)")
    if heights.ndim != 1 or heights.size != reflectivity.shape[0]:
        raise ValueError("heights_km must have one value per reflectivity level")
    reducer = RadarVolumeReducer(
        tuple(reflectivity.shape[-2:]), fill_values=fill_values, grid_hash=grid_hash
    )
    for level_index, height in enumerate(heights):
        reducer.update_level(reflectivity[level_index], float(height))
    return reducer.finalize()


def reduce_reflectivity_time_series(
    reflectivity_dbz: np.ndarray,
    heights_km: Sequence[float],
    time_offsets_minutes: Sequence[float],
    *,
    fill_values: Iterable[float] = DEFAULT_FILL_VALUES,
    grid_hash: str | None = None,
) -> Iterator[tuple[float, RadarInstantaneousFeatures]]:
    """Yield reduced frames from one ``(time, level, lat, lon)`` file array.

    The generator processes one time view at a time and never copies or retains
    all raw time volumes.  Call it separately for each hourly file instead of
    concatenating the four files.
    """

    reflectivity = np.ma.asarray(reflectivity_dbz)
    offsets = np.asarray(time_offsets_minutes, dtype=np.float64)
    if reflectivity.ndim != 4:
        raise ValueError("time series must have shape (time, level, latitude, longitude)")
    if reflectivity.shape[0] > 10:
        raise ValueError(
            "at most one 10-volume hourly radar file may be reduced at once; "
            "process hourly files separately"
        )
    if offsets.ndim != 1 or offsets.size != reflectivity.shape[0]:
        raise ValueError("time_offsets_minutes must have one value per time volume")
    if not np.all(np.isfinite(offsets)):
        raise ValueError("time offsets must be finite")
    for time_index, offset in enumerate(offsets):
        yield float(offset), reduce_reflectivity_volume(
            reflectivity[time_index],
            heights_km,
            fill_values=fill_values,
            grid_hash=grid_hash,
        )


def _regular_axis(
    bounds: tuple[float, float], resolution: float, origin: float
) -> np.ndarray:
    lower, upper = sorted((float(bounds[0]), float(bounds[1])))
    if not all(math.isfinite(value) for value in (lower, upper, resolution, origin)):
        raise ValueError("grid bounds, resolution, and origin must be finite")
    if resolution <= 0 or upper <= lower:
        raise ValueError("resolution and coordinate bounds must be positive")
    tolerance = 1e-9
    first_index = math.ceil((lower - origin) / resolution - tolerance)
    last_index = math.floor((upper - origin) / resolution + tolerance)
    if last_index - first_index + 1 < 2:
        raise ValueError("requested bounds contain fewer than two target coordinates")
    indices = np.arange(first_index, last_index + 1, dtype=np.float64)
    return origin + indices * resolution


def make_regular_radar_grid(
    source: GridSpec,
    *,
    resolution_degrees: float = 0.02,
    latitude_bounds: tuple[float, float] | None = None,
    longitude_bounds: tuple[float, float] | None = None,
    lattice_origin_degrees: float = 0.0,
    name: str = "radar_0p02",
) -> GridSpec:
    """Build a coordinate-defined regular target grid, never a shape resize."""

    latitude = _regular_axis(
        latitude_bounds or source.latitude_bounds,
        float(resolution_degrees),
        float(lattice_origin_degrees),
    )
    longitude = _regular_axis(
        longitude_bounds or source.longitude_bounds,
        float(resolution_degrees),
        float(lattice_origin_degrees),
    )
    return GridSpec(latitude, longitude, name=name)


def map_radar_instantaneous(
    frame: RadarInstantaneousFeatures,
    mapping: GridMap,
    *,
    minimum_valid_weight: float = 1e-6,
) -> RadarInstantaneousFeatures:
    """Map one reduced frame with validity-renormalised bilinear sampling."""

    if mapping.method != "bilinear" or mapping.categorical:
        raise ValueError("Radar Lite continuous fields require a bilinear GridMap")
    if frame.shape != mapping.source.shape:
        raise ValueError(
            f"frame shape {frame.shape} does not match mapping source {mapping.source.shape}"
        )
    if frame.grid_hash is None:
        raise ValueError(
            "frame lacks source coordinate identity; shape-only radar mapping is forbidden"
        )
    if frame.grid_hash != mapping.source.coordinate_hash:
        raise ValueError("frame grid hash does not match the GridMap source coordinates")
    if not math.isfinite(minimum_valid_weight) or minimum_valid_weight <= 0:
        raise ValueError("minimum_valid_weight must be finite and positive")

    outside = mapping.outside_mask
    mapped_values: dict[str, np.ndarray] = {}
    mapped_masks: dict[str, np.ndarray] = {}
    for name in INSTANTANEOUS_FEATURES:
        values = frame.values[name]
        source_valid = frame.valid_masks[name]
        if name == "valid_fraction":
            mapped, _ = mapping.apply_numpy(values)
            mapped = np.asarray(mapped, dtype=np.float32)
            mapped[outside] = 0.0
            valid = np.logical_not(outside)
        else:
            numerator, _ = mapping.apply_numpy(values * source_valid)
            weight, _ = mapping.apply_numpy(source_valid.astype(np.float32))
            numerator = np.asarray(numerator, dtype=np.float32)
            weight = np.asarray(weight, dtype=np.float32)
            valid = (weight >= minimum_valid_weight) & ~outside
            mapped = np.zeros(mapping.target.shape, dtype=np.float32)
            np.divide(numerator, weight, out=mapped, where=valid)
        mapped_values[name] = mapped
        mapped_masks[name] = valid
    return RadarInstantaneousFeatures(
        mapped_values, mapped_masks, mapping.target.coordinate_hash
    )


def _window_reduce(
    frames: Sequence[RadarInstantaneousFeatures],
    indices: np.ndarray,
    feature: str,
    operation: str,
) -> tuple[np.ndarray, np.ndarray]:
    shape = frames[0].shape
    if indices.size == 0:
        return np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=bool)
    values = np.stack([frames[int(index)].values[feature] for index in indices], axis=0)
    valid = np.stack(
        [frames[int(index)].valid_masks[feature] for index in indices], axis=0
    )
    output_valid = np.any(valid, axis=0)
    if operation == "max":
        prepared = np.where(valid, values, -np.inf)
        output = np.max(prepared, axis=0)
        output[~output_valid] = 0.0
    elif operation == "mean":
        numerator = np.sum(np.where(valid, values, 0.0), axis=0, dtype=np.float64)
        denominator = np.sum(valid, axis=0, dtype=np.int32)
        output = np.zeros(shape, dtype=np.float64)
        np.divide(numerator, denominator, out=output, where=output_valid)
    else:
        raise ValueError("operation must be max or mean")
    return np.asarray(output, dtype=np.float32), output_valid


def _time_valid_fraction(
    frames: Sequence[RadarInstantaneousFeatures], indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    shape = frames[0].shape
    if indices.size == 0:
        return np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=bool)
    observed = np.stack(
        [frames[int(index)].valid_masks["zmax"] for index in indices], axis=0
    )
    available = np.stack(
        [frames[int(index)].valid_masks["valid_fraction"] for index in indices],
        axis=0,
    )
    denominator = np.sum(available, axis=0, dtype=np.int32)
    output_valid = denominator > 0
    numerator = np.count_nonzero(observed & available, axis=0)
    output = np.zeros(shape, dtype=np.float32)
    np.divide(numerator, denominator, out=output, where=output_valid)
    return output, output_valid


def _linear_trend(
    frames: Sequence[RadarInstantaneousFeatures],
    offsets_minutes: np.ndarray,
    indices: np.ndarray,
    feature: str,
) -> tuple[np.ndarray, np.ndarray]:
    shape = frames[0].shape
    if indices.size < 2:
        return np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=bool)
    values = np.stack([frames[int(index)].values[feature] for index in indices], axis=0)
    valid = np.stack(
        [frames[int(index)].valid_masks[feature] for index in indices], axis=0
    )
    times = offsets_minutes[indices].astype(np.float64) / 60.0
    time_grid = times.reshape((-1, 1, 1))
    weights = valid.astype(np.float64)
    count = np.sum(weights, axis=0)
    sum_t = np.sum(weights * time_grid, axis=0)
    sum_y = np.sum(np.where(valid, values, 0.0), axis=0, dtype=np.float64)
    sum_tt = np.sum(weights * time_grid * time_grid, axis=0)
    sum_ty = np.sum(
        np.where(valid, values * time_grid, 0.0), axis=0, dtype=np.float64
    )
    denominator = count * sum_tt - sum_t * sum_t
    output_valid = (count >= 2) & (np.abs(denominator) > 1e-12)
    output = np.zeros(shape, dtype=np.float64)
    numerator = count * sum_ty - sum_t * sum_y
    np.divide(numerator, denominator, out=output, where=output_valid)
    return np.asarray(output, dtype=np.float32), output_valid


def aggregate_radar_lite(
    frames: Sequence[RadarInstantaneousFeatures],
    time_offsets_minutes: Sequence[float],
) -> RadarLiteSummary:
    """Form latest, 30/60/180-minute, and 60-minute trend features.

    Offsets are relative to the anchor.  Any positive offset is rejected as a
    potential future leak.  Window endpoints are inclusive.
    """

    frame_list = list(frames)
    offsets = np.asarray(time_offsets_minutes, dtype=np.float64)
    if not frame_list or offsets.ndim != 1 or offsets.size != len(frame_list):
        raise ValueError("one finite time offset is required for every radar frame")
    if not np.all(np.isfinite(offsets)):
        raise ValueError("time offsets must be finite")
    if bool(np.any(offsets > 1e-9)):
        raise ValueError("future radar time offsets are forbidden")
    order = np.argsort(offsets, kind="stable")
    offsets = offsets[order]
    frame_list = [frame_list[int(index)] for index in order]
    if bool(np.any(np.diff(offsets) <= 1e-9)):
        raise ValueError("radar time offsets must be unique")
    shape = frame_list[0].shape
    grid_hash = frame_list[0].grid_hash
    for frame in frame_list[1:]:
        if frame.shape != shape or frame.grid_hash != grid_hash:
            raise ValueError("all reduced radar frames must share one coordinate grid")

    values: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    latest = frame_list[-1]
    for feature in INSTANTANEOUS_FEATURES:
        name = f"latest_{feature}"
        values[name] = latest.values[feature]
        masks[name] = latest.valid_masks[feature]

    window_counts: dict[str, int] = {}
    window_indices: dict[int, np.ndarray] = {}
    for window in DEFAULT_WINDOWS_MINUTES:
        indices = np.flatnonzero((offsets >= -float(window) - 1e-9) & (offsets <= 1e-9))
        window_indices[window] = indices
        prefix = f"w{window:03d}"
        window_counts[str(window)] = int(indices.size)
        for feature in INSTANTANEOUS_FEATURES:
            name = f"{prefix}_{feature}_max"
            values[name], masks[name] = _window_reduce(
                frame_list, indices, feature, "max"
            )
        mean_name = f"{prefix}_zmax_mean"
        values[mean_name], masks[mean_name] = _window_reduce(
            frame_list, indices, "zmax", "mean"
        )
        valid_name = f"{prefix}_time_valid_fraction"
        values[valid_name], masks[valid_name] = _time_valid_fraction(
            frame_list, indices
        )

    trend_indices = window_indices[60]
    for feature in ("zmax", "etop35"):
        name = f"w060_{feature}_trend_per_hour"
        values[name], masks[name] = _linear_trend(
            frame_list, offsets, trend_indices, feature
        )

    # Rebuild in the fixed schema order.  This also detects accidental omissions
    # before cache creation.
    ordered_values = {name: values[name] for name in RADAR_LITE_SUMMARY_FEATURES}
    ordered_masks = {name: masks[name] for name in RADAR_LITE_SUMMARY_FEATURES}
    metadata = {
        "schema": "precip-radar-lite-summary-v1",
        "feature_version": RADAR_LITE_FEATURE_VERSION,
        "feature_hash": RADAR_LITE_FEATURE_HASH,
        "grid_hash": grid_hash,
        "frame_count": len(frame_list),
        "time_offsets_minutes": [float(value) for value in offsets],
        "latest_offset_minutes": float(offsets[-1]),
        "window_interval": "closed [-window, 0]",
        "window_frame_counts": window_counts,
        "feature_names": list(RADAR_LITE_SUMMARY_FEATURES),
    }
    return RadarLiteSummary(ordered_values, ordered_masks, metadata)


class RadarLiteSequenceAccumulator:
    """Hold reduced/mapped frames only, never raw four-hour radar tensors."""

    def __init__(self) -> None:
        self._frames: list[RadarInstantaneousFeatures] = []
        self._offsets: list[float] = []

    def add_reduced_frame(
        self, time_offset_minutes: float, frame: RadarInstantaneousFeatures
    ) -> None:
        offset = float(time_offset_minutes)
        if not math.isfinite(offset):
            raise ValueError("time offset must be finite")
        if offset > 1e-9:
            raise ValueError("future radar time offsets are forbidden")
        if any(abs(offset - existing) <= 1e-9 for existing in self._offsets):
            raise ValueError("duplicate radar time offset")
        if self._frames and (
            frame.shape != self._frames[0].shape
            or frame.grid_hash != self._frames[0].grid_hash
        ):
            raise ValueError("all sequence frames must share one coordinate grid")
        self._offsets.append(offset)
        self._frames.append(frame)

    def add_volume(
        self,
        time_offset_minutes: float,
        reflectivity_dbz: np.ndarray,
        heights_km: Sequence[float],
        *,
        fill_values: Iterable[float] = DEFAULT_FILL_VALUES,
        source_grid: GridSpec | None = None,
        mapping: GridMap | None = None,
    ) -> None:
        if mapping is not None and source_grid is None:
            source_grid = mapping.source
        grid_hash = source_grid.coordinate_hash if source_grid is not None else None
        frame = reduce_reflectivity_volume(
            reflectivity_dbz,
            heights_km,
            fill_values=fill_values,
            grid_hash=grid_hash,
        )
        if mapping is not None:
            frame = map_radar_instantaneous(frame, mapping)
        self.add_reduced_frame(time_offset_minutes, frame)

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    def finalize(self) -> RadarLiteSummary:
        return aggregate_radar_lite(self._frames, self._offsets)
