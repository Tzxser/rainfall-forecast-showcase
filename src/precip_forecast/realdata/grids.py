"""Coordinate-first grid descriptions for competition data.

The classes in this module deliberately know nothing about array shape ratios.  A
``GridSpec`` is defined by the actual one-dimensional latitude and longitude
coordinate values, and its hash is therefore suitable for cache provenance.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

import numpy as np


def _validated_coordinate(values: np.ndarray, name: str) -> np.ndarray:
    coordinate = np.asarray(values, dtype=np.float64)
    if coordinate.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if coordinate.size < 2:
        raise ValueError(f"{name} must contain at least two points")
    if not np.all(np.isfinite(coordinate)):
        raise ValueError(f"{name} contains non-finite values")
    differences = np.diff(coordinate)
    if not (np.all(differences > 0) or np.all(differences < 0)):
        raise ValueError(f"{name} must be strictly monotonic")

    # Always hash a canonical little-endian float64 representation.  Copying and
    # making it read-only prevents a caller from invalidating a cached hash.
    canonical = np.ascontiguousarray(coordinate, dtype="<f8").copy()
    canonical.setflags(write=False)
    return canonical


def coordinate_direction(values: np.ndarray) -> str:
    """Return ``ascending`` or ``descending`` for a validated coordinate."""

    coordinate = _validated_coordinate(values, "coordinate")
    return "ascending" if coordinate[-1] > coordinate[0] else "descending"


def _regular_resolution(values: np.ndarray) -> float | None:
    differences = np.abs(np.diff(values))
    resolution = float(np.median(differences))
    # Float32 coordinate vectors at 0.01 degree have small quantisation jitter
    # in their first differences; tolerate that without accepting genuinely
    # irregular spacing.
    if np.allclose(differences, resolution, rtol=5e-4, atol=1e-8):
        return resolution
    return None


@dataclass(frozen=True, eq=False)
class GridSpec:
    """A rectilinear latitude/longitude grid with immutable coordinates."""

    latitude: np.ndarray
    longitude: np.ndarray
    name: str = "unnamed"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("grid name must be a non-empty string")
        object.__setattr__(self, "latitude", _validated_coordinate(self.latitude, "latitude"))
        object.__setattr__(
            self, "longitude", _validated_coordinate(self.longitude, "longitude")
        )

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.latitude.size), int(self.longitude.size))

    @property
    def latitude_direction(self) -> str:
        return "ascending" if self.latitude[-1] > self.latitude[0] else "descending"

    @property
    def longitude_direction(self) -> str:
        return "ascending" if self.longitude[-1] > self.longitude[0] else "descending"

    @property
    def latitude_bounds(self) -> tuple[float, float]:
        return (float(np.min(self.latitude)), float(np.max(self.latitude)))

    @property
    def longitude_bounds(self) -> tuple[float, float]:
        return (float(np.min(self.longitude)), float(np.max(self.longitude)))

    @property
    def latitude_resolution(self) -> float | None:
        return _regular_resolution(self.latitude)

    @property
    def longitude_resolution(self) -> float | None:
        return _regular_resolution(self.longitude)

    @property
    def coordinate_hash(self) -> str:
        """SHA256 over exact coordinate values and an explicit schema header."""

        header = {
            "schema": "precip-grid-spec-v1",
            "latitude_count": int(self.latitude.size),
            "longitude_count": int(self.longitude.size),
            "dtype": "little-endian-float64",
        }
        digest = hashlib.sha256(
            json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        digest.update(b"\0latitude\0")
        digest.update(self.latitude.tobytes(order="C"))
        digest.update(b"\0longitude\0")
        digest.update(self.longitude.tobytes(order="C"))
        return digest.hexdigest()

    def to_summary(self) -> dict[str, Any]:
        """Return bounded metadata; coordinate arrays themselves stay out of reports."""

        return {
            "name": self.name,
            "shape": list(self.shape),
            "latitude": {
                "first": float(self.latitude[0]),
                "last": float(self.latitude[-1]),
                "min": self.latitude_bounds[0],
                "max": self.latitude_bounds[1],
                "direction": self.latitude_direction,
                "regular_resolution": self.latitude_resolution,
            },
            "longitude": {
                "first": float(self.longitude[0]),
                "last": float(self.longitude[-1]),
                "min": self.longitude_bounds[0],
                "max": self.longitude_bounds[1],
                "direction": self.longitude_direction,
                "regular_resolution": self.longitude_resolution,
            },
            "coordinate_hash": self.coordinate_hash,
        }

    def coordinates_equal(self, other: "GridSpec", *, atol: float = 0.0) -> bool:
        if not isinstance(other, GridSpec) or self.shape != other.shape:
            return False
        return bool(
            np.allclose(self.latitude, other.latitude, rtol=0.0, atol=atol)
            and np.allclose(self.longitude, other.longitude, rtol=0.0, atol=atol)
        )
