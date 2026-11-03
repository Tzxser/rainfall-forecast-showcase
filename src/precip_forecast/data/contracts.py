"""Abstract contracts that deliberately avoid real-data variable assumptions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Mapping


class Modality(str, Enum):
    PRECIP_HISTORY = "precip_history"
    RADAR_3D = "radar_3d"
    CONVECTIVE_INITIATION = "convective_initiation"
    REGIONAL_NWP = "regional_nwp"
    SATELLITE = "satellite"
    GLOBAL_NWP = "global_nwp"


@dataclass(frozen=True)
class SpatialExtent:
    south: float
    north: float
    west: float
    east: float

    def validate(self) -> None:
        if not self.south < self.north:
            raise ValueError("south must be less than north")
        if not self.west < self.east:
            raise ValueError("west must be less than east")


@dataclass(frozen=True)
class DataAsset:
    modality: Modality
    path: str
    variable: str | None = None
    dimensions: tuple[str, ...] = ()
    coordinates: Mapping[str, Any] = field(default_factory=dict)
    init_time: datetime | None = None
    valid_time: datetime | None = None
    lead_time: timedelta | None = None
    level: str | float | None = None
    unit: str | None = None
    missing_value: float | int | str | None = None
    spatial_resolution: tuple[float, float] | None = None
    spatial_extent: SpatialExtent | None = None
    source_cycle: str | None = None
    quality_flags: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.path:
            raise ValueError("path is required")
        if self.init_time is not None and self.valid_time is not None and self.lead_time is not None:
            if self.init_time + self.lead_time != self.valid_time:
                raise ValueError("valid_time must equal init_time + lead_time")
        if self.spatial_resolution is not None and any(value <= 0 for value in self.spatial_resolution):
            raise ValueError("spatial_resolution values must be positive")
        if self.spatial_extent is not None:
            self.spatial_extent.validate()

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["modality"] = self.modality.value
        for key in ("init_time", "valid_time"):
            if result[key] is not None:
                result[key] = result[key].isoformat()
        if self.lead_time is not None:
            result["lead_time_seconds"] = self.lead_time.total_seconds()
            del result["lead_time"]
        return result


REQUIRED_MODALITIES = tuple(Modality)
