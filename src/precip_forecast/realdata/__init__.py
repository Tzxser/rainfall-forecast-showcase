"""Real-data contracts that are safe to import without opening competition data."""

from precip_forecast.realdata.anchors import (
    WINDOW_OFFSETS,
    build_anchor_index,
    read_anchor_index,
)
from precip_forecast.realdata.catalog import build_variable_catalog
from precip_forecast.realdata.grids import GridSpec
from precip_forecast.realdata.pseudotime import PseudoTime
from precip_forecast.realdata.target_mapping import GridMap

__all__ = [
    "GridMap",
    "GridSpec",
    "PseudoTime",
    "WINDOW_OFFSETS",
    "build_anchor_index",
    "build_variable_catalog",
    "read_anchor_index",
]
