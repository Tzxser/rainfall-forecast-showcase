"""Versioned, real-data feature reducers.

Feature modules operate on bounded arrays or streams.  They do not discover
competition files and they never write to the read-only data root.
"""

from precip_forecast.features.precipitation import (
    PrecipitationHour,
    prepare_precipitation,
    read_precipitation_domain_mask,
    read_precipitation_file,
)
from precip_forecast.features.radar import (
    INSTANTANEOUS_FEATURES,
    RADAR_LITE_FEATURE_HASH,
    RADAR_LITE_FEATURE_VERSION,
    RADAR_LITE_SUMMARY_FEATURES,
    RadarInstantaneousFeatures,
    RadarLiteSequenceAccumulator,
    RadarLiteSummary,
    RadarVolumeReducer,
    aggregate_radar_lite,
    make_regular_radar_grid,
    map_radar_instantaneous,
    radar_lite_feature_hash,
    radar_lite_feature_spec,
    reduce_reflectivity_time_series,
    reduce_reflectivity_volume,
)

__all__ = [
    "INSTANTANEOUS_FEATURES",
    "PrecipitationHour",
    "RADAR_LITE_FEATURE_HASH",
    "RADAR_LITE_FEATURE_VERSION",
    "RADAR_LITE_SUMMARY_FEATURES",
    "RadarInstantaneousFeatures",
    "RadarLiteSequenceAccumulator",
    "RadarLiteSummary",
    "RadarVolumeReducer",
    "aggregate_radar_lite",
    "make_regular_radar_grid",
    "map_radar_instantaneous",
    "prepare_precipitation",
    "read_precipitation_domain_mask",
    "radar_lite_feature_hash",
    "radar_lite_feature_spec",
    "reduce_reflectivity_time_series",
    "reduce_reflectivity_volume",
    "read_precipitation_file",
]
