"""Leakage-safe sample anchors and temporal block split helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Iterable, Mapping, Sequence

from .contracts import DataAsset, Modality


TARGET_LEADS = (timedelta(hours=4), timedelta(hours=5), timedelta(hours=6))


def require_aware(value: datetime, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class SampleIndex:
    sample_id: str
    anchor_time: datetime
    inputs: Mapping[Modality, tuple[DataAsset, ...]] = field(default_factory=dict)
    target_times: tuple[datetime, datetime, datetime] | None = None

    def __post_init__(self) -> None:
        anchor = require_aware(self.anchor_time, "anchor_time")
        object.__setattr__(self, "anchor_time", anchor)
        expected = tuple(anchor + lead for lead in TARGET_LEADS)
        if self.target_times is None:
            object.__setattr__(self, "target_times", expected)
        else:
            normalised = tuple(require_aware(value, "target_time") for value in self.target_times)
            if normalised != expected:
                raise ValueError("target_times must be exactly anchor + 4/5/6 hours")
            object.__setattr__(self, "target_times", normalised)
        self.assert_no_future_leakage()

    def assert_no_future_leakage(self) -> None:
        for modality, assets in self.inputs.items():
            for asset in assets:
                if asset.modality != modality:
                    raise ValueError("input modality key and asset modality disagree")
                available_time = asset.init_time
                if modality not in {Modality.REGIONAL_NWP, Modality.GLOBAL_NWP}:
                    available_time = asset.valid_time or asset.init_time
                if available_time is None:
                    raise ValueError(f"{modality.value} asset lacks an availability time")
                if require_aware(available_time, "asset availability") > self.anchor_time:
                    raise ValueError(
                        f"future leakage: {modality.value} asset becomes available after anchor"
                    )

    def frame_keys(self) -> set[tuple[str, str, str | None]]:
        result = set()
        for modality, assets in self.inputs.items():
            for asset in assets:
                timestamp = asset.valid_time or asset.init_time
                result.add((modality.value, asset.path, None if timestamp is None else timestamp.isoformat()))
        return result


def shared_frame_count(left: SampleIndex, right: SampleIndex) -> int:
    return len(left.frame_keys() & right.frame_keys())


def split_time_blocks(
    samples: Sequence[SampleIndex],
    *,
    validation_start: datetime,
    test_start: datetime,
    buffer: timedelta = timedelta(hours=24),
) -> dict[str, list[SampleIndex]]:
    validation_start = require_aware(validation_start, "validation_start")
    test_start = require_aware(test_start, "test_start")
    if validation_start >= test_start:
        raise ValueError("validation_start must precede test_start")
    if buffer < timedelta(0):
        raise ValueError("buffer cannot be negative")
    result = {"train": [], "validation": [], "local_test": [], "buffered_out": []}
    for sample in sorted(samples, key=lambda item: item.anchor_time):
        anchor = sample.anchor_time
        if anchor < validation_start - buffer:
            result["train"].append(sample)
        elif validation_start <= anchor < test_start - buffer:
            result["validation"].append(sample)
        elif anchor >= test_start:
            result["local_test"].append(sample)
        else:
            result["buffered_out"].append(sample)
    return result


def build_anchor_times(
    start: datetime,
    end: datetime,
    interval: timedelta = timedelta(hours=1),
) -> list[datetime]:
    start, end = require_aware(start, "start"), require_aware(end, "end")
    if interval <= timedelta(0):
        raise ValueError("interval must be positive")
    anchors = []
    cursor = start
    while cursor <= end:
        anchors.append(cursor)
        cursor += interval
    return anchors
