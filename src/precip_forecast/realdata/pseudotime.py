"""Pseudo-year-safe hourly time helpers for the released competition samples.

The sample uses years 0001 and 0002.  Those years are supported by
``datetime`` but not by several popular tabular time implementations.  This
module deliberately keeps the representation small, timezone-free and
independent of pandas/NumPy time types.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator


_TOKEN_RE = re.compile(r"(?<!\d)(\d{10}|\d{12})(?!\d)")
_ISO_RE = re.compile(
    r"^\s*(\d{4})-(\d{2})-(\d{2})[ T](\d{2})(?::(\d{2}))?(?::(\d{2}))?\s*$"
)


@dataclass(frozen=True, order=True)
class PseudoTime:
    """A validated civil time whose year may be as small as one.

    The competition observations are hourly.  Minutes are retained so the
    same type can describe sub-hourly coordinates during audits, while anchor
    construction explicitly requires ``minute == second == 0``.
    """

    year: int
    month: int
    day: int
    hour: int
    minute: int = 0
    second: int = 0

    def __post_init__(self) -> None:
        # ``datetime`` provides the calendar validation, including leap days.
        datetime(self.year, self.month, self.day, self.hour, self.minute, self.second)

    @classmethod
    def parse(cls, value: str | "PseudoTime" | datetime) -> "PseudoTime":
        if isinstance(value, cls):
            return value
        if isinstance(value, datetime):
            if value.tzinfo is not None:
                raise ValueError("pseudo time must be timezone-free")
            return cls(
                value.year,
                value.month,
                value.day,
                value.hour,
                value.minute,
                value.second,
            )
        text = str(value).strip()
        if len(text) in (10, 12) and text.isdigit():
            minute = int(text[10:12]) if len(text) == 12 else 0
            return cls(
                int(text[0:4]),
                int(text[4:6]),
                int(text[6:8]),
                int(text[8:10]),
                minute,
                0,
            )
        match = _ISO_RE.fullmatch(text)
        if not match:
            raise ValueError(f"unsupported pseudo-time value: {value!r}")
        year, month, day, hour, minute, second = match.groups()
        return cls(
            int(year),
            int(month),
            int(day),
            int(hour),
            int(minute or 0),
            int(second or 0),
        )

    def to_datetime(self) -> datetime:
        return datetime(self.year, self.month, self.day, self.hour, self.minute, self.second)

    def add_hours(self, hours: int) -> "PseudoTime":
        if not isinstance(hours, int):
            raise TypeError("hours must be an integer")
        return type(self).parse(self.to_datetime() + timedelta(hours=hours))

    def token(self, *, include_minutes: bool = False) -> str:
        base = f"{self.year:04d}{self.month:02d}{self.day:02d}{self.hour:02d}"
        return f"{base}{self.minute:02d}" if include_minutes else base

    def isoformat(self) -> str:
        return (
            f"{self.year:04d}-{self.month:02d}-{self.day:02d}T"
            f"{self.hour:02d}:{self.minute:02d}:{self.second:02d}"
        )

    def ordinal_hour(self) -> int:
        if self.minute or self.second:
            raise ValueError("ordinal_hour requires an hour-aligned pseudo time")
        return (self.to_datetime().toordinal() - 1) * 24 + self.hour


def extract_time_token(path_or_name: str | Path, *, require_hourly: bool = True) -> str:
    """Extract the unique 10/12-digit civil-time token from a filename.

    Ambiguous filenames are rejected instead of silently choosing a token.
    """

    name = Path(path_or_name).name
    matches = _TOKEN_RE.findall(name)
    if len(matches) != 1:
        raise ValueError(f"expected one pseudo-time token in {name!r}, found {matches!r}")
    parsed = PseudoTime.parse(matches[0])
    if require_hourly and (parsed.minute or parsed.second):
        raise ValueError(f"filename token is not hour-aligned: {matches[0]}")
    return parsed.token(include_minutes=not require_hourly and len(matches[0]) == 12)


def add_hours(token: str, hours: int) -> str:
    return PseudoTime.parse(token).add_hours(hours).token()


def hours_between(later: str | PseudoTime, earlier: str | PseudoTime) -> int:
    delta = PseudoTime.parse(later).to_datetime() - PseudoTime.parse(earlier).to_datetime()
    seconds = delta.total_seconds()
    if seconds % 3600:
        raise ValueError("pseudo times are not separated by a whole number of hours")
    return int(seconds // 3600)


def iter_hour_tokens(start: str | PseudoTime, stop: str | PseudoTime) -> Iterator[str]:
    """Yield an inclusive, hourly token sequence."""

    current = PseudoTime.parse(start)
    final = PseudoTime.parse(stop)
    if current.minute or current.second or final.minute or final.second:
        raise ValueError("iter_hour_tokens requires hour-aligned endpoints")
    if final < current:
        raise ValueError("stop precedes start")
    while current <= final:
        yield current.token()
        current = current.add_hours(1)
