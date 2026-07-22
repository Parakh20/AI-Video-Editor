"""Typed data models shared across the GPS analysis package."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict


class TrackPoint(BaseModel):
    """A single GPS fix, optionally enriched with sensor data.

    ``speed_mps``/``cadence_rpm``/``power_w`` are the values embedded in the
    source file (e.g. a FIT file's speed/cadence/power fields) when present.
    :func:`ai_video_editor.gps.metrics.compute_metrics` derives them from
    position/time when they are missing, so callers should not assume these
    are populated on points fresh out of a parser.
    """

    model_config = ConfigDict(frozen=True)

    timestamp: datetime
    lat: float
    lon: float
    elevation_m: float | None = None
    heart_rate_bpm: int | None = None
    cadence_rpm: float | None = None
    power_w: float | None = None
    speed_mps: float | None = None


class Track(BaseModel):
    """An ordered GPS track loaded from a GPX or FIT file."""

    model_config = ConfigDict(frozen=True)

    source_path: str
    points: list[TrackPoint]

    @property
    def is_empty(self) -> bool:
        return len(self.points) == 0


class MetricSample(BaseModel):
    """Time-indexed derived ride metrics at one track point."""

    model_config = ConfigDict(frozen=True)

    timestamp: datetime
    elapsed_s: float
    lat: float
    lon: float
    elevation_m: float | None
    speed_mps: float
    cadence_rpm: float | None
    heart_rate_bpm: int | None
    power_w: float | None
    gradient_pct: float
    distance_m: float
    """Cumulative distance travelled since the start of the track, in metres."""


class RideMetrics(BaseModel):
    """The full derived-metric time series for a track."""

    model_config = ConfigDict(frozen=True)

    samples: list[MetricSample]

    @property
    def total_distance_m(self) -> float:
        return self.samples[-1].distance_m if self.samples else 0.0

    @property
    def duration_s(self) -> float:
        return self.samples[-1].elapsed_s if self.samples else 0.0


class EventType(str, Enum):
    CLIMB = "climb"
    DESCENT = "descent"
    HIGH_SPEED = "high_speed"
    STOP = "stop"
    INTERESTING_SEGMENT = "interesting_segment"


class RideEvent(BaseModel):
    """A labeled interval of interest over the ride timeline."""

    model_config = ConfigDict(frozen=True)

    event_type: EventType
    start_s: float
    end_s: float
    """Elapsed-seconds interval, inclusive, relative to the track start."""
    detail: str
    """Human-readable explanation of why this interval was flagged."""

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s
