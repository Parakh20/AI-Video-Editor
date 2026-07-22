"""Detect climbs, descents, high-speed sections, stops, and other ride events."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ai_video_editor.gps.models import EventType, MetricSample, RideEvent, RideMetrics
from ai_video_editor.utils.logging import get_logger

logger = get_logger(__name__)

_MPS_PER_KMH = 1.0 / 3.6


@dataclass(frozen=True)
class EventThresholds:
    """Tunable thresholds for :func:`detect_events`. Defaults suit road cycling."""

    climb_gradient_pct: float = 3.0
    descent_gradient_pct: float = -3.0
    min_climb_descent_duration_s: float = 15.0

    high_speed_kmh: float = 40.0
    min_high_speed_duration_s: float = 5.0

    stop_speed_kmh: float = 2.0
    min_stop_duration_s: float = 3.0

    heading_change_deg: float = 45.0
    heading_window_s: float = 4.0
    min_interesting_duration_s: float = 2.0


def detect_events(metrics: RideMetrics, thresholds: EventThresholds | None = None) -> list[RideEvent]:
    """Detect all event types over a ride's metric series and return them
    sorted by start time. Returns an empty list for an empty/None-backed
    series (e.g. when no GPS data was available for a ride).
    """
    thresholds = thresholds or EventThresholds()
    if len(metrics.samples) < 2:
        return []

    events: list[RideEvent] = []
    events.extend(_detect_gradient_events(metrics.samples, thresholds))
    events.extend(_detect_high_speed_events(metrics.samples, thresholds))
    events.extend(_detect_stop_events(metrics.samples, thresholds))
    events.extend(_detect_interesting_segments(metrics.samples, thresholds))

    events.sort(key=lambda e: e.start_s)
    logger.info("Detected %d ride event(s)", len(events))
    return events


def _merge_runs(
    samples: list[MetricSample],
    predicate,
    min_duration_s: float,
    event_type: EventType,
    detail_fn,
) -> list[RideEvent]:
    """Merge consecutive samples satisfying ``predicate`` into intervals,
    dropping runs shorter than ``min_duration_s``.
    """
    events: list[RideEvent | None] = []
    run_start_idx: int | None = None

    for i, sample in enumerate(samples):
        matches = predicate(sample)
        if matches and run_start_idx is None:
            run_start_idx = i
        elif not matches and run_start_idx is not None:
            events.append(_finalize_run(samples, run_start_idx, i - 1, min_duration_s, event_type, detail_fn))
            run_start_idx = None

    if run_start_idx is not None:
        events.append(
            _finalize_run(samples, run_start_idx, len(samples) - 1, min_duration_s, event_type, detail_fn)
        )

    return [e for e in events if e is not None]


def _finalize_run(
    samples: list[MetricSample],
    start_idx: int,
    end_idx: int,
    min_duration_s: float,
    event_type: EventType,
    detail_fn,
) -> RideEvent | None:
    start_s = samples[start_idx].elapsed_s
    end_s = samples[end_idx].elapsed_s
    if end_s - start_s < min_duration_s:
        return None
    return RideEvent(
        event_type=event_type,
        start_s=start_s,
        end_s=end_s,
        detail=detail_fn(samples[start_idx : end_idx + 1]),
    )


def _detect_gradient_events(samples: list[MetricSample], t: EventThresholds) -> list[RideEvent]:
    climbs = _merge_runs(
        samples,
        lambda s: s.gradient_pct >= t.climb_gradient_pct,
        t.min_climb_descent_duration_s,
        EventType.CLIMB,
        lambda run: f"avg gradient {_avg(run, lambda s: s.gradient_pct):.1f}%, "
        f"{run[-1].distance_m - run[0].distance_m:.0f}m",
    )
    descents = _merge_runs(
        samples,
        lambda s: s.gradient_pct <= t.descent_gradient_pct,
        t.min_climb_descent_duration_s,
        EventType.DESCENT,
        lambda run: f"avg gradient {_avg(run, lambda s: s.gradient_pct):.1f}%, "
        f"{run[-1].distance_m - run[0].distance_m:.0f}m",
    )
    return climbs + descents


def _detect_high_speed_events(samples: list[MetricSample], t: EventThresholds) -> list[RideEvent]:
    threshold_mps = t.high_speed_kmh * _MPS_PER_KMH
    return _merge_runs(
        samples,
        lambda s: s.speed_mps >= threshold_mps,
        t.min_high_speed_duration_s,
        EventType.HIGH_SPEED,
        lambda run: f"avg speed {_avg(run, lambda s: s.speed_mps) / _MPS_PER_KMH:.1f} km/h, "
        f"peak {max(s.speed_mps for s in run) / _MPS_PER_KMH:.1f} km/h",
    )


def _detect_stop_events(samples: list[MetricSample], t: EventThresholds) -> list[RideEvent]:
    threshold_mps = t.stop_speed_kmh * _MPS_PER_KMH
    return _merge_runs(
        samples,
        lambda s: s.speed_mps <= threshold_mps,
        t.min_stop_duration_s,
        EventType.STOP,
        lambda run: f"stopped for {run[-1].elapsed_s - run[0].elapsed_s:.0f}s",
    )


def _bearing_deg(a: MetricSample, b: MetricSample) -> float:
    lat1, lat2 = math.radians(a.lat), math.radians(b.lat)
    d_lon = math.radians(b.lon - a.lon)
    x = math.sin(d_lon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(d_lon)
    return math.degrees(math.atan2(x, y)) % 360.0


def _angular_diff_deg(a: float, b: float) -> float:
    diff = abs(a - b) % 360.0
    return diff if diff <= 180.0 else 360.0 - diff


def _detect_interesting_segments(samples: list[MetricSample], t: EventThresholds) -> list[RideEvent]:
    """Flag sharp turns: a large heading change within a short time window.

    Straight-line stretches (near-zero displacement between fixes, e.g. GPS
    noise while stationary) are skipped so noise doesn't register as a turn.
    """
    bearings: list[float | None] = [None]
    for i in range(1, len(samples)):
        prev, cur = samples[i - 1], samples[i]
        moved_m = cur.distance_m - prev.distance_m
        bearings.append(_bearing_deg(prev, cur) if moved_m > 1.0 else None)

    def predicate(i: int) -> bool:
        window_start = samples[i].elapsed_s - t.heading_window_s
        j = i
        while j > 0 and samples[j - 1].elapsed_s >= window_start:
            j -= 1
        bearing_j, bearing_i = bearings[j], bearings[i]
        if bearing_j is None or bearing_i is None:
            return False
        return _angular_diff_deg(bearing_j, bearing_i) >= t.heading_change_deg

    events: list[RideEvent] = []
    run_start_idx: int | None = None
    for i in range(len(samples)):
        matches = predicate(i)
        if matches and run_start_idx is None:
            run_start_idx = i
        elif not matches and run_start_idx is not None:
            event = _finalize_run(
                samples,
                run_start_idx,
                i - 1,
                t.min_interesting_duration_s,
                EventType.INTERESTING_SEGMENT,
                lambda run: "sharp turn / heading change",
            )
            if event is not None:
                events.append(event)
            run_start_idx = None
    if run_start_idx is not None:
        event = _finalize_run(
            samples,
            run_start_idx,
            len(samples) - 1,
            t.min_interesting_duration_s,
            EventType.INTERESTING_SEGMENT,
            lambda run: "sharp turn / heading change",
        )
        if event is not None:
            events.append(event)

    return events


def _avg(samples: list[MetricSample], key) -> float:
    values = [key(s) for s in samples]
    return sum(values) / len(values) if values else 0.0
