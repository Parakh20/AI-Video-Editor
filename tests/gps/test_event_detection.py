"""Tests for ai_video_editor.gps.event_detection, against the synthetic fixture ride."""

from __future__ import annotations

from pathlib import Path

from ai_video_editor.gps.event_detection import EventThresholds, detect_events
from ai_video_editor.gps.metrics import compute_metrics
from ai_video_editor.gps.models import EventType, RideMetrics
from ai_video_editor.gps.parsers import load_gpx

FIXTURE_GPX = Path(__file__).resolve().parents[1] / "fixtures" / "sample_ride.gpx"


def _fixture_metrics() -> RideMetrics:
    return compute_metrics(load_gpx(FIXTURE_GPX))


def test_detect_events_empty_metrics_returns_empty():
    assert detect_events(RideMetrics(samples=[])) == []


def test_detect_events_finds_climb_descent_stop_and_high_speed():
    events = detect_events(_fixture_metrics())
    types = {e.event_type for e in events}

    assert EventType.CLIMB in types
    assert EventType.DESCENT in types
    assert EventType.STOP in types
    assert EventType.HIGH_SPEED in types


def test_detect_events_finds_sharp_turn():
    events = detect_events(_fixture_metrics())
    turns = [e for e in events if e.event_type == EventType.INTERESTING_SEGMENT]
    assert len(turns) >= 1


def test_detect_events_climb_precedes_descent_and_high_speed_finish():
    events = detect_events(_fixture_metrics())
    climb = next(e for e in events if e.event_type == EventType.CLIMB)
    descent = next(e for e in events if e.event_type == EventType.DESCENT)
    high_speed = next(e for e in events if e.event_type == EventType.HIGH_SPEED)

    assert climb.start_s < descent.start_s < high_speed.start_s


def test_detect_events_respects_min_duration_threshold():
    metrics = _fixture_metrics()
    # A very long minimum duration should suppress the short sharp-turn event.
    strict = EventThresholds(min_interesting_duration_s=1000.0)
    events = detect_events(metrics, strict)
    assert not any(e.event_type == EventType.INTERESTING_SEGMENT for e in events)


def test_ride_event_duration_property():
    metrics = _fixture_metrics()
    events = detect_events(metrics)
    for event in events:
        assert event.duration_s == event.end_s - event.start_s
        assert event.duration_s > 0
