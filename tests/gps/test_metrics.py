"""Tests for ai_video_editor.gps.metrics."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from ai_video_editor.gps.metrics import compute_metrics, haversine_distance_m
from ai_video_editor.gps.models import Track, TrackPoint


def _point(lat: float, lon: float, elevation: float, t: datetime, **kwargs) -> TrackPoint:
    return TrackPoint(timestamp=t, lat=lat, lon=lon, elevation_m=elevation, **kwargs)


def test_haversine_known_distance():
    # ~1 degree of latitude is ~111.19 km at the equator.
    distance = haversine_distance_m(0.0, 0.0, 1.0, 0.0)
    assert distance == pytest.approx(111_195, rel=0.01)


def test_haversine_zero_distance_for_identical_points():
    assert haversine_distance_m(12.34, 56.78, 12.34, 56.78) == pytest.approx(0.0, abs=1e-6)


def test_compute_metrics_empty_track_returns_empty():
    metrics = compute_metrics(Track(source_path="none", points=[]))
    assert metrics.samples == []
    assert metrics.total_distance_m == 0.0
    assert metrics.duration_s == 0.0


def test_compute_metrics_derives_speed_from_position_and_time():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    # Move ~100m due north (roughly 0.0009 degrees latitude) in 10 seconds -> 10 m/s.
    lat_delta = 100.0 / 111_195.0
    track = Track(
        source_path="synthetic",
        points=[
            _point(12.0, 77.0, 100.0, start),
            _point(12.0 + lat_delta, 77.0, 100.0, start + timedelta(seconds=10)),
        ],
    )

    metrics = compute_metrics(track)

    assert len(metrics.samples) == 2
    assert metrics.samples[0].speed_mps == 0.0  # first point has no prior segment
    assert metrics.samples[1].speed_mps == pytest.approx(10.0, rel=0.02)
    assert metrics.total_distance_m == pytest.approx(100.0, rel=0.02)
    assert metrics.duration_s == 10.0


def test_compute_metrics_prefers_source_speed_when_present():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    lat_delta = 100.0 / 111_195.0
    track = Track(
        source_path="synthetic",
        points=[
            _point(12.0, 77.0, 100.0, start, speed_mps=1.0),
            _point(12.0 + lat_delta, 77.0, 100.0, start + timedelta(seconds=10), speed_mps=99.0),
        ],
    )

    metrics = compute_metrics(track)

    # source-reported speed (99.0) wins over the derived ~10 m/s.
    assert metrics.samples[1].speed_mps == 99.0


def test_compute_metrics_gradient_sign_and_magnitude():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    lat_delta = 100.0 / 111_195.0  # ~100m horizontal
    track = Track(
        source_path="synthetic",
        points=[
            _point(12.0, 77.0, 100.0, start),
            _point(12.0 + lat_delta, 77.0, 105.0, start + timedelta(seconds=10)),  # +5m over ~100m
        ],
    )

    metrics = compute_metrics(track)

    assert metrics.samples[1].gradient_pct == pytest.approx(5.0, rel=0.05)


def test_compute_metrics_cumulative_distance_monotonic():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    lat_delta = 50.0 / 111_195.0
    points = [_point(12.0 + i * lat_delta, 77.0, 100.0, start + timedelta(seconds=i * 5)) for i in range(5)]
    metrics = compute_metrics(Track(source_path="synthetic", points=points))

    distances = [s.distance_m for s in metrics.samples]
    assert distances == sorted(distances)
    assert distances[0] == 0.0
    assert distances[-1] > 0.0
