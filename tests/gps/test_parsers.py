"""Tests for ai_video_editor.gps.parsers."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import gpxpy
import gpxpy.gpx
import pytest

from ai_video_editor.gps.parsers import load_fit, load_gpx, load_track

FIXTURE_GPX = Path(__file__).resolve().parents[1] / "fixtures" / "sample_ride.gpx"


def _write_gpx(tmp_path: Path, points: list[tuple[float, float, float, datetime]]) -> Path:
    gpx = gpxpy.gpx.GPX()
    track = gpxpy.gpx.GPXTrack()
    gpx.tracks.append(track)
    segment = gpxpy.gpx.GPXTrackSegment()
    track.segments.append(segment)
    for lat, lon, ele, t in points:
        segment.points.append(gpxpy.gpx.GPXTrackPoint(latitude=lat, longitude=lon, elevation=ele, time=t))
    path = tmp_path / "track.gpx"
    path.write_text(gpx.to_xml(), encoding="utf-8")
    return path


def test_load_gpx_fixture_ride_has_expected_shape():
    track = load_gpx(FIXTURE_GPX)

    assert not track.is_empty
    assert len(track.points) == 111
    assert track.points[0].heart_rate_bpm == 110
    assert track.points[0].cadence_rpm == 80.0
    assert track.points[0].power_w == 150.0
    # points must be chronologically sorted
    timestamps = [p.timestamp for p in track.points]
    assert timestamps == sorted(timestamps)


def test_load_gpx_skips_points_missing_time_or_position(tmp_path):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    good_points = [
        (12.0, 77.0, 100.0, start),
        (12.001, 77.001, 101.0, start + timedelta(seconds=5)),
    ]
    path = _write_gpx(tmp_path, good_points)

    # Inject one point with no <time> element by hand, which gpxpy still parses
    # as a point with time=None -- append directly via gpxpy API.
    gpx_text = path.read_text(encoding="utf-8")
    gpx = gpxpy.parse(gpx_text)
    gpx.tracks[0].segments[0].points.append(
        gpxpy.gpx.GPXTrackPoint(latitude=12.002, longitude=77.002, elevation=102.0, time=None)
    )
    path.write_text(gpx.to_xml(), encoding="utf-8")

    track = load_gpx(path)

    assert len(track.points) == 2  # the untimed point was skipped, not fatal


def test_load_gpx_empty_track(tmp_path):
    path = _write_gpx(tmp_path, [])
    track = load_gpx(path)
    assert track.is_empty
    assert track.points == []


def test_load_track_dispatches_by_extension(tmp_path):
    path = _write_gpx(tmp_path, [(12.0, 77.0, 100.0, datetime(2026, 1, 1, tzinfo=timezone.utc))])
    track = load_track(path)
    assert len(track.points) == 1


def test_load_track_rejects_unsupported_extension(tmp_path):
    path = tmp_path / "track.kml"
    path.write_text("<kml></kml>", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported GPS file extension"):
        load_track(path)


@pytest.mark.skip(
    reason=(
        "fitparse is read-only and ships no sample .fit test data with this "
        "install; writing a real binary FIT file needs a separate encoder "
        "library not in this project's dependencies. load_fit's field-mapping "
        "logic (timestamp/position_lat/position_long/altitude/heart_rate/"
        "cadence/power, semicircle->degree conversion) is exercised manually "
        "against real device FIT files during integration testing instead."
    )
)
def test_load_fit_round_trip():
    load_fit("unused.fit")
