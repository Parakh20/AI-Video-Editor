"""Unit tests for cycling telemetry overlays (Phase 11)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from ai_video_editor.gps.metrics import compute_metrics
from ai_video_editor.gps.models import MetricSample, RideMetrics
from ai_video_editor.gps.overlays import (
    build_timesynced_overlay,
    cadence_overlay_text,
    distance_overlay_text,
    drawtext_filter,
    gradient_overlay_text,
    heart_rate_overlay_text,
    power_overlay_text,
    render_elevation_profile,
    render_route_map,
    render_ride_stats_card,
    speed_overlay_text,
)
from ai_video_editor.gps.parsers import load_gpx

FIXTURE_GPX = Path(__file__).resolve().parents[1] / "fixtures" / "sample_ride.gpx"


@pytest.fixture(scope="module")
def real_metrics() -> RideMetrics:
    track = load_gpx(FIXTURE_GPX)
    return compute_metrics(track)


def test_speed_overlay_text_none_without_metrics() -> None:
    assert speed_overlay_text(None, 10.0) is None


def test_speed_overlay_text_empty_metrics() -> None:
    assert speed_overlay_text(RideMetrics(samples=[]), 10.0) is None


def test_speed_overlay_text_real_fixture(real_metrics: RideMetrics) -> None:
    mid_time = real_metrics.duration_s / 2
    text = speed_overlay_text(real_metrics, mid_time)
    assert text is not None
    assert text.endswith("km/h")


def test_distance_overlay_text_increases_over_time(real_metrics: RideMetrics) -> None:
    early = distance_overlay_text(real_metrics, 5.0)
    late = distance_overlay_text(real_metrics, real_metrics.duration_s - 5.0)
    assert early is not None and late is not None
    early_km = float(early.split()[0])
    late_km = float(late.split()[0])
    assert late_km >= early_km


def test_gradient_overlay_text_signed(real_metrics: RideMetrics) -> None:
    text = gradient_overlay_text(real_metrics, real_metrics.duration_s / 4)
    assert text is not None
    assert text[0] in "+-"


def test_power_hr_cadence_overlays_present_in_fixture(real_metrics: RideMetrics) -> None:
    # sample_ride.gpx includes power/HR/cadence Garmin TrackPointExtension data.
    assert power_overlay_text(real_metrics, 10.0) is not None
    assert power_overlay_text(real_metrics, 10.0).endswith("W")
    assert heart_rate_overlay_text(real_metrics, 10.0) is not None
    assert heart_rate_overlay_text(real_metrics, 10.0).endswith("bpm")
    assert cadence_overlay_text(real_metrics, 10.0) is not None
    assert cadence_overlay_text(real_metrics, 10.0).endswith("rpm")


def test_power_hr_cadence_overlays_none_without_metrics() -> None:
    assert power_overlay_text(None, 10.0) is None
    assert heart_rate_overlay_text(None, 10.0) is None
    assert cadence_overlay_text(None, 10.0) is None


def test_power_overlay_present_when_source_has_it() -> None:
    samples = [
        MetricSample(
            timestamp=datetime(2026, 1, 1, tzinfo=UTC),
            elapsed_s=0.0,
            lat=0.0,
            lon=0.0,
            elevation_m=None,
            speed_mps=5.0,
            cadence_rpm=80.0,
            heart_rate_bpm=140,
            power_w=210.0,
            gradient_pct=0.0,
            distance_m=0.0,
        )
    ]
    metrics = RideMetrics(samples=samples)
    assert power_overlay_text(metrics, 0.0) == "210 W"
    assert heart_rate_overlay_text(metrics, 0.0) == "140 bpm"
    assert cadence_overlay_text(metrics, 0.0) == "80 rpm"


def test_drawtext_filter_escapes_special_chars() -> None:
    result = drawtext_filter("Kolar: 42 km/h")
    assert "drawtext=text=" in result
    assert "\\:" in result


def test_render_elevation_profile_none_without_metrics(tmp_path: Path) -> None:
    assert render_elevation_profile(None, tmp_path / "elev.png") is None


def test_render_elevation_profile_real_fixture(real_metrics: RideMetrics, tmp_path: Path) -> None:
    out = tmp_path / "elev.png"
    result = render_elevation_profile(real_metrics, out)
    assert result is not None
    assert out.exists()
    assert out.stat().st_size > 0


def test_render_route_map_real_fixture(real_metrics: RideMetrics, tmp_path: Path) -> None:
    out = tmp_path / "route.png"
    result = render_route_map(real_metrics, out)
    assert result is not None
    assert out.exists()
    assert out.stat().st_size > 0


def test_render_ride_stats_card_real_fixture(real_metrics: RideMetrics, tmp_path: Path) -> None:
    out = tmp_path / "stats.png"
    result = render_ride_stats_card(real_metrics, out)
    assert result is not None
    assert out.exists()
    assert out.stat().st_size > 0


def test_render_functions_none_without_gps(tmp_path: Path) -> None:
    assert render_route_map(None, tmp_path / "a.png") is None
    assert render_ride_stats_card(None, tmp_path / "b.png") is None


# --- Time-synced (per-frame) telemetry overlay ------------------------------


def test_time_windows_basic() -> None:
    from ai_video_editor.gps.overlays import _time_windows

    windows = _time_windows(6.0, 60.0, 2.0, 600)
    assert len(windows) == 3
    assert windows[0].start_s == 0.0
    assert windows[0].end_s == 2.0
    assert windows[-1].end_s == 6.0
    assert windows[0].ride_elapsed_s == 0.0
    assert windows[-1].ride_elapsed_s == pytest.approx(40.0)


def test_time_windows_caps_at_max_windows() -> None:
    from ai_video_editor.gps.overlays import _time_windows

    windows = _time_windows(100_000.0, 100.0, 0.001, max_windows=50)
    assert len(windows) <= 50


def test_build_timesynced_overlay_none_without_metrics() -> None:
    assert build_timesynced_overlay(None, 10.0) is None


def test_build_timesynced_overlay_none_empty_metrics() -> None:
    assert build_timesynced_overlay(RideMetrics(samples=[]), 10.0) is None


def test_build_timesynced_overlay_none_for_nonpositive_duration(
    real_metrics: RideMetrics,
) -> None:
    assert build_timesynced_overlay(real_metrics, 0.0) is None
    assert build_timesynced_overlay(real_metrics, -5.0) is None


def test_build_timesynced_overlay_produces_time_gated_chain(
    real_metrics: RideMetrics,
) -> None:
    overlay = build_timesynced_overlay(real_metrics, 10.0, update_interval_s=1.0)
    assert overlay is not None
    # One drawtext per update window, each gated on a time window.
    assert overlay.count("drawtext=text=") == 10
    assert overlay.count("enable='between(t,") == 10
    # Windows are chained with commas into a single filterchain fragment.
    assert overlay.count("km/h") == 10


def test_build_timesynced_overlay_windows_cover_full_duration(
    real_metrics: RideMetrics,
) -> None:
    overlay = build_timesynced_overlay(real_metrics, 6.0, update_interval_s=2.0)
    assert overlay is not None
    # 3 windows: [0,2], [2,4], [4,6].
    assert "enable='between(t,0.000,2.000)'" in overlay
    assert "enable='between(t,2.000,4.000)'" in overlay
    assert "enable='between(t,4.000,6.000)'" in overlay


def test_build_timesynced_overlay_bounds_window_count(
    real_metrics: RideMetrics,
) -> None:
    # A very long output with a tiny interval must not explode the filter graph.
    overlay = build_timesynced_overlay(
        real_metrics, 100_000.0, update_interval_s=0.001, max_windows=50
    )
    assert overlay is not None
    assert overlay.count("drawtext=text=") <= 50


def test_build_timesynced_overlay_stretches_ride_across_output(
    real_metrics: RideMetrics,
) -> None:
    # Distance should increase from the first window to the last, because the
    # full ride is mapped linearly across the (shorter) rendered output.
    overlay = build_timesynced_overlay(
        real_metrics, 8.0, update_interval_s=2.0, fields=("distance",)
    )
    assert overlay is not None
    import re

    km_values = [float(m) for m in re.findall(r"([0-9.]+) km", overlay)]
    assert len(km_values) == 4
    assert km_values[-1] >= km_values[0]
