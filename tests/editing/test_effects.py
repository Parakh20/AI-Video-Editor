"""Unit tests for ffmpeg filter-graph fragment builders (no real ffmpeg calls)."""

from __future__ import annotations

import pytest

from ai_video_editor.editing.effects import (
    GradeStyle,
    crop_filter,
    crossfade_filter,
    fade_filter,
    letterbox_filter,
    lut3d_filter,
    speed_ramp_filter,
    suggest_color_grade,
    zoom_filter,
)
from ai_video_editor.vision.day_night_classifier import LightingCondition


def test_speed_ramp_filter_speeds_up() -> None:
    assert speed_ramp_filter(2.0) == "setpts=0.500000*PTS"


def test_speed_ramp_filter_slows_down() -> None:
    assert speed_ramp_filter(0.5) == "setpts=2.000000*PTS"


def test_speed_ramp_filter_rejects_non_positive() -> None:
    with pytest.raises(ValueError):
        speed_ramp_filter(0.0)
    with pytest.raises(ValueError):
        speed_ramp_filter(-1.0)


def test_crossfade_filter_contains_duration_and_offset() -> None:
    result = crossfade_filter(0.5, 12.4)
    assert "xfade" in result
    assert "duration=0.500" in result
    assert "offset=12.400" in result


def test_fade_filter_validates_type() -> None:
    assert fade_filter(fade_type="in", start_s=0.0, duration_s=1.0) == "fade=t=in:st=0.000:d=1.000"
    with pytest.raises(ValueError):
        fade_filter(fade_type="sideways", start_s=0.0, duration_s=1.0)  # type: ignore[arg-type]


def test_zoom_filter_rejects_shrink() -> None:
    with pytest.raises(ValueError):
        zoom_filter(zoom_factor=0.9, duration_frames=30, frame_rate=30.0)


def test_zoom_filter_produces_zoompan() -> None:
    result = zoom_filter(zoom_factor=1.2, duration_frames=60, frame_rate=30.0)
    assert result.startswith("zoompan=")
    assert "d=60" in result


def test_crop_filter_centers_by_default() -> None:
    result = crop_filter(width=1080, height=1920)
    assert "crop=1080:1920:(in_w-out_w)/2:(in_h-out_h)/2" == result


def test_crop_filter_explicit_offset() -> None:
    result = crop_filter(width=1080, height=1920, x=10, y=20)
    assert result == "crop=1080:1920:10:20"


def test_letterbox_filter() -> None:
    result = letterbox_filter(target_width=1920, target_height=1080)
    assert result.startswith("pad=1920:1080:")


def test_lut3d_filter_quotes_path() -> None:
    result = lut3d_filter("/tmp/some lut.cube")
    assert result.startswith("lut3d=file='")
    assert "some lut.cube" in result


def test_suggest_color_grade_night() -> None:
    suggestion = suggest_color_grade(LightingCondition.NIGHT)
    assert suggestion.style is GradeStyle.NIGHT_LIFTED
    assert "hqdn3d" in suggestion.filter_chain


def test_suggest_color_grade_golden_hour() -> None:
    suggestion = suggest_color_grade(LightingCondition.DAY, warm_ratio=1.3)
    assert suggestion.style is GradeStyle.GOLDEN_HOUR


def test_suggest_color_grade_day_natural() -> None:
    suggestion = suggest_color_grade(LightingCondition.DAY, warm_ratio=1.0)
    assert suggestion.style is GradeStyle.DAY_NATURAL
