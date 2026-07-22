"""Unit tests for render presets (Phase 12)."""

from __future__ import annotations

import pytest

from ai_video_editor.render.presets import PRESETS, AspectMode, get_preset


def test_all_presets_have_positive_dimensions() -> None:
    for preset in PRESETS.values():
        assert preset.width > 0
        assert preset.height > 0
        assert preset.fps > 0


def test_get_preset_case_insensitive() -> None:
    assert get_preset("1080P") is get_preset("1080p")


def test_get_preset_unknown_raises_with_valid_names() -> None:
    with pytest.raises(KeyError, match="youtube"):
        get_preset("nonexistent")


def test_vertical_presets_use_center_crop() -> None:
    reel = get_preset("instagram_reel")
    shorts = get_preset("youtube_shorts")
    assert reel.aspect_mode is AspectMode.CENTER_CROP
    assert shorts.aspect_mode is AspectMode.CENTER_CROP
    assert reel.height > reel.width
    assert shorts.height > shorts.width


def test_4k_has_highest_bitrate_of_landscape_presets() -> None:
    landscape = [get_preset(n) for n in ("1080p", "1440p", "4k")]
    bitrates = [int(p.video_bitrate.rstrip("M")) for p in landscape]
    assert bitrates == sorted(bitrates)
