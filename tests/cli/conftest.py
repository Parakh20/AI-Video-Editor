"""Fixtures shared by CLI tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REAL_RAW_DIR = PROJECT_ROOT / "assets" / "raw"
SAMPLE_GPX = PROJECT_ROOT / "tests" / "fixtures" / "sample_ride.gpx"


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def smallest_real_clip() -> Path:
    """The smallest real clip in assets/raw, for fast real-data smoke tests."""
    # Real footage is multi-GB and not committed; is_file() also drops dangling
    # symlinks, so a fresh clone (or CI) skips these tests instead of erroring.
    candidates = REAL_RAW_DIR.iterdir() if REAL_RAW_DIR.is_dir() else iter(())
    clips = sorted(
        (p for p in candidates if p.suffix.upper() == ".MP4" and p.is_file()),
        key=lambda p: p.stat().st_size,
    )
    if not clips:
        pytest.skip("No real clips available under assets/raw (footage is not committed)")
    return clips[0]


@pytest.fixture
def synthetic_scores_json(tmp_path: Path) -> Path:
    """A hand-built scores.json (2 keep-worthy shots from a fake clip) --
    avoids a slow real video decode for tests that only exercise the
    timeline/render commands' JSON-in, JSON/file-out wiring."""
    fake_video = tmp_path / "fake_clip.mp4"
    fake_video.touch()  # timeline/render only need the path string, not real media

    sub_scores = {
        "visual_quality": 80.0,
        "stability": 90.0,
        "motion": 70.0,
        "motion_raw": 70.0,
        "scenery": 90.0,
        "scene_category": "sky",
        "lighting": 85.0,
        "gps_speed": None,
        "gps_events": None,
        "gps_event_details": [],
        "audio_quality": None,
        "speech_penalty": None,
        "music_sync": None,
    }
    scores = [
        {
            "video_path": str(fake_video),
            "shot_index": 0,
            "start_seconds": 0.0,
            "end_seconds": 3.0,
            "overall_score": 82.0,
            "sub_scores": sub_scores,
            "gps_available": False,
            "audio_available": False,
            "music_available": False,
        },
        {
            "video_path": str(fake_video),
            "shot_index": 1,
            "start_seconds": 3.0,
            "end_seconds": 6.0,
            "overall_score": 78.0,
            "sub_scores": sub_scores,
            "gps_available": False,
            "audio_available": False,
            "music_available": False,
        },
    ]
    path = tmp_path / "scores.json"
    path.write_text(json.dumps(scores, indent=2), encoding="utf-8")
    return path
