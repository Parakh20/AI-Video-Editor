"""Tests for the weighted clip scoring engine."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from ai_video_editor.analysis.scoring import (
    ScoreWeights,
    _motion_quality,
    score_shot,
    score_video,
    write_scores_json,
)
from ai_video_editor.audio.music_analysis import MusicAnalysis, MusicSection
from ai_video_editor.audio.silence_detection import SilenceInterval
from ai_video_editor.audio.speech_detection import SpeechSegment
from ai_video_editor.gps.models import EventType, MetricSample, RideEvent, RideMetrics
from ai_video_editor.vision.scene_detection import Shot


def test_motion_quality_rewards_stable_motion_over_shaky_motion() -> None:
    """Design decision: same raw motion, but stability scales it down."""
    stable = _motion_quality(motion_raw=80.0, stability=95.0)
    shaky = _motion_quality(motion_raw=80.0, stability=20.0)
    assert stable > shaky
    assert stable == pytest.approx(80.0 * 0.95)
    assert shaky == pytest.approx(80.0 * 0.20)


def test_score_video_detects_and_scores_shots(two_shot_video: Path) -> None:
    scores = score_video(two_shot_video)
    assert len(scores) >= 1
    for s in scores:
        assert 0.0 <= s.overall_score <= 100.0
        assert s.gps_available is False
        assert s.audio_available is False
        assert s.music_available is False
        assert s.sub_scores.gps_speed is None
        assert s.sub_scores.audio_quality is None
        assert s.sub_scores.music_sync is None


def test_score_video_second_shot_scores_higher_when_split(two_shot_video: Path) -> None:
    """The bright/moving/sharp half should out-score the dark/static half."""
    scores = score_video(two_shot_video, scene_threshold=15.0)
    if len(scores) < 2:
        pytest.skip("scene detector merged the synthetic cut into one shot")
    by_start = sorted(scores, key=lambda s: s.start_seconds)
    assert by_start[-1].overall_score > by_start[0].overall_score


def test_score_shot_gps_signals_included_when_track_present(two_shot_video: Path) -> None:
    scores_without_gps = score_video(two_shot_video, max_seconds=2.0)
    assert scores_without_gps
    shot = Shot(index=0, start_seconds=0.0, end_seconds=2.0)

    base_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    samples = [
        MetricSample(
            timestamp=base_time + timedelta(seconds=i),
            elapsed_s=float(i),
            lat=0.0,
            lon=0.0,
            elevation_m=100.0,
            speed_mps=15.0,
            cadence_rpm=None,
            heart_rate_bpm=None,
            power_w=None,
            gradient_pct=5.0,
            distance_m=float(i) * 15.0,
        )
        for i in range(3)
    ]
    metrics = RideMetrics(samples=samples)
    events = [
        RideEvent(
            event_type=EventType.CLIMB, start_s=0.0, end_s=2.0, detail="6% climb over 2s (synthetic)"
        )
    ]

    scored = score_shot(two_shot_video, shot, gps_metrics=metrics, gps_events=events)
    assert scored.gps_available is True
    assert scored.sub_scores.gps_speed is not None
    assert scored.sub_scores.gps_events is not None
    assert scored.sub_scores.gps_events >= 80.0  # climb event should boost this
    assert scored.sub_scores.gps_event_details


def test_score_shot_gps_absent_when_metrics_empty(two_shot_video: Path) -> None:
    shot = Shot(index=0, start_seconds=0.0, end_seconds=2.0)
    empty_metrics = RideMetrics(samples=[])
    scored = score_shot(two_shot_video, shot, gps_metrics=empty_metrics)
    assert scored.gps_available is False
    assert scored.sub_scores.gps_speed is None


def test_score_shot_audio_and_music_signals(two_shot_video: Path) -> None:
    shot = Shot(index=0, start_seconds=0.0, end_seconds=2.0)
    silence = [SilenceInterval(start=0.0, end=0.1)]
    speech = [SpeechSegment(start=0.5, end=0.7)]
    music = MusicAnalysis(
        duration_seconds=200.0,
        rms_times=[0.0, 1.0],
        rms_curve=[0.2, 0.9],
        spectral_centroid_mean_hz=2000.0,
        spectral_bandwidth_mean_hz=1500.0,
        sections=[
            MusicSection(start=0.0, end=100.0, energy_label="low"),
            MusicSection(start=100.0, end=200.0, energy_label="high"),
        ],
    )

    scored = score_shot(
        two_shot_video, shot, silence_intervals=silence, speech_segments=speech, music=music
    )
    assert scored.audio_available is True
    assert scored.sub_scores.audio_quality is not None
    assert scored.sub_scores.speech_penalty is not None
    assert scored.sub_scores.speech_penalty < 100.0  # speech present -> some penalty
    assert scored.music_available is True
    assert scored.sub_scores.music_sync is not None


def test_weights_sum_to_one() -> None:
    weights = ScoreWeights()
    total = sum(weights.as_dict().values())
    assert total == pytest.approx(1.0, abs=1e-6)


def test_write_scores_json_roundtrip(two_shot_video: Path, tmp_path: Path) -> None:
    scores = score_video(two_shot_video, max_seconds=2.0)
    out_path = tmp_path / "scores.json"
    write_scores_json(scores, out_path)
    assert out_path.exists()

    import json

    loaded = json.loads(out_path.read_text())
    assert isinstance(loaded, list)
    assert len(loaded) == len(scores)
    assert "overall_score" in loaded[0]
    assert "sub_scores" in loaded[0]
