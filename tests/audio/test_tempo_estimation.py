from pathlib import Path

from ai_video_editor.audio.tempo_estimation import estimate_tempo


def test_estimate_tempo_on_clicktrack_is_plausible(clicktrack_wav: Path) -> None:
    result = estimate_tempo(clicktrack_wav)
    assert 40.0 <= result.bpm <= 220.0
    assert 0.0 <= result.confidence <= 1.0


def test_estimate_tempo_on_silence_is_zero(silent_wav: Path) -> None:
    result = estimate_tempo(silent_wav)
    assert result.bpm == 0.0
    assert result.confidence == 0.0
