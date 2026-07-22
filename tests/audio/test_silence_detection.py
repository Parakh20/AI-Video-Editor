from pathlib import Path

from ai_video_editor.audio.silence_detection import detect_silence


def test_detect_silence_finds_middle_gap(loud_then_silent_wav: Path) -> None:
    silences = detect_silence(loud_then_silent_wav, threshold_db=-35.0, min_duration=0.3)
    assert len(silences) >= 1
    middle = max(silences, key=lambda s: s.duration)
    assert 1.5 <= middle.start <= 2.5
    assert 3.5 <= middle.end <= 4.5


def test_detect_silence_on_fully_silent_track(silent_wav: Path) -> None:
    silences = detect_silence(silent_wav, min_duration=0.1)
    assert len(silences) == 1
    assert silences[0].duration >= 2.5


def test_detect_silence_on_loud_tone_finds_none(tone_wav: Path) -> None:
    silences = detect_silence(tone_wav, threshold_db=-40.0)
    assert silences == []
