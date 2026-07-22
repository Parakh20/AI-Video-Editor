from pathlib import Path

from ai_video_editor.audio.beat_detection import detect_beats


def test_detect_beats_finds_beats_in_clicktrack(clicktrack_wav: Path) -> None:
    grid = detect_beats(clicktrack_wav)
    assert grid.beat_count > 0
    assert grid.duration_seconds > 0
    assert grid.tempo_bpm > 0
    assert all(0 <= t <= grid.duration_seconds for t in grid.beat_times)
    assert grid.beat_times == sorted(grid.beat_times)


def test_detect_beats_on_silence_returns_empty(silent_wav: Path) -> None:
    grid = detect_beats(silent_wav)
    assert grid.beat_times == []
    assert grid.tempo_bpm == 0.0
