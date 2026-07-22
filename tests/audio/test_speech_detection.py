from pathlib import Path

from ai_video_editor.audio.speech_detection import detect_speech_segments


def test_detect_speech_segments_on_speech_like_signal(speech_like_wav: Path) -> None:
    segments = detect_speech_segments(speech_like_wav, energy_threshold_db=-35.0)
    assert len(segments) >= 1
    assert all(s.end > s.start for s in segments)


def test_detect_speech_segments_on_silence_is_empty(silent_wav: Path) -> None:
    segments = detect_speech_segments(silent_wav)
    assert segments == []
