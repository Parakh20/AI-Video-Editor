from pathlib import Path

from ai_video_editor.audio.music_analysis import analyze_music


def test_analyze_music_on_clicktrack(clicktrack_wav: Path) -> None:
    analysis = analyze_music(clicktrack_wav)
    assert analysis.duration_seconds > 0
    assert len(analysis.rms_curve) == len(analysis.rms_times)
    assert len(analysis.rms_curve) > 0
    assert analysis.spectral_centroid_mean_hz > 0
    assert analysis.spectral_bandwidth_mean_hz > 0
    for section in analysis.sections:
        assert section.energy_label in {"low", "mid", "high"}
        assert section.end > section.start


def test_analyze_music_on_silence_is_empty(silent_wav: Path) -> None:
    analysis = analyze_music(silent_wav)
    assert analysis.rms_curve == []
    assert analysis.sections == []
