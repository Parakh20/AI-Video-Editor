import os
from pathlib import Path

import pytest

from ai_video_editor.audio.subtitle_generation import generate_subtitles


def _whisper_model_cached(name: str) -> bool:
    """Whether whisper.load_model(name) would load from disk instead of downloading."""
    cache_root = Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache"))
    return (cache_root / "whisper" / f"{name}.pt").is_file()


pytestmark = pytest.mark.skipif(
    not _whisper_model_cached("tiny"),
    reason="Whisper 'tiny' model is not cached locally; skipped to avoid a model download "
    "in CI. Enable with: python -c \"import whisper; whisper.load_model('tiny')\"",
)


@pytest.mark.slow
def test_generate_subtitles_on_silence_has_no_speech(silent_wav: Path) -> None:
    result = generate_subtitles(silent_wav, model_size="tiny")
    assert result.has_speech is False
    assert result.segments == []
    assert result.srt_text == ""


@pytest.mark.slow
def test_generate_subtitles_srt_format_when_speech_present(speech_like_wav: Path) -> None:
    # A synthetic tone-modulated signal is not real speech, so Whisper may or may
    # may not emit segments; this test only checks that the pipeline runs
    # end-to-end without error and that any emitted SRT is well-formed.
    result = generate_subtitles(speech_like_wav, model_size="tiny")
    if result.segments:
        assert "-->" in result.srt_text
        assert result.srt_text.strip().startswith("1")
    else:
        assert result.srt_text == ""
