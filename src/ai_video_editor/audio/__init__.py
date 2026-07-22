"""Audio analysis: beats, tempo, speech, subtitles, music structure, silence."""

from ai_video_editor.audio.beat_detection import BeatGrid, detect_beats
from ai_video_editor.audio.music_analysis import MusicAnalysis, MusicSection, analyze_music
from ai_video_editor.audio.silence_detection import SilenceInterval, detect_silence
from ai_video_editor.audio.speech_detection import SpeechSegment, detect_speech_segments
from ai_video_editor.audio.subtitle_generation import (
    SubtitleResult,
    TranscriptSegment,
    generate_subtitles,
)
from ai_video_editor.audio.tempo_estimation import TempoEstimate, estimate_tempo

__all__ = [
    "BeatGrid",
    "detect_beats",
    "TempoEstimate",
    "estimate_tempo",
    "SpeechSegment",
    "detect_speech_segments",
    "SubtitleResult",
    "TranscriptSegment",
    "generate_subtitles",
    "MusicAnalysis",
    "MusicSection",
    "analyze_music",
    "SilenceInterval",
    "detect_silence",
]
