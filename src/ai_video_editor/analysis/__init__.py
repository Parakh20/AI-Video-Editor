"""Clip scoring: combines vision, audio, and GPS signals into per-shot scores."""

from ai_video_editor.analysis.scoring import (
    ScoreWeights,
    SegmentScore,
    SubScores,
    score_shot,
    score_video,
    write_scores_json,
)

__all__ = [
    "ScoreWeights",
    "SegmentScore",
    "SubScores",
    "score_shot",
    "score_video",
    "write_scores_json",
]
