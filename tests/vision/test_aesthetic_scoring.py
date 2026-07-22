import numpy as np

from ai_video_editor.vision.aesthetic_scoring import score_frame_aesthetics


def test_flat_gray_frame_scores_low() -> None:
    flat = np.full((120, 160, 3), 128, dtype=np.uint8)
    result = score_frame_aesthetics(flat)
    assert result.score < 20.0


def test_colorful_high_contrast_frame_scores_higher_than_flat() -> None:
    flat = np.full((120, 160, 3), 128, dtype=np.uint8)
    rng = np.random.default_rng(1)
    varied = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)

    flat_score = score_frame_aesthetics(flat).score
    varied_score = score_frame_aesthetics(varied).score
    assert varied_score > flat_score


def test_aesthetic_score_bounded() -> None:
    rng = np.random.default_rng(2)
    frame = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
    result = score_frame_aesthetics(frame)
    assert 0.0 <= result.score <= 100.0
