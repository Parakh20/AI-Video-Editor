from ai_video_editor.vision.blur_detection import analyze_video_blur
from ai_video_editor.vision.sharpness_scoring import score_clip_sharpness


def test_blurry_video_flagged_as_blurry(blurry_video) -> None:
    report = analyze_video_blur(blurry_video, stride=1)
    assert report.blurry_fraction > 0.8


def test_sharp_video_not_flagged_as_blurry(sharp_checkerboard_video) -> None:
    report = analyze_video_blur(sharp_checkerboard_video, stride=1)
    assert report.blurry_fraction < 0.2


def test_sharp_video_scores_higher_than_blurry(sharp_checkerboard_video, blurry_video) -> None:
    sharp = score_clip_sharpness(sharp_checkerboard_video, stride=1)
    blurry = score_clip_sharpness(blurry_video, stride=1)
    assert sharp.mean_score > blurry.mean_score


def test_sharpness_score_bounded(sharp_checkerboard_video) -> None:
    report = score_clip_sharpness(sharp_checkerboard_video, stride=1)
    assert 0.0 <= report.mean_score <= 100.0
