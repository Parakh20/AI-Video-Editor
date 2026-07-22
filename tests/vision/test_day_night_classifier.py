from ai_video_editor.vision.day_night_classifier import (
    LightingCondition,
    classify_video_lighting,
)


def test_bright_video_classified_as_day(bright_day_video) -> None:
    results = classify_video_lighting(bright_day_video, stride=1)
    assert results
    assert all(r.condition == LightingCondition.DAY for r in results)


def test_dark_video_classified_as_night(dark_night_video) -> None:
    results = classify_video_lighting(dark_night_video, stride=1)
    assert results
    assert all(r.condition == LightingCondition.NIGHT for r in results)
