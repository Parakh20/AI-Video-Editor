from ai_video_editor.vision.camera_shake import analyze_video_shake
from ai_video_editor.vision.motion_intensity import analyze_video_motion_intensity
from ai_video_editor.vision.optical_flow import analyze_video_flow


def test_static_video_has_near_zero_flow(static_gray_video) -> None:
    flow = analyze_video_flow(static_gray_video, stride=1)
    assert flow, "expected at least one flow pair"
    assert all(f.mean_magnitude < 0.5 for f in flow)


def test_moving_video_has_higher_motion_than_static(moving_gradient_video, static_gray_video) -> None:
    moving = analyze_video_motion_intensity(moving_gradient_video, stride=1)
    static = analyze_video_motion_intensity(static_gray_video, stride=1)
    assert moving.intensity_score > static.intensity_score


def test_static_video_is_very_stable(static_gray_video) -> None:
    report = analyze_video_shake(static_gray_video, stride=1)
    assert report.stability_score > 90.0


def test_motion_intensity_bounded(moving_gradient_video) -> None:
    report = analyze_video_motion_intensity(moving_gradient_video, stride=1)
    assert 0.0 <= report.intensity_score <= 100.0
