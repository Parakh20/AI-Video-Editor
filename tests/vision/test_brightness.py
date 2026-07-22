from ai_video_editor.vision.brightness import analyze_video_brightness, frame_brightness
import numpy as np


def test_frame_brightness_black_is_zero() -> None:
    black = np.zeros((10, 10, 3), dtype=np.uint8)
    assert frame_brightness(black) == 0.0


def test_frame_brightness_white_is_one() -> None:
    white = np.full((10, 10, 3), 255, dtype=np.uint8)
    assert frame_brightness(white) == 1.0


def test_bright_video_has_high_mean_brightness(bright_day_video) -> None:
    report = analyze_video_brightness(bright_day_video, stride=1)
    assert report.mean_brightness > 0.7


def test_dark_video_has_low_mean_brightness(dark_night_video) -> None:
    report = analyze_video_brightness(dark_night_video, stride=1)
    assert report.mean_brightness < 0.3


def test_bright_video_brighter_than_dark_video(bright_day_video, dark_night_video) -> None:
    bright = analyze_video_brightness(bright_day_video, stride=1)
    dark = analyze_video_brightness(dark_night_video, stride=1)
    assert bright.mean_brightness > dark.mean_brightness
