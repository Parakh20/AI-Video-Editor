import cv2
import numpy as np

from ai_video_editor.vision.scene_classifier import SceneCategory, classify_frame_scene


def test_blue_sky_like_frame_classified_as_sky() -> None:
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    hsv = np.zeros((120, 160, 3), dtype=np.uint8)
    hsv[:, :, 0] = 105  # blue hue
    hsv[:, :, 1] = 60
    hsv[:, :, 2] = 220
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    result = classify_frame_scene(frame)
    assert result.category == SceneCategory.SKY


def test_green_frame_classified_as_foliage() -> None:
    hsv = np.zeros((120, 160, 3), dtype=np.uint8)
    hsv[:, :, 0] = 60  # green hue
    hsv[:, :, 1] = 180
    hsv[:, :, 2] = 150
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    result = classify_frame_scene(frame)
    assert result.category == SceneCategory.FOLIAGE


def test_scene_classification_fractions_bounded() -> None:
    frame = np.random.default_rng(0).integers(0, 255, (120, 160, 3), dtype=np.uint8)
    result = classify_frame_scene(frame)
    assert 0.0 <= result.sky_fraction <= 1.0
    assert 0.0 <= result.foliage_fraction <= 1.0
    assert 0.0 <= result.edge_density <= 1.0
