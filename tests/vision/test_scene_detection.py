from ai_video_editor.vision.scene_detection import detect_shots


def test_scene_cut_video_produces_two_shots(scene_cut_video) -> None:
    shots = detect_shots(scene_cut_video)
    assert len(shots) == 2
    assert shots[0].start_seconds == 0.0
    assert shots[1].start_seconds > shots[0].start_seconds


def test_static_video_produces_one_shot(static_gray_video) -> None:
    shots = detect_shots(static_gray_video)
    assert len(shots) == 1
