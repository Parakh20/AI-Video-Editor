from ai_video_editor.vision.duplicate_frame_detection import (
    average_hash,
    find_duplicate_frames,
    hamming_distance,
)


def test_identical_frames_have_zero_hamming_distance(static_gray_video) -> None:
    import cv2

    cap = cv2.VideoCapture(str(static_gray_video))
    ok1, frame1 = cap.read()
    ok2, frame2 = cap.read()
    cap.release()
    assert ok1 and ok2
    assert hamming_distance(average_hash(frame1), average_hash(frame2)) == 0


def test_static_video_is_all_one_duplicate_run(static_gray_video) -> None:
    groups = find_duplicate_frames(static_gray_video, stride=1)
    assert len(groups) == 1
    assert len(groups[0].frame_indices) >= 25


def test_scene_cut_video_has_two_duplicate_runs(scene_cut_video) -> None:
    groups = find_duplicate_frames(scene_cut_video, stride=1)
    assert len(groups) == 2
