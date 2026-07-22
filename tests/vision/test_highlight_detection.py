from ai_video_editor.vision.highlight_detection import detect_highlights


def test_detect_highlights_returns_scored_shots(scene_cut_video) -> None:
    highlights = detect_highlights(scene_cut_video)
    assert len(highlights) == 2
    for h in highlights:
        assert 0.0 <= h.highlight_score <= 100.0


def test_highlights_sorted_descending(scene_cut_video) -> None:
    highlights = detect_highlights(scene_cut_video)
    scores = [h.highlight_score for h in highlights]
    assert scores == sorted(scores, reverse=True)
