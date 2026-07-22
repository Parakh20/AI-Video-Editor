"""Fast unit tests for filter-graph string assembly (no ffmpeg invoked)."""

from __future__ import annotations

from pathlib import Path

from ai_video_editor.editing.decision_engine import EditAction
from ai_video_editor.gps.overlays import MarkerWindow
from ai_video_editor.render.presets import get_preset
from ai_video_editor.render.renderer import (
    GraphicOverlaySpec,
    MarkerOverlaySpec,
    _build_filter_complex,
    _build_inputs,
)


def _one_kept_action() -> EditAction:
    return EditAction(
        kind="keep",
        video_path="clip.mp4",
        start_seconds=0.0,
        end_seconds=2.0,
        source_shot_indices=[0],
        explanation="test clip",
    )


def test_build_filter_complex_without_graphic_overlays_unchanged(tmp_path: Path) -> None:
    kept = [_one_kept_action()]
    input_args, clip_inputs = _build_inputs(kept)
    filter_complex, v_label, a_label = _build_filter_complex(
        kept, clip_inputs, get_preset("1080p"), crossfade_s=0.5, overlay_drawtext=None
    )
    assert "movie=" not in filter_complex
    assert v_label == "v0"


def test_build_filter_complex_composites_background_image(tmp_path: Path) -> None:
    bg = tmp_path / "bg.png"
    bg.touch()
    kept = [_one_kept_action()]
    input_args, clip_inputs = _build_inputs(kept)
    spec = GraphicOverlaySpec(image_path=bg, x=40, y=800)
    filter_complex, v_label, a_label = _build_filter_complex(
        kept,
        clip_inputs,
        get_preset("1080p"),
        crossfade_s=0.5,
        overlay_drawtext=None,
        graphic_overlays=[spec],
    )
    assert f"movie={bg}" in filter_complex
    assert "overlay=40:800" in filter_complex
    assert v_label != "v0"


def test_build_filter_complex_composites_windowed_marker(tmp_path: Path) -> None:
    bg = tmp_path / "bg.png"
    dot = tmp_path / "dot.png"
    bg.touch()
    dot.touch()
    kept = [_one_kept_action()]
    input_args, clip_inputs = _build_inputs(kept)
    windows = [
        MarkerWindow(start_s=0.0, end_s=1.0, x_px=10.0, y_px=20.0),
        MarkerWindow(start_s=1.0, end_s=2.0, x_px=30.0, y_px=40.0),
    ]
    spec = GraphicOverlaySpec(
        image_path=bg, x=40, y=800, marker=MarkerOverlaySpec(image_path=dot, windows=windows)
    )
    filter_complex, v_label, a_label = _build_filter_complex(
        kept,
        clip_inputs,
        get_preset("1080p"),
        crossfade_s=0.5,
        overlay_drawtext=None,
        graphic_overlays=[spec],
    )
    assert f"movie={dot}" in filter_complex
    assert filter_complex.count("enable='between(t,") == 2
    assert "split=2" in filter_complex
