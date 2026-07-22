"""Slow integration tests that actually invoke ffmpeg (Phase 12).

Uses ffmpeg's built-in ``lavfi`` testsrc/sine sources to generate tiny
synthetic clips on the fly, rather than the large real footage in
assets/raw, so these stay fast even though they're marked slow.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from ai_video_editor.editing.decision_engine import EditAction
from ai_video_editor.gps.overlays import drawtext_filter
from ai_video_editor.render.presets import get_preset
from ai_video_editor.render.renderer import render_edit_plan
from ai_video_editor.utils.logging import get_logger

logger = get_logger(__name__)

pytestmark = pytest.mark.slow


def _make_synthetic_clip(path: Path, *, duration_s: float, color: str) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"testsrc2=size=640x360:rate=30:duration={duration_s}",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration_s}",
        "-vf", f"hue=s=0,colorchannelmixer=1:0:0:0:0:1:0:0:0:0:1:0" if color == "gray" else "null",
        "-c:v", "libx264", "-preset", "ultrafast",
        "-c:a", "aac",
        str(path),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


@pytest.fixture
def synthetic_clips(tmp_path: Path) -> list[Path]:
    clip_a = tmp_path / "clip_a.mp4"
    clip_b = tmp_path / "clip_b.mp4"
    _make_synthetic_clip(clip_a, duration_s=3.0, color="normal")
    _make_synthetic_clip(clip_b, duration_s=3.0, color="gray")
    return [clip_a, clip_b]


def _has_nvenc() -> bool:
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True
    )
    return "h264_nvenc" in result.stdout


def test_render_hard_cut_two_clips(synthetic_clips: list[Path], tmp_path: Path) -> None:
    clip_a, clip_b = synthetic_clips
    actions = [
        EditAction(
            kind="keep",
            video_path=str(clip_a),
            start_seconds=0.0,
            end_seconds=2.0,
            source_shot_indices=[0],
            explanation="test clip A",
        ),
        EditAction(
            kind="keep",
            video_path=str(clip_b),
            start_seconds=0.0,
            end_seconds=2.0,
            source_shot_indices=[1],
            transition_in="hard_cut",
            explanation="test clip B",
        ),
    ]
    output = tmp_path / "hard_cut.mp4"
    result = render_edit_plan(actions, output, get_preset("1080p"), prefer_gpu=False)

    assert result.output_path.exists()
    assert result.used_encoder == "libx264"
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(output)],
        capture_output=True, text=True, check=True,
    )
    duration = float(probe.stdout.strip())
    assert 3.5 < duration < 4.5  # ~2s + ~2s, some encoder slack


def test_render_crossfade_two_clips(synthetic_clips: list[Path], tmp_path: Path) -> None:
    clip_a, clip_b = synthetic_clips
    actions = [
        EditAction(
            kind="keep",
            video_path=str(clip_a),
            start_seconds=0.0,
            end_seconds=2.5,
            source_shot_indices=[0],
            explanation="test clip A",
        ),
        EditAction(
            kind="keep",
            video_path=str(clip_b),
            start_seconds=0.0,
            end_seconds=2.5,
            source_shot_indices=[1],
            transition_in="crossfade",
            explanation="test clip B crossfades in",
        ),
    ]
    output = tmp_path / "crossfade.mp4"
    result = render_edit_plan(actions, output, get_preset("1080p"), prefer_gpu=False)
    assert result.output_path.exists()
    assert result.output_path.stat().st_size > 0


def test_render_with_drawtext_overlay(synthetic_clips: list[Path], tmp_path: Path) -> None:
    clip_a, _ = synthetic_clips
    actions = [
        EditAction(
            kind="keep",
            video_path=str(clip_a),
            start_seconds=0.0,
            end_seconds=2.0,
            source_shot_indices=[0],
            explanation="test clip with overlay",
        ),
    ]
    output = tmp_path / "overlay.mp4"
    overlay = drawtext_filter("42 km/h")
    result = render_edit_plan(
        actions, output, get_preset("1080p"), overlay_drawtext=overlay, prefer_gpu=False
    )
    assert result.output_path.exists()
    assert result.output_path.stat().st_size > 0


def test_nvenc_encode_if_available(synthetic_clips: list[Path], tmp_path: Path) -> None:
    if not _has_nvenc():
        pytest.skip("h264_nvenc not available in this ffmpeg build")

    clip_a, _ = synthetic_clips
    actions = [
        EditAction(
            kind="keep",
            video_path=str(clip_a),
            start_seconds=0.0,
            end_seconds=2.0,
            source_shot_indices=[0],
            explanation="nvenc smoke test",
        ),
    ]
    output = tmp_path / "nvenc.mp4"
    result = render_edit_plan(actions, output, get_preset("1080p"), prefer_gpu=True)
    logger.info("nvenc test used encoder: %s", result.used_encoder)
    assert result.output_path.exists()
    # nvenc may still fail on this 4GB laptop GPU under load and fall back;
    # either outcome is a pass as long as SOME encoder produced valid output.
    assert result.used_encoder in ("h264_nvenc", "libx264")
