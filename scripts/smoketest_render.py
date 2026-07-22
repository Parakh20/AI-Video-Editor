"""Real-footage smoke test for Phase 10-12 (effects, overlays, rendering).

Renders a short clip from two real Kolar clips with a crossfade transition
and a burned-in drawtext overlay (a static stat, since no real GPS data
exists for this footage), then separately proves the speed-ramp effect
filter against a single real clip via a direct ffmpeg call.

Not part of the pytest suite (touches real, large source files and takes
longer than a unit test should) — run manually:
    .venv/bin/python scripts/smoketest_render.py
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from ai_video_editor.editing.decision_engine import EditAction
from ai_video_editor.editing.effects import speed_ramp_filter
from ai_video_editor.gps.overlays import drawtext_filter
from ai_video_editor.render.presets import get_preset
from ai_video_editor.render.renderer import render_edit_plan
from ai_video_editor.utils.logging import get_logger

logger = get_logger("scripts.smoketest_render")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "assets" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "output"


def main() -> None:
    clip_a = RAW_DIR / "DJI_20260712185038_0133_D.MP4"
    clip_b = RAW_DIR / "DJI_20260712213106_0141_D.MP4"
    assert clip_a.exists() and clip_b.exists(), "expected real Kolar clips not found"

    actions = [
        EditAction(
            kind="keep",
            video_path=str(clip_a),
            start_seconds=1.0,
            end_seconds=6.0,
            source_shot_indices=[0],
            explanation="smoke test clip A (first real clip)",
        ),
        EditAction(
            kind="keep",
            video_path=str(clip_b),
            start_seconds=1.0,
            end_seconds=6.0,
            source_shot_indices=[1],
            transition_in="crossfade",
            explanation="smoke test clip B, crossfades in over clip A",
        ),
    ]

    overlay = drawtext_filter("42 km/h")
    output_path = OUTPUT_DIR / "render_smoketest.mp4"

    logger.info("Rendering real-footage smoke test to %s", output_path)
    result = render_edit_plan(
        actions,
        output_path,
        get_preset("1080p"),
        overlay_drawtext=overlay,
        prefer_gpu=True,
    )
    logger.info("Render complete: encoder=%s clips=%d", result.used_encoder, result.clip_count)

    probe = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration,size",
            "-show_entries", "stream=width,height,codec_name",
            "-of", "default=noprint_wrappers=1",
            str(output_path),
        ],
        capture_output=True, text=True, check=True,
    )
    logger.info("ffprobe output:\n%s", probe.stdout)

    # Separately prove the speed-ramp effect filter against one real clip.
    speed_output = OUTPUT_DIR / "speed_ramp_smoketest.mp4"
    speed_filter = speed_ramp_filter(2.0)
    cmd = [
        "ffmpeg", "-y",
        "-ss", "1.0", "-t", "5.0", "-i", str(clip_a),
        "-vf", speed_filter,
        "-an",
        "-c:v", "h264_nvenc", "-preset", "p5",
        str(speed_output),
    ]
    logger.info("Rendering speed-ramp smoke test to %s", speed_output)
    speed_result = subprocess.run(cmd, capture_output=True, text=True)
    if speed_result.returncode != 0:
        logger.warning("nvenc speed-ramp render failed, retrying with libx264: %s", speed_result.stderr[-500:])
        cmd[cmd.index("h264_nvenc")] = "libx264"
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    logger.info("Speed-ramp smoke test complete: %s", speed_output)

    print(f"OK: {output_path} ({output_path.stat().st_size} bytes)")
    print(f"OK: {speed_output} ({speed_output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
