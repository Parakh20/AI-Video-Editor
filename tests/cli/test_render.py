"""`video-editor render --dry-run` against a synthetic edit plan (no real ffmpeg encode)."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from ai_video_editor.cli.main import app


def _write_edit_plan(tmp_path: Path) -> Path:
    fake_video = tmp_path / "fake_clip.mp4"
    fake_video.touch()
    actions = [
        {
            "kind": "keep",
            "video_path": str(fake_video),
            "start_seconds": 0.0,
            "end_seconds": 3.0,
            "source_shot_indices": [0],
            "score": 82.0,
            "transition_in": "hard_cut",
            "beat_aligned": False,
            "chapter_marker": "Sky — 0.0s",
            "explanation": "kept: score=82.0",
        }
    ]
    path = tmp_path / "edit_plan.json"
    path.write_text(json.dumps(actions, indent=2), encoding="utf-8")
    return path


def test_render_dry_run_does_not_invoke_ffmpeg(runner: CliRunner, tmp_path: Path) -> None:
    edit_plan = _write_edit_plan(tmp_path)
    output = tmp_path / "out.mp4"

    result = runner.invoke(
        app,
        ["render", "--edit-plan", str(edit_plan), "--preset", "1080p", "--output", str(output), "--dry-run"],
    )
    assert result.exit_code == 0, result.output
    assert not output.exists()
    assert "dry run" in result.output.lower()


def test_render_unknown_preset_fails(runner: CliRunner, tmp_path: Path) -> None:
    edit_plan = _write_edit_plan(tmp_path)
    result = runner.invoke(
        app,
        ["render", "--edit-plan", str(edit_plan), "--preset", "not_a_real_preset", "--dry-run"],
    )
    assert result.exit_code != 0


def test_render_missing_edit_plan_exits_nonzero(runner: CliRunner, tmp_path: Path) -> None:
    result = runner.invoke(app, ["render", "--edit-plan", str(tmp_path / "missing.json")])
    assert result.exit_code != 0


FIXTURE_GPX = Path(__file__).resolve().parents[1] / "fixtures" / "sample_ride.gpx"


def test_render_dry_run_reports_gps_overlay(runner: CliRunner, tmp_path: Path) -> None:
    edit_plan = _write_edit_plan(tmp_path)
    result = runner.invoke(
        app,
        [
            "render",
            "--edit-plan", str(edit_plan),
            "--preset", "1080p",
            "--gps", str(FIXTURE_GPX),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    # Rich wraps long lines at the terminal width (CI paths are long), so
    # compare with whitespace collapsed.
    assert "with gps overlay" in " ".join(result.output.lower().split())


def test_gps_overlay_fragment_is_time_synced(tmp_path: Path) -> None:
    from ai_video_editor.cli.commands.render import _gps_overlay_fragment, load_edit_plan_json

    kept = load_edit_plan_json(_write_edit_plan(tmp_path))
    overlay = _gps_overlay_fragment(FIXTURE_GPX, kept, crossfade_s=0.5)
    assert overlay is not None
    # Time-synced: multiple drawtext filters, each gated on an output-time window.
    assert overlay.count("drawtext=text=") >= 2
    assert "enable='between(t," in overlay


def test_gps_overlay_fragment_none_without_gps(tmp_path: Path) -> None:
    from ai_video_editor.cli.commands.render import _gps_overlay_fragment, load_edit_plan_json

    kept = load_edit_plan_json(_write_edit_plan(tmp_path))
    assert _gps_overlay_fragment(None, kept, crossfade_s=0.5) is None


def test_gps_graphic_overlays_builds_specs_with_gps(tmp_path: Path) -> None:
    from ai_video_editor.cli.commands.render import _gps_graphic_overlays, load_edit_plan_json
    from ai_video_editor.render.presets import get_preset

    kept = load_edit_plan_json(_write_edit_plan(tmp_path))
    preset = get_preset("1080p")
    specs = _gps_graphic_overlays(FIXTURE_GPX, kept, crossfade_s=0.5, preset=preset, work_dir=tmp_path)
    assert len(specs) == 2
    for spec in specs:
        assert spec.image_path.exists()
        if spec.marker is not None:
            assert spec.marker.image_path.exists()
            assert len(spec.marker.windows) > 0


def test_gps_graphic_overlays_empty_without_gps(tmp_path: Path) -> None:
    from ai_video_editor.cli.commands.render import _gps_graphic_overlays, load_edit_plan_json
    from ai_video_editor.render.presets import get_preset

    kept = load_edit_plan_json(_write_edit_plan(tmp_path))
    preset = get_preset("1080p")
    assert _gps_graphic_overlays(None, kept, crossfade_s=0.5, preset=preset, work_dir=tmp_path) == []


def _spec_bbox(spec) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1) bounding box for a GraphicOverlaySpec, read from its PNG."""
    from PIL import Image

    with Image.open(spec.image_path) as img:
        width, height = img.size
    return (spec.x, spec.y, spec.x + width, spec.y + height)


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1


def test_gps_graphic_overlays_no_collision_on_vertical_preset(tmp_path: Path) -> None:
    """Regression test: elevation strip and route map must not overlap on
    narrow/vertical presets (e.g. instagram_reel, 1080x1920), where the
    side-by-side bottom-center/bottom-right layout used for 16:9 presets
    doesn't leave enough horizontal room."""
    from ai_video_editor.cli.commands.render import _gps_graphic_overlays, load_edit_plan_json
    from ai_video_editor.render.presets import get_preset

    kept = load_edit_plan_json(_write_edit_plan(tmp_path))
    preset = get_preset("instagram_reel")
    specs = _gps_graphic_overlays(FIXTURE_GPX, kept, crossfade_s=0.5, preset=preset, work_dir=tmp_path)
    assert len(specs) == 2

    boxes = [_spec_bbox(spec) for spec in specs]
    assert not _boxes_overlap(boxes[0], boxes[1])

    for x0, y0, x1, y1 in boxes:
        assert x0 >= 0
        assert y0 >= 0
        assert x1 <= preset.width
        assert y1 <= preset.height


def test_gps_graphic_overlays_landscape_layout_unchanged(tmp_path: Path) -> None:
    """Landscape presets have enough width for the existing side-by-side
    bottom-center (elevation) / bottom-right (route map) layout -- confirm
    the fix doesn't regress the already-approved behavior."""
    from ai_video_editor.cli.commands.render import _gps_graphic_overlays, load_edit_plan_json
    from ai_video_editor.render.presets import get_preset

    kept = load_edit_plan_json(_write_edit_plan(tmp_path))
    preset = get_preset("1080p")
    specs = _gps_graphic_overlays(FIXTURE_GPX, kept, crossfade_s=0.5, preset=preset, work_dir=tmp_path)
    assert len(specs) == 2

    elevation_spec, route_spec = specs
    boxes = [_spec_bbox(spec) for spec in specs]
    assert not _boxes_overlap(boxes[0], boxes[1])

    # Side-by-side (not stacked): each box is independently bottom-anchored
    # with the standard margin, rather than the route map sitting directly
    # above the elevation strip.
    margin = 20
    _, elev_y0, _, elev_y1 = boxes[0]
    _, route_y0, _, route_y1 = boxes[1]
    assert elev_y1 == preset.height - margin
    assert route_y1 == preset.height - margin
    assert route_y0 != elev_y1 + margin  # not stacked directly above the strip

    for x0, y0, x1, y1 in boxes:
        assert x0 >= 0
        assert y0 >= 0
        assert x1 <= preset.width
        assert y1 <= preset.height
