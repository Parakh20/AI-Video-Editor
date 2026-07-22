"""Build an editor-agnostic OpenTimelineIO timeline from an edit plan (Phase 8).

Consumes :class:`ai_video_editor.editing.decision_engine.EditAction` (kept
actions only — dropped actions are not represented on the timeline, since
OTIO has no notion of "considered and rejected") and produces a plain
:class:`opentimelineio.schema.Timeline`: one video track with clips, a
transition between clips where the edit plan called for a crossfade,
markers for chapter points, and generation metadata on the timeline itself.

No editor-specific extensions are used here — this is meant to be a neutral
interchange format that :mod:`ai_video_editor.timeline.kdenlive_export` (or,
in future, a DaVinci Resolve or Premiere exporter) can consume.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

import opentimelineio as otio

from ai_video_editor.editing.decision_engine import EditAction
from ai_video_editor.utils.logging import get_logger

logger = get_logger(__name__)

# Frame rate assumed for RationalTime conversion. The real source footage is
# ~29.97fps (see EDIT-PLAN.md precedent); this is a parameter, not hardcoded
# for correctness, but this is the sane default for this project's footage.
DEFAULT_FPS = 29.97

_CROSSFADE_DURATION_SECONDS = 0.5


def _seconds_to_rational_time(seconds: float, fps: float) -> otio.opentime.RationalTime:
    return otio.opentime.RationalTime(seconds * fps, fps)


def build_otio_timeline(
    edit_actions: list[EditAction],
    *,
    name: str = "AI Video Editor Timeline",
    fps: float = DEFAULT_FPS,
    song_path: str | None = None,
    source_project: str | None = None,
) -> otio.schema.Timeline:
    """Build a single-video-track OTIO timeline from kept edit actions.

    Only ``kind="keep"`` actions produce clips; ``kind="drop"`` actions are
    skipped (they exist in the edit plan purely for auditability). Actions
    are placed on the timeline in the order given, back-to-back.
    """
    kept = [a for a in edit_actions if a.kind == "keep"]
    if not kept:
        logger.warning("build_otio_timeline: no kept actions in edit plan; timeline is empty")

    timeline = otio.schema.Timeline(name=name)
    track = otio.schema.Track(name="V1", kind=otio.schema.TrackKind.Video)
    timeline.tracks.append(track)

    record_cursor = otio.opentime.RationalTime(0, fps)

    for index, action in enumerate(kept):
        source_start = _seconds_to_rational_time(action.start_seconds, fps)
        source_duration = _seconds_to_rational_time(action.duration_seconds, fps)

        media_ref = otio.schema.ExternalReference(
            target_url=action.video_path
            if action.video_path.startswith("file://")
            else Path(action.video_path).resolve().as_uri(),
            available_range=otio.opentime.TimeRange(
                start_time=otio.opentime.RationalTime(0, fps),
                duration=source_duration + source_start,
            ),
        )

        clip = otio.schema.Clip(
            name=f"{Path(action.video_path).stem}_{index:03d}",
            media_reference=media_ref,
            source_range=otio.opentime.TimeRange(
                start_time=source_start,
                duration=source_duration,
            ),
        )
        clip.metadata["ai_video_editor"] = {
            "score": action.score,
            "explanation": action.explanation,
            "beat_aligned": action.beat_aligned,
            "source_shot_indices": action.source_shot_indices,
        }

        track.append(clip)

        if action.transition_in == "crossfade" and index > 0:
            transition_duration = _seconds_to_rational_time(_CROSSFADE_DURATION_SECONDS, fps)
            transition = otio.schema.Transition(
                name=f"crossfade_{index:03d}",
                transition_type=otio.schema.TransitionTypes.SMPTE_Dissolve,
                in_offset=transition_duration,
                out_offset=transition_duration,
            )
            # Insert immediately before the clip we just appended.
            track.insert(len(track) - 1, transition)

        if action.chapter_marker:
            marker = otio.schema.Marker(
                name=action.chapter_marker,
                marked_range=otio.opentime.TimeRange(
                    start_time=record_cursor,
                    duration=otio.opentime.RationalTime(0, fps),
                ),
                color=otio.schema.MarkerColor.GREEN,
            )
            clip.markers.append(marker)

        record_cursor += source_duration

    metadata: dict[str, Any] = {
        "generated_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "generator": "ai_video_editor.timeline.otio_builder",
        "clip_count": len(kept),
    }
    if song_path:
        metadata["song_path"] = song_path
    if source_project:
        metadata["source_project"] = source_project
    timeline.metadata["ai_video_editor"] = metadata

    logger.info(
        "build_otio_timeline: built timeline '%s' with %d clips (%d dropped)",
        name,
        len(kept),
        len(edit_actions) - len(kept),
    )
    return timeline


def write_otio(timeline: otio.schema.Timeline, path: Path) -> None:
    """Write a timeline to disk as a ``.otio`` (OTIO JSON) file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    otio.adapters.write_to_file(timeline, str(path))
    logger.info("Wrote OTIO timeline to %s", path)
