"""Generate a Kdenlive project file from an edit plan (Phase 9).

Kdenlive projects are MLT XML (``<mlt>`` root, ``<producer>`` per source
clip, a ``<playlist>`` referencing them in order, a ``<tractor>`` wrapping
the playlist as the timeline, and ``<transition>`` elements for dissolves).
:mod:`opentimelineio` has no MLT/Kdenlive adapter in this environment
(confirmed via ``otio.adapters.available_adapter_names()`` — only
``otio_json``, ``otioz``, ``otiod`` are registered), so this module
hand-builds the XML directly against the documented MLT project format
rather than going through OTIO.

Caveat: Kdenlive is not installed in this development environment (apt
install requires the user's sudo password and is not automatable), so this
module's output has been validated as well-formed XML with the expected
producer/playlist/tractor structure, but has **not** been opened in the
Kdenlive GUI. To verify: install Kdenlive (``sudo apt install kdenlive``)
and run ``kdenlive <path-to-project.kdenlive>``.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.sax.saxutils import escape

from ai_video_editor.editing.decision_engine import EditAction
from ai_video_editor.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_FPS = 29.97
DEFAULT_WIDTH = 2688
DEFAULT_HEIGHT = 1512

_CROSSFADE_DURATION_SECONDS = 0.5


def _seconds_to_frames(seconds: float, fps: float) -> int:
    return max(0, round(seconds * fps))


def _add_profile(mlt: Element, fps: float, width: int, height: int) -> None:
    # MLT expresses fps as a fraction; 29.97 -> 30000/1001, else n/1.
    if abs(fps - 29.97) < 1e-3:
        num, den = 30000, 1001
    elif abs(fps - 59.94) < 1e-3:
        num, den = 60000, 1001
    else:
        num, den = int(round(fps * 1000)), 1000
    SubElement(
        mlt,
        "profile",
        {
            "description": "ai_video_editor_profile",
            "width": str(width),
            "height": str(height),
            "progressive": "1",
            "sample_aspect_num": "1",
            "sample_aspect_den": "1",
            "display_aspect_num": str(width),
            "display_aspect_den": str(height),
            "frame_rate_num": str(num),
            "frame_rate_den": str(den),
            "colorspace": "709",
        },
    )


def build_kdenlive_project(
    edit_actions: list[EditAction],
    *,
    fps: float = DEFAULT_FPS,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
) -> str:
    """Build a Kdenlive (MLT XML) project document as a string.

    Only ``kind="keep"`` actions become clips, mirroring
    :func:`ai_video_editor.timeline.otio_builder.build_otio_timeline`.
    """
    kept = [a for a in edit_actions if a.kind == "keep"]
    if not kept:
        logger.warning("build_kdenlive_project: no kept actions in edit plan; project is empty")

    mlt = Element(
        "mlt",
        {
            "LC_NUMERIC": "C",
            "version": "7.20.0",
            "title": "AI Video Editor Project",
            "producer": "main_bin",
        },
    )
    _add_profile(mlt, fps, width, height)

    playlist = Element("playlist", {"id": "playlist0"})
    producer_ids: list[str] = []

    for index, action in enumerate(kept):
        producer_id = f"producer{index}"
        producer_ids.append(producer_id)
        in_frame = _seconds_to_frames(action.start_seconds, fps)
        out_frame = max(in_frame, _seconds_to_frames(action.end_seconds, fps) - 1)

        producer = Element(
            "producer",
            {"id": producer_id, "in": "0", "out": str(out_frame)},
        )
        SubElement(producer, "property", {"name": "resource"}).text = escape(action.video_path)
        SubElement(producer, "property", {"name": "mlt_service"}).text = "avformat-novalidate"
        SubElement(producer, "property", {"name": "kdenlive:id"}).text = producer_id
        SubElement(producer, "property", {"name": "kdenlive:comment"}).text = escape(
            action.explanation
        )
        mlt.append(producer)

        SubElement(
            playlist,
            "entry",
            {"producer": producer_id, "in": str(in_frame), "out": str(out_frame)},
        )
        # Chapter markers are emitted as tractor-level kdenlive:marker properties below.

        if action.transition_in == "crossfade" and index > 0:
            fade_frames = _seconds_to_frames(_CROSSFADE_DURATION_SECONDS, fps)
            transition = SubElement(
                mlt,
                "transition",
                {"id": f"transition{index}"},
            )
            SubElement(transition, "property", {"name": "mlt_service"}).text = "luma"
            SubElement(transition, "property", {"name": "a_track"}).text = "0"
            SubElement(transition, "property", {"name": "b_track"}).text = "1"
            SubElement(transition, "property", {"name": "in"}).text = str(in_frame)
            SubElement(transition, "property", {"name": "out"}).text = str(
                in_frame + fade_frames
            )

    mlt.append(playlist)

    tractor = Element("tractor", {"id": "tractor0", "title": "AI Video Editor Timeline"})
    SubElement(tractor, "track", {"producer": "playlist0"})

    for index, action in enumerate(kept):
        if action.chapter_marker:
            marker_frame = _seconds_to_frames(action.start_seconds, fps)
            SubElement(
                tractor,
                "property",
                {"name": f"kdenlive:marker:{marker_frame}"},
            ).text = escape(f"{action.chapter_marker}|{uuid.uuid4().hex[:8]}")

    mlt.append(tractor)

    xml_bytes = tostring(mlt, encoding="utf-8", xml_declaration=True)
    logger.info(
        "build_kdenlive_project: built project with %d producers (%d dropped)",
        len(producer_ids),
        len(edit_actions) - len(kept),
    )
    return xml_bytes.decode("utf-8")


def write_kdenlive_project(
    edit_actions: list[EditAction],
    path: Path,
    *,
    fps: float = DEFAULT_FPS,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
) -> None:
    """Build and write a ``.kdenlive`` project file."""
    xml_text = build_kdenlive_project(edit_actions, fps=fps, width=width, height=height)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(xml_text, encoding="utf-8")
    logger.info("Wrote Kdenlive project to %s", path)
