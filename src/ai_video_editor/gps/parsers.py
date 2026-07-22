"""Load GPX and FIT ride files into a common :class:`Track` representation.

If no GPS file is available for a given clip or ride (common for action-cam
footage with no paired computer/head unit), every downstream consumer in
``ai_video_editor.gps`` accepts ``Track | None`` and treats ``None`` as "skip
GPS-derived features for this ride" rather than an error. Callers that don't
have a GPS file simply never call these parsers.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import fitparse
import gpxpy

from ai_video_editor.gps.models import Track, TrackPoint
from ai_video_editor.utils.logging import get_logger

logger = get_logger(__name__)


def load_gpx(path: str | Path) -> Track:
    """Parse a GPX file into a :class:`Track`.

    Points missing a timestamp or coordinates are skipped and logged; a
    malformed file that cannot be parsed at all raises ``gpxpy`` errors, but
    per-point issues never abort the whole load.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        gpx = gpxpy.parse(f)

    points: list[TrackPoint] = []
    skipped = 0
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                if point.time is None or point.latitude is None or point.longitude is None:
                    skipped += 1
                    continue

                extensions = _extract_gpx_extensions(point)
                raw_hr = extensions.get("heart_rate_bpm")
                points.append(
                    TrackPoint(
                        timestamp=_ensure_utc(point.time),
                        lat=point.latitude,
                        lon=point.longitude,
                        elevation_m=point.elevation,
                        heart_rate_bpm=int(raw_hr) if raw_hr is not None else None,
                        cadence_rpm=extensions.get("cadence_rpm"),
                        power_w=extensions.get("power_w"),
                    )
                )

    if skipped:
        logger.warning("Skipped %d GPX point(s) missing time/lat/lon in %s", skipped, path)

    points.sort(key=lambda p: p.timestamp)
    logger.info("Loaded %d point(s) from GPX %s", len(points), path)
    return Track(source_path=str(path), points=points)


def load_fit(path: str | Path) -> Track:
    """Parse a FIT file into a :class:`Track`, reading ``record`` messages.

    Points missing a timestamp or coordinates (FIT stores lat/lon as
    semicircles; some records, e.g. pure HR-strap samples, carry no position)
    are skipped and logged rather than aborting the load.
    """
    path = Path(path)
    fit = fitparse.FitFile(str(path))

    points: list[TrackPoint] = []
    skipped = 0
    for record in fit.get_messages("record"):
        values = {field.name: field.value for field in record}

        timestamp = values.get("timestamp")
        lat_semi = values.get("position_lat")
        lon_semi = values.get("position_long")
        if timestamp is None or lat_semi is None or lon_semi is None:
            skipped += 1
            continue

        points.append(
            TrackPoint(
                timestamp=_ensure_utc(timestamp),
                lat=_semicircles_to_degrees(lat_semi),
                lon=_semicircles_to_degrees(lon_semi),
                elevation_m=values.get("altitude") or values.get("enhanced_altitude"),
                heart_rate_bpm=values.get("heart_rate"),
                cadence_rpm=values.get("cadence"),
                power_w=values.get("power"),
                speed_mps=values.get("speed") or values.get("enhanced_speed"),
            )
        )

    if skipped:
        logger.warning("Skipped %d FIT record(s) missing time/position in %s", skipped, path)

    points.sort(key=lambda p: p.timestamp)
    logger.info("Loaded %d point(s) from FIT %s", len(points), path)
    return Track(source_path=str(path), points=points)


def load_track(path: str | Path) -> Track:
    """Dispatch to :func:`load_gpx` or :func:`load_fit` based on file extension."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".gpx":
        return load_gpx(path)
    if suffix == ".fit":
        return load_fit(path)
    raise ValueError(f"Unsupported GPS file extension {suffix!r} for {path}")


def _ensure_utc(dt: datetime) -> datetime:
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def _semicircles_to_degrees(value: int) -> float:
    return value * (180.0 / 2**31)


def _extract_gpx_extensions(point: gpxpy.gpx.GPXTrackPoint) -> dict[str, float | int]:
    """Best-effort extraction of Garmin TrackPointExtension HR/cadence/power.

    GPX has no standard field for these, so devices embed them as namespaced
    XML extensions. We scan for the common tag names rather than depending on
    a specific namespace prefix, since it varies by device/exporter.
    """
    result: dict[str, float | int] = {}
    for extension in point.extensions:
        _walk_extension(extension, result)
    return result


def _walk_extension(element, result: dict[str, float | int]) -> None:
    tag = element.tag.rsplit("}", 1)[-1].lower() if hasattr(element, "tag") else ""
    text = getattr(element, "text", None)
    if text is not None:
        text = text.strip()
    if text:
        if tag in ("hr", "heartrate") and "heart_rate_bpm" not in result:
            result["heart_rate_bpm"] = int(float(text))
        elif tag == "cad" and "cadence_rpm" not in result:
            result["cadence_rpm"] = float(text)
        elif tag == "power" and "power_w" not in result:
            result["power_w"] = float(text)
    for child in list(element):
        _walk_extension(child, result)
