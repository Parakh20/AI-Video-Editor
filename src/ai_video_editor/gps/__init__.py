"""GPS/telemetry parsing, metric derivation, and ride-event detection.

Every public entry point here accepts an optional GPS source and is meant to
be skipped cleanly when a ride has no GPX/FIT file: parse the track with
:func:`load_track`, or don't call it at all and pass ``None`` downstream.
"""

from ai_video_editor.gps.event_detection import EventThresholds, detect_events
from ai_video_editor.gps.metrics import compute_metrics, haversine_distance_m
from ai_video_editor.gps.models import (
    EventType,
    MetricSample,
    RideEvent,
    RideMetrics,
    Track,
    TrackPoint,
)
from ai_video_editor.gps.overlays import (
    OverlayImage,
    cadence_overlay_text,
    distance_overlay_text,
    drawtext_filter,
    gradient_overlay_text,
    heart_rate_overlay_text,
    power_overlay_text,
    render_elevation_profile,
    render_ride_stats_card,
    render_route_map,
    speed_overlay_text,
)
from ai_video_editor.gps.parsers import load_fit, load_gpx, load_track

__all__ = [
    "EventThresholds",
    "EventType",
    "MetricSample",
    "OverlayImage",
    "RideEvent",
    "RideMetrics",
    "Track",
    "TrackPoint",
    "cadence_overlay_text",
    "compute_metrics",
    "detect_events",
    "distance_overlay_text",
    "drawtext_filter",
    "gradient_overlay_text",
    "haversine_distance_m",
    "heart_rate_overlay_text",
    "load_fit",
    "load_gpx",
    "load_track",
    "power_overlay_text",
    "render_elevation_profile",
    "render_ride_stats_card",
    "render_route_map",
    "speed_overlay_text",
]
