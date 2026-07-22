"""Derive speed/cadence/heart-rate/power/elevation/gradient/distance series."""

from __future__ import annotations

import math

from ai_video_editor.gps.models import MetricSample, RideMetrics, Track, TrackPoint
from ai_video_editor.utils.logging import get_logger

logger = get_logger(__name__)

_EARTH_RADIUS_M = 6_371_000.0
_MIN_INTERVAL_S = 1e-3
"""Guards against divide-by-zero on duplicate/near-duplicate timestamps."""


def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two lat/lon points, in metres."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * _EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(a)))


def compute_metrics(track: Track) -> RideMetrics:
    """Derive a per-point time series of ride metrics from a raw track.

    Speed is taken from the source point when present (e.g. FIT
    speed/enhanced_speed), otherwise computed from consecutive-point
    distance/time. Gradient is the elevation change over the horizontal
    distance of each segment, as a percentage. Distance is cumulative from
    the start of the track.
    """
    if track.is_empty:
        return RideMetrics(samples=[])

    samples: list[MetricSample] = []
    cumulative_distance_m = 0.0
    start_time = track.points[0].timestamp

    prev: TrackPoint | None = None
    for point in track.points:
        elapsed_s = (point.timestamp - start_time).total_seconds()

        if prev is None:
            speed_mps = point.speed_mps or 0.0
            gradient_pct = 0.0
        else:
            horizontal_m = haversine_distance_m(prev.lat, prev.lon, point.lat, point.lon)
            dt_s = max((point.timestamp - prev.timestamp).total_seconds(), _MIN_INTERVAL_S)

            speed_mps = point.speed_mps if point.speed_mps is not None else horizontal_m / dt_s
            cumulative_distance_m += horizontal_m

            if prev.elevation_m is not None and point.elevation_m is not None and horizontal_m > 0:
                rise_m = point.elevation_m - prev.elevation_m
                gradient_pct = (rise_m / horizontal_m) * 100.0
            else:
                gradient_pct = 0.0

        samples.append(
            MetricSample(
                timestamp=point.timestamp,
                elapsed_s=elapsed_s,
                lat=point.lat,
                lon=point.lon,
                elevation_m=point.elevation_m,
                speed_mps=speed_mps,
                cadence_rpm=point.cadence_rpm,
                heart_rate_bpm=point.heart_rate_bpm,
                power_w=point.power_w,
                gradient_pct=gradient_pct,
                distance_m=cumulative_distance_m,
            )
        )
        prev = point

    logger.info(
        "Computed metrics for %d sample(s): %.1fm over %.1fs",
        len(samples),
        samples[-1].distance_m,
        samples[-1].elapsed_s,
    )
    return RideMetrics(samples=samples)
