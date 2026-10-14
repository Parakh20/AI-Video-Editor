# Graphical GPS Overlays — Design

**Status:** Approved
**Date:** 2026-07-23
**Scope:** Item 1 of the post-final-report roadmap (`docs/final_report.md` §5) — add a
moving elevation-profile marker and a moving route-map dot to the rendered output,
alongside the existing text telemetry overlay.

## Problem

`gps/overlays.py` already renders static `elevation_profile.png` and `route_map.png`
strips (via `render_elevation_profile` / `render_route_map`) and a continuously
updating text overlay (`build_timesynced_overlay`), but the graphical overlays are
never actually composited into a render, and neither shows *where on the ride* the
video currently is. This closes that gap: a small dot moves along the elevation
curve and along the route shape in sync with the same ride-time mapping already
used for the text overlay.

## Non-goals

- Real map tiles / basemap imagery (route map stays a plain plotted line, as today).
- True clip-time↔GPS-time correlation (tracked separately — item 4 on the roadmap).
- Smooth (per-frame/continuous) marker motion — see "Marker mechanism" below.

## Marker mechanism: windowed discrete overlay

Reuses the pattern already established by `build_timesynced_overlay` for text: the
render's output duration is divided into time windows (interval capped so the
window count never exceeds `max_windows`), and each window is assigned a fixed
pixel position, gated in the ffmpeg filter graph with `enable='between(t,A,B)'`.

Rejected alternative: driving marker `x`/`y` with continuous ffmpeg expressions
(`overlay=x='if(lt(t,5),...)'`). This would give perfectly smooth motion but
requires generating a correct multi-segment ffmpeg expression string for
potentially hundreds of GPS samples — fragile to build, hard to unit-test, and a
divergent pattern from the rest of the overlay code. The windowed approach keeps
all position math in pure, testable Python and mirrors the existing text-overlay
architecture.

## Components

### `gps/overlays.py`

- **`_time_windows(output_duration_s, ride_duration_s, update_interval_s, max_windows) -> list[TimeWindow]`**
  Extracted from the body of `build_timesynced_overlay` (which currently inlines
  this loop) into a shared helper. `TimeWindow` = `(start_s, end_s, ride_elapsed_s)`.
  Both `build_timesynced_overlay` and the two new marker-position builders below
  call this, so window boundaries stay identical across text and graphics.

- **`OverlayGeometry`** (new frozen dataclass): `x_data_range: tuple[float, float]`,
  `y_data_range: tuple[float, float]`, `bbox_px: tuple[float, float, float, float]`
  (left, top, right, bottom, in top-left-origin image-pixel space).

- **`OverlayImage`** gains a new field: `geometry: OverlayGeometry | None = None`.

- **`render_elevation_profile`** / **`render_route_map`**: after plotting, call
  `fig.canvas.draw()`, then read `ax.get_xlim()` / `ax.get_ylim()` and
  `ax.get_window_extent(renderer=fig.canvas.get_renderer())` to compute `bbox_px`
  (converting matplotlib's bottom-left-origin display coords to top-left-origin
  image-pixel coords via `height_px - y`). Populate `OverlayImage.geometry` with
  this before returning.

- **`_data_to_pixel(x, y, geometry) -> tuple[float, float]`**: pure linear
  interpolation from data space to pixel space using `geometry`. No matplotlib
  object needed at call time — this is what makes marker-position tests fast and
  independent of rendering. Guards against a degenerate (zero-width/zero-height)
  data range by returning the bbox center.

- **`render_marker_dot(output_path, *, radius_px=10, color="red") -> OverlayImage`**:
  draws one small filled-circle PNG via PIL with a transparent background. Called
  once per render; the same dot image is reused across all windows.

- **`build_elevation_marker_positions(metrics, geometry, output_duration_s, *, update_interval_s=1.0, max_windows=600) -> list[MarkerWindow]`**
  `MarkerWindow` = `(start_s, end_s, x_px, y_px)`. For each `TimeWindow`,
  interpolates cumulative distance and elevation at `ride_elapsed_s` (reusing the
  existing `_interpolate` helper), maps `(distance_km, elevation_m)` through
  `_data_to_pixel`. Returns `[]` when `metrics` is `None`/empty or has no
  elevation data (mirrors `render_elevation_profile`'s own no-op check).

- **`build_route_map_marker_positions(metrics, geometry, output_duration_s, *, update_interval_s=1.0, max_windows=600) -> list[MarkerWindow]`**
  Same shape, mapping `(lon, lat)` through `_data_to_pixel`. Returns `[]` when
  `metrics` is `None`/empty.

### `render/renderer.py`

- New frozen dataclass **`GraphicOverlaySpec`**: `image_path: Path`, `x: int`,
  `y: int` (fixed on-screen placement of the background image), and
  `marker: MarkerOverlaySpec | None`, where `MarkerOverlaySpec` = `image_path: Path`
  (the dot PNG) + `windows: list[MarkerWindow]` (positions are relative to the
  background image's own pixel space; final screen position is
  `background.x + window.x_px - dot_radius_px`, same for y).

- `_build_filter_complex` gains `graphic_overlays: Sequence[GraphicOverlaySpec] = ()`.
  For each spec: add a `movie=<image_path>` filter input for the background,
  chain an `overlay=x:y[final_v]` fragment; if `marker` is set, add a second
  `movie=<dot_path>` input and chain one `overlay=x:y:enable='between(t,A,B)'`
  fragment per window (mirrors how `overlay_drawtext` fragments are chained
  today). Applied after the existing text-overlay fragment.

- `render_edit_plan` gains a `graphic_overlays: Sequence[GraphicOverlaySpec] = ()`
  parameter, threaded through to `_build_filter_complex`, with the corresponding
  ffmpeg `-i` inputs added for each background/dot image.

### `cli/commands/render.py`

- New helper **`_gps_graphic_overlays(gps_path, kept, crossfade_s, work_dir) -> list[GraphicOverlaySpec]`**:
  parses the track (reusing `load_track`/`compute_metrics`, same as
  `_gps_overlay_fragment`), renders `elevation_profile.png`, `route_map.png`, and
  `marker_dot.png` into `work_dir`, builds both marker-window lists, and returns
  the assembled specs. Returns `[]` when `gps_path` is `None` or the track/metrics
  are empty — same no-op convention as `_gps_overlay_fragment`.
- Wired into the existing render call alongside `overlay_drawtext`.

## Placement defaults

- Elevation-profile strip: bottom-center, small screen margin.
- Route-map: bottom-right corner, small screen margin.
- Existing text overlay stays top-left, so nothing overlaps by default.

## Error handling

Every new function follows the codebase's established convention: return
`None`/`[]` (not an exception) when GPS data, elevation data, or lat/lon data is
unavailable — exactly as `render_elevation_profile`/`render_route_map` do today.
The only new arithmetic risk is division-by-zero in `_data_to_pixel` on a
degenerate data range (e.g. perfectly flat elevation, or a single GPS point);
guarded by returning the bbox center in that case.

## Testing

- `tests/gps/test_overlays.py`:
  - Pure-math unit tests for `_data_to_pixel` (normal case, degenerate-range
    guard) and `_time_windows` (window count/boundaries, `max_windows` capping)
    — no ffmpeg/matplotlib rendering needed.
  - Tests against `tests/fixtures/sample_ride.gpx` verifying: correct marker
    window count, monotonic pixel movement along the route/elevation as ride time
    advances, and `[]`/`None` no-op behavior when GPS data is absent.
  - `render_elevation_profile`/`render_route_map` tests extended to assert
    `OverlayImage.geometry` is populated with sane data ranges/bbox.
- `tests/cli/test_render.py`:
  - `_gps_graphic_overlays` produces the expected number of `GraphicOverlaySpec`s
    for the fixture track, and `[]` with no `--gps` flag.
  - `render_edit_plan` accepts and wires `graphic_overlays` without a real ffmpeg
    run (mocked subprocess, consistent with existing render tests).

## Open questions / follow-ups (not blocking this design)

- Route-map marker accuracy depends on matplotlib's `ax.get_window_extent()`
  being stable across matplotlib versions/backends; if it proves brittle in
  testing, the fallback is to compute the bbox manually from known figure
  margins (`fig.subplotpars`) instead of asking matplotlib for it post-draw.
- Dot color/size are hardcoded defaults (`radius_px=10`, `color="red"`) — no
  config surface proposed here; can be added later if needed.
