# Graphical GPS Overlays Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a moving elevation-profile marker and a moving route-map dot to the rendered output, composited alongside the existing text telemetry overlay.

**Architecture:** Reuse the windowed-discrete-overlay pattern already established by `build_timesynced_overlay` (text): output duration is split into time windows, each window gets a fixed pixel position, gated in the ffmpeg filter graph with `enable='between(t,A,B)'`. All new position math is pure Python (unit-testable without ffmpeg/matplotlib at call time); ffmpeg's `movie=` filter loads the background/marker PNGs directly inside `-filter_complex`, so no new `-i` input plumbing is needed.

**Tech Stack:** Python 3.12, pytest, matplotlib (Agg backend), PIL, ffmpeg `-filter_complex` (`movie`, `overlay`, `split` filters).

## Global Constraints

- Design spec: `docs/superpowers/specs/2026-07-23-graphical-gps-overlays-design.md` — follow it exactly; this plan implements it task-by-task.
- Every new function follows the codebase's existing no-op convention: return `None`/`[]` (never raise) when GPS/elevation/lat-lon data is unavailable.
- All new dataclasses are `@dataclass(frozen=True)`, matching every existing dataclass in `gps/models.py`, `gps/overlays.py`, and `render/renderer.py`.
- Type annotations on all function signatures (project-wide convention, see `src/ai_video_editor/gps/overlays.py` for the established style).
- Run the full test file after each task, not just the new test, to catch regressions: `pytest tests/gps/test_overlays.py -v` (Tasks 1-6), `pytest tests/render/ -v -m "not slow"` (Tasks 7-8 fast tests), `pytest tests/cli/test_render.py -v` (Task 9).
- Virtualenv: use `.venv/bin/python -m pytest ...` (per `README.md` Development section) if `pytest` is not on PATH directly.

---

### Task 1: Extract `_time_windows` shared helper

**Files:**
- Modify: `src/ai_video_editor/gps/overlays.py` (add `TimeWindow` dataclass + `_time_windows` function near line 158, before `_overlay_line`; refactor `build_timesynced_overlay` body, lines 206-235, to use it)
- Test: `tests/gps/test_overlays.py`

**Interfaces:**
- Produces: `TimeWindow(start_s: float, end_s: float, ride_elapsed_s: float)` (frozen dataclass); `_time_windows(output_duration_s: float, ride_duration_s: float, update_interval_s: float, max_windows: int) -> list[TimeWindow]`.
- Consumes: nothing new (pure function of its four float/int args).

- [ ] **Step 1: Write the failing tests**

Add to `tests/gps/test_overlays.py` (near the "Time-synced (per-frame) telemetry overlay" section, before `test_build_timesynced_overlay_none_without_metrics`):

```python
def test_time_windows_basic() -> None:
    from ai_video_editor.gps.overlays import _time_windows

    windows = _time_windows(6.0, 60.0, 2.0, 600)
    assert len(windows) == 3
    assert windows[0].start_s == 0.0
    assert windows[0].end_s == 2.0
    assert windows[-1].end_s == 6.0
    assert windows[0].ride_elapsed_s == 0.0
    assert windows[-1].ride_elapsed_s == pytest.approx(40.0)


def test_time_windows_caps_at_max_windows() -> None:
    from ai_video_editor.gps.overlays import _time_windows

    windows = _time_windows(100_000.0, 100.0, 0.001, max_windows=50)
    assert len(windows) <= 50
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/gps/test_overlays.py::test_time_windows_basic -v`
Expected: FAIL with `ImportError: cannot import name '_time_windows'`

- [ ] **Step 3: Implement `TimeWindow` + `_time_windows`, refactor `build_timesynced_overlay`**

In `src/ai_video_editor/gps/overlays.py`, add just before `def _overlay_line(...)` (around line 162):

```python
@dataclass(frozen=True)
class TimeWindow:
    start_s: float
    end_s: float
    ride_elapsed_s: float
    """The point on the ride's own timeline that this output window maps to."""


def _time_windows(
    output_duration_s: float,
    ride_duration_s: float,
    update_interval_s: float,
    max_windows: int,
) -> list[TimeWindow]:
    """Divide ``output_duration_s`` into windows, each mapped onto ride time.

    Shared by the text overlay and the graphical markers so window
    boundaries stay identical across both. The full ride is mapped
    linearly onto the output timeline: window ``k`` starting at output time
    ``start`` maps to ride-elapsed time ``(start / output_duration_s) *
    ride_duration_s``.
    """
    window_count = math.ceil(output_duration_s / update_interval_s)
    if window_count > max_windows:
        window_count = max_windows
    dt = output_duration_s / window_count

    windows: list[TimeWindow] = []
    for k in range(window_count):
        start = k * dt
        end = output_duration_s if k == window_count - 1 else (k + 1) * dt
        ride_elapsed = (start / output_duration_s) * ride_duration_s
        windows.append(TimeWindow(start_s=start, end_s=end, ride_elapsed_s=ride_elapsed))
    return windows
```

Then replace the body of `build_timesynced_overlay` (the block from `ride_duration_s = metrics.duration_s` through the `for k in range(window_count):` loop, lines 213-231) with:

```python
    ride_duration_s = metrics.duration_s
    windows = _time_windows(output_duration_s, ride_duration_s, update_interval_s, max_windows)

    fragments: list[str] = []
    for w in windows:
        line = _overlay_line(metrics, w.ride_elapsed_s, fields)
        if not line:
            continue
        fragment = drawtext_filter(
            line, x=x, y=y, font_size=font_size, font_color=font_color
        )
        fragments.append(f"{fragment}:enable='between(t,{w.start_s:.3f},{w.end_s:.3f})'")
```

(The `if not fragments: return None` / `return ",".join(fragments)` lines that follow stay unchanged.)

- [ ] **Step 4: Run tests to verify everything passes**

Run: `pytest tests/gps/test_overlays.py -v`
Expected: PASS (all existing `build_timesynced_overlay` tests plus the two new `_time_windows` tests)

- [ ] **Step 5: Commit**

```bash
git add src/ai_video_editor/gps/overlays.py tests/gps/test_overlays.py
git commit -m "refactor: extract _time_windows helper from build_timesynced_overlay"
```

---

### Task 2: `OverlayGeometry` + `_data_to_pixel`

**Files:**
- Modify: `src/ai_video_editor/gps/overlays.py` (add near the `OverlayImage` dataclass, around line 238)
- Test: `tests/gps/test_overlays.py`

**Interfaces:**
- Produces: `OverlayGeometry(x_data_range: tuple[float, float], y_data_range: tuple[float, float], bbox_px: tuple[float, float, float, float])` (frozen dataclass; `bbox_px` = `(left, top, right, bottom)` in top-left-origin image-pixel space); `_data_to_pixel(x: float, y: float, geometry: OverlayGeometry) -> tuple[float, float]`.
- Consumes: nothing new.

- [ ] **Step 1: Write the failing tests**

Add to `tests/gps/test_overlays.py`, after the `test_render_functions_none_without_gps` test:

```python
def test_data_to_pixel_maps_corners() -> None:
    from ai_video_editor.gps.overlays import OverlayGeometry, _data_to_pixel

    geometry = OverlayGeometry(
        x_data_range=(0.0, 10.0),
        y_data_range=(0.0, 100.0),
        bbox_px=(20.0, 10.0, 220.0, 110.0),
    )
    px, py = _data_to_pixel(0.0, 0.0, geometry)
    assert px == pytest.approx(20.0)
    assert py == pytest.approx(110.0)

    px, py = _data_to_pixel(10.0, 100.0, geometry)
    assert px == pytest.approx(220.0)
    assert py == pytest.approx(10.0)

    px, py = _data_to_pixel(5.0, 50.0, geometry)
    assert px == pytest.approx(120.0)
    assert py == pytest.approx(60.0)


def test_data_to_pixel_degenerate_range_returns_bbox_center() -> None:
    from ai_video_editor.gps.overlays import OverlayGeometry, _data_to_pixel

    geometry = OverlayGeometry(
        x_data_range=(5.0, 5.0),
        y_data_range=(0.0, 0.0),
        bbox_px=(0.0, 0.0, 100.0, 50.0),
    )
    px, py = _data_to_pixel(5.0, 0.0, geometry)
    assert px == pytest.approx(50.0)
    assert py == pytest.approx(25.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/gps/test_overlays.py::test_data_to_pixel_maps_corners -v`
Expected: FAIL with `ImportError: cannot import name 'OverlayGeometry'`

- [ ] **Step 3: Implement `OverlayGeometry` + `_data_to_pixel`**

In `src/ai_video_editor/gps/overlays.py`, add immediately before the `@dataclass(frozen=True)` / `class OverlayImage:` block (around line 238):

```python
@dataclass(frozen=True)
class OverlayGeometry:
    """Maps a matplotlib axes' data space onto its rendered image's pixels."""

    x_data_range: tuple[float, float]
    y_data_range: tuple[float, float]
    bbox_px: tuple[float, float, float, float]
    """(left, top, right, bottom) in top-left-origin image-pixel space."""


def _data_to_pixel(x: float, y: float, geometry: OverlayGeometry) -> tuple[float, float]:
    """Map a data-space point to image-pixel coordinates via ``geometry``.

    Falls back to the bbox center on a degenerate (zero-width or
    zero-height) data range, to avoid a divide-by-zero.
    """
    x_lo, x_hi = geometry.x_data_range
    y_lo, y_hi = geometry.y_data_range
    left, top, right, bottom = geometry.bbox_px

    x_span = x_hi - x_lo
    if x_span == 0:
        px = (left + right) / 2
    else:
        px = left + ((x - x_lo) / x_span) * (right - left)

    y_span = y_hi - y_lo
    if y_span == 0:
        py = (top + bottom) / 2
    else:
        # Data y increases upward; image y increases downward.
        py = bottom - ((y - y_lo) / y_span) * (bottom - top)

    return px, py
```

Then extend `OverlayImage` (currently `path: Path`, `width: int`, `height: int`) with a fourth, defaulted field:

```python
@dataclass(frozen=True)
class OverlayImage:
    path: Path
    width: int
    height: int
    geometry: OverlayGeometry | None = None
```

- [ ] **Step 4: Run tests to verify everything passes**

Run: `pytest tests/gps/test_overlays.py -v`
Expected: PASS (all tests, including the two new ones)

- [ ] **Step 5: Commit**

```bash
git add src/ai_video_editor/gps/overlays.py tests/gps/test_overlays.py
git commit -m "feat: add OverlayGeometry and _data_to_pixel for marker positioning"
```

---

### Task 3: Capture geometry in `render_elevation_profile` / `render_route_map`

**Files:**
- Modify: `src/ai_video_editor/gps/overlays.py` (`render_elevation_profile`, currently lines 245-282; `render_route_map`, currently lines 285-318)
- Test: `tests/gps/test_overlays.py`

**Interfaces:**
- Consumes: `OverlayGeometry` (Task 2).
- Produces: `render_elevation_profile(...)` and `render_route_map(...)` now populate `OverlayImage.geometry` (previously always `None`).

- [ ] **Step 1: Write the failing tests**

Add to `tests/gps/test_overlays.py`, right after `test_render_elevation_profile_real_fixture`:

```python
def test_render_elevation_profile_populates_geometry(real_metrics: RideMetrics, tmp_path: Path) -> None:
    out = tmp_path / "elev.png"
    result = render_elevation_profile(real_metrics, out)
    assert result is not None
    assert result.geometry is not None
    x_lo, x_hi = result.geometry.x_data_range
    assert x_hi > x_lo
    left, top, right, bottom = result.geometry.bbox_px
    assert right > left
    assert bottom > top
```

Add right after `test_render_route_map_real_fixture`:

```python
def test_render_route_map_populates_geometry(real_metrics: RideMetrics, tmp_path: Path) -> None:
    out = tmp_path / "route.png"
    result = render_route_map(real_metrics, out)
    assert result is not None
    assert result.geometry is not None
    left, top, right, bottom = result.geometry.bbox_px
    assert right > left
    assert bottom > top
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/gps/test_overlays.py::test_render_elevation_profile_populates_geometry tests/gps/test_overlays.py::test_render_route_map_populates_geometry -v`
Expected: FAIL — `assert result.geometry is not None` fails (`geometry` is `None`, the dataclass default)

- [ ] **Step 3: Wire geometry capture into both render functions**

In `render_elevation_profile`, insert this block after `fig.tight_layout(pad=0)` and before `output_path.parent.mkdir(...)` (around line 276):

```python
    fig.canvas.draw()
    bbox = ax.get_window_extent(renderer=fig.canvas.get_renderer())
    geometry = OverlayGeometry(
        x_data_range=ax.get_xlim(),
        y_data_range=ax.get_ylim(),
        bbox_px=(bbox.x0, height_px - bbox.y1, bbox.x1, height_px - bbox.y0),
    )
```

Then change the final `return` statement from:

```python
    return OverlayImage(path=output_path, width=width_px, height=height_px)
```

to:

```python
    return OverlayImage(path=output_path, width=width_px, height=height_px, geometry=geometry)
```

Apply the identical pattern to `render_route_map` (insert after its `fig.tight_layout(pad=0)`, using `size_px` in place of `height_px`/`width_px` since that function is square):

```python
    fig.canvas.draw()
    bbox = ax.get_window_extent(renderer=fig.canvas.get_renderer())
    geometry = OverlayGeometry(
        x_data_range=ax.get_xlim(),
        y_data_range=ax.get_ylim(),
        bbox_px=(bbox.x0, size_px - bbox.y1, bbox.x1, size_px - bbox.y0),
    )
```

and change its final `return` to:

```python
    return OverlayImage(path=output_path, width=size_px, height=size_px, geometry=geometry)
```

- [ ] **Step 4: Run tests to verify everything passes**

Run: `pytest tests/gps/test_overlays.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add src/ai_video_editor/gps/overlays.py tests/gps/test_overlays.py
git commit -m "feat: capture axes geometry in render_elevation_profile and render_route_map"
```

---

### Task 4: `render_marker_dot`

**Files:**
- Modify: `src/ai_video_editor/gps/overlays.py` (add after `render_route_map`, before `render_ride_stats_card`, around line 320)
- Test: `tests/gps/test_overlays.py`

**Interfaces:**
- Consumes: `OverlayImage` (existing).
- Produces: `render_marker_dot(output_path: Path, *, radius_px: int = 10, color: str = "red") -> OverlayImage`.

- [ ] **Step 1: Write the failing test**

Add to `tests/gps/test_overlays.py`, after `test_render_route_map_populates_geometry`:

```python
def test_render_marker_dot_creates_transparent_png(tmp_path: Path) -> None:
    out = tmp_path / "dot.png"
    result = render_marker_dot(out, radius_px=8, color="red")
    assert out.exists()
    assert out.stat().st_size > 0
    assert result.width == 16
    assert result.height == 16
```

Add `render_marker_dot` to the existing `from ai_video_editor.gps.overlays import (...)` block at the top of the test file.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/gps/test_overlays.py::test_render_marker_dot_creates_transparent_png -v`
Expected: FAIL with `ImportError: cannot import name 'render_marker_dot'`

- [ ] **Step 3: Implement `render_marker_dot`**

In `src/ai_video_editor/gps/overlays.py`, add after `render_route_map` (before `render_ride_stats_card`):

```python
def render_marker_dot(
    output_path: Path,
    *,
    radius_px: int = 10,
    color: str = "red",
) -> OverlayImage:
    """Render one small filled-circle PNG (transparent background).

    Rendered once per render and reused across every window of a moving
    marker overlay (see :func:`build_elevation_marker_positions` /
    :func:`build_route_map_marker_positions`).
    """
    size = radius_px * 2
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse((0, 0, size - 1, size - 1), fill=color)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)

    logger.info("render_marker_dot: wrote %s", output_path)
    return OverlayImage(path=output_path, width=size, height=size)
```

- [ ] **Step 4: Run tests to verify everything passes**

Run: `pytest tests/gps/test_overlays.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add src/ai_video_editor/gps/overlays.py tests/gps/test_overlays.py
git commit -m "feat: add render_marker_dot for graphic overlay markers"
```

---

### Task 5: `MarkerWindow` + `build_elevation_marker_positions`

**Files:**
- Modify: `src/ai_video_editor/gps/overlays.py` (add at end of file, after `render_ride_stats_card`)
- Test: `tests/gps/test_overlays.py`

**Interfaces:**
- Consumes: `_time_windows` (Task 1), `OverlayGeometry` / `_data_to_pixel` (Task 2), `_interpolate` (existing, line 33).
- Produces: `MarkerWindow(start_s: float, end_s: float, x_px: float, y_px: float)` (frozen dataclass); `build_elevation_marker_positions(metrics: RideMetrics | None, geometry: OverlayGeometry, output_duration_s: float, *, update_interval_s: float = 1.0, max_windows: int = 600) -> list[MarkerWindow]`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/gps/test_overlays.py`, at the end of the file:

```python
# --- Graphical marker positions ----------------------------------------------


def test_build_elevation_marker_positions_moves_with_distance(real_metrics: RideMetrics) -> None:
    from ai_video_editor.gps.overlays import OverlayGeometry, build_elevation_marker_positions

    geometry = OverlayGeometry(
        x_data_range=(0.0, real_metrics.total_distance_m / 1000),
        y_data_range=(0.0, 500.0),
        bbox_px=(0.0, 0.0, 800.0, 200.0),
    )
    windows = build_elevation_marker_positions(real_metrics, geometry, output_duration_s=6.0, update_interval_s=2.0)
    assert len(windows) == 3
    assert windows[-1].x_px >= windows[0].x_px


def test_build_elevation_marker_positions_empty_without_metrics() -> None:
    from ai_video_editor.gps.overlays import OverlayGeometry, build_elevation_marker_positions

    geometry = OverlayGeometry(x_data_range=(0.0, 1.0), y_data_range=(0.0, 1.0), bbox_px=(0.0, 0.0, 1.0, 1.0))
    assert build_elevation_marker_positions(None, geometry, 10.0) == []
    assert build_elevation_marker_positions(RideMetrics(samples=[]), geometry, 10.0) == []


def test_build_elevation_marker_positions_empty_for_nonpositive_duration(real_metrics: RideMetrics) -> None:
    from ai_video_editor.gps.overlays import OverlayGeometry, build_elevation_marker_positions

    geometry = OverlayGeometry(x_data_range=(0.0, 1.0), y_data_range=(0.0, 1.0), bbox_px=(0.0, 0.0, 1.0, 1.0))
    assert build_elevation_marker_positions(real_metrics, geometry, 0.0) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/gps/test_overlays.py::test_build_elevation_marker_positions_moves_with_distance -v`
Expected: FAIL with `ImportError: cannot import name 'build_elevation_marker_positions'`

- [ ] **Step 3: Implement `MarkerWindow` + `build_elevation_marker_positions`**

Append to the end of `src/ai_video_editor/gps/overlays.py`:

```python
@dataclass(frozen=True)
class MarkerWindow:
    """One time-gated pixel position for a moving marker overlay."""

    start_s: float
    end_s: float
    x_px: float
    y_px: float


def build_elevation_marker_positions(
    metrics: RideMetrics | None,
    geometry: OverlayGeometry,
    output_duration_s: float,
    *,
    update_interval_s: float = 1.0,
    max_windows: int = _DEFAULT_MAX_WINDOWS,
) -> list[MarkerWindow]:
    """Per-window pixel positions for a dot moving along the elevation curve.

    Tracks cumulative ride distance against the elevation-profile plot's
    ``geometry`` (see :func:`render_elevation_profile`). Returns ``[]`` when
    ``metrics`` has no samples, no elevation data at all, or
    ``output_duration_s`` is non-positive.
    """
    if metrics is None or not metrics.samples:
        return []
    if all(s.elevation_m is None for s in metrics.samples):
        return []
    if output_duration_s <= 0:
        return []

    windows = _time_windows(output_duration_s, metrics.duration_s, update_interval_s, max_windows)

    result: list[MarkerWindow] = []
    for w in windows:
        distance_m = _interpolate(metrics, w.ride_elapsed_s, "distance_m")
        elevation_m = _interpolate(metrics, w.ride_elapsed_s, "elevation_m")
        if distance_m is None or elevation_m is None:
            continue
        x_px, y_px = _data_to_pixel(distance_m / 1000, elevation_m, geometry)
        result.append(MarkerWindow(start_s=w.start_s, end_s=w.end_s, x_px=x_px, y_px=y_px))
    return result
```

- [ ] **Step 4: Run tests to verify everything passes**

Run: `pytest tests/gps/test_overlays.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add src/ai_video_editor/gps/overlays.py tests/gps/test_overlays.py
git commit -m "feat: add build_elevation_marker_positions for moving elevation marker"
```

---

### Task 6: `build_route_map_marker_positions`

**Files:**
- Modify: `src/ai_video_editor/gps/overlays.py` (add at end of file, after Task 5's additions)
- Test: `tests/gps/test_overlays.py`

**Interfaces:**
- Consumes: `MarkerWindow`, `_time_windows`, `_data_to_pixel`, `OverlayGeometry` (Tasks 1, 2, 5).
- Produces: `build_route_map_marker_positions(metrics: RideMetrics | None, geometry: OverlayGeometry, output_duration_s: float, *, update_interval_s: float = 1.0, max_windows: int = 600) -> list[MarkerWindow]`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/gps/test_overlays.py`, at the end of the file:

```python
def test_build_route_map_marker_positions_returns_one_window_per_interval(real_metrics: RideMetrics) -> None:
    from ai_video_editor.gps.overlays import OverlayGeometry, build_route_map_marker_positions

    lats = [s.lat for s in real_metrics.samples]
    lons = [s.lon for s in real_metrics.samples]
    geometry = OverlayGeometry(
        x_data_range=(min(lons), max(lons)),
        y_data_range=(min(lats), max(lats)),
        bbox_px=(0.0, 0.0, 300.0, 300.0),
    )
    windows = build_route_map_marker_positions(real_metrics, geometry, output_duration_s=6.0, update_interval_s=2.0)
    assert len(windows) == 3
    for w in windows:
        assert 0.0 <= w.x_px <= 300.0
        assert 0.0 <= w.y_px <= 300.0


def test_build_route_map_marker_positions_empty_without_metrics() -> None:
    from ai_video_editor.gps.overlays import OverlayGeometry, build_route_map_marker_positions

    geometry = OverlayGeometry(x_data_range=(0.0, 1.0), y_data_range=(0.0, 1.0), bbox_px=(0.0, 0.0, 1.0, 1.0))
    assert build_route_map_marker_positions(None, geometry, 10.0) == []
    assert build_route_map_marker_positions(RideMetrics(samples=[]), geometry, 10.0) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/gps/test_overlays.py::test_build_route_map_marker_positions_returns_one_window_per_interval -v`
Expected: FAIL with `ImportError: cannot import name 'build_route_map_marker_positions'`

- [ ] **Step 3: Implement `build_route_map_marker_positions`**

Append to the end of `src/ai_video_editor/gps/overlays.py`:

```python
def build_route_map_marker_positions(
    metrics: RideMetrics | None,
    geometry: OverlayGeometry,
    output_duration_s: float,
    *,
    update_interval_s: float = 1.0,
    max_windows: int = _DEFAULT_MAX_WINDOWS,
) -> list[MarkerWindow]:
    """Per-window pixel positions for a dot moving along the route-map plot.

    Tracks lat/lon against the route-map plot's ``geometry`` (see
    :func:`render_route_map`). Returns ``[]`` when ``metrics`` has no
    samples or ``output_duration_s`` is non-positive.
    """
    if metrics is None or not metrics.samples:
        return []
    if output_duration_s <= 0:
        return []

    windows = _time_windows(output_duration_s, metrics.duration_s, update_interval_s, max_windows)

    result: list[MarkerWindow] = []
    for w in windows:
        lon = _interpolate(metrics, w.ride_elapsed_s, "lon")
        lat = _interpolate(metrics, w.ride_elapsed_s, "lat")
        if lon is None or lat is None:
            continue
        x_px, y_px = _data_to_pixel(lon, lat, geometry)
        result.append(MarkerWindow(start_s=w.start_s, end_s=w.end_s, x_px=x_px, y_px=y_px))
    return result
```

- [ ] **Step 4: Run tests to verify everything passes**

Run: `pytest tests/gps/test_overlays.py -v`
Expected: PASS (all tests — this is the last task touching `gps/overlays.py`, so also spot-check the full module: `pytest tests/gps/ -v`)

- [ ] **Step 5: Commit**

```bash
git add src/ai_video_editor/gps/overlays.py tests/gps/test_overlays.py
git commit -m "feat: add build_route_map_marker_positions for moving route-map dot"
```

---

### Task 7: `GraphicOverlaySpec` + filter-graph compositing in `renderer.py`

**Files:**
- Modify: `src/ai_video_editor/render/renderer.py` (add dataclasses near `_ClipInputs`, around line 61; add `_build_graphic_overlay_fragments` before `_build_filter_complex`, around line 137; extend `_build_filter_complex`'s signature and body, lines 137-182)
- Test: Create `tests/render/test_renderer.py` (new file — fast, no ffmpeg required)

**Interfaces:**
- Consumes: `MarkerWindow` (Task 5) from `ai_video_editor.gps.overlays`.
- Produces: `MarkerOverlaySpec(image_path: Path, windows: list[MarkerWindow])`; `GraphicOverlaySpec(image_path: Path, x: int, y: int, marker: MarkerOverlaySpec | None = None)`; `_build_filter_complex(..., graphic_overlays: Sequence[GraphicOverlaySpec] = ())` (new keyword parameter, same return shape as before: `tuple[str, str, str]`).

- [ ] **Step 1: Write the failing tests**

Create `tests/render/test_renderer.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/render/test_renderer.py -v`
Expected: FAIL with `ImportError: cannot import name 'GraphicOverlaySpec'`

- [ ] **Step 3: Implement `GraphicOverlaySpec`, `MarkerOverlaySpec`, and filter-graph compositing**

In `src/ai_video_editor/render/renderer.py`, add this import near the top (with the other `ai_video_editor` imports, around line 30):

```python
from collections.abc import Sequence

from ai_video_editor.gps.overlays import MarkerWindow
```

Add these dataclasses after `_ClipInputs` (around line 68, before `_build_inputs`):

```python
@dataclass(frozen=True)
class MarkerOverlaySpec:
    """A moving-dot marker composited over a :class:`GraphicOverlaySpec`."""

    image_path: Path
    windows: list[MarkerWindow]
    """Positions are relative to the parent GraphicOverlaySpec's own (x, y);
    the caller is responsible for centering (e.g. subtracting the dot's own
    half-width/half-height) before constructing this spec."""


@dataclass(frozen=True)
class GraphicOverlaySpec:
    """A static background image (e.g. elevation profile) plus an optional
    moving marker, composited into the render at a fixed screen position."""

    image_path: Path
    x: int
    y: int
    marker: MarkerOverlaySpec | None = None
```

Add this function before `_build_filter_complex` (around line 137):

```python
def _build_graphic_overlay_fragments(
    acc_v: str,
    graphic_overlays: Sequence[GraphicOverlaySpec],
) -> tuple[list[str], str]:
    """Build filter fragments compositing background images + moving markers.

    Each background image loads via ``movie=`` (no extra ffmpeg ``-i``
    needed) and composites once at a fixed screen position. An optional
    marker dot is split into one copy per time window and chained as
    individually time-gated ``overlay`` filters, mirroring how the text
    overlay chains gated ``drawtext`` filters.
    """
    parts: list[str] = []
    for i, spec in enumerate(graphic_overlays):
        bg_label = f"bg{i}"
        parts.append(f"movie={spec.image_path}[{bg_label}]")
        next_v = f"gv{i}"
        parts.append(f"[{acc_v}][{bg_label}]overlay={spec.x}:{spec.y}[{next_v}]")
        acc_v = next_v

        if spec.marker is None or not spec.marker.windows:
            continue

        windows = spec.marker.windows
        dot_src = f"dotsrc{i}"
        parts.append(f"movie={spec.marker.image_path}[{dot_src}]")
        dot_labels = [f"dot{i}_{j}" for j in range(len(windows))]
        if len(dot_labels) == 1:
            parts.append(f"[{dot_src}]copy[{dot_labels[0]}]")
        else:
            fanout = "".join(f"[{label}]" for label in dot_labels)
            parts.append(f"[{dot_src}]split={len(dot_labels)}{fanout}")

        for j, (window, dot_label) in enumerate(zip(windows, dot_labels, strict=True)):
            marker_out = f"mk{i}_{j}"
            x = spec.x + window.x_px
            y = spec.y + window.y_px
            parts.append(
                f"[{acc_v}][{dot_label}]overlay={x:.1f}:{y:.1f}:"
                f"enable='between(t,{window.start_s:.3f},{window.end_s:.3f})'[{marker_out}]"
            )
            acc_v = marker_out

    return parts, acc_v
```

Change `_build_filter_complex`'s signature (currently lines 137-144) from:

```python
def _build_filter_complex(
    kept: list[EditAction],
    clip_inputs: list[_ClipInputs],
    preset: RenderPreset,
    *,
    crossfade_s: float,
    overlay_drawtext: str | None,
) -> tuple[str, str, str]:
```

to:

```python
def _build_filter_complex(
    kept: list[EditAction],
    clip_inputs: list[_ClipInputs],
    preset: RenderPreset,
    *,
    crossfade_s: float,
    overlay_drawtext: str | None,
    graphic_overlays: Sequence[GraphicOverlaySpec] = (),
) -> tuple[str, str, str]:
```

And change the tail of the function (currently lines 177-182):

```python
    if overlay_drawtext:
        final_v = "vout"
        parts.append(f"[{acc_v}]{overlay_drawtext}[{final_v}]")
        acc_v = final_v

    return ";".join(parts), acc_v, acc_a
```

to:

```python
    if overlay_drawtext:
        final_v = "vout"
        parts.append(f"[{acc_v}]{overlay_drawtext}[{final_v}]")
        acc_v = final_v

    graphic_parts, acc_v = _build_graphic_overlay_fragments(acc_v, graphic_overlays)
    parts += graphic_parts

    return ";".join(parts), acc_v, acc_a
```

- [ ] **Step 4: Run tests to verify everything passes**

Run: `pytest tests/render/test_renderer.py -v`
Expected: PASS (all 3 new tests)

Also run: `pytest tests/render/ -v -m "not slow"` to confirm no regressions in the fast suite.

- [ ] **Step 5: Commit**

```bash
git add src/ai_video_editor/render/renderer.py tests/render/test_renderer.py
git commit -m "feat: composite graphic GPS overlays into the ffmpeg filter graph"
```

---

### Task 8: Thread `graphic_overlays` through `render_edit_plan`

**Files:**
- Modify: `src/ai_video_editor/render/renderer.py` (`render_edit_plan`, currently lines 199-224)
- Test: `tests/render/test_renderer_ffmpeg.py` (slow, real-ffmpeg integration test)

**Interfaces:**
- Consumes: `GraphicOverlaySpec` (Task 7).
- Produces: `render_edit_plan(..., graphic_overlays: Sequence[GraphicOverlaySpec] = ())` (new keyword parameter; return type unchanged, still `RenderResult`).

- [ ] **Step 1: Write the failing test**

Add to `tests/render/test_renderer_ffmpeg.py`, after `test_render_with_drawtext_overlay`:

```python
def test_render_with_graphic_overlay(synthetic_clips: list[Path], tmp_path: Path) -> None:
    from PIL import Image

    from ai_video_editor.render.renderer import GraphicOverlaySpec

    bg_path = tmp_path / "bg.png"
    Image.new("RGBA", (100, 50), (255, 0, 0, 180)).save(bg_path)

    clip_a, _ = synthetic_clips
    actions = [
        EditAction(
            kind="keep",
            video_path=str(clip_a),
            start_seconds=0.0,
            end_seconds=2.0,
            source_shot_indices=[0],
            explanation="test clip with graphic overlay",
        ),
    ]
    output = tmp_path / "graphic_overlay.mp4"
    spec = GraphicOverlaySpec(image_path=bg_path, x=20, y=20)
    result = render_edit_plan(
        actions, output, get_preset("1080p"), graphic_overlays=[spec], prefer_gpu=False
    )
    assert result.output_path.exists()
    assert result.output_path.stat().st_size > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/render/test_renderer_ffmpeg.py::test_render_with_graphic_overlay -v -m slow`
Expected: FAIL with `TypeError: render_edit_plan() got an unexpected keyword argument 'graphic_overlays'`

- [ ] **Step 3: Thread `graphic_overlays` through `render_edit_plan`**

In `src/ai_video_editor/render/renderer.py`, change `render_edit_plan`'s signature (currently lines 199-208) from:

```python
def render_edit_plan(
    edit_actions: list[EditAction],
    output_path: Path,
    preset: RenderPreset,
    *,
    crossfade_s: float = _DEFAULT_CROSSFADE_S,
    overlay_drawtext: str | None = None,
    prefer_gpu: bool = True,
    ffmpeg_bin: str = "ffmpeg",
) -> RenderResult:
```

to:

```python
def render_edit_plan(
    edit_actions: list[EditAction],
    output_path: Path,
    preset: RenderPreset,
    *,
    crossfade_s: float = _DEFAULT_CROSSFADE_S,
    overlay_drawtext: str | None = None,
    graphic_overlays: Sequence[GraphicOverlaySpec] = (),
    prefer_gpu: bool = True,
    ffmpeg_bin: str = "ffmpeg",
) -> RenderResult:
```

And change the `_build_filter_complex` call (currently lines 222-224) from:

```python
    filter_complex, v_label, a_label = _build_filter_complex(
        kept, clip_inputs, preset, crossfade_s=crossfade_s, overlay_drawtext=overlay_drawtext
    )
```

to:

```python
    filter_complex, v_label, a_label = _build_filter_complex(
        kept,
        clip_inputs,
        preset,
        crossfade_s=crossfade_s,
        overlay_drawtext=overlay_drawtext,
        graphic_overlays=graphic_overlays,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/render/test_renderer_ffmpeg.py::test_render_with_graphic_overlay -v -m slow`
Expected: PASS

Also run the full renderer suite to confirm no regressions: `pytest tests/render/ -v`

- [ ] **Step 5: Commit**

```bash
git add src/ai_video_editor/render/renderer.py tests/render/test_renderer_ffmpeg.py
git commit -m "feat: thread graphic_overlays through render_edit_plan"
```

---

### Task 9: Wire `_gps_graphic_overlays` into the `render` CLI command

**Files:**
- Modify: `src/ai_video_editor/cli/commands/render.py` (imports at top; add `_gps_graphic_overlays` after `_gps_overlay_fragment`, currently lines 50-66; wire into `run_render`, currently lines 69-105)
- Test: `tests/cli/test_render.py`

**Interfaces:**
- Consumes: `render_marker_dot`, `build_elevation_marker_positions`, `build_route_map_marker_positions`, `render_elevation_profile`, `render_route_map` (Tasks 3-6, all in `ai_video_editor.gps.overlays`); `GraphicOverlaySpec`, `MarkerOverlaySpec` (Task 7); `render_edit_plan(..., graphic_overlays=...)` (Task 8).
- Produces: `_gps_graphic_overlays(gps_path: Path | None, kept: list[EditAction], crossfade_s: float, preset: RenderPreset, work_dir: Path) -> list[GraphicOverlaySpec]`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/cli/test_render.py`, at the end of the file:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/cli/test_render.py::test_gps_graphic_overlays_builds_specs_with_gps -v`
Expected: FAIL with `ImportError: cannot import name '_gps_graphic_overlays'`

- [ ] **Step 3: Implement `_gps_graphic_overlays` and wire it into `run_render`**

In `src/ai_video_editor/cli/commands/render.py`, add `import dataclasses` near the top (with the other stdlib imports, after `import json`), and change:

```python
from ai_video_editor.gps.overlays import build_timesynced_overlay
```

to:

```python
from ai_video_editor.gps.overlays import (
    build_elevation_marker_positions,
    build_route_map_marker_positions,
    build_timesynced_overlay,
    render_elevation_profile,
    render_marker_dot,
    render_route_map,
)
```

and change:

```python
from ai_video_editor.render.presets import get_preset
from ai_video_editor.render.renderer import RenderResult, render_edit_plan
```

to:

```python
from ai_video_editor.render.presets import RenderPreset, get_preset
from ai_video_editor.render.renderer import GraphicOverlaySpec, MarkerOverlaySpec, RenderResult, render_edit_plan
```

Add this function after `_gps_overlay_fragment` (after line 66):

```python
def _gps_graphic_overlays(
    gps_path: Path | None,
    kept: list[EditAction],
    crossfade_s: float,
    preset: RenderPreset,
    work_dir: Path,
) -> list[GraphicOverlaySpec]:
    """Build the elevation-profile and route-map graphic overlays.

    Renders the elevation-profile strip, route-map plot, and marker dot
    into ``work_dir`` and returns the composited overlay specs -- empty
    when GPS data is unavailable, mirroring :func:`_gps_overlay_fragment`.
    """
    if gps_path is None:
        return []
    track = load_track(gps_path)
    metrics = compute_metrics(track)
    if not metrics.samples:
        return []

    output_duration = _rendered_output_duration(kept, crossfade_s)
    margin = 20
    specs: list[GraphicOverlaySpec] = []

    dot = render_marker_dot(work_dir / "marker_dot.png", radius_px=8)
    dot_radius = dot.width / 2

    elevation_image = render_elevation_profile(metrics, work_dir / "elevation_profile.png")
    if elevation_image is not None and elevation_image.geometry is not None:
        elev_x = (preset.width - elevation_image.width) // 2
        elev_y = preset.height - elevation_image.height - margin
        windows = build_elevation_marker_positions(metrics, elevation_image.geometry, output_duration)
        centered = [
            dataclasses.replace(w, x_px=w.x_px - dot_radius, y_px=w.y_px - dot_radius) for w in windows
        ]
        marker = MarkerOverlaySpec(image_path=dot.path, windows=centered) if centered else None
        specs.append(GraphicOverlaySpec(image_path=elevation_image.path, x=elev_x, y=elev_y, marker=marker))

    route_image = render_route_map(metrics, work_dir / "route_map.png")
    if route_image is not None and route_image.geometry is not None:
        route_x = preset.width - route_image.width - margin
        route_y = preset.height - route_image.height - margin
        windows = build_route_map_marker_positions(metrics, route_image.geometry, output_duration)
        centered = [
            dataclasses.replace(w, x_px=w.x_px - dot_radius, y_px=w.y_px - dot_radius) for w in windows
        ]
        marker = MarkerOverlaySpec(image_path=dot.path, windows=centered) if centered else None
        specs.append(GraphicOverlaySpec(image_path=route_image.path, x=route_x, y=route_y, marker=marker))

    return specs
```

Then in `run_render`, change the block after the `dry_run` early return (currently lines 97-104):

```python
    result = render_edit_plan(
        actions,
        output_path,
        preset,
        crossfade_s=crossfade_s,
        overlay_drawtext=overlay,
        prefer_gpu=True,
    )
    return result
```

to:

```python
    graphic_overlays = _gps_graphic_overlays(gps, kept_actions, crossfade_s, preset, output_path.parent)

    result = render_edit_plan(
        actions,
        output_path,
        preset,
        crossfade_s=crossfade_s,
        overlay_drawtext=overlay,
        graphic_overlays=graphic_overlays,
        prefer_gpu=True,
    )
    return result
```

- [ ] **Step 4: Run tests to verify everything passes**

Run: `pytest tests/cli/test_render.py -v`
Expected: PASS (all tests, including the two new ones)

Then run the full fast suite to confirm no project-wide regressions: `pytest tests/ -v -m "not slow"`

- [ ] **Step 5: Commit**

```bash
git add src/ai_video_editor/cli/commands/render.py tests/cli/test_render.py
git commit -m "feat: wire graphical GPS overlays into the render CLI command"
```

---

## Post-plan verification

After Task 9 lands, run the full suite once (including slow tests) to confirm the whole feature works end-to-end:

```bash
.venv/bin/python -m pytest tests/ --cov=ai_video_editor --cov-report=term-missing
.venv/bin/python -m ruff check src/
.venv/bin/python -m mypy src/
```

Then update `docs/final_report.md` §5 "Future improvements" to move this item from the roadmap into a "done" note (matching the existing `~~Time-synced, per-frame GPS overlays~~ **(done)**` pattern already in that file), and `README.md`'s "GPS" bullet if it needs updating to mention the graphical markers.
