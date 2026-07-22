# Editor integration

The edit decision engine (Phase 7, `ai_video_editor.editing.decision_engine`)
produces an editor-agnostic `list[EditAction]`. Everything downstream of
that is a pure export step — no editor-specific logic leaks upstream of the
`timeline/` package.

## `timeline.otio` (OpenTimelineIO)

`ai_video_editor.timeline.otio_builder.build_otio_timeline()` builds a plain
`opentimelineio.schema.Timeline`: one video track, one `Clip` per kept
`EditAction` (dropped actions are omitted — OTIO has no "considered and
rejected" concept), a `Transition` where the edit plan specifies a
crossfade, a chapter `Marker` per action that has one, and timeline-level
metadata (generation timestamp, source song, source project). No
editor-specific extensions are used, so this file is a neutral interchange
format.

`write_otio(timeline, path)` writes it via OTIO's own JSON adapter
(`otio_json` — the only relevant adapter registered in this environment;
`otio.adapters.available_adapter_names()` also lists `otioz`/`otiod`, which
are archive/bundle variants of the same format, not different editors).

## `project.kdenlive` (Kdenlive)

Kdenlive projects are MLT XML: a `<mlt>` root, one `<producer>` per source
clip (with `resource`/`mlt_service` properties), a `<playlist>` referencing
the producers in order, and a `<tractor>` wrapping the playlist as the
timeline, with `<transition>` elements for crossfades and
`kdenlive:marker:<frame>` properties for chapter markers.

**There is no MLT/Kdenlive adapter available in this project's OTIO
install** — confirmed by checking `otio.adapters.available_adapter_names()`
(only `otio_json`, `otioz`, `otiod`). So `ai_video_editor.timeline.kdenlive_export`
hand-builds the XML directly against the documented MLT project format,
rather than converting from an OTIO timeline.

**Caveat — not verified in the Kdenlive GUI.** Kdenlive itself is not
installed in this development environment (`apt install kdenlive` requires
the user's sudo password, which isn't available to an automated session).
The generated XML has been validated as well-formed
(`xml.etree.ElementTree.fromstring` parses it without error) with the
expected producer/playlist/tractor structure, and a real smoke test against
actual project data (`scripts/smoketest_timeline.py`) produces a file that
re-parses correctly. But it has not been opened in Kdenlive's UI. To
verify:

```bash
sudo apt install kdenlive
kdenlive ~/AI-Video-Editor/output/project.kdenlive
```

If anything fails to open, the fix belongs in `kdenlive_export.py`, not in
a workaround elsewhere.

## Adding another editor later

Because the interchange point is a plain `list[EditAction]` (or the OTIO
timeline built from it), adding another editor means adding one more
`build_<editor>_project()` / `write_<editor>_project()` pair in
`timeline/`, not touching `analysis/` or `editing/`.

- **DaVinci Resolve Studio**: there is already a fully installed and
  configured `davinci-resolve-mcp` server on this machine
  (`~/.local/share/davinci-resolve-mcp/`) with extensive scripting tools for
  media pool, timeline, color, render, etc. It was verified during an
  earlier session that DaVinci Resolve's scripting API only activates under
  a paid **Studio** license — only the free edition is installed on this
  machine, so `dvr.scriptapp("Resolve")` currently returns `None` and the
  MCP tools can't connect. Once Resolve Studio is activated
  (Preferences → General → "External scripting using" → Local, which only
  appears in Studio), a `timeline/resolve_export.py` could either replay an
  `EditAction` list directly through the MCP server's timeline-building
  tools, or import `timeline.otio` via Resolve's own OTIO import support.
- **Adobe Premiere**: Premiere can import OTIO (via its own adapter or
  through a conversion step to Premiere's XML/FCPXML interchange formats);
  a `timeline/premiere_export.py` would follow the same pattern as
  `kdenlive_export.py`.
