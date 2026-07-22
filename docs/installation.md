# Installation

## Prerequisites

- **OS**: Ubuntu 24.04 (developed and tested here; other recent Linux
  distros likely work but are untested).
- **Python 3.12** (the project pins `requires-python = ">=3.12,<3.13"`).
- **ffmpeg** with `h264_nvenc`/`hevc_nvenc`/`av1_nvenc` encoders and the
  `vidstab` filter (`libvidstab`) for GPU encoding and stabilization.
  Ubuntu 24.04's system `ffmpeg` package (6.1.1) already includes both.
- **NVIDIA GPU + driver** (optional but recommended) for NVENC hardware
  encoding and CUDA-accelerated PyTorch/Whisper inference. The renderer
  falls back to `libx264`/`libx265` automatically if no GPU is present.
- **`uv`** for dependency management — `scripts/setup.sh` installs it
  automatically if missing (to `~/.local/bin`, no sudo required).

## Setup

```bash
./scripts/setup.sh
```

This is idempotent (safe to re-run): it installs `uv` if needed, runs
`uv sync --dev` to create `.venv/` with every pinned dependency (including
CUDA-enabled PyTorch via the `pytorch-cu124` index if available), and
creates the `assets/`, `cache/`, `logs/`, `output/` directories.

## Pending manual step

`kdenlive`, `imagemagick`, `clinfo`, and `vulkan-tools` are apt packages
that were **not** installed by this project's automated setup, because
`apt install` requires the user's sudo password, which an automated
session cannot provide. Only `kdenlive` and `imagemagick` matter for this
project directly — and note `imagemagick` is **not actually required**
(see `docs/architecture.md`: all text/overlay rendering goes through ffmpeg
`drawtext` and PIL/matplotlib specifically to avoid that dependency).
`kdenlive` itself is only needed if you want to open/verify the generated
`project.kdenlive` in its GUI:

```bash
sudo apt update && sudo apt install -y kdenlive
kdenlive ~/AI-Video-Editor/output/project.kdenlive
```

## Verifying the install

```bash
source .venv/bin/activate
video-editor --help                  # or: ./scripts/run.sh --help

# Run the test suite (real end-to-end checks included, ~2-3 minutes)
.venv/bin/python -m pytest tests/ -v

# A quick, bounded, real pipeline run
video-editor full assets/raw --limit 2 --max-clip-seconds 20 \
    --music "assets/music/<your soundtrack>.mp3" --skip-subtitles
```

`torch.cuda.is_available()` should be `True` on a machine with a working
NVIDIA driver; if not, the pipeline still runs correctly on CPU, just
slower for Whisper transcription and video encoding.
