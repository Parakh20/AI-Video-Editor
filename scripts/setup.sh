#!/usr/bin/env bash
# Idempotent project setup: installs uv if missing, syncs the venv.
# Safe to re-run at any time.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "==> AI Video Editor setup"

if ! command -v uv >/dev/null 2>&1; then
    if [ -x "$HOME/.local/bin/uv" ]; then
        export PATH="$HOME/.local/bin:$PATH"
    else
        echo "==> uv not found, installing to ~/.local/bin ..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv installation failed or ~/.local/bin is not on PATH." >&2
    exit 1
fi
echo "==> uv: $(uv --version)"

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "ERROR: ffmpeg is required but not found on PATH." >&2
    echo "       Install it via your system package manager (e.g. 'sudo apt install ffmpeg') and re-run this script." >&2
    exit 1
fi
echo "==> ffmpeg: $(ffmpeg -version | head -n1)"

echo "==> Syncing dependencies (uv sync --dev) ..."
uv sync --dev

mkdir -p assets/raw assets/music assets/fonts assets/luts configs cache models logs output

echo ""
echo "==> Setup complete."
echo "    Run the CLI with:   ./scripts/run.sh --help"
echo "    Or activate venv:   source .venv/bin/activate && video-editor --help"
echo "    Try the pipeline:   ./scripts/run.sh full assets/raw --limit 2 --dry-run"
