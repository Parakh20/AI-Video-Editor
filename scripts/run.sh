#!/usr/bin/env bash
# Thin wrapper so the CLI can be invoked without manually activating the venv:
#   ./scripts/run.sh full assets/raw --limit 4
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_BIN="$PROJECT_ROOT/.venv/bin/video-editor"

if [ ! -x "$VENV_BIN" ]; then
    echo "ERROR: $VENV_BIN not found. Run ./scripts/setup.sh first." >&2
    exit 1
fi

exec "$VENV_BIN" "$@"
