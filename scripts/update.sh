#!/usr/bin/env bash
# Upgrade pinned dependencies within pyproject.toml's constraints and
# regenerate requirements.txt for anyone not using uv directly.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

if ! command -v uv >/dev/null 2>&1; then
    export PATH="$HOME/.local/bin:$PATH"
fi
if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv not found. Run ./scripts/setup.sh first." >&2
    exit 1
fi

echo "==> Upgrading dependencies (uv sync --dev --upgrade) ..."
uv sync --dev --upgrade

echo "==> Regenerating requirements.txt ..."
uv export --no-dev --no-hashes --format requirements-txt --quiet > requirements.txt

echo "==> Update complete. requirements.txt regenerated from the uv lockfile."
