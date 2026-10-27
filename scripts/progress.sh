#!/usr/bin/env bash
# Read-only monitor. Ctrl+C never signals a background experiment.
set -euo pipefail
PROJECT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
exec "$PROJECT_DIRECTORY/runtime/app-env/bin/python" -B "$PROJECT_DIRECTORY/scripts/blueprint_status.py" "$@"
