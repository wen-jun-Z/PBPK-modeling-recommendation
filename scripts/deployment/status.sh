#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "$PROJECT_ROOT/runtime/app-env/bin/python" "$PROJECT_ROOT/scripts/deployment/services.py" status
