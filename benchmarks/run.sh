#!/bin/bash
# Thin shell wrapper around run_fixture.py so developers can invoke benchmarks
# the same way across environments. See run_fixture.py for the real logic.
set -euo pipefail
BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$BENCH_DIR/run_fixture.py" "$@"
