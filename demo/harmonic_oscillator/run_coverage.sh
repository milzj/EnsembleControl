#!/usr/bin/env bash
# Run the harmonic-oscillator coverage test: Monte-Carlo validation of the
# plug-in confidence interval for the SAA optimal value.
#
# Writes output/coverage/coverage.{json,tex,txt} and prints the LaTeX table of
# estimated coverage probabilities. Uses the repo virtualenv interpreter with a
# headless matplotlib backend, run from the demo's own directory.
#
# Any extra arguments are forwarded to the demo (a later flag overrides the
# defaults below), e.g.:
#   ./run_coverage.sh --workers 8
#   ./run_coverage.sh --R 100 --n-ref 512     # cheap smoke run
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PY="$ROOT/.venv/bin/python"

cd "$HERE"
MPLBACKEND=Agg "$PY" coverage_harmonic_oscillator.py --R 5000 --n-ref 4096 "$@"
