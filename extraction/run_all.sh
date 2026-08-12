#!/usr/bin/env bash
# Run all 14 section-combination extractions in parallel, one API key each.
set -euo pipefail
cd "$(dirname "$0")/.."
for mode in $(seq 1 14); do
  python -m extraction.extract_without_lut \
    --mode "$mode" --key $(( (mode - 1) % 7 )) --concurrency 20 --resume &
done
wait
