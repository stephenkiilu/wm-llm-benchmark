#!/bin/bash
# Rebuild every table and figure in the paper from the saved predictions.
# No API calls; about a minute.   Usage: ./reproduce.sh   (PY=python3 ./reproduce.sh)
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-python}"

for step in evaluate field_metrics sections complexity efficiency vocab_compliance error_sample; do
    echo "== analysis/$step.py"
    "$PY" "analysis/$step.py" > /dev/null
done
if [ -f data/full_text/articles.json ]; then      # full texts are not redistributed
    echo "== analysis/name_variants.py"
    "$PY" analysis/name_variants.py > /dev/null
fi
for fig in figure1 figure2 figure_s1_complexity figure_s2_pipeline; do
    echo "== figures/$fig.py"
    "$PY" "figures/$fig.py" > /dev/null
done
echo "done: tables in results/tables, figures in results/figures"
