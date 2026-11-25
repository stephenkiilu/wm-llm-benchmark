#!/bin/bash
# All full-text runs: 5 models x 2 arms x 622 articles = 6,220 calls (US$34 logged).
# Each arm is retried up to MAX_PASSES times with --resume, so an interruption
# loses only the calls in flight; rerun the same command to continue.
#
#   caffeinate -i extraction/run_all.sh 2>&1 | tee run_$(date +%F).log
#   extraction/run_all.sh gpt-5.6-luna          # a subset of models
#   LIMIT=10 extraction/run_all.sh              # smoke test on 10 articles

set -u
cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"
EFFORT="medium"
MAX_PASSES=3          # attempts per arm; each retry resumes, it does not restart
SEED=12345            # fixed sampling seed; verified deterministic across repeat calls
LIMIT="${LIMIT:-}"

MODELS=("gpt-4o-mini" "gpt-5.4-nano" "gpt-5.4-mini" "gpt-5.6-luna" "gpt-5-mini")
[ $# -gt 0 ] && MODELS=("$@")

# Requests in flight per model: fast models get fewer, so all stay under rate limits.
concurrency_for() {
    case "$1" in
        gpt-4o-mini)  echo 8  ;;   # 3.7s/call  -> ~131 rpm   fastest = heaviest load
        gpt-5.4-nano) echo 10 ;;   # ~7s/call   ->  ~86 rpm   (projected)
        gpt-5.4-mini) echo 10 ;;   # ~7s/call   ->  ~86 rpm   (projected)
        gpt-5.6-luna) echo 10 ;;   # 7.4s/call  ->  ~81 rpm
        gpt-5-mini)   echo 25 ;;   # 32.6s/call ->  ~46 rpm   slow calls, safe to push
        *)
            echo "unknown model '$1' - add it to concurrency_for() in $(basename "$0")" >&2
            return 1 ;;
    esac
}

started=$(date +%s)
declare -a COMPLETED=() FAILED=()

for model in "${MODELS[@]}"; do
    conc=$(concurrency_for "$model") || exit 1
    for arm in without_vocabulary with_vocabulary; do
        label="$model/$arm"
        ok=0
        for pass in $(seq 1 $MAX_PASSES); do
            echo ""
            echo "=================================================================="
            echo " $label | effort=$EFFORT | seed=$SEED | conc=$conc | pass $pass/$MAX_PASSES"
            echo " $(date '+%F %T')"
            echo "=================================================================="
            "$PY" extraction/run_extraction.py \
                --model "$model" --arm "$arm" --effort "$EFFORT" \
                --concurrency "$conc" --seed "$SEED" --resume \
                ${LIMIT:+--limit $LIMIT}
            if [ $? -eq 0 ]; then ok=1; break; fi
            echo "!! $label pass $pass failed. Rows already written are safe."
            [ "$pass" -lt "$MAX_PASSES" ] && { echo "!! waiting 120s before resuming..."; sleep 120; }
        done
        if [ "$ok" -eq 1 ]; then COMPLETED+=("$label"); else FAILED+=("$label"); fi
    done
done

echo ""
echo "=================================================================="
echo " finished in $(( ($(date +%s) - started) / 60 )) min"
echo "=================================================================="
echo "complete (${#COMPLETED[@]}/10):"
for x in "${COMPLETED[@]:-}"; do [ -n "$x" ] && echo "   ok    $x"; done
if [ "${#FAILED[@]}" -gt 0 ]; then
    echo "INCOMPLETE (${#FAILED[@]}):"
    for x in "${FAILED[@]}"; do echo "   FAIL  $x"; done
    echo ""
    echo "re-run this script to resume the incomplete arms."
fi
echo ""
echo "row counts written:"
for f in predictions/full_text/*.csv; do
    [ -e "$f" ] && printf "   %-46s %s\n" "$(basename "$f")" "$(( $(wc -l < "$f") - 1 ))/622"
done
echo ""
echo "score with:  ./reproduce.sh"
