#!/bin/bash
# =============================================================================
# run_variation_test.sh — Run-to-run performance variation test
#
# Runs N_RUNS identical evaluations of N_QUERIES queries and measures the
# run-to-run spread in per-query latency. If the relative spread exceeds
# VARIATION_THRESHOLD, automatically triggers an extra run with reasoning=low
# to check whether deterministic reasoning reduces variance.
#
# Usage:
#   bash run_variation_test.sh [N_QUERIES [NUM_WORKERS]]
#
# Optional env vars:
#   N_RUNS               Number of baseline runs         (default: 3)
#   VARIATION_THRESHOLD  Relative spread (0–1) that      (default: 0.15 = 15%)
#                        triggers a low-reasoning run
#   CONFIG               Path to config.sh               (default: config.sh)
#
# Examples:
#   bash run_variation_test.sh 50 10
#   N_RUNS=5 VARIATION_THRESHOLD=0.10 bash run_variation_test.sh 50 10
#   INFERENCE_PERF_TEST_MODE=logs_result.json bash run_variation_test.sh 50 10
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG="${CONFIG:-config.sh}"
if [[ -f "${CONFIG}" ]]; then
    source "${CONFIG}"
else
    echo "WARNING: ${CONFIG} not found; using built-in defaults" >&2
fi

N_QUERIES="${1:-50}"
NUM_WORKERS="${2:-10}"
N_RUNS="${N_RUNS:-3}"
VARIATION_THRESHOLD="${VARIATION_THRESHOLD:-0.05}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
VARIATION_DIR="variation_test_${TIMESTAMP}"
mkdir -p "${VARIATION_DIR}"

SUMMARY_FILE="${VARIATION_DIR}/summary.txt"

# ── Helpers ───────────────────────────────────────────────────────────────────

# Extract the mean per-query latency (seconds) from a captured run log.
# Parses: "Per-query latency (query-to-answer):  mean=228.21s  ..."
extract_mean() {
    local log_file="$1"
    grep 'Per-query latency (query-to-answer)' "${log_file}" \
        | sed 's/.*mean=\([0-9.]*\)s.*/\1/' \
        | tail -1
}

# Run scripts/run_multi_shot.sh, capturing all output to a log file.
# Optional extra KEY=VALUE env var pairs can be passed after the log file arg.
run_experiment() {
    local log_file="$1"
    shift
    # Any remaining args are treated as VAR=VALUE env overrides.
    env "$@" bash scripts/run_multi_shot.sh "${N_QUERIES}" "${NUM_WORKERS}" \
        > "${log_file}" 2>&1
    # Print the last few summary lines so progress is visible on the terminal.
    tail -20 "${log_file}"
}

log() { echo "$@" | tee -a "${SUMMARY_FILE}"; }

# ── Header ────────────────────────────────────────────────────────────────────
log "================================================================"
log "  Run-to-run variation test  ($(date))"
log "  Queries per run   : ${N_QUERIES}"
log "  Workers           : ${NUM_WORKERS}"
log "  Baseline runs     : ${N_RUNS}"
log "  Variation threshold (relative spread): $(python3 -c "print(f'{${VARIATION_THRESHOLD}*100:.0f}%')")"
log "  Output dir        : ${VARIATION_DIR}/"
if [[ -n "${INFERENCE_PERF_TEST_MODE:-}" ]]; then
    log "  Perf-test mode    : ${INFERENCE_PERF_TEST_MODE}"
fi
log "================================================================"
log ""

# ── Baseline runs ─────────────────────────────────────────────────────────────
declare -a mean_lats=()

for i in $(seq 1 "${N_RUNS}"); do
    log "────────────────────────────────────────────────────────────────"
    log "  Baseline run ${i} / ${N_RUNS}  ($(date +%H:%M:%S))"
    log "────────────────────────────────────────────────────────────────"

    RUN_LOG="${VARIATION_DIR}/run_${i}.log"
    run_experiment "${RUN_LOG}"

    mean=$(extract_mean "${RUN_LOG}" || echo "")
    if [[ -z "${mean}" ]]; then
        echo "ERROR: could not parse mean latency from run ${i} log (${RUN_LOG})" >&2
        mean="0"
    fi
    mean_lats+=("${mean}")
    log ""
    log "  → Run ${i} mean latency: ${mean}s"
    log ""
done

# ── Variation analysis ────────────────────────────────────────────────────────
log "================================================================"
log "  VARIATION SUMMARY"
log "================================================================"

# Compute stats in Python and embed a machine-readable OVER_THRESHOLD tag.
ANALYSIS=$(VARIATION_THRESHOLD="${VARIATION_THRESHOLD}" \
python3 - "${mean_lats[@]}" << 'PYEOF'
import sys, os, statistics

vals = list(map(float, sys.argv[1:]))
n    = len(vals)
mn   = statistics.mean(vals)
rng  = max(vals) - min(vals)
rel  = rng / mn if mn > 0 else 0
cv   = statistics.stdev(vals) / mn if n > 1 and mn > 0 else 0

for i, v in enumerate(vals, 1):
    print(f"  Run {i:>2}: mean latency = {v:.2f}s")
print()
print(f"  Mean across runs    : {mn:.2f}s")
print(f"  Min / Max           : {min(vals):.2f}s / {max(vals):.2f}s")
print(f"  Relative spread     : {rel*100:.1f}%   (max−min) / mean")
print(f"  Coeff of variation  : {cv*100:.1f}%   std / mean")

thresh = float(os.environ.get('VARIATION_THRESHOLD', '0.15'))
print()
if rel > thresh:
    print(f"  ⚠  Relative spread {rel*100:.1f}% EXCEEDS threshold {thresh*100:.0f}%")
else:
    print(f"  ✓  Relative spread {rel*100:.1f}% is within threshold {thresh*100:.0f}%")

# Machine-readable tag — stripped before display
print(f"__OVER_THRESHOLD__={'yes' if rel > thresh else 'no'}")
PYEOF
)

# Print human-readable lines (strip the tag line)
echo "${ANALYSIS}" | grep -v '^__' | tee -a "${SUMMARY_FILE}"

OVER_THRESHOLD=$(echo "${ANALYSIS}" | grep '^__OVER_THRESHOLD__=' | cut -d= -f2)

# ── Optional low-reasoning run ────────────────────────────────────────────────
if [[ "${OVER_THRESHOLD}" == "yes" ]]; then
    log ""
    log "────────────────────────────────────────────────────────────────"
    log "  Low-reasoning run  (reasoning=low)  ($(date +%H:%M:%S))"
    log "────────────────────────────────────────────────────────────────"

    LOW_LOG="${VARIATION_DIR}/run_low_reasoning.log"
    run_experiment "${LOW_LOG}" INFERENCE_REASONING=low

    low_mean=$(extract_mean "${LOW_LOG}" || echo "N/A")
    log ""
    log "  → Low-reasoning mean latency: ${low_mean}s"
    log ""
    log "  Compare with baseline mean to assess whether reasoning=low"
    log "  reduces latency and/or variation."
else
    log ""
    log "  No additional run needed."
fi

# ── Footer ────────────────────────────────────────────────────────────────────
log ""
log "================================================================"
log "  Done.  ($(date))"
log "  All run logs : ${VARIATION_DIR}/run_N.log"
log "  Summary      : ${SUMMARY_FILE}"
log "================================================================"
