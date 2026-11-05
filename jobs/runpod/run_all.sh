#!/bin/sh
# End-to-end rented-GPU mapping job.
#
# Sequence: dictionary build -> gate (stops on exit 3, before any bulk
# embedding) -> bulk index -> lexical -> embed -> fuse -> optional fine-tune
# and re-index -> metrics -> McNemar -> GPU-vs-CPU validation -> adjudication
# sheet build. Every step logs its literal exit code and duration to the run
# log and stops the job on any non-zero exit. Run unpiped; see RUNBOOK.md.
#
# Inputs come from environment variables (defaults in brackets):
#   OMOPMAP_CONCEPTS       (required) Athena CONCEPT table (.csv/.tsv/.parquet)
#   OMOPMAP_QUERIES        (required) source-term table (source_term column;
#                          with the fine-tune path it also needs a gold
#                          concept_id column for stats mcnemar)
#   OMOPMAP_MODEL          encoder key: sapbert | biolord [sapbert]
#   OMOPMAP_LEXICAL_LABELS extra concept_id/label surfaces for the lexical run
#   OMOPMAP_BUDGET_HOURS   gate budget in hours [1.5]
#   OMOPMAP_FIXED_SECONDS  gate allowance for unscaled steps [900]
#   OMOPMAP_FINETUNE       1 enables the fine-tune + re-index block (also
#                          requires OMOPMAP_PAIRS)
#   OMOPMAP_PAIRS          source_term/concept_id/split table for fine-tuning
#   OMOPMAP_GOLD           adjudicated frame of record for metrics compute
#   OMOPMAP_SPLIT          restrict metrics to one split
#   OMOPMAP_CASES          adjudication cases table (query_id, system, code)
#   OMOPMAP_RUNS_DIR       output root [artifacts/runpod]
#   HF_TOKEN               optional; read from the environment only, never
#                          echoed or written to disk
set -eu

repo_root=$(CDPATH='' cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$repo_root"

# Keep the CUDA torch build setup.sh installed: no re-sync to the CPU lock.
export UV_NO_SYNC=1

RUNS_DIR=${OMOPMAP_RUNS_DIR:-artifacts/runpod}
RUN_LOG=${OMOPMAP_RUN_LOG:-"$RUNS_DIR/run_all.log"}
mkdir -p "$RUNS_DIR"
: > "$RUN_LOG"

log() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$RUN_LOG"
}

# Records start time, literal exit code and duration of each step, then
# returns the step's exit code.
run_step() {
  step=$1
  shift
  log "STEP $step: start"
  t0=$(date +%s)
  rc=0
  "$@" || rc=$?
  t1=$(date +%s)
  log "STEP $step: exit=$rc duration=$((t1 - t0))s"
  LAST_RC=$rc
  return "$rc"
}

die() {
  log "STOP: $* (exit $LAST_RC)"
  exit "$LAST_RC"
}

: "${OMOPMAP_CONCEPTS:?Set OMOPMAP_CONCEPTS to the Athena CONCEPT table}"
: "${OMOPMAP_QUERIES:?Set OMOPMAP_QUERIES to the source-term table}"

MODEL=${OMOPMAP_MODEL:-sapbert}
BUDGET_HOURS=${OMOPMAP_BUDGET_HOURS:-1.5}
FIXED_SECONDS=${OMOPMAP_FIXED_SECONDS:-900}

log "run_all: start (model=$MODEL budget_hours=$BUDGET_HOURS runs_dir=$RUNS_DIR)"

# --- 1: eligible-target dictionary ---
run_step 01-dictionary-build \
  uv run omopmap dictionary build \
  --concepts "$OMOPMAP_CONCEPTS" \
  --out "$RUNS_DIR/dictionary" \
  || die "dictionary build failed"

# --- 2: compute gate, before any bulk embedding; exit 3 stops the job ---
run_step 02-gate \
  uv run omopmap gate \
  --concepts "$RUNS_DIR/dictionary/targets.parquet" \
  --queries "$OMOPMAP_QUERIES" \
  --out "$RUNS_DIR/gate" \
  --model "$MODEL" \
  --budget-hours "$BUDGET_HOURS" \
  --fixed-seconds "$FIXED_SECONDS" \
  || {
    if [ "$LAST_RC" -eq 3 ]; then
      log "GATE STOP: projected wall time exceeded $BUDGET_HOURS x 2 h. Download $RUNS_DIR/gate/gate.json and $RUNS_DIR/gate/GATE_STOP.json, terminate the machine, and report; do not continue."
    else
      log "STOP: gate failed with exit $LAST_RC"
    fi
    exit "$LAST_RC"
  }

# --- 3: bulk embedding into a frozen index ---
run_step 03-index \
  uv run omopmap retrieve index \
  --concepts "$RUNS_DIR/dictionary/targets.parquet" \
  --out "$RUNS_DIR/index" \
  --model "$MODEL" \
  || die "index build failed"

# --- 4: lexical channel ---
set -- uv run omopmap retrieve lexical \
  --concepts "$RUNS_DIR/dictionary/targets.parquet" \
  --queries "$OMOPMAP_QUERIES" \
  --out "$RUNS_DIR/lexical"
[ -n "${OMOPMAP_LEXICAL_LABELS:-}" ] && set -- "$@" --labels "$OMOPMAP_LEXICAL_LABELS"
run_step 04-lexical "$@" || die "lexical run failed"

# --- 5: embedding channel against the frozen index ---
run_step 05-embed \
  uv run omopmap retrieve embed \
  --index "$RUNS_DIR/index" \
  --queries "$OMOPMAP_QUERIES" \
  --out "$RUNS_DIR/embed" \
  --model "$MODEL" \
  || die "embed run failed"

# --- 6: RRF fusion ---
run_step 06-fuse \
  uv run omopmap retrieve fuse \
  --run "$RUNS_DIR/lexical" \
  --run "$RUNS_DIR/embed" \
  --out "$RUNS_DIR/fused" \
  || die "fusion failed"

# --- 7: optional fine-tune and re-index ---
if [ "${OMOPMAP_FINETUNE:-0}" = "1" ]; then
  : "${OMOPMAP_PAIRS:?OMOPMAP_FINETUNE=1 requires OMOPMAP_PAIRS (source_term/concept_id/split table)}"

  run_step 07a-finetune \
    uv run omopmap finetune \
    --pairs "$OMOPMAP_PAIRS" \
    --concepts "$RUNS_DIR/dictionary/targets.parquet" \
    --model "$MODEL" \
    --out "$RUNS_DIR/finetune" \
    || die "fine-tune failed"

  run_step 07b-index-ft \
    uv run omopmap retrieve index \
    --concepts "$RUNS_DIR/dictionary/targets.parquet" \
    --out "$RUNS_DIR/index-ft" \
    --model-dir "$RUNS_DIR/finetune" \
    --model "$MODEL" \
    || die "fine-tuned index build failed"

  run_step 07c-embed-ft \
    uv run omopmap retrieve embed \
    --index "$RUNS_DIR/index-ft" \
    --queries "$OMOPMAP_QUERIES" \
    --out "$RUNS_DIR/embed-ft" \
    --model-dir "$RUNS_DIR/finetune" \
    --model "$MODEL" \
    || die "fine-tuned embed run failed"
else
  log "STEP 07-finetune: skipped (OMOPMAP_FINETUNE is not 1)"
fi

# --- 8: metrics over an adjudicated frame of record ---
if [ -n "${OMOPMAP_GOLD:-}" ]; then
  set -- uv run omopmap metrics compute \
    --run "lexical=$RUNS_DIR/lexical" \
    --run "embed=$RUNS_DIR/embed" \
    --run "fused=$RUNS_DIR/fused" \
    --gold "$OMOPMAP_GOLD" \
    --concepts "$RUNS_DIR/dictionary/targets.parquet" \
    --out "$RUNS_DIR/metrics"
  [ -n "${OMOPMAP_SPLIT:-}" ] && set -- "$@" --split "$OMOPMAP_SPLIT"
  run_step 08-metrics "$@" || die "metrics compute failed"
else
  log "STEP 08-metrics: skipped (OMOPMAP_GOLD, the adjudicated frame of record, is not set)"
fi

# --- 9: exact paired McNemar, fine-tuned vs baseline ---
if [ "${OMOPMAP_FINETUNE:-0}" = "1" ]; then
  run_step 09-mcnemar \
    uv run omopmap stats mcnemar \
    --baseline "$RUNS_DIR/embed" \
    --candidate "$RUNS_DIR/embed-ft" \
    --gold "$OMOPMAP_QUERIES" \
    --out "$RUNS_DIR/mcnemar" \
    || die "stats mcnemar failed"
else
  log "STEP 09-mcnemar: skipped (no fine-tuned candidate; set OMOPMAP_FINETUNE=1)"
fi

# --- 10: GPU-vs-CPU spot check on the embedding channel ---
run_step 10a-embed-cpu \
  env CUDA_VISIBLE_DEVICES= uv run omopmap retrieve embed \
  --index "$RUNS_DIR/index" \
  --queries "$OMOPMAP_QUERIES" \
  --out "$RUNS_DIR/embed-cpu" \
  --model "$MODEL" \
  || die "CPU spot-check embed run failed"

run_step 10b-validate \
  uv run omopmap validate runs \
  --first "$RUNS_DIR/embed" \
  --second "$RUNS_DIR/embed-cpu" \
  --out "$RUNS_DIR/validation" \
  || die "GPU-vs-CPU validation diverged or failed"

# --- 11: blinded adjudication sheet build ---
if [ -n "${OMOPMAP_CASES:-}" ]; then
  run_step 11-adjudicate-build \
    uv run omopmap adjudicate build \
    --run "embed=$RUNS_DIR/embed" \
    --run "lexical=$RUNS_DIR/lexical" \
    --cases "$OMOPMAP_CASES" \
    --concepts "$RUNS_DIR/dictionary/targets.parquet" \
    --sheet-out "$RUNS_DIR/adjudication/sheet" \
    --key-out "$RUNS_DIR/adjudication/key" \
    || die "adjudication sheet build failed"
else
  log "STEP 11-adjudicate-build: skipped (OMOPMAP_CASES is not set)"
fi

log "run_all: done"
echo "run_all done; run log: $RUN_LOG"
