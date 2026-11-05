#!/bin/sh
# Opt-in real-model smoke for the rented-GPU job folder.
#
# Everything here downloads or loads real model weights, so it is never run
# by default. It exercises the paths the hermetic suite cannot: the pinned
# CLMBR dictionary import, a real BioLORD fine-tune, re-indexing from the
# hash-verified checkpoint (--model-dir), and the real translator.
#
# Required environment:
#   OMOPMAP_CLMBR_DIR  local copy of the pinned dictionary.msgpack; the
#                      importer itself verifies its SHA-256 and fails closed
#                      on any other file
#   HF_TOKEN           optional; read from the environment only, never
#                      echoed or written to disk
#
# Estimated cost: model downloads dominate (BioLORD a few hundred MB, the
# translator model several GB); the compute itself is minutes on a GPU.
set -eu

repo_root=$(CDPATH='' cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$repo_root"

# Keep the CUDA torch build setup.sh installed: no re-sync to the CPU lock.
export UV_NO_SYNC=1

RUNS_DIR=${OMOPMAP_RUNS_DIR:-artifacts/runpod}
RUN_LOG=${OMOPMAP_RUN_LOG:-"$RUNS_DIR/real_model_smoke.log"}
mkdir -p "$RUNS_DIR"
: > "$RUN_LOG"

log() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$RUN_LOG"
}

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

: "${OMOPMAP_CLMBR_DIR:?Set OMOPMAP_CLMBR_DIR to a local copy of the pinned dictionary; nothing here runs without it}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "FAIL: nvidia-smi not found; the smoke requires a GPU machine" >&2
  exit 1
fi

# --- 1: the real_model-marked pytest tests (pinned dictionary import) ---
run_step 01-real-model-pytest \
  uv run pytest -m real_model -rs \
  || die "real_model pytest selection failed"

# --- 2: a tiny real BioLORD fine-tune on the committed synthetic fixtures ---
run_step 02-finetune-biolord \
  uv run omopmap finetune \
  --pairs tests/fixtures/source_terms.csv \
  --concepts tests/fixtures/concepts.csv \
  --model biolord \
  --out "$RUNS_DIR/smoke/finetune" \
  || die "real fine-tune failed"

# --- 3: re-index from the hash-verified checkpoint (--model-dir) ---
run_step 03-index-from-checkpoint \
  uv run omopmap retrieve index \
  --concepts tests/fixtures/concepts.csv \
  --all-concepts \
  --model-dir "$RUNS_DIR/smoke/finetune" \
  --model biolord \
  --out "$RUNS_DIR/smoke/index-ft" \
  || die "re-index from checkpoint failed"

# --- 4: real translator smoke (the command's own smoke gate applies) ---
run_step 04-translate \
  uv run omopmap translate run \
  --queries tests/fixtures/source_terms.csv \
  --out "$RUNS_DIR/smoke/translate" \
  || die "translator smoke failed"

log "real_model_smoke: done"
echo "real_model smoke done; run log: $RUN_LOG"
