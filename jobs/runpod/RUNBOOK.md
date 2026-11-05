# RUNBOOK — rented-GPU mapping job

The vocabulary-mapping embedding and fine-tune experiments that this
package's retrieval methods follow
([ludvigalden/vocabmod-se](https://github.com/ludvigalden/vocabmod-se)) ran
on rented GPUs at Runpod. This folder is the generic, sanitised form of that
job: a setup script, one end-to-end run script with a cost gate in front of
the bulk embedding, an opt-in real-model smoke, and this runbook. No
credentials or account details are embedded anywhere; every input comes from
environment variables.

Scripts (run from anywhere; each locates the repository root relative to
itself):

- `setup.sh` — installs uv, restores the locked environment
  (`uv sync --frozen --extra embed --extra hf`), replaces the lock's CPU-only
  torch with the CUDA build of the same version (index from
  `OMOPMAP_TORCH_INDEX`, default the PyTorch CUDA 13.2 index), and verifies a
  usable CUDA GPU (`nvidia-smi` plus `torch.cuda`). Fails closed if any
  prerequisite is missing. The job scripts export `UV_NO_SYNC=1`, so `uv run`
  keeps that CUDA build; run any manual `uv run` on the pod with `--no-sync`
  for the same reason.
- `run_all.sh` — the full job, described below.
- `real_model_smoke.sh` — opt-in; runs the `real_model`-marked pytest tests
  (the pinned CLMBR dictionary import from `OMOPMAP_CLMBR_DIR`), a tiny real
  BioLORD fine-tune, re-indexing from the hash-verified checkpoint
  (`--model-dir`), and the real translator smoke. It is never run by default:
  every step downloads or loads real model weights.

## Step 0 — licensing gates (confirm before uploading anything)

- **Athena vocabulary: bring your own download.** The candidate labels are
  OMOP concept names from licensed terminologies (SNOMED CT, LOINC, RxNorm
  and others). Uploading these label strings to a third-party host is a
  distribution decision only you can make under your Athena licence. If it
  is not acceptable, do not run this job; the retrieval pipeline also runs
  locally on CPU, more slowly.
- **Gated model terms.** The CLMBR dictionary release is gated
  (credentialed-access licence; see
  [clmbr-concept-set.md](../../context/reference/clmbr-concept-set.md)). The
  encoder and translator models used by this job (SapBERT, BioLORD,
  translategemma) are not gated, but their model licences still apply. The
  pinned CLMBR route is not exercised by `run_all.sh`; the real-model smoke
  uses a local dictionary copy and verifies its hash itself.
- **Token handling.** `HF_TOKEN`, when you set it, is read from the
  environment only. No script echoes, logs, or writes it to disk, and no
  secret is stored in the repository.

## Step 1 — rent a machine

Any CUDA GPU host works; the source experiments used a mid-range 24 GB card
billed hourly. A CPU-only machine is of no use here: `setup.sh` fails
closed without a GPU. Choose at least 8 vCPUs (the lexical stage and parquet
I/O are CPU-bound) and enough disk for the models, the CONCEPT table, and
the frozen index (tens of GB is ample).

Before renting, write down the estimate and the stop rule (Step 3): the
gate makes the wall-time budget an explicit, recorded decision instead of
something discovered on the invoice.

## Step 2 — upload and set up

Upload or sync the repository to the machine, then:

    sh jobs/runpod/setup.sh

## Step 3 — configure the job

Environment variables (all documented in the header of `run_all.sh`):

| Variable | Meaning | Default |
|---|---|---|
| `OMOPMAP_CONCEPTS` | Athena CONCEPT table (required) | — |
| `OMOPMAP_QUERIES` | source-term table, `source_term` column (required) | — |
| `OMOPMAP_MODEL` | `sapbert` or `biolord` | `sapbert` |
| `OMOPMAP_LEXICAL_LABELS` | extra concept_id/label surfaces | unset |
| `OMOPMAP_BUDGET_HOURS` | gate budget, hours | `1.5` |
| `OMOPMAP_FIXED_SECONDS` | gate allowance for unscaled steps | `900` |
| `OMOPMAP_FINETUNE` | `1` enables fine-tune + re-index + McNemar | off |
| `OMOPMAP_PAIRS` | source_term/concept_id/split table for fine-tuning (required when `OMOPMAP_FINETUNE=1`) | — |
| `OMOPMAP_GOLD` | adjudicated frame of record for `metrics compute` | unset |
| `OMOPMAP_SPLIT` | restrict metrics to one split | unset |
| `OMOPMAP_CASES` | adjudication cases table (`query_id`, `system`, `code`) | unset |
| `OMOPMAP_RUNS_DIR` | output root | `artifacts/runpod` |
| `HF_TOKEN` | optional, environment only | unset |

Optional stages are skipped with the reason logged when their input variable
is unset; nothing is silently omitted.

## Step 4 — run under screen/tmux

    screen -S job
    sh jobs/runpod/run_all.sh

Run it unpiped: each step's literal exit code and duration are written to
`$OMOPMAP_RUNS_DIR/run_all.log`, and the job stops on any non-zero exit. Do
not pipe the script through `tee` or `head`; a pipe would report the
shaper's exit status instead of the job's.

Watch for:

- `STEP 02-gate: exit=0` — the gate passed and bulk embedding may start.
- `STEP 02-gate: exit=3` — **gate stop.** The projected wall time exceeded
  2x the budget. Download `gate.json` and `GATE_STOP.json`, terminate the
  machine, and report; do not continue. Details in
  [compute-gates.md](../../context/reference/compute-gates.md).
- `STEP 10b-validate: exit=5` — the CPU spot check diverged from the GPU
  run; the job stops (see
  [reproducibility.md](../../context/reference/reproducibility.md)).

## Step 5 — copy results back

Copy back only what is small and provenance-bearing:

- every `run_manifest.json`, plus `gate.json`, `mcnemar.json`,
  `metrics.json`, `validation.json`, `finetune.json`, and `run_all.log`.

Do **not** copy back data unless your licence permits redistribution of
concept text: the frozen index (`matrix.npy`), `ids.txt`, `targets.parquet`,
the query and pairs tables, and the adjudication `sheet.csv` all carry
licensed label strings or your own source terms. The sealed adjudication
key may be copied back only together with a completed sheet, since a key
alone is useless; treat both as licensed content under the same rule.

## Step 6 — teardown

Terminate the machine, then delete its storage volume. The volume keeps
billing if left behind, and it holds licensed label strings and your source
tables.

## Failure modes

| Symptom | Meaning | Action |
|---|---|---|
| `setup.sh` exits on `nvidia-smi` or `torch.cuda` | no usable GPU attached | redeploy with a GPU template |
| exit 3 at `STEP 02-gate` | throughput far below the estimate | stop, download the gate JSONs, terminate, report |
| exit 4 at any retrieval step | a frozen index hash mismatch | the index is corrupt or was rebuilt; re-run from `STEP 03-index` |
| exit 5 at `STEP 10b-validate` | GPU and CPU rankings diverged beyond tolerance | stop and report; do not use the run |
| a step logs `skipped` | its optional input env var is unset | expected behaviour; set the variable to include the stage |
