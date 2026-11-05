# Compute gates

Embedding a large concept dictionary is the one step of the mapping work that
can run for hours on a GPU, and in the source experiments that GPU was rented
by the hour. `omopmap gate` measures before it spends: it times the encoder on
a small sample, projects the full job, and stops if the projection is far
beyond the budget, before any bulk work starts.

The vocabulary-mapping experiments this follows
([ludvigalden/vocabmod-se](https://github.com/ludvigalden/vocabmod-se)) ran on
rented GPUs at Runpod. The record-standardisation pipeline of
[GMLF](https://urn.kb.se/resolve?urn=urn:nbn:se:uu:diva-590690) ran on Bianca at
UPPMAX, and the gate is not part of it. On a fixed allocation the stop rule is
still useful as a sanity check, but wall time there is not billed per hour.

## Rule

```
projected = fixed_seconds + Σ over stages: full_items × measured_seconds / measured_items
threshold = factor × budget            (factor = 2 by default)
stop      = projected > threshold
```

- **Stages** are measured separately, because labels and queries have
  different lengths and throughputs: `labels` (the eligible concept names) and
  `queries` (the non-empty source terms). Each is timed on its first
  `--sample` texts (default 1,000) and extrapolated linearly to its full size.
- **The model is loaded before timing** (one warm-up encode), so download and
  load time do not inflate the per-item rate. Steps that do not scale with the
  sample (model download, retrieval, I/O) belong in `--fixed-seconds`.
- **Why twice the budget.** Throughput on a sample is noisy, and a projection
  within 2× of the estimate is ordinary variance. A projection beyond 2× means
  something is structurally wrong: the hardware is slower than assumed, the
  input is larger than assumed, or the estimate is. Continuing would spend money
  nobody budgeted for, on a run that will probably have to be repeated. The
  boundary is strict: a projection exactly at 2× passes.

## Outputs and exit codes

| File | Written when | Content |
|---|---|---|
| `gate.json` | always | stages with measured items, seconds and per-stage projection; fixed allowance; budget; factor; total projection; threshold; `stop` |
| `GATE_STOP.json` | only on stop | the same content as `gate.json` |
| `run_manifest.json` | always, last | encoder identity, parameters, input hashes ([retrieval-methods.md](retrieval-methods.md#run-manifests)) |

Exit **3** means the gate stopped the job; exit 0 means it may proceed. A job
script should check the exit code before the bulk embedding step. The stop
file, written whether or not the caller checks it, is the record an operator
downloads before terminating the machine. A passing run keeps its measured
rates in `gate.json`, so the real throughput is on record either way.

```
omopmap gate --concepts CONCEPT.csv --queries terms.tsv --out runs/gate \
  --model sapbert --budget-hours 1.5 --fixed-seconds 900
```

## The gate as a job step

The rented-GPU job folder wires the gate in as an explicit step:
[jobs/runpod/run_all.sh](../../jobs/runpod/run_all.sh) runs it after the
dictionary build and before any bulk embedding, and stops the whole job when
it exits 3, leaving the projection and the stop file on disk to copy back.
The folder's operational runbook — setup, environment-variable inputs, cost
and wall-time checks before renting, what to copy back, and teardown — is
[jobs/runpod/RUNBOOK.md](../../jobs/runpod/RUNBOOK.md).

## The source runbook's numbers

These are the figures the source runbook used for its first rented-GPU
retrieval job. They are quoted as recorded; they are not measurements of this
package.

- **Budget:** 1.5 h estimated in total, so the stop threshold was 2 × 1.5 h =
  3 h, with a 0.25 h retrieval allowance added to the projected embedding
  time.
- **Sample:** 1,000 labels and 1,000 queries.
- **Throughput floor:** a Colab T4 smoke run measured 572 labels/s and 4,042
  queries/s with zero truncations at 128 tokens. The runbook used that rate as
  a conservative floor and deliberately did not credit the faster rented card's
  higher fp16 throughput.
- **Host:** an RTX A5000 at about $0.27/h. The runbook expected 35-60 minutes
  of work, budgeted 1.5 h, and put the cost at about $0.16-0.27.

Later experiments in the same work used the same rule in minutes, with a fixed
allowance for the non-scaling steps.

## Why gate at all

The gate makes the cost of a run an explicit, recorded decision rather than
something discovered on the invoice. It also fails closed: a run that cannot
show a passing projection does not start its bulk work. This mirrors the hash
check on frozen indexes (exit 4): both are cheap checks placed in front of an
expensive or irreversible step, with a dedicated exit code so a script cannot
mistake a stop for a crash.
