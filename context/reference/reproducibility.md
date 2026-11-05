# Reproducibility

A mapping result is only useful if it can be traced to the exact model, the
exact vocabulary bytes and the exact queries that produced it, and recomputed
from them. This page describes how the code makes that true: pinned model
revisions, hash-asserted frozen matrices, run manifests, and a GPU-versus-CPU
check that tells floating-point noise apart from a real change.

## Two compute environments, kept apart

The vocabulary-mapping experiments that this repository's retrieval,
fine-tuning and statistics methods come from
([ludvigalden/vocabmod-se](https://github.com/ludvigalden/vocabmod-se)) ran on
rented GPUs at Runpod. They work on vocabulary strings only, never on patient
records. The record-standardisation pipeline of
[GMLF](https://urn.kb.se/resolve?urn=urn:nbn:se:uu:diva-590690) (OMOP CDM to
MEDS/FEMR artifacts) ran on Bianca at UPPMAX, a system for sensitive data.
The two never shared a machine, and nothing on this page implies that a
mapping experiment saw patient data.

## Pinned model revisions

Every model is pinned to a full 40-hex commit, never a branch name, and the
code refuses anything else:

| Model | Use | Revision |
|---|---|---|
| `cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR` | embedding channel, CLS pooling | `47b6bd041ba61311584bb2494edfda5c7d9b719f` |
| `FremyCompany/BioLORD-2023-M` | embedding channel and fine-tune base, mean pooling | `4ea2ea2c89ef63365f7fcd91406a0cd8ac36d2e2` |
| `google/translategemma-4b-it` | optional sv→en translation | `10042cb0e6e7fdce748996a71dc3dc432a4e0c89` |
| `StanfordShahLab/clmbr-t-base` | CLMBR dictionary (`dictionary.msgpack`) | pinned with file and concept-list SHA-256 in `pins.toml` ([clmbr-concept-set.md](clmbr-concept-set.md)) |

The revision travels into every run manifest, so a result names the model
commit it came from.

## Frozen matrices with hash asserts

Embedding the dictionary is the expensive step, and a dictionary matrix is
only meaningful together with its row-to-concept map. `retrieve index` writes
both (`matrix.npy`, `ids.txt`) and records their SHA-256 in its manifest.
`retrieve embed` re-opens an index only after both blobs hash to the recorded
values, and exits 4 before computing a single score if either differs. It also
refuses (exit 1) to query an index with a different encoder than the one that
built it. The source experiments used the same discipline: a reproduction run
asserted the dictionary matrix hash and stopped with exit 4 on a mismatch.
Vectors are stored as float32 whatever the device, so a matrix's bytes do not
depend on fp16 compute.

## Run manifests

Every retrieval, dictionary, translation, fine-tune, statistics and
validation step writes `run_manifest.json` last, so a directory without one is
an incomplete run. A manifest records:

- the run type and the package version;
- the encoder or translator: model id, full revision, pooling, maximum
  length, dimension, device, GPU name, precision and library versions;
- the step's parameters and seed;
- the SHA-256 of every input (concept table, queries, pair table, index
  blobs, upstream rankings) and of every output file.

The JSON is deterministic (sorted keys, fixed indent, trailing newline), and
`--created-at` fixes the timestamp, so a rerun on the same inputs gives
byte-identical outputs; the tests check this for the retrieval, dictionary,
translation, fine-tune dry-run and statistics commands. Downstream
steps read upstream manifests: fusion, `stats mcnemar`, `validate runs` and
`finetune --negatives` all fail closed when a rankings file no longer hashes
to its manifest, or when two runs did not rank the same query table.

## GPU versus CPU: `omopmap validate runs`

The same pinned model on a GPU (fp16) and on a CPU (fp32) does not produce
byte-identical scores. It should still produce the same rankings, apart from
candidates whose scores are so close that half-precision noise swaps them.
`omopmap validate runs --first GPU_RUN --second CPU_RUN --out DIR` checks
exactly that, per query, over the top-`k` lists (`--k`, default 1, 5, 10):

| Measure | Meaning |
|---|---|
| `overlap@k` | share of the first run's top `k` also in the second run's top `k` |
| `first_divergence` | first rank at which the lists name different concepts (none if identical) |
| `max_rank_shift` | largest rank change of a concept in both top-`k` lists |
| `max_score_delta` | largest absolute score difference over concepts both runs scored |
| `tie_explained` | same top-`k` set, and every pair ordered differently scored within `--tie-eps` in the first run |

The verdict passes when every query is tie-explained and the largest score
difference is within `--score-tol`. Both tolerances default to `2e-3`, the
values of the source experiments' CPU regression gate, which re-scored
queries on a CPU in fp32 against the GPU's fp16 dictionary matrix and
required the same top-10 membership, the same order up to near-ties below
`2e-3`, and score deviations below `2e-3`. The two runs must be the same
channel and model revision over the same query table; only the device may
differ. The command writes `validation.json` (aggregates and every query's
measures) and a manifest, exits 0 on agreement, 5 on divergence, and 1 on
unusable input.

## Offline and enclave use

Part of this repository is meant to run inside a sensitive-data enclave with
no outbound network, as the record-standardisation side did on Bianca at
UPPMAX. What holds in the code:

- The CLMBR importer has a local route (`--model-dir`) for a byte-identical
  copy of the pinned release file; it checks the same file and concept-list
  hashes as the download route and does no network I/O
  ([clmbr-concept-set.md](clmbr-concept-set.md)).
- `omopmap artifacts verify`, the MEDS writer and converter, label building,
  dictionary building, `stats mcnemar`, `validate runs` and
  `finetune --dry-run` read and write local files only.
- `retrieve index|embed --model-dir` load a fine-tuned checkpoint from a
  local `finetune` output directory with `local_files_only`, after checking
  every checkpoint file against its recorded SHA-256 (exit 4 on a mismatch);
  no network access ([fine-tuning.md](fine-tuning.md)).
- `metrics compute` reads and writes local files only.
- Commands that load an encoder or translator by name (`retrieve index|embed`
  with `--model`, `gate`, `translate run`, `finetune` in full mode) load it by
  model id and pinned revision through `transformers`. Offline, that needs the pinned
  revision already in the local Hugging Face cache, with `HF_HUB_OFFLINE=1`
  set; nothing in this repository downloads it implicitly beyond that call.
- Dependencies are pinned in the committed `uv.lock`. On Linux the lock
  resolves torch to the CPU-only build from the PyTorch CPU index, so a plain
  sync never installs the CUDA stack; `jobs/runpod/setup.sh` swaps in the CUDA
  build of the same torch version on a GPU host. CI installs with
  `uv sync --frozen` and the `embed` and `hf` extras, type-checks against the
  real packages, runs the tests on CPU with `HF_HUB_OFFLINE=1` and
  `TRANSFORMERS_OFFLINE=1`, and deselects the `real_model` tests; no hermetic
  test needs a model download. Without the extras, the tests that need torch,
  transformers or huggingface_hub skip with a stated reason.
- The only network access in the code is that pinned-revision model load
  and the CLMBR importer's download route.
