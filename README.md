# ehr-omop-meds-femr

Production-ready infrastructure for the Swedish emergency-department
EHR → OMOP CDM → MEDS/FEMR → CLMBR pipeline: the terminological side —
mapping Swedish clinical source terms onto OMOP standard concepts — plus
the model-ready artifacts a CLMBR continued-pretraining pipeline consumes.

## Context

A feasibility study in the thesis ["Global Models, Local
Friction: Interoperability, Adaptation, and Deployment of Clinical
Foundation Models in Swedish Healthcare" (GMLF)](https://urn.kb.se/resolve?urn=urn:nbn:se:uu:diva-590690)
brought CLMBR-T-Base — a clinical foundation model trained on American
records — to Swedish emergency care. When the transformed Swedish records
entered the model, **fewer than 20% of distinct codes were recognised by
the model's vocabulary**: the Swedish code systems (ICD-10-SE, KVÅ, ATC,
NPU) are absent from the American standard vocabularies the model knows,
so every code reaches it only through a mapping to SNOMED CT, RxNorm or
LOINC. This repository is that mapping and standardisation chain, rebuilt
as reproducible tooling; the full abstract and the thesis's adaptation-cost
framework are in [project-context.md](context/reference/project-context.md).

## What is in the box

- Model-ready artifacts: MEDS extracts, label sets and the CLMBR concept set, schema-checked and hash-manifested, from synthetic cohorts or OMOP CDM tables with deterministic splits ([artifacts.md](context/reference/artifacts.md))
- CLMBR dictionary import and coverage: pins the gated CLMBR-T-Base vocabulary by SHA-256 and measures how much of an extract it recognises ([clmbr-concept-set.md](context/reference/clmbr-concept-set.md))
- Mapping retrieval: lexical containment, multilingual-embedding and rank-fused ranking over the eligible OMOP targets ([retrieval-methods.md](context/reference/retrieval-methods.md), [dictionary-eligibility.md](context/reference/dictionary-eligibility.md))
- Encoder fine-tuning (multiple-negatives, hard negatives, dev early stopping) and the optional sv→en translation step ([fine-tuning.md](context/reference/fine-tuning.md), [translation.md](context/reference/translation.md))
- Blinded adjudication sheets and calibrated H/M/L confidence tiers ([human-adjudication.md](context/reference/human-adjudication.md))
- Evaluation and statistics: recall@k with Wilson intervals behind fail-closed gates, exact paired McNemar ([evaluation-and-statistics.md](context/reference/evaluation-and-statistics.md))
- Reproducibility and compute gates: pinned revisions, hash-asserted frozen indexes, run manifests, GPU-vs-CPU validation, a cost gate in front of bulk embedding ([reproducibility.md](context/reference/reproducibility.md), [compute-gates.md](context/reference/compute-gates.md))

## Quickstart (synthetic data)

```sh
uv sync --all-extras
uv run omopmap meds synth --out artifacts/meds/demo --subjects 200 --seed 1
uv run omopmap artifacts verify artifacts/meds/demo
uv run omopmap dictionary build --concepts tests/fixtures/concepts.csv \
    --vocabulary DEMOCOND --vocabulary DEMODRUG --vocabulary DEMOPROC \
    --vocabulary DEMOLAB --vocabulary DEMOVOCAB --out artifacts/targets
uv run omopmap retrieve lexical --concepts artifacts/targets/targets.parquet \
    --queries tests/fixtures/source_terms.csv --out artifacts/runs/lexical
uv run omopmap finetune --dry-run --pairs tests/fixtures/source_terms.csv \
    --concepts artifacts/targets/targets.parquet --out artifacts/ft-dry
```

Every command verifies its inputs, refuses an occupied output directory,
and writes a run manifest last. No vocabulary data ships with the repo;
the committed fixtures carry invented codes only.

## Results of the source mapping experiments

Aggregate results of the vocabulary-mapping experiments on real Swedish
source terms that these methods follow ([results.md](context/reference/results.md); none are
claims about this code): of 400 held-out adjudicated queries, 315
(78.75%; 74.48–82.48%) accepted a standard-concept mapping after
documented correction, and 19.75% had no eligible target at all. On the
accepted queries, retrieval hit@10 was 7.3% for lexical containment
against 39.1% (33.8–44.5%) for the multilingual SapBERT encoder, and
encoder fine-tuning moved recall@1 from 91 to 103 and recall@5 from 138
to 150 (exact paired p = 0.036 and 0.0075), while recall@10 (159 → 162,
p = 0.25) was not significant, all on a fresh held-out sample.

## Running on real data

- Vocabularies: bring your own Athena download — concept tables and names are licensed text, nothing is committed, and every run manifest records its inputs' SHA-256 ([athena-vocabularies.md](context/reference/athena-vocabularies.md))
- CLMBR dictionary: gated release, pinned — import needs acceptance of the model's access terms and `HF_TOKEN` in the environment, or a local copy of the pinned file (`--model-dir`): byte-identical, hash-checked, no network access, the route for an offline sensitive-data enclave ([clmbr-concept-set.md](context/reference/clmbr-concept-set.md))
- Bulk embedding on rented GPUs runs through [jobs/runpod/RUNBOOK.md](jobs/runpod/RUNBOOK.md), with the compute gate in front of the expensive step

## The artifact contract

[ehr-clmbr-training](https://github.com/ludvigalden/ehr-clmbr-training)
consumes this repository's outputs read-only: MEDS/FEMR extracts, the
CLMBR concept set, coverage reports and label sets. Their layout, schemas,
dtypes, hashing and versioning are the contract in
[artifacts.md](context/reference/artifacts.md); `omopmap artifacts verify`
checks any artifact directory against it. Continued pretraining, task
heads, evaluation and inference live in that sibling repository; the
Swedish mapping-method material this repo rewrites comes from
[ludvigalden/vocabmod-se](https://github.com/ludvigalden/vocabmod-se).

## Reference index

Design notes for every component live in `context/reference/`:
[project-context.md](context/reference/project-context.md) (the research, the abstract, the adaptation-cost framework, compute environments), [artifacts.md](context/reference/artifacts.md) (the artifact contract), [clmbr-concept-set.md](context/reference/clmbr-concept-set.md) (the dictionary, its pin, coverage, licensing), [retrieval-methods.md](context/reference/retrieval-methods.md), [dictionary-eligibility.md](context/reference/dictionary-eligibility.md), [translation.md](context/reference/translation.md), [fine-tuning.md](context/reference/fine-tuning.md), [human-adjudication.md](context/reference/human-adjudication.md), [evaluation-and-statistics.md](context/reference/evaluation-and-statistics.md), [reproducibility.md](context/reference/reproducibility.md), [compute-gates.md](context/reference/compute-gates.md), [results.md](context/reference/results.md) (aggregate results of the source experiments), [athena-vocabularies.md](context/reference/athena-vocabularies.md) (obtaining and loading OMOP vocabularies), [swedish-coding-systems.md](context/reference/swedish-coding-systems.md) (ICD-10-SE, KVÅ, ATC, NPU and their OMOP mapping situation).

## Licence and citation

Code is Apache-2.0 (`LICENSE`). No mapping data ships with this
repository; any future own-work mapping data would be CC-BY-4.0. Cite
this software via [CITATION.cff](CITATION.cff).
