# Repository conventions

The rules a change to this repository follows, beyond what the code and
its tests enforce. They apply to every contributor, human or automated.

## Layout

| Path | Contents |
|---|---|
| `src/omopmap/` | The library and the `omopmap` CLI: the eligible-target dictionary builder, the lexical, embedding and fused retrievers, the optional sv→en translation step, the fine-tuning harness, metrics, statistics, calibration, adjudication, the compute gate, run manifests, the MEDS/FEMR artifact writers and the CLMBR importer (`clmbr/`, with its pins in `pins.toml`) |
| `jobs/runpod/` | The generic rented-GPU job: setup, the end-to-end run with its cost gate, the opt-in real-model smoke and the runbook |
| `vocab/` | The vocabulary KDL tree: `module.kdl`, the node schema `schema.kdl`, and the committed `clmbr/manifest.json` |
| `tests/` | Hermetic tests over synthetic fixtures |
| `context/reference/` | These reference notes; the README names every one of them |

## What is never committed

- **Vocabulary content.** No concept lists, no licensed designations
  (SNOMED CT, LOINC or CPT4 text) and no Athena downloads, in code,
  tests or documentation. The CLMBR importer, its typed schema,
  `pins.toml` and `vocab/clmbr/manifest.json` (counts and hashes) are
  committed; the generated `vocab/clmbr/*.kdl` documents and any
  `concept_set.parquet` stay local and are git-ignored
  ([clmbr-concept-set.md](clmbr-concept-set.md),
  [athena-vocabularies.md](athena-vocabularies.md)).
- **Real records and real terms.** Tests and fixtures use invented codes
  and invented strings only.
- **Identifying detail.** No institution, site, ward or unit names
  beyond the research provenance in
  [project-context.md](project-context.md), no allocation ids,
  usernames, hostnames or host paths, and no personal email addresses.
  Author names appear only in the technical author fields
  (`pyproject.toml`, `CITATION.cff`).
- **Secrets.** Every input that needs a credential reads it from the
  environment.

## Documentation

The committed documentation describes the current state of the code:
no decision records, changelogs, "why we chose" prose, planned or
upcoming work, or deadlines. Results quoted from the source experiments
are labelled as theirs ([results.md](results.md)); nothing claims a
result on real data for this code.

## Commits

Commit messages follow Conventional Commits: a lowercase
`type(scope): subject` of at most 72 characters, with no body and no
trailers.
