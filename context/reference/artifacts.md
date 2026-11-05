# Artifact contract

This repository produces the model-ready artifacts that
[ehr-clmbr-training](https://github.com/ludvigalden/ehr-clmbr-training)
consumes for CLMBR continued pretraining, task heads and evaluation. The
consumer reads them as files. It does not import or duplicate the code
that writes them. This document is that interface: the on-disk layout,
the schema of every file, the integrity manifest, the versioning rules
and the commands that produce each artifact. The tables marked as
generated below are rendered from `omopmap.artifacts`. A test fails when
they differ from the code.

<!-- contract:versions -->
Contract version `1.0.0`; MEDS `0.3.3`; code pattern `^[^/\s](?:[^/\n]*[^/\s])?/\S(?:.*\S)?$`.
<!-- /contract:versions -->

## Where the artifacts sit in the pipeline

The pipeline follows the research setting of
[GMLF](https://urn.kb.se/resolve?urn=urn:nbn:se:uu:diva-590690)
("Global Models, Local Friction: Interoperability, Adaptation, and Deployment of Clinical Foundation Models in Swedish Healthcare"):
Swedish emergency-department records are standardised into the OMOP CDM,
converted to MEDS, and fed to CLMBR-T-Base, a foundation model trained on
American records. That work found that fewer than 20% of distinct codes in
the transformed Swedish records are recognised by the model's vocabulary.
The artifacts below make that chain reproducible and that number
measurable.

| Artifact | Pipeline step it serves | Consumer step |
|---|---|---|
| A. MEDS extract | Record standardisation: OMOP CDM to MEDS, with subject splits | Builds a `meds_reader` database, tokenises the `train` split for continued pretraining, evaluates on `tuning` / `held_out` |
| B. CLMBR concept set and manifest | Terminological cost: which codes the pinned CLMBR dictionary knows | Pins and checks the dictionary its base checkpoint carries; analyses vocabulary overlap |
| B'. Coverage report | Terminological cost, measured: the share of an extract's distinct codes in the dictionary | Reports coverage next to training results |
| C. Label set | Prediction tasks over an extract | Trains and evaluates task heads |

In the research project the ETL and model runs took place inside a
sensitive-data enclave (Bianca at UPPMAX). The contract tooling reflects
that setting: `omopmap artifacts verify` reads only local files and needs
no network access, and the manifests let an artifact be moved between
environments and checked byte for byte on arrival.

## Common rules

- **Paths.** Every artifact is a directory under `artifacts/` at the
  repository root (git-ignored; artifacts are never committed). The only
  committed artifact file is `vocab/clmbr/manifest.json`.
- **Code strings.** Every event code is `<vocabulary_id>/<concept_code>`:
  the OMOP `vocabulary_id` and `concept_code`, joined by `/`. A string is
  split on its **first** `/`, so a concept code may itself contain `/`.
  This is the format of the CLMBR dictionary (FEMR's `dictionary.msgpack`),
  so an extract code and a dictionary entry compare as plain strings. Both
  parts are non-empty and carry no leading or trailing whitespace. The only
  codes without a vocabulary prefix are the MEDS reserved codes
  `MEDS_BIRTH` and `MEDS_DEATH`. An event whose source code has no standard
  concept keeps its source vocabulary and code (for example a Swedish
  source vocabulary), never the OMOP placeholder concept `0`, so coverage
  can count it.
- **Parquet types.** Columns, column order, Arrow types and nullability
  must equal the tables below exactly. Strings are Arrow `string`, not
  `large_string` (Polars writes `large_string` by default; the producer
  casts every table to the contract schema before writing). Timestamps are
  timezone-naive `timestamp[us]`.
- **JSON.** UTF-8, keys sorted, two-space indent, one trailing newline.
  Unknown keys are rejected. Timestamps are UTC `YYYY-MM-DDTHH:MM:SSZ`.
- **Hashes.** SHA-256, lowercase hex, over the exact file bytes.
- **Fail closed.** Every check raises a typed error
  (`omopmap.artifacts.errors`); nothing is downgraded to a warning, and an
  artifact that fails any check is not consumed.

## A. MEDS extract

Follows [MEDS](https://github.com/Medical-Event-Data-Standard/meds)
`0.3.3`, the release FEMR 0.2.x requires (`femr==0.2.4` pins
`meds==0.3.3`).

### Layout

```
artifacts/meds/<dataset_name>/
  artifact_manifest.json
  data/
    train/0.parquet  train/1.parquet  ...
    tuning/0.parquet  ...
    held_out/0.parquet  ...
  metadata/
    dataset.json
    codes.parquet
    subject_splits.parquet
```

- All three split directories exist and hold shards numbered `0..n-1`
  without gaps; no other files are allowed.
- A subject's rows live in exactly one shard, and that shard's split
  directory equals the subject's split in `subject_splits.parquet`.
- Within a shard, rows are sorted by `(subject_id, time)` with null
  times (static events) first in each subject.
- Every subject has exactly one `MEDS_BIRTH` row, with a time, and no
  timed event before it. FEMR derives age from that event. The schema is
  identical in every shard and split, which is what `meds_reader_convert`
  requires, so the consumer needs no normalisation step (timezone
  stripping, birth-row injection, re-sorting or `numeric_value` casts)
  before conversion.

### `data/<split>/<n>.parquet` (generated)

<!-- contract:meds_data -->
| Column | Arrow type | Nullable | Semantics |
|---|---|---|---|
| `subject_id` | `int64` | no | Pseudonymous subject id; every row of a subject lives in exactly one shard. |
| `time` | `timestamp[us]` | yes | Event time, timezone-naive, microseconds. Null only for static events, which sort first within the subject. |
| `code` | `string` | no | `<vocabulary_id>/<concept_code>` (OMOP vocabulary id and concept code, split on the first `/`), or a reserved code `MEDS_BIRTH` / `MEDS_DEATH`. |
| `numeric_value` | `float32` | yes | Numeric measurement value; null when the event carries none. |
| `text_value` | `string` | yes | Text value; null when the event carries none. |
<!-- /contract:meds_data -->

### `metadata/codes.parquet` (generated)

Exactly one row per distinct code in `data/`, sorted by `code`.

<!-- contract:meds_codes -->
| Column | Arrow type | Nullable | Semantics |
|---|---|---|---|
| `code` | `string` | no | One row per distinct code in `data/`; unique. |
| `description` | `string` | yes | Human-readable description, or null. Generated from a local vocabulary; never committed. |
| `parent_codes` | `list<item: string>` | yes | Direct parents as code strings in the same format, or null / empty. |
<!-- /contract:meds_codes -->

### `metadata/subject_splits.parquet` (generated)

Exactly the subjects in `data/`, sorted by `subject_id`. The split names
are the MEDS ones: `train` (continued pretraining and head training),
`tuning` (model selection) and `held_out` (final evaluation only).

<!-- contract:subject_splits -->
| Column | Arrow type | Nullable | Semantics |
|---|---|---|---|
| `subject_id` | `int64` | no | Every subject present in `data/`, exactly once. |
| `split` | `string` | no | `train`, `tuning` or `held_out`; equals the `data/<split>/` directory that holds the subject. |
<!-- /contract:subject_splits -->

### `metadata/dataset.json` (generated)

<!-- contract:dataset_metadata -->
| Field | Type | Meaning |
|---|---|---|
| `dataset_name` | `str` | Extract name; the directory name. |
| `dataset_version` | `str` | Extract version label. |
| `etl_name` | `'omopmap'` | Always `omopmap`. |
| `etl_version` | `str` | Producing package version. |
| `meds_version` | `'0.3.3'` | Always `0.3.3`. |
| `created_at` | `str` | UTC creation time. |
<!-- /contract:dataset_metadata -->

### FEMR side: the `meds_reader` database

The consumer builds the `meds_reader` database from the extract:

```
meds_reader_convert artifacts/meds/<dataset_name> <database_dir>
```

`meds_reader_convert` comes with the `meds-reader` package. The stack the
pipeline was built and run with is `femr==0.2.4` with `meds==0.3.3`, which
resolves to `meds-reader` 0.1.x (`meds-reader` 0.2.0 requires
`meds>=0.4.1`). The database is then opened with
`meds_reader.SubjectDatabase(<database_dir>)`, and the `train` subjects
are selected through `metadata/subject_splits.parquet`.

The consumer owns this step, for three reasons:

1. The database is a binary index whose format belongs to the
   `meds-reader` version that the consumer's `femr` loads. Building it on
   the consumer side keeps writer and reader on one pinned version.
2. The producer stays free of `femr`, `torch` and `meds-reader`; the
   `meds-reader` 0.1.x line ships as source that needs a Bazel toolchain
   to build.
3. The MEDS Parquet extract is the standard, tool-neutral interface. The
   contract makes it convertible by construction: the layout, sort order,
   birth events and uniform schema above are exactly the preconditions of
   `meds_reader_convert`.

## B. CLMBR dictionary artifacts

The CLMBR-T-Base release (`StanfordShahLab/clmbr-t-base` on Hugging Face,
gated) carries its vocabulary as FEMR's `dictionary.msgpack`. The importer
reads that file from the pinned release revision (`--hf-repo`) or from a
local copy of the same file (`--model-dir`), checks its SHA-256 against
the single pin in `src/omopmap/clmbr/pins.toml`, checks the derived
concept list against the pinned `concept_list_sha256`, and writes the
artifacts below. Both routes check the identical pins and must reproduce
the same concept set. A hash mismatch on either pin, or an unreachable
download, stops the import with exit code 2 before anything is written. Concept lists and designations are never committed: only
counts and hashes are. The design, the pins and the licensing are in
[clmbr-concept-set.md](clmbr-concept-set.md).

### Layout

```
vocab/clmbr/
  manifest.json            committed: counts and hashes only
  <vocabulary_id>.kdl      generated, git-ignored: one file per vocabulary
artifacts/clmbr/
  artifact_manifest.json
  concept_set.parquet      generated, git-ignored
```

- `<vocabulary_id>.kdl` (KDL 2, node conventions in
  `vocab/schema.kdl`) holds one `concept` node per distinct concept
  of that vocabulary, sorted by code point, with its dictionary entry types
  (and, when an Athena `CONCEPT` table is supplied, its OMOP `concept_id`,
  `domain_id` and `standard_concept`). It holds no designations, weights,
  value bins or age statistics.
- `concept_set.parquet` is the same concept set in a form the consumer
  can load without a KDL parser.
- Both exclude dictionary entries of type `unused` and the site-local
  `CARE_SITE/*` ids of the release, which are not standard concepts; the
  manifest records how many `CARE_SITE` refs were excluded.

### `vocab/clmbr/manifest.json` (generated)

<!-- contract:clmbr_manifest -->
| Field | Type | Meaning |
|---|---|---|
| `contract_version` | `str` | Artifact contract version. |
| `source_repo` | `str` | Hugging Face repo id of the release. |
| `revision` | `str` | Pinned commit of that repo, which both acquisition routes are pinned to. |
| `filename` | `str` | Dictionary file name within the release. |
| `file_sha256` | `str` | SHA-256 of the dictionary file. |
| `entry_type_counts` | `dict["code" \| "numeric" \| "text" \| "unused", int]` | Dictionary entries per entry type, all four types. |
| `vocabulary_counts` | `dict[str, int]` | Distinct concepts per vocabulary id, `CARE_SITE` excluded. |
| `total` | `int` | Distinct concepts, `CARE_SITE` excluded. |
| `care_site_excluded` | `int` | Distinct `CARE_SITE` refs excluded. |
| `concept_list_sha256` | `str` | Hash of the concept list (see below). |
<!-- /contract:clmbr_manifest -->

`concept_list_sha256` is the SHA-256 of the UTF-8 bytes of the distinct
`<vocabulary_id>/<concept_code>` strings of the concept set, sorted by
code point and joined with `\n` (no trailing newline). It identifies the
concept set without publishing it. `omopmap.artifacts.concept_list_sha256`
implements it.

### `artifacts/clmbr/concept_set.parquet` (generated)

<!-- contract:concept_set -->
| Column | Arrow type | Nullable | Semantics |
|---|---|---|---|
| `vocabulary_id` | `string` | no | OMOP vocabulary id as written in the CLMBR dictionary (for example `LOINC`). |
| `concept_code` | `string` | no | Concept code within the vocabulary. |
| `code` | `string` | no | `<vocabulary_id>/<concept_code>`: the MEDS code string; unique; rows sorted by it. |
| `entry_types` | `list<item: string>` | no | Sorted, distinct dictionary entry types for the code (`code`, `numeric`, `text`). |
<!-- /contract:concept_set -->

### B'. Coverage report

`omopmap clmbr coverage` measures the share of an extract's distinct
codes that the concept set contains: the kind of measurement behind the
"fewer than 20%" finding above. It reports what it measures on the extract
it is given and reproduces no published figure. The extract and the
concept-set artifact are both verified against their manifests before they
are read. The report lists codes only, never descriptions, at most
`unrecognised_top_n` of them (`--top-n`, default 20), and it never lists a
code seen in fewer than `unrecognised_min_events` events (`--min-events`,
default 10), so rare codes from a small cohort are not exposed. It is
written to `artifacts/coverage/<dataset_name>.json`; an existing report is
never overwritten.

<!-- contract:coverage_report -->
| Field | Type | Meaning |
|---|---|---|
| `contract_version` | `str` | Artifact contract version. |
| `producer` | `Producer` | Producing tool and version. |
| `created_at` | `str` | UTC creation time. |
| `extract_manifest_sha256` | `str` | SHA-256 of the extract's manifest. |
| `concept_list_sha256` | `str` | Concept-list hash of the dictionary used. |
| `distinct_codes` | `int` | Distinct vocabulary codes in the extract (reserved `MEDS_*` codes excluded). |
| `recognised_codes` | `int` | Of those, codes present in the concept set. |
| `share` | `float` | `recognised_codes / distinct_codes` (0 when empty). |
| `by_vocabulary` | `tuple[VocabularyCoverage, ...]` | Per vocabulary id, sorted by `vocabulary_id`. |
| `unrecognised_top_n` | `int` | Maximum length of `unrecognised`. |
| `unrecognised_min_events` | `int` | Codes with fewer events are never listed. |
| `unrecognised` | `tuple[UnrecognisedCode, ...]` | Most frequent unrecognised codes: by `events` descending, then `code`. |
<!-- /contract:coverage_report -->

`by_vocabulary` entries:

<!-- contract:vocabulary_coverage -->
| Field | Type | Meaning |
|---|---|---|
| `vocabulary_id` | `str` | Vocabulary id from the code prefix. |
| `distinct_codes` | `int` | Distinct extract codes in the vocabulary. |
| `recognised_codes` | `int` | Of those, codes in the concept set. |
| `share` | `float` | `recognised_codes / distinct_codes`. |
<!-- /contract:vocabulary_coverage -->

`unrecognised` entries:

<!-- contract:unrecognised_code -->
| Field | Type | Meaning |
|---|---|---|
| `code` | `str` | Code string; no description or other text. |
| `events` | `int` | Rows in the extract carrying the code. |
<!-- /contract:unrecognised_code -->

## C. Label sets

### Layout

```
artifacts/labels/<dataset_name>/<task_name>/
  artifact_manifest.json
  task.json
  labels.parquet
```

- One directory per task; `<task_name>` matches `^[a-z0-9][a-z0-9_]*$`
  and equals `task.json` `name`. `<dataset_name>` names the extract the
  labels refer to, whose manifest hash the label manifest records.
- One `labels.parquet` per task covers all splits. It has no split
  column: a label's split is its subject's split in the extract's
  `subject_splits.parquet`, so labels can never leak across splits.
  Every labelled subject is in that file.
- Rows are sorted by `(subject_id, prediction_time)`, and each pair
  occurs once.
- Exactly one value column is set, the one matching `task.json`
  `label_type`; the other three are null in every row.

### `labels.parquet` (generated)

The MEDS 0.3.3 label schema.

<!-- contract:labels -->
| Column | Arrow type | Nullable | Semantics |
|---|---|---|---|
| `subject_id` | `int64` | no | A subject of the referenced extract, listed in its `subject_splits.parquet`. |
| `prediction_time` | `timestamp[us]` | no | Timezone-naive; the model may use events with `time <= prediction_time`. |
| `boolean_value` | `bool` | yes | Set (non-null) exactly when the task's `label_type` is `boolean`. |
| `integer_value` | `int64` | yes | Set exactly when `label_type` is `integer`. |
| `float_value` | `float64` | yes | Set exactly when `label_type` is `float`. |
| `categorical_value` | `string` | yes | Set exactly when `label_type` is `categorical`. |
<!-- /contract:labels -->

### `task.json` (generated)

<!-- contract:task_definition -->
| Field | Type | Meaning |
|---|---|---|
| `name` | `str` | Task name; equals the label-set directory name. |
| `label_type` | `"boolean" \| "integer" \| "float" \| "categorical"` | Which label value column is set. |
| `description` | `str` | Plain-language definition of the label. |
| `categories` | `tuple[str, ...] \| None` | Allowed `categorical_value` values, sorted; null unless categorical. |
<!-- /contract:task_definition -->

## Artifact manifest

Every artifact directory (A, B's `artifacts/clmbr/`, C) has an
`artifact_manifest.json` at its root. It lists every other file with its
size, SHA-256 and, for Parquet, row count, and it records the inputs the
artifact was derived from and type-specific counts.

<!-- contract:artifact_manifest -->
| Field | Type | Meaning |
|---|---|---|
| `contract_version` | `str` | Artifact contract version the files follow. |
| `artifact_type` | `"meds_extract" \| "label_set" \| "clmbr_concept_set"` | Kind of artifact; equals `summary`'s. |
| `producer` | `Producer` | Producing tool and version. |
| `created_at` | `str` | UTC creation time, `YYYY-MM-DDTHH:MM:SSZ`. |
| `inputs` | `tuple[InputRecord, ...]` | Hashed inputs, sorted by `name`; names unique. |
| `files` | `tuple[FileRecord, ...]` | Every file under the root except this manifest, sorted by `path`. |
| `summary` | `MedsExtractSummary \| LabelSetSummary \| ConceptSetSummary` | Type-specific counts. |
<!-- /contract:artifact_manifest -->

`producer`:

<!-- contract:producer -->
| Field | Type | Meaning |
|---|---|---|
| `name` | `'omopmap'` | Producing package; always `omopmap`. |
| `version` | `str` | Producing package version. |
<!-- /contract:producer -->

`files` entries:

<!-- contract:file_record -->
| Field | Type | Meaning |
|---|---|---|
| `path` | `str` | POSIX path relative to the artifact root. |
| `sha256` | `str` | SHA-256 of the file bytes. |
| `bytes` | `int` | File size in bytes. |
| `rows` | `int \| None` | Row count for Parquet files (from the footer); null for other files. |
<!-- /contract:file_record -->

`inputs` entries:

<!-- contract:input_record -->
| Field | Type | Meaning |
|---|---|---|
| `name` | `str` | Role of the input, for example `omop:person` or `dictionary`. |
| `sha256` | `str` | SHA-256 of the input file, or of its own manifest. |
<!-- /contract:input_record -->

Input names in use: `omop:<table>` for each OMOP CDM Parquet table an
extract was converted from, `synthetic:<seed>` for a generated cohort,
`vocabulary:<file>` for each Athena file read, and `dictionary` for the
CLMBR dictionary file. A label set names its extract through
`summary.extract_manifest_sha256`.

`summary` for a MEDS extract:

<!-- contract:meds_extract_summary -->
| Field | Type | Meaning |
|---|---|---|
| `artifact_type` | `'meds_extract'` | Discriminator. |
| `meds_version` | `'0.3.3'` | MEDS release the data follows. |
| `subjects` | `int` | Distinct subjects in `data/`. |
| `events` | `int` | Rows in `data/`. |
| `distinct_codes` | `int` | Distinct `code` values in `data/`. |
| `subjects_by_split` | `dict["train" \| "tuning" \| "held_out", int]` | Subjects per split; all splits. |
| `events_by_split` | `dict["train" \| "tuning" \| "held_out", int]` | Rows per split; all splits. |
<!-- /contract:meds_extract_summary -->

`summary` for a label set:

<!-- contract:label_set_summary -->
| Field | Type | Meaning |
|---|---|---|
| `artifact_type` | `'label_set'` | Discriminator. |
| `task_name` | `str` | Equals `task.json` `name` and the directory name. |
| `label_type` | `"boolean" \| "integer" \| "float" \| "categorical"` | Equals `task.json` `label_type`. |
| `extract_manifest_sha256` | `str` | SHA-256 of the labelled extract's `artifact_manifest.json`. |
| `labels` | `int` | Rows in `labels.parquet`. |
| `subjects` | `int` | Distinct labelled subjects. |
| `labels_by_split` | `dict["train" \| "tuning" \| "held_out", int]` | Rows per split of the labelled subject; all splits. |
<!-- /contract:label_set_summary -->

`summary` for a concept set:

<!-- contract:concept_set_summary -->
| Field | Type | Meaning |
|---|---|---|
| `artifact_type` | `'clmbr_concept_set'` | Discriminator. |
| `dictionary_sha256` | `str` | SHA-256 of the source dictionary file. |
| `concepts` | `int` | Rows in `concept_set.parquet`. |
| `concept_list_sha256` | `str` | Concept-list hash; equals the CLMBR manifest's `concept_list_sha256`. |
| `vocabulary_counts` | `dict[str, int]` | Concepts per vocabulary id. |
<!-- /contract:concept_set_summary -->

### Verification

```
omopmap artifacts verify artifacts/meds/<dataset_name>
omopmap artifacts verify artifacts/labels/<dataset_name>/<task_name> --extract artifacts/meds/<dataset_name>
omopmap artifacts verify artifacts/clmbr
omopmap clmbr verify --manifest vocab/clmbr/manifest.json
```

`artifacts verify` checks, in order: the manifest parses and its contract
version is readable; the file set equals the manifest's list (no missing
and no unlisted files); every size, SHA-256 and row count matches; the
layout, schemas and content rules above hold; and the recorded `summary`
equals the one computed from the files. It exits 1 with the error type
and reason on the first violation. Without `omopmap` installed, the
per-file hashes can be checked with `sha256sum` against `files[].sha256`.

## Versioning

The contract has a semantic version, recorded in every manifest and
report as `contract_version`.

- **MAJOR** for any change that can make a previously valid artifact
  invalid or change its meaning: renaming, removing or retyping a column
  or field, making a nullable column non-nullable, changing a path,
  the code-string format, the split names, the sort order, or a hash
  definition, or adding a required file.
- **MINOR** for additions that a reader of the new version accepts on
  every older artifact of the same major version, for example a new
  artifact type or a new optional manifest input.
- **PATCH** for corrections that change neither the set of valid
  artifacts nor their meaning (documentation, error messages).

A reader accepts an artifact whose contract version has the same MAJOR
and a MINOR no newer than its own, and refuses anything else. The
consumer pins the contract version it was built against.

## Producing the artifacts

| Command | Produces |
|---|---|
| `omopmap meds synth --out artifacts/meds/<dataset_name> --subjects <n> --seed <s>` | A: a seeded synthetic cohort with splits, from invented codes only |
| `omopmap meds convert --omop <omop_dir> --concept <CONCEPT.csv> --out artifacts/meds/<dataset_name> [--concept-ancestor <CONCEPT_ANCESTOR.csv>] [--source-vocabulary <table>=<vocabulary_id>] [--timezone <zone>] [--seed <s>]` | A: an extract from OMOP CDM tables, with deterministic seeded subject splits |
| `omopmap labels build --extract artifacts/meds/<dataset_name> --task <task_name> --index <code> --outcome <code> --window-days <n> --out artifacts/labels/<dataset_name>/<task_name>` | C: prediction times and labels for one task |
| `omopmap clmbr import (--model-dir <dir> \| --hf-repo <repo> [--revision <sha>]) --out vocab/clmbr [--concept-set artifacts/clmbr] [--athena <CONCEPT.csv>]` | B: KDL files and `manifest.json`; with `--concept-set`, `concept_set.parquet` |
| `omopmap clmbr verify --manifest vocab/clmbr/manifest.json` | B: re-derives counts and hashes from the KDL files and checks them |
| `omopmap clmbr coverage --extract artifacts/meds/<dataset_name> --concept-set artifacts/clmbr --out artifacts/coverage/<dataset_name>.json [--top-n <n>] [--min-events <n>]` | B': the coverage report |
| `omopmap artifacts verify <artifact_dir> [--extract <dir>]` | Verification of A, B and C |

Each producing command validates its output against this contract and
writes the manifest last, so a directory without a manifest is
incomplete and is not consumed.

Every producing command refuses an output directory that exists and is not
empty, so an artifact is never partially overwritten.

### `meds synth`

Each subject is generated from its own generator seeded with the cohort
seed and the subject id, and `created_at` is the cohort's `--as-of` date, so
the same arguments write byte-identical files. Codes are invented
(`DEMOCOND/`, `DEMODRUG/`, `DEMOLAB/`, `DEMOPROC/`, `DEMOOBS/`, and
`DEMOSRC/` for local source codes without a standard concept) or
OMOP-generated (`Gender/`, `Visit/`). Part of each invented family is in a
synthetic dictionary (`omopmap.meds.synth.synthetic_dictionary_codes`), so
dictionary coverage on a synthetic cohort lies strictly between 0 and 1.
The manifest input is `synthetic:<seed>`, hashed over the full generator
configuration.

### `meds convert`

OMOP tables are read from `<omop_dir>` as `<table>.parquet`, `.csv` or
`.tsv`: `person` (required), `visit_occurrence`, `condition_occurrence`,
`drug_exposure`, `procedure_occurrence`, `measurement`, `observation` and
`death`. Each event's code is, first match wins:

1. the standard concept (`<domain>_concept_id`), looked up in `CONCEPT`;
2. the source concept (`<domain>_source_concept_id`), looked up in `CONCEPT`;
3. the source value, as `<vocabulary_id>/<value>`, where the vocabulary id is
   set per table with `--source-vocabulary` (default `<table>_source`).

Concept `0` never becomes a code. Event time is the `*_datetime` column, or
the `*_date` column at midnight; timezone-aware inputs are converted to
`--timezone` (default `UTC`) and made naive. `MEDS_BIRTH` comes from
`birth_datetime`, or from `year_of_birth` with month and day defaulting to 1.
`death` yields `MEDS_DEATH` at the earliest death time plus the cause code
when one resolves. `value_as_number` becomes `numeric_value`;
`value_as_string`, or else the code of `value_as_concept_id`, becomes
`text_value`. Rows without a code or a time, rows of persons absent from
`person` or without a derivable birth, and timed rows before birth are
dropped and counted in the printed conversion report. Subject splits are
`--train` / `--tuning` / `--held-out` fractions (default 0.8 / 0.1 / 0.1)
over a SHA-256 of `<seed>:<subject_id>`, so a subject's split never depends
on the other subjects. `--descriptions` copies concept names into
`codes.parquet`; `--concept-ancestor` fills `parent_codes` with direct
parents. Manifest inputs are `omop:<table>` per table read and
`vocabulary:<file>` per Athena file.

### `labels build`

A binary task: is there an outcome event within `--window-days` after an
index event? `--index` and `--outcome` take code strings or `prefix*`
patterns and repeat. The prediction time is the index event's time; the
window excludes the prediction time and includes its end. A prediction time
without an outcome whose subject's last event precedes the window end is
dropped (censored). By default prediction times at or after the subject's
first outcome are dropped (`--keep-prior-outcome` keeps them);
`--first-index-only` keeps one prediction time per subject, and
`--min-age-years` drops prediction times before that age. The extract is
verified against its manifest before it is read.
