# Human adjudication and confidence tiers

Retrieval proposes candidate concepts. Deciding whether a candidate is right
is a human judgement, and every recall figure in
[evaluation-and-statistics.md](evaluation-and-statistics.md) rests on it. This
page covers how that judgement is collected without bias
(`omopmap adjudicate build|ingest`) and how proposed mappings get confidence
tiers whose stated accuracy is honest (`omopmap calibrate fit|apply`).

The protocol follows the Swedish vocabulary-mapping experiments in
[ludvigalden/vocabmod-se](https://github.com/ludvigalden/vocabmod-se), which
ran on rented GPUs at Runpod. It serves the terminological side of
[GMLF](https://urn.kb.se/resolve?urn=urn:nbn:se:uu:diva-590690): mapping
Swedish codes onto the standard vocabularies the model knows.

## Why blind the reviewer

A reviewer who can see that a candidate was ranked first, scored high, or came
from the favoured model is anchored by it. Borderline candidates near the top
get accepted, borderline candidates further down get rejected, and a model's
reputation leaks into its own evaluation. The resulting recall measures the
reviewer's trust in the system, not the quality of the mapping, and it
inflates exactly the comparison the evaluation exists to make (system A
against system B, before against after fine-tuning).

The sheet a reviewer works from therefore carries only what is needed to judge
the mapping on its merits:

| Column | Content |
|---|---|
| `case_id` | `C0001`, `C0002`, ... in case-table order; not the source query id |
| `row` | position within the case, after shuffling |
| `source_term` | the term to map |
| `concept_id`, `concept_name`, `vocabulary_id`, `domain_id` | the candidate |
| `acceptable` | blank, to fill: `y`, `n` or blank |
| `other_targets` | blank, to fill: acceptable concept ids found by searching outside the list, `;`-separated |
| `case_verdict` | blank, to fill: `accept`, `no-target` or `defer` |
| `notes` | free text |

Hidden: rank, score, run name, channel, model id and revision, and the source
query id. Candidates are pooled across all runs and deduplicated per case, so
a concept proposed by two systems appears once and agreement between systems
is hidden too.

## Seeded shuffle

Within a case the pooled candidates are first put in concept-id order, then
shuffled by Python's `random.Random` seeded with the string
`"<seed>|<system>|<code>"`, where `system` and `code` identify the source code
of the case (the case table needs `system` and `code` columns). The default
seed is 20261104, the one the source experiments used
(`coverage/w4_build.py:6,18` in ludvigalden/vocabmod-se).

- **Deterministic.** The order is a pure function of the seed and the case.
  Rebuilding gives byte-identical sheets, so a sheet can be regenerated and
  checked, and two people building from the same runs hand out the same
  sheet. String seeds are hashed with SHA-512 by `random.Random`, which is
  stable across platforms and Python versions.
- **Independent of the runs.** Sorting before shuffling means the order does
  not depend on run order, ranks or scores. Nothing about position leaks
  which system liked a candidate.
- **Per case.** Each case gets its own permutation, so a reviewer cannot learn
  that "the right answer tends to be row 3".

Repeated `(system, code)` pairs are refused, because they would share a
shuffle.

## The sealed key

`build` writes two directories that must differ: the sheet directory, which
reviewers get, and the key directory, which they do not. The key
(`adjudication_key.json`) maps every sheet row back to its query (row, id,
split) and to every run, rank and score that proposed the candidate. It also
records:

- the SHA-256 of the blank sheet, and of its read-only columns;
- the row count;
- the seed, the depth and each run's rankings digest;
- the SHA-256 of the case table the runs ranked.

Its own digest is recorded in the key directory's `run_manifest.json`.

## Ingest

`ingest` re-joins a completed sheet with the key and writes `adjudicated.tsv`,
a frame of record that `omopmap metrics compute` reads directly
(`query_id`, `system`, `code`, `source_term`, `split`, `verdict`, `targets`),
plus `ingest_report.json`.

The frame keeps the case table's row order and source terms, but it is a new
file, and `metrics compute` requires runs that recorded the frame's own
SHA-256. Re-rank it before computing metrics: `retrieve lexical` over it, and
`retrieve embed` against the existing frozen index (only the queries are
encoded). The rankings are the same as over the case table, now tied to the
frame that holds the verdicts.

| Check | Failure | Exit |
|---|---|---|
| the key hashes to its manifest | key edited | 4 |
| header, row count and read-only cells match the build | rows added, removed, re-sorted or edited | 4 |
| every case decided: a `y`, an `other_targets` id, or `case_verdict` `no-target`/`defer` | undecided case | 6 |
| no contradiction: targets with `no-target`/`defer`, or `accept` without targets, or two verdicts | contradictory case | 6 |
| values well-formed (`y`/`n`/blank; integer ids) | malformed entry | 6 |

The read-only check hashes parsed cell values, not file bytes, so a
spreadsheet that re-quotes the file on save still ingests; changing any
read-only value does not.

A case with acceptable targets is `accept`, with the union of the rows marked
`y` and the `other_targets`. The report counts accepted cases whose targets
were all found by search outside the list (`accepted_by_search_only`).

## Calibration notes as methodology

The source's development batch (`pilot/adjudication/CALIBRATION_NOTES.md:3-19`
in ludvigalden/vocabmod-se) shaped three choices here:

- **Three verdicts.** Of 100 development concepts, 52 were accepted, 45 had no
  suitable target and 3 could not be decided. `no-target` and `defer` are
  distinct outcomes, not failures to record; they become the `no-target`
  and `defer` shares in the metrics.
- **Search beyond the list.** Of the 52 accepted mappings, 35 had acceptable
  targets that were not in the presented candidate list; the reviewer found
  them by independent terminology search. Acceptability over the presented
  list is therefore a lower bound on what a reviewer with search access
  achieves. The `other_targets` column exists so those targets are recorded,
  and the ingest report keeps them separate.
- **Verdict rules before the batch.** The notes amend the rubric after the
  development pass and bind it for the held-out waves. A rubric change after
  held-out review has started is a correction, and belongs in the documented
  overrides file of `metrics compute`, not in an edited sheet.

## Confidence tiers

A mapping pipeline commits proposals at different levels of certainty, and a
reviewer's time is best spent where certainty is lowest. `calibrate` assigns
each proposed mapping (a `query_id, concept_id` row of a proposals table,
optionally with a `path` and an `external_concept_id`) a tier from its
retrieval evidence.

### Evidence features

Read from the primary embedding run (and an optional lexical run) over the
frame of record:

| Feature | Definition |
|---|---|
| `rank` | 1-based rank of the proposed concept in the primary run; empty if absent |
| `s1`, `s2`, `margin` | the query's top-1 and top-2 scores, and `s1 − s2` |
| `lexical_rank`, `lexical_agrees` | rank in the lexical run; whether it is the lexical top-1 (cross-channel agreement) |
| `external_agrees` | whether another tool's suggestion for the query (column `external_concept_id`) names the same concept |
| `eligible` | the concept is in the target dictionary (standard, valid) |
| `has_text` | the query has a source term to retrieve with |
| `path` | how the mapping was produced, e.g. `committed` or `truncation` |

Features are written per proposal to `tiers.tsv`. The tier rule uses rank,
`s1`, eligibility, text and path; margin and the agreement features are
recorded for analysis and for richer rules.

### The tier rule

Evaluated top-down, as in the source (`coverage/w3_tier.py:13-29`):

1. **U** (unassessable, with a reason) when the target is not eligible, the
   query has no text, or the primary run returned nothing.
2. **H** when `rank = 1` and `s1 ≥ 0.85`. A proposal whose path is a capped
   path (default `truncation`, a mapping made by truncating a more specific
   source code) is capped at **M**.
3. **M** when `rank ≤ 3`.
4. **L** otherwise, including a proposal absent from the run's list.

### Train-only fitting

Each tier's **score** is the 95% Wilson lower bound of its exact-match rate on
the training split, rounded to two decimals: the rate a reader can rely on
without trusting a point estimate. A proposal is an exact match when its
query was accepted and the concept is one of the acceptable targets; deferred
queries are counted but left out of the denominators.

`calibrate fit` uses training rows only. Rows of any other split in the same
files are ignored, and the library function refuses non-training rows
outright. The threshold `s1 ≥ 0.85` is fixed by default, as in the source
(`coverage/w3_calibrate.py:2-6`: fit on 588 training cases, evaluated on 291
held-out cases, with tier scores equal to the training Wilson lower bounds).
`--learn-s1` instead learns the threshold from training rows: the smallest
`s1` at which rank-1 training proposals reach a Wilson lower bound of
`--target-precision` on at least `--min-n` decided rows. This option is this
implementation's, not the source's. The rule, the scores and the encoder that
produced the scores are frozen in `calibration.json`.

`calibrate apply` tiers dev or held-out rows with the frozen calibration and
reports each tier's observed exact and any-accept rates, with intervals, next
to the frozen scores. It never refits.

The reason is optimism. A threshold chosen on the rows it is judged on fits
their noise, so the tier looks more precise than it will be on new data; a
tier score computed on those same rows inherits the bias. Fitting on training
rows and reporting held-out rates against the frozen scores shows the gap
instead of hiding it.

| Gate | Exit |
|---|---|
| no training proposals; no learnable threshold | 6 |
| `apply --split train` | 6 |
| the applied run comes from a different channel or encoder build | 6 |
| `calibration.json` differs from its fit manifest | 4 |
| unreadable input, or an occupied output directory | 1 |

Scores are on the encoder's own cosine scale, so a calibration is valid only
for the encoder build it was fit on; a fine-tuned checkpoint needs its own
calibration.

## Commands

```sh
omopmap adjudicate build --run lexical=RUN_DIR --run embed=RUN_DIR \
  --cases CASES.tsv --concepts TARGETS --sheet-out SHEET_DIR --key-out KEY_DIR \
  [--seed 20261104] [--depth 10]
omopmap adjudicate ingest --sheet COMPLETED.csv --key KEY_DIR --out OUT_DIR
omopmap calibrate fit --gold FRAME.tsv --proposals PROPOSALS.tsv --run EMBED_RUN \
  [--lexical LEXICAL_RUN] --concepts TARGETS --out FIT_DIR \
  [--s1-high 0.85 | --learn-s1 --target-precision 0.9 --min-n 10]
omopmap calibrate apply --calibration FIT_DIR --gold FRAME.tsv --proposals PROPOSALS.tsv \
  --run EMBED_RUN [--lexical LEXICAL_RUN] --concepts TARGETS --split heldout --out OUT_DIR
```

The runs must have ranked exactly the case table or frame given: their
manifests record its SHA-256, and a mismatch is refused. Every output
directory must be new, and every step writes a `run_manifest.json` with the
digests of its inputs and outputs.
