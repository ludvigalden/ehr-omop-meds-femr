# Dictionary eligibility

This page documents the eligible-target dictionary: the subset of a
user-supplied OMOP vocabulary `CONCEPT` table that source terms may be mapped
to, and the target table (`omopmap dictionary build`) that freezes that
decision for every downstream step. The method and its recorded real-data
counts come from the Swedish vocabulary-mapping experiments in
[ludvigalden/vocabmod-se](https://github.com/ludvigalden/vocabmod-se), which
ran on rented GPUs at Runpod.

## What is eligible, and why

A concept is a mapping target when three conditions hold, each evaluated as a
pure filter over the table:

1. **`standard_concept` is `S`.** OMOP organises its vocabularies so that the
   interoperable meaning of a clinical idea lives in exactly one *standard*
   concept per vocabulary release; everything else is vocabulary-local
   (classification or source-level) content. Mapping a Swedish source term to
   a non-standard concept would give the downstream record two vocab-local
   answers where the standard space expects one interoperable one, and MEDS
   code strings (`<vocabulary_id>/<concept_code>`) and the CLMBR dictionary
   are keyed on the standard space. The standard space is therefore the
   mapping target, and the target space is only as large as it is.
2. **The concept is current: `invalid_reason` is null or empty and
   `valid_end_date` is not in the past.** An invalidated (retired) concept no
   longer resolves in the current vocabulary release; mapping to it would
   point a source term at a code that the vocabulary itself has withdrawn.
   OMOP sets `invalid_reason` when a concept retires, and the current release
   moves its meaning to a successor concept. The end-date clause, an
   addition of this implementation, makes the same judgement explicit at a
   reference date (`--as-of`, default today, so
   pin it for byte-stable reruns) and is inclusive at the boundary: a concept
   valid up to and including the reference date stays eligible. Athena's
   `YYYYMMDD` dates and ISO `YYYY-MM-DD` dates are both read correctly.
3. **The vocabulary and domain are in scope.** Scope is a configurable map
   from vocabulary id to an optional domain restriction (`--vocabulary VOC`
   for all domains, `--vocabulary VOC=DOM1,DOM2` for a restriction). The
   default scope is the source experiment's contract: the clinical
   vocabularies SNOMED, RxNorm, LOINC and UCUM over the clinical metadata
   domains (Condition, Device, Drug, Measurement, Metadata, Observation,
   Procedure, Provider, Spec Anatomic Site, Unit), plus the OpenStreetMap
   vocabulary over Geography (the source run admitted every standard,
   valid OSM concept).

Why a frozen target table at all: the eligible set is the denominator of every
coverage and recall number the mapping pipeline reports. If each step filtered
the raw `CONCEPT` table itself, a scope change would silently change every
previously computed number. Building it once gives the retrieval channels one
shared candidate set (`retrieve index` consumes `targets.parquet` directly —
the table carries no eligibility columns, so it is taken as already filtered),
and the manifest records the scope and counts that produced each run.

## The target table and its manifest

`omopmap dictionary build --concepts CONCEPT.csv --out DIR` applies the
contract above inside a lazy polars plan collected with the streaming engine,
so only eligible rows are ever materialised, never the full export. It then deduplicates on `concept_id` (a duplicate fails closed) and
writes:

- `targets.parquet` — one row per eligible target:
  `concept_id` (int64), `concept_name`, `vocabulary_id`, `domain_id`,
  `concept_code` (all strings), sorted by `concept_id`;
- `run_manifest.json`, written last (a run without it is incomplete):
  `run_type` `"dictionary"`, the producer and timestamp, the SHA-256 of the
  input concept table and of `targets.parquet`, and parameters:
  `as_of`, `scope` (vocabulary-to-domain string), `eligible` (total target
  count), and `vocabulary:<id>` counts.

No concept text ships with this repository: you supply your own Athena
download, and the licence notes in [clmbr-concept-set.md](clmbr-concept-set.md)
apply to it.

## Recorded real-data counts

These are facts about the source experiments, attributed to their reports
(the wave-5 dictionary metadata of
[ludvigalden/vocabmod-se](https://github.com/ludvigalden/vocabmod-se)); they
are not defaults of this code and no real-data run is claimed here:

| Quantity | Value |
|---|---|
| Concept rows streamed from Athena | 6,587,793 (~6.58M) |
| Eligible targets | 811,546 |
| of which SNOMED, RxNorm, LOINC, UCUM in the clinical domains | 608,207 |
| SNOMED | 349,524 |
| OSM (Geography) | 203,339 |
| RxNorm | 158,086 |
| LOINC | 99,556 |
| UCUM | 1,041 |

The eligibility expression recorded by that run: `standard_concept='S' AND
invalid_reason='' AND (vocabulary_id IN {SNOMED, RxNorm, LOINC, UCUM} AND
domain_id IN the clinical metadata domains) OR (vocabulary_id='OSM' AND
domain_id='Geography')`. About one concept row in eight of the Athena
download (811,546 of 6,587,793, 12.3%) survives the contract. The retrieval
channels rank over that space, which is why building it once, with recorded
counts, matters for every downstream number.
