# Athena vocabularies

This repository never ships OMOP vocabulary content: concept tables and
concept names are licensed text, and every command that reads one expects
you to supply your own download from the OHDSI [Athena](https://athena.ohdsi.org)
portal. This page records what a download is, what may and may not be
redistributed, and how the code consumes it. It is background for
[dictionary-eligibility.md](dictionary-eligibility.md),
[clmbr-concept-set.md](clmbr-concept-set.md) and
[artifacts.md](artifacts.md).

## What Athena is

Athena is OHDSI's vocabulary portal: after a free account registration it
builds a downloadable bundle of the OMOP CDM vocabulary files for the
vocabularies you select. Some vocabularies (SNOMED CT, LOINC and others)
require a UMLS licence, accepted through the Athena profile before the
download is released. The bundle is generated on request and delivered as
a ZIP archive.

The vocabularies relevant to Swedish-to-standard mapping include:

| Vocabulary | Role in the mapping |
|---|---|
| SNOMED | Standard concept target for diagnoses and procedures |
| RxNorm / RxNorm Extension | Standard drug vocabulary; the Extension carries non-US drug mappings (ATC to RxNorm) |
| LOINC | Standard laboratory and measurement vocabulary |
| ICD10 / ICD10CM | Base vocabulary for diagnosis mapping chains; ICD10CM is the US clinical modification the CLMBR model was trained around |
| ATC | WHO drug classification, the Swedish source side for drugs |
| UCUM | Units of measure |
| Gender / Race / Ethnicity | OMOP-generated standard demographic vocabularies |

A bundle is a versioned snapshot: each download carries its release date
in `VOCABULARY.csv`, and two downloads from different dates are different
candidate universes. Record the download date with the data; every command
here that reads a concept table hashes the file into its run manifest, and
`omopmap dictionary build --as-of` pins the validity reference date so a
rerun against the same file stays byte-identical
([dictionary-eligibility.md](dictionary-eligibility.md)).

## What may not be redistributed

The bundle mixes openly reusable content with licensed terminologies. As
a rule of thumb (an assessment, not legal advice):

- **Licensed text stays local.** SNOMED CT concept descriptions fall under
  the SNOMED International affiliate licence outside member territories;
  LOINC carries redistribution-with-notice terms; CPT4 codes are claimed
  under copyright by the AMA. Concept names from these vocabularies are
  licensed strings: uploading them to a third-party compute host, or
  committing them to a repository, is a distribution decision under your
  licence, and this repository's default is that no concept text ships at
  all.
- **Counts and hashes are fine.** Every committed artifact here holds
  counts and SHA-256 hashes only — never codes of licensed vocabularies
  and never names ([clmbr-concept-set.md](clmbr-concept-set.md) applies
  the same rule to the CLMBR dictionary).
- **The CLMBR dictionary is separately gated.** It is a component of a
  credentialed model release, so the same discipline applies there plus a
  file-level pin; see
  [clmbr-concept-set.md](clmbr-concept-set.md).

Keep downloaded bundle files out of the repository. The run manifests
already record each input file's SHA-256, which is what makes a run
reproducible without keeping the input.

## File format

Athena writes the vocabulary tables as header-row text files that are
tab-delimited despite the `.csv` suffix, UTF-8 encoded with Unix line
endings. Fields may be double-quoted, and null values are empty strings.
The expected `CONCEPT` columns are `concept_id`, `concept_name`,
`domain_id`, `vocabulary_id`, `concept_class_id`, `standard_concept`,
`concept_code`, `valid_start_date`, `valid_end_date`, `invalid_reason`,
with dates written as `YYYYMMDD`.

This repository's readers sniff the delimiter and read all columns as
strings, so the tab-delimited form and the `YYYYMMDD` dates work as-is
(the validity comparison accepts both `YYYYMMDD` and ISO dates). The core
files and their sizes, approximately, for a full bundle:

| File | Size | Content |
|---|---|---|
| `CONCEPT.csv` | ~0.8 GB | One row per concept (~8M rows) |
| `CONCEPT_RELATIONSHIP.csv` | ~1 GB | Concept-to-concept relationships, including `Maps to` |
| `CONCEPT_ANCESTOR.csv` | ~0.6 GB | Transitive ancestor closure |
| `CONCEPT_SYNONYM.csv` | ~0.2 GB | Alternative concept names |
| `VOCABULARY.csv`, `DOMAIN.csv`, `CONCEPT_CLASS.csv`, `RELATIONSHIP.csv` | small | Metadata for the id spaces |
| `DRUG_STRENGTH.csv` | ~50 MB | Drug strength facts |

Only `CONCEPT.csv` is required by the commands here;
`CONCEPT_ANCESTOR.csv` is optional for `omopmap meds convert
--concept-ancestor` (direct parents in `codes.parquet`).

## Loading: how the code consumes a download

| Command | Files read | What it does with them |
|---|---|---|
| `omopmap dictionary build --concepts CONCEPT.csv` | `CONCEPT.csv` | Filters to the eligible targets and writes `targets.parquet` plus a run manifest ([dictionary-eligibility.md](dictionary-eligibility.md)) |
| `omopmap retrieve lexical/index --concepts ...` | the built target table, or `CONCEPT.csv` directly | Ranks targets for source terms ([retrieval-methods.md](retrieval-methods.md)) |
| `omopmap meds convert --omop <dir> --concept CONCEPT.csv` | OMOP CDM tables + `CONCEPT.csv` | Resolves event codes and writes the MEDS extract ([artifacts.md](artifacts.md)) |
| `omopmap clmbr import --athena CONCEPT.csv` | `CONCEPT.csv` | Enriches the CLMBR concept set with `concept_id`, `domain_id`, `standard_concept` ([clmbr-concept-set.md](clmbr-concept-set.md)) |

Each command records the input file's SHA-256 in its run manifest before
anything downstream can forget which vocabulary snapshot produced a
result.
