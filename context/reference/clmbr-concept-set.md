# CLMBR concept set

CLMBR-T-Base is a clinical foundation model trained on American records in
the OMOP CDM. It knows a fixed vocabulary of codes, and an event whose code
is not in that vocabulary carries no meaning for the model. In the research
behind this repository,
[GMLF](https://urn.kb.se/resolve?urn=urn:nbn:se:uu:diva-590690), fewer than
20% of the distinct codes in the transformed Swedish records were recognised
by that vocabulary. This document describes how the repository pins the
vocabulary, derives the concept set from it, and measures how much of an
extract it covers, without publishing the concept list itself.

## The dictionary

FEMR stores the CLMBR vocabulary as `dictionary.msgpack` in the checkpoint
directory. It is a map with three keys:

- `vocab`: the ordered list of vocabulary entries, one per model token;
- `is_hierarchical`: whether entries carry a `property` field (see below);
- `age_stats`: the mean and standard deviation used to normalise age.

Each entry has a `code_string` in the MEDS format
`<vocabulary_id>/<concept_code>`, a `type`, a `weight`, a value bin
(`val_start`, `val_end`), a `text_string` and, when `is_hierarchical` is
true, a `property` on `numeric` and `text` entries. The entry types are:

| Type | Meaning |
|---|---|
| `code` | The code itself is a token. |
| `numeric` | One value bin of a measured code; a code can have several. |
| `text` | One text value of a code. |
| `unused` | A reserved slot with no learned meaning. |

A code can appear under several entry types and several times, so the number
of distinct codes is far smaller than the number of entries. The vocabulary
ids are OMOP `vocabulary_id`s, and the codes are OMOP `concept_code`s, not
`concept_id`s. The release also uses the id `CARE_SITE` for site-local ids
of its training data, which are not standard concepts.

The loader (`omopmap.clmbr.model`) accepts three shapes of the same
dictionary: msgpack with `is_hierarchical` true and `property` on entries,
msgpack with `is_hierarchical` false and no `property` (the shape of the
release's `dictionary.msgpack`), and JSON with the entry list under
`regular`. It rejects unknown keys at every level and unknown entry types,
so a format change fails instead of being read partially. The import itself
reads only the pinned `dictionary.msgpack`.

## From dictionary to concept set

`omopmap.clmbr.importer` builds the concept set:

1. entries of type `unused` are excluded;
2. `CARE_SITE/*` codes are counted and kept out;
3. the remaining codes are deduplicated, each keeping the sorted list of entry
   types it appears under;
4. optionally, an Athena `CONCEPT` table that you supply adds each concept's
   OMOP `concept_id`, `domain_id` and `standard_concept`. A concept the table
   does not contain stays in the set; the import prints how many are
   unresolved. Concept names are never read into the output.

The concept set is identified by `concept_list_sha256`: the SHA-256 of its
distinct code strings, sorted by code point and joined with `\n`. Two imports
with the same hash hold exactly the same concepts.

## Commands

```
omopmap clmbr import --model-dir <checkpoint dir> --out vocab/clmbr \
    [--concept-set artifacts/clmbr] [--athena <CONCEPT.csv>]
omopmap clmbr import --hf-repo StanfordShahLab/clmbr-t-base [--revision <sha>] \
    --out vocab/clmbr [--concept-set artifacts/clmbr]
omopmap clmbr verify --manifest vocab/clmbr/manifest.json
omopmap clmbr coverage --extract artifacts/meds/<dataset_name> \
    --concept-set artifacts/clmbr --out artifacts/coverage/<dataset_name>.json \
    [--top-n 20] [--min-events 10]
```

- **`import`** writes one KDL 2 document per vocabulary
  (`vocab/clmbr/<vocabulary_id>.kdl`, git-ignored) and
  `manifest.json` (committed), then runs `verify` on what it wrote. With
  `--concept-set` it also writes the `concept_set.parquet` artifact with its
  `artifact_manifest.json`. The output directory may hold only a previous
  `manifest.json`, which is replaced; leftover KDL files from an earlier
  import stop it, so two imports are never mixed.
- **`verify`** re-reads the KDL documents, checks that there is exactly one
  per vocabulary in the manifest, that each names the manifest's dictionary
  hash and lists unique, sorted codes, and recomputes the per-vocabulary
  counts, the total and `concept_list_sha256`. Any difference exits 1.
- **`coverage`** verifies the extract and the concept-set artifact against
  their manifests, then reports the share of the extract's distinct codes
  (reserved `MEDS_*` codes excluded) that the concept set contains, overall
  and per vocabulary id, and the most frequent unrecognised codes. It lists
  code strings only, and never a code seen in fewer than `--min-events`
  events (default 10). It measures the extract it is given; it does not
  reproduce the published figure, which came from records that are not part
  of this repository.

Exit codes: 0 on success; 1 when an artifact or contract check fails; 2 when
the pinned dictionary cannot be obtained or does not match its pin.

## Pinned acquisition routes and offline use

`src/omopmap/clmbr/pins.toml` holds one pin that both routes check:

| Field | Value |
|---|---|
| `repo` | `StanfordShahLab/clmbr-t-base` |
| `revision` | `a0f2b36ef3976faf032bfbce23030cb640b5e97f` |
| `filename` | `dictionary.msgpack` |
| `sha256` | `481ddac70a37e79bbfadde1411ad674544a161e3c1948149989cd159a34203ee` |
| `concept_list_sha256` | `71f396167f40096c9ff11dc1b762746e946a7ddaec195f11a813a341bffedbc1` |

- **Hugging Face** (`--hf-repo`): `hf_hub_download` of `filename` at the
  pinned `revision`. The model is gated: accept its access terms on Hugging
  Face (the gated EHRSHOT credentialed licence; see below), then set
  `HF_TOKEN` in the environment. The token is read from the environment
  only, passed to the download, redacted from error messages, and never
  printed, logged or written to any file. Needs the `hf` extra.
- **Local copy** (`--model-dir`): `<dir>/dictionary.msgpack`, a byte-identical
  copy of the same release file. The import performs no network access. This
  is the route for an offline, sensitive-data enclave such as Bianca at
  UPPMAX, where the research project's ETL and model runs took place: copy
  the file in, and the SHA-256 check proves it is the pinned one. The
  dependencies come from the committed `uv.lock`.

Both routes then apply the second, content-level pin: the derived concept
set's `concept_list_sha256` must equal the pinned value, so both routes must
reproduce the same 25,749 concepts. Any difference on either pin, a different
repo or revision, a missing file, or a refused or unreachable download stops
the import with exit 2 before anything is written. There is no fallback from
one route to the other and no unchecked path. Another file of the same
release, for example a re-serialised dictionary with the same concepts, is
refused by the file pin.

The committed manifest records the pinned revision and file hash, and the
concept-list hash lets a result be checked against the same concept set on
another machine without moving the concept list.

## What is committed, and why

Only `vocab/clmbr/manifest.json` is committed: the source, the file
hash, the entry-type counts, the per-vocabulary concept counts, the total,
the number of excluded `CARE_SITE` refs and `concept_list_sha256`. The KDL
documents and `concept_set.parquet` are generated locally and git-ignored.
Tests use invented codes only.

The reasons (an assessment, not legal advice):

- **The release is gated.** Downloading CLMBR-T-Base requires accepting the
  EHRSHOT Credentialed Health Data License, a research-ethics training
  certificate and a non-commercial-use confirmation. Publishing its
  vocabulary as an open list would sidestep that gate.
- **The licence statements disagree.** The model card declares
  CC-BY-NC-4.0, while the `LICENSE` file shipped with the release is
  CC BY-NC-SA 4.0. Either way the use is non-commercial, and the share-alike
  term of the second would bind any redistributed derivative.
- **Some vocabularies are licensed in their own right.** CPT4 codes are
  claimed under copyright by the AMA, SNOMED CT identifiers fall under the
  SNOMED International affiliate licence outside member territories, and
  ICD-O-3 belongs to the WHO. RxNorm, LOINC (with notice), the CMS
  vocabularies (ICD10PCS, HCPCS, ICD9Proc), CVX and the OMOP-generated
  vocabularies carry lower redistribution risk, but they are part of the same
  gated list.
- **Designations never appear.** The dictionary holds no concept names; the
  Athena enrichment reads ids and flags only.

Counts and hashes are facts about the list, not the list, so the manifest can
be public while the list stays with whoever has accepted the release terms.

## Counts of the pinned dictionary

From `vocab/clmbr/manifest.json` for the pinned file:

- 65,536 entries (`is_hierarchical` false): 25,667 `code`, 11,183
  `numeric`, 2,961 `text`, 25,725 `unused`;
- 26,145 distinct codes, of which 396 are `CARE_SITE` ids, leaving
  **25,749 concepts** in 20 vocabularies;
- `concept_list_sha256`
  `71f396167f40096c9ff11dc1b762746e946a7ddaec195f11a813a341bffedbc1`.

| Vocabulary | Concepts |
|---|---:|
| SNOMED | 12,157 |
| RxNorm | 4,678 |
| LOINC | 4,394 |
| CPT4 | 3,647 |
| RxNorm Extension | 255 |
| ICD10PCS | 233 |
| ICD9Proc | 196 |
| HCPCS | 54 |
| ICDO3 | 52 |
| CVX | 41 |
| Cancer Modifier | 19 |
| Visit | 6 |
| Race | 5 |
| OMOP Extension | 3 |
| CMS Place of Service | 2 |
| Ethnicity | 2 |
| Gender | 2 |
| Condition Type | 1 |
| Domain | 1 |
| Medicare Specialty | 1 |

The vocabulary is dominated by SNOMED, RxNorm, LOINC and CPT4, the coding
systems of American care. Swedish source vocabularies (ICD-10-SE, KVÅ, ATC,
NPU) are absent, so a Swedish code reaches the model only through a mapping
to one of these standard vocabularies; `clmbr coverage` measures how far that
mapping got for a given extract.
