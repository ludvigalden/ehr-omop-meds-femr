# Retrieval methods

This page describes the retrieval core in `omopmap.retrieval`: how a Swedish
clinical source term becomes a ranked list of candidate OMOP standard concepts.
It covers what each component does, why it is built the way it is, and where it
sits in the adaptation-cost framing of
[GMLF](https://urn.kb.se/resolve?urn=urn:nbn:se:uu:diva-590690).

## Where this sits

GMLF names three costs of bringing an international clinical foundation model
into Swedish care: terminological, distributional and implementation cost.
Retrieval belongs to the **terminological** side. Swedish code systems and
their free-text designations have to land on the standard vocabularies the
model was trained on (SNOMED CT, RxNorm, LOINC), and each term that lands on no
standard concept is lost to the model. Retrieval does not decide a mapping. It
proposes a short, ranked candidate list, so that a reviewer or a later
calibration step chooses from a handful of concepts instead of hundreds of
thousands.

The methods follow the Swedish vocabulary-mapping experiments in
[ludvigalden/vocabmod-se](https://github.com/ludvigalden/vocabmod-se). Those
experiments ran on rented GPUs at Runpod, not on the sensitive-data system
(Bianca at UPPMAX) that ran the record-standardisation pipeline. The two
environments are kept apart: mapping experiments work on vocabulary strings,
never on patient records.

## Pipeline

```
concept table ──► lexical index ─────────────┐
      │                                       ├─► RRF fusion ─► fused candidates
      └─► encoder ─► frozen matrix + id map ──┘
                         ▲
source terms ────────────┴─ (encoder, then cosine top-k)
```

| Command | Module | Output |
|---|---|---|
| `omopmap retrieve lexical` | `lexical.py` | `rankings.tsv`, `run_manifest.json` |
| `omopmap retrieve index` | `embed.py`, `retrieve.py` | `matrix.npy`, `ids.txt`, `run_manifest.json` |
| `omopmap retrieve embed` | `embed.py`, `retrieve.py` | `rankings.tsv`, `run_manifest.json` |
| `omopmap retrieve fuse` | `fusion.py` | `rankings.tsv`, `run_manifest.json` |
| `omopmap gate` | `gate.py` | `gate.json` (and `GATE_STOP.json`), `run_manifest.json` ([compute-gates.md](compute-gates.md)) |

By default the candidate set is the standard (`standard_concept = 'S'`), valid
(`invalid_reason` empty) rows of a user-supplied OMOP `CONCEPT` table.
`--all-concepts` lifts that filter. No vocabulary ships with this repository:
you supply your own Athena download (see the licence notes in
[clmbr-concept-set.md](clmbr-concept-set.md)).

Every command refuses a non-empty output directory and writes its run manifest
last. A directory without `run_manifest.json` is therefore an incomplete run.

## Lexical containment (the baseline)

A query matches a label by the share of the query's distinct tokens that occur
in the label:

```
containment(query, label) = |tokens(query) ∩ tokens(label)| / |tokens(query)|
```

- **Normalisation:** Unicode NFKC, then `casefold`, then whitespace collapsed
  to single spaces, then edge punctuation stripped (`.,;:!?()[]{}"'«»`), then
  a whitespace split into a set of distinct tokens.
- **Diacritics are a parameter** (`--diacritics keep|strip`, default `keep`).
  In Swedish, å, ä and ö are separate letters, not accented variants: *får*
  (sheep, gets) and *far* (father) are different words, so folding them by
  default would merge unrelated terms. `strip` exists for sources that have
  already lost their diacritics, such as ASCII-only exports.
- **The denominator is every distinct query token,** including tokens that
  occur in no label. A query that is half unknown words can score at most 0.5.
- **A concept's score is its best label.** A concept can carry several surface
  strings: its OMOP name, plus any extra `concept_id`/`label` rows supplied with
  `--labels`, for example trusted Swedish synonyms. Extra surfaces can only add
  strings to concepts that are already candidates; they never widen the
  candidate set.
- **Ties break on concept id,** ascending.

**No corpus statistics** (no IDF, BM25 or document frequency) are used, on
purpose. With them, a candidate's score would move whenever the candidate set
changed, for example through a new vocabulary release or a different domain
filter, and the channel would stop being a fixed reference point. Containment
depends only on the query and the label, so the lexical channel is the stable
baseline that the learned channels are measured against. In the source
experiments it is also the weak channel on English concept names: on 315
accepted held-out cases, lexical hit@10 was 23/315 (7.3%), against 123/315
(39.1%) for the multilingual SapBERT channel (source: the pilot metrics of
[ludvigalden/vocabmod-se](https://github.com/ludvigalden/vocabmod-se)). That gap is the reason to embed at all, and the baseline is how
the gap is measured.

## Encoders: one protocol, pinned models

`Encoder` is a two-method protocol: `encode(texts)` returns an `(n, dim)`
float32 matrix of L2-normalised rows plus a truncation count, and `info()`
returns the model identity for manifests. Every retrieval step depends only on
this protocol. The three near-identical encoder libraries of the source
experiments collapse into one implementation, `TransformerEncoder`,
parameterised by a `ModelSpec`.

| Key | Model | Revision (full commit) | Pooling |
|---|---|---|---|
| `sapbert` | `cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR` | `47b6bd041ba61311584bb2494edfda5c7d9b719f` | `[CLS]` |
| `biolord` | `FremyCompany/BioLORD-2023-M` | `4ea2ea2c89ef63365f7fcd91406a0cd8ac36d2e2` | attention-mask mean |

- **Why multilingual SapBERT.** The queries are Swedish and the candidate names
  are English. SapBERT's all-languages checkpoint is built on XLM-R and trained
  to place synonymous biomedical names (UMLS synonyms across languages) close
  together. A Swedish term can therefore land near its English concept name
  without a translation step, and without the token overlap the lexical channel
  needs.
- **Pooling belongs to the model.** SapBERT is trained on the `[CLS]` vector.
  BioLORD's model card prescribes attention-mask mean pooling. Using the wrong
  pooling still produces vectors, just worse ones, so pooling is part of each
  pin rather than a global flag.
- **Pinned to a full revision.** A model id names a moving branch, and a 40-hex
  commit names the exact weights. `ModelSpec` rejects anything shorter, and the
  run manifest records the revision, so every vector traces to one checkpoint.
- **Fixed settings:** maximum length 128 tokens, with truncations counted (the
  tokenizer is run once untruncated to count them) and recorded in the manifest
  rather than happening silently. fp16 on a GPU, fp32 on a CPU. Outputs are
  L2-normalised and returned as float32. A non-finite value or a wrong shape
  raises `EncoderError`.
- **Lazy, optional dependency.** torch and transformers come from the `embed`
  extra and are imported only on the first `encode`. The rest of the package,
  and every test, runs without them. The tests use a deterministic hashing
  encoder behind the same protocol.

## Frozen matrix and hash-asserted id map

`retrieve index` embeds the candidate names once and writes two files:

- `matrix.npy`: `(n, dim)` float32, L2-normalised rows;
- `ids.txt`: the concept id of each matrix row, one per line.

Row *i* belongs to line *i*. That pairing is the weak point of any embedding
index: re-sorting, filtering or re-embedding one side without the other gives
results that still look plausible but are wrong, and nothing in the scores
reveals it. So `retrieve embed` opens an index only after both files hash
(SHA-256) to the digests its build manifest recorded. On a mismatch it exits
**4** before reading a vector, and writes nothing. It also refuses an index
built with a different encoder (model id or revision), because cosine scores
between vectors from two models mean nothing. A fine-tuned checkpoint
(`--model-dir`, [fine-tuning.md](fine-tuning.md)) counts as a different
encoder from its base model: the index records the checkpoint's digest, and
`retrieve embed` must load the same checkpoint.

Freezing also pays for itself. Embedding a dictionary of hundreds of thousands
of concept names is the expensive step of a run, and queries are cheap. A
verified frozen matrix lets every later query batch, model comparison and
re-run reuse the same bytes, so a difference between two runs comes from the
queries or the method, never from a silently re-embedded dictionary. The source
experiments asserted a dictionary matrix hash against earlier runs for the same
reason, and stopped with exit 4 when it differed.

Top-k is a cosine (dot-product) ranking, computed in query chunks so the full
queries × labels matrix is never built at once. Ties break on concept id, and
candidates tied at the cut-off are resolved by id rather than by partition
order, so the output is the same on every machine.

## Reciprocal rank fusion

```
rrf(c) = Σ over channels where c is in the top `depth`: 1 / (k + rank_channel(c))
```

with `k = 60`, `depth = 200` (each channel truncated to its first 200 entries),
and ranks starting at 1. Fused output is sorted by RRF score descending, then
concept id. The defaults are the source experiments' settings.

**Raw scores are never mixed,** and the types enforce it. A containment score
in [0, 1] counts shared tokens. A cosine score depends on the model, and
different encoders put their similarities in different ranges. Adding or
averaging them would let whichever channel has the wider spread dominate, with
no meaning behind the weights. Rank order is the only quantity the channels
have in common. Retrievers therefore return a `ScoredRanking` (ids and scores),
and `fuse` accepts only a `Ranking` (ids alone). The conversion,
`ScoredRanking.ranking()`, drops the scores, and passing a `ScoredRanking` to
`fuse` is a type error in strict checking and a `TypeError` at run time. The
`rankings.tsv` files keep their scores for inspection, but the decoder used by
`retrieve fuse` drops them on read.

`retrieve fuse` checks each channel's `rankings.tsv` against its run manifest,
refuses runs computed on different query tables (by query-table hash and query
count), and refuses duplicate channels. The result does not depend on the order
of the `--run` arguments, because contributions are summed with an exactly
rounded `math.fsum`.

Fusion is useful because the channels fail differently. Lexical containment is
precise when a surface string is shared and blind otherwise. The embedding
channel generalises across languages and spellings but can rank a fluent
near-miss first. RRF rewards candidates both channels rank highly, without
asking either channel to be calibrated against the other.

## Run manifests

Every run writes `run_manifest.json`, deterministic JSON with sorted keys:

| Field | Content |
|---|---|
| `run_type` | `lexical`, `index`, `embed`, `fuse` or `gate` (other commands write their own types, e.g. `finetune`, `mcnemar`, `metrics`) |
| `producer` | `omopmap` and its version |
| `created_at` | UTC timestamp (`--created-at` fixes it for byte-identical reruns) |
| `encoder` | model id, full revision, pooling, max length, dim, device, GPU name (null on CPU), precision, library versions, and `checkpoint_sha256` for a fine-tuned checkpoint (null for the base model) |
| `parameters` | the run's parameters, including the query count |
| `inputs` / `outputs` | name and SHA-256 of every input and output file |
| `truncated` | inputs truncated by the encoder, if an encoder ran |
| `seed` | the seed, when a step is seeded |

The `inputs` and `producer` records reuse the artifact-contract types from
`omopmap.artifacts` ([artifacts.md](artifacts.md)), so retrieval runs and
model-ready artifacts describe hashes and versions the same way. With the same
inputs, encoder and `--created-at`, a rerun reproduces `rankings.tsv` and the
manifest byte for byte. The tests assert this.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | success |
| 1 | bad input, a non-empty output directory, a manifest mismatch, an encoder mismatch, or another failed check |
| 3 | the compute gate stopped the job ([compute-gates.md](compute-gates.md)) |
| 4 | a frozen index's `matrix.npy` or `ids.txt`, or a `--model-dir` checkpoint's files, do not match their recorded SHA-256 |
