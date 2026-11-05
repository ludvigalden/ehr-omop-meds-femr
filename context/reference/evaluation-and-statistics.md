# Evaluation and statistics

This page covers how the mapping pipeline's retrieval results are measured
and compared. `omopmap metrics compute` turns ranking runs and an adjudicated
query table into verdict shares and recall@k with confidence intervals, behind
gates that refuse inconsistent input. `omopmap stats mcnemar` decides whether
one retrieval system beats another on the same queries. The aggregate numbers
of the source experiments are in [results.md](results.md).

The methods follow the Swedish vocabulary-mapping experiments in
[ludvigalden/vocabmod-se](https://github.com/ludvigalden/vocabmod-se), which
ran on rented GPUs at Runpod.

## Metrics

### The frame of record

A recall figure is only as meaningful as its denominator, and the denominator
is where mapping evaluations quietly go wrong: drop the queries nobody could
map and recall rises; count them as misses and it falls; correct a reviewer's
verdict in place and nobody can say which version a number came from. So the
denominator is an explicit input, the **frame of record**: the query table the
runs ranked, one row per sampled query, with the outcome of human review.
`omopmap adjudicate ingest` produces it from blinded review sheets; see
[human-adjudication.md](human-adjudication.md).

| Column | Content |
|---|---|
| `query_id` (or `--id-column`) | unique, non-empty id of the query |
| `verdict` | `accept`, `no-target` or `defer` |
| `targets` | for `accept`: the acceptable target concept ids, `;`-separated; empty otherwise |
| `split` | optional; `--split` evaluates one split |
| (any other column) | ignored, e.g. the source term the runs ranked |

- `accept`: at least one concept in the target dictionary is an acceptable
  mapping. Several acceptable targets are allowed; a hit on any one counts.
- `no-target`: reviewed, and nothing in the target dictionary fits. These are
  the unmappable queries.
- `defer`: reviewed, and no decision could be reached.

Every row counts in the verdict shares (`accept`, `no-target`, `defer`, each
with an interval). Only accepted queries enter the recall denominator: a query
with no acceptable target cannot be hit by any retriever, so counting it as a
miss would mix "the retriever failed" with "there is nothing to find". The
unmappable queries are not lost, they are reported as the `no-target` share. A
reader sees both how much of the sample is mappable at all and how well the
retrievers do on the mappable part, and the two cannot be traded against each
other by changing what is counted.

The runs must have ranked this exact table: every lexical or embedding run
records the SHA-256 of its query table, and `metrics compute` requires it to
be the frame's. Query identity is the row position, so a run's ranking of row
*i* is matched to the frame's row *i* with no join on text.

### Documented overrides

Review results get corrected after the fact: an arbitration ruling changes a
verdict, a re-review replaces a target, a query turns out to be a duplicate.
Editing the frame in place destroys the record of what changed and why. So
corrections live in a separate TOML file (`--overrides`), and the frame stays
as adjudicated:

```toml
[[override]]
query_id = "q17"
from = "no-target"          # the verdict the frame must have now
verdict = "accept"
targets = [4100021]
reason = "arbitration: the dictionary concept fits"

[[override]]
query_id = "q31"
from = "defer"
exclude = true              # not counted anywhere; still listed
reason = "duplicate of q12 in the sample"

[expect.heldout]            # optional: totals after correction
accept = 315
"no-target" = 79
defer = 6
```

The rules exist so that an override can never apply silently or wrongly:

- **Every override carries a non-empty `reason`.** A number that moved must
  say why.
- **`from` asserts the current verdict.** An override written against one
  version of the frame cannot be applied to another without failing.
- **Each query has at most one override, the query must exist, and the
  override must change something.** An override that changes nothing is a
  sign the frame already moved underneath it.
- **Exactly one action**: a new `verdict` (with `targets` for `accept`, none
  otherwise), or `exclude = true`. Excluded queries are listed in the output
  but counted nowhere, so removing a query from the sample is a documented
  decision, not a deleted row.
- **All of them are applied and counted.** `metrics.json` reports the number
  applied, the count per kind (`verdict`, `targets`, `exclude`) and each
  override with its reason; the verdict totals before and after correction;
  and `frame_of_record.tsv` marks each corrected row. The overrides file's
  SHA-256 is in the run manifest.
- **`[expect.<split>]` (or `[expect.all]`) totals are asserted** after
  correction, so a reviewed set of corrections reproduces the totals it was
  reviewed against or the run stops. The source experiments asserted their
  per-batch totals before and after correction the same way.

### Recall@k and its interval

A query is a hit at `k` when any of its acceptable targets is among the run's
first `k` candidates. Recall@k is hits over accepted queries. Each run also
reports `no_retrieval`, the accepted queries it returned nothing for.

Every proportion (recall@k, and each verdict share) carries a 95% **Wilson
score interval** without continuity correction, the interval the source
experiments used. With `p = x/n` and `z = 1.959963984540054`:

```
centre     = (p + z²/(2n)) / (1 + z²/n)
half-width = z · √( p(1−p)/n + z²/(4n²) ) / (1 + z²/n)
interval   = centre ± half-width
```

Wilson rather than the normal ("Wald") interval because mapping recall is
often near 0 or 1 and on a few hundred queries. There the Wald interval is too
narrow and can leave [0, 1], and at `p = 0` it collapses to a single point,
while the Wilson interval keeps sensible coverage. The tests check the
implementation against the formula with an independently derived `z`,
against a hand-worked case, and against the intervals stored by the source
experiments.

The intervals are per run. They describe the uncertainty of one system's
recall; whether two systems differ on the same queries is the paired question
answered by `stats mcnemar` below, not by comparing two intervals.

### Gates

`metrics compute` checks everything before writing anything. A failed gate
names its error class and exits **6**; the output directory is not created.

| Gate | Error | Why |
|---|---|---|
| a query id is empty or repeats | `DuplicateQueryIdError` | a repeated id would count one query twice and make overrides ambiguous |
| unknown verdict; `accept` without targets; targets on another verdict | `VerdictError` | an accepted query with no target can never be hit and would silently deflate recall |
| an override is malformed, has no reason, names an unknown query, expects another verdict, repeats, or changes nothing | `OverrideError` | corrections must apply exactly as reviewed |
| post-correction totals differ from `[expect.*]` | `ExpectedTotalsError` | the corrections in force are not the reviewed ones |
| an accepted target is not in the target dictionary (`--concepts`) | `GoldNotInDictionaryError` | no run could return it; the query would be a guaranteed miss that looks like a retrieval failure. Fix the gold, or exclude the query with a reason |
| a run did not rank this frame table (query-table hash or count differs); a fused run is given without all its channel runs | `RunCoverageError` | run and gold would refer to different queries |
| `k` is larger than the depth a run kept | `DepthExceededError` | recall@k would be capped by truncation, not by the retriever |
| a score is NaN or infinite | `NonFiniteScoreError` | a NaN breaks the ranking order without any visible error |
| a concept is ranked twice for one query | `DuplicateCandidateError` | the run is corrupt; ranks after the repeat are shifted |
| ranks out of sequence, or scores not descending | `RankingOrderError` | the stored order is not the order the scores imply |
| no accepted query is counted | `EmptyDenominatorError` | recall is undefined; reporting 0 would be a false number |

A fused run records the digests of its channel runs, not the query table, so
it is accepted only alongside every one of its channel runs, which are checked
against the frame.

### Exit codes

| Code | Meaning |
|---|---|
| 0 | success: `metrics.json`, `frame_of_record.tsv`, then `run_manifest.json` written |
| 1 | unreadable input (missing file or column, bad `--k`), or a non-empty output directory |
| 4 | a run's `rankings.tsv` no longer hashes to its run manifest |
| 6 | an evaluation gate failed (table above) |

Exit 4 matches its use elsewhere: bytes that differ from their recorded
digest. Gates get their own code so a pipeline can tell "the evaluation inputs
are inconsistent" from "a file is unreadable".

### `omopmap metrics compute`

```
omopmap metrics compute \
    --run lexical=runs/lex --run sapbert=runs/emb --run rrf=runs/fused \
    --gold frame.tsv --concepts targets/targets.parquet \
    --overrides overrides.toml --split heldout --k 1,5,10 --out metrics/heldout
```

- `--run NAME=DIR` (or `DIR`, named after its channel): lexical, embedding or
  fused runs ([retrieval-methods.md](retrieval-methods.md)); repeat for each.
- `--concepts`: the target dictionary the runs ranked from; standard, valid
  concepts are taken from a raw CONCEPT table, a built target table is used
  as-is.
- Output: `metrics.json` (deterministic JSON: the interval method, the frame
  summary with every share and its interval, before/after totals per split,
  the overrides with reasons, and per run its type, channel, rankings
  SHA-256, depth, `no_retrieval` and recall at each `k`); `frame_of_record.tsv`
  (per query: verdict before and after, targets, override kind, exclusion,
  and the first rank of an acceptable target in each run); and
  `run_manifest.json`, written last, with the SHA-256 of the frame, the
  dictionary, the overrides and each run. With `--created-at`, reruns are
  byte-identical.

## Statistics

### Hits and recall@k

A query is a **hit at k** for a system when at least one of the query's gold
concepts appears among the system's first `k` ranked candidates. Recall@k is
the share of queries that are hits. A query may have several gold concepts
(repeated rows in the gold table); any one of them counts.

### Why a paired test

Two systems that ranked the **same** queries are not two independent samples:
most queries are easy for both or hard for both. Comparing their recall@k as
if the samples were independent wastes that pairing and overstates the
uncertainty. McNemar's test uses it. Per query and cut-off, the two outcomes
form a 2×2 table:

|  | candidate hit | candidate miss |
|---|---|---|
| **baseline hit** | `a` (both) | `b` (baseline only) |
| **baseline miss** | `c` (candidate only) | `d` (neither) |

The concordant cells `a` and `d` carry no information about which system is
better. Only the `n = b + c` discordant queries do.

### The exact test

Under the null hypothesis that both systems have the same hit rate, each
discordant query is equally likely to favour either system, so
`c ~ Binomial(n, 1/2)`. The p-value is the exact two-sided binomial test of
`c` against 1/2 (`scipy.stats.binomtest`). For this symmetric null that equals
twice the probability of the smaller tail, capped at 1:

```
p = min(1, 2 · Σ_{i ≤ min(b, c)} C(n, i) / 2^n)
```

- No continuity correction and no mid-p: the exact test needs neither, and
  either would change the reported number.
- Exact rather than the chi-squared approximation: discordant counts in
  mapping evaluations are small (3 in one of the results below), where the
  approximation is poor.
- `b = c = 0` (no discordant query) gives `p = 1`: the systems cannot be told
  apart on this data.
- The test is two-sided: it asks whether the systems differ, not whether a
  chosen one is better. The direction is read from `b` and `c`.

### Traceability

The table is built from per-query outcomes, and every query keeps its id: the
gold table's row index by default, or a column named by `--id-column`. The
output lists, per cut-off, the ids of the baseline-only and candidate-only
queries, and all per-query outcomes. So every discordant cell can be traced
to the exact queries behind it and reviewed.

### `omopmap stats mcnemar`

```
omopmap stats mcnemar --baseline runs/base --candidate runs/tuned \
    --gold queries.tsv --split heldout --k 1,5,10 --out stats/heldout
```

- `--baseline` and `--candidate` are lexical or embedding run directories
  ([retrieval-methods.md](retrieval-methods.md)). Each run's rankings must
  still hash to what its manifest recorded, and each manifest must record the
  SHA-256 of the `--gold` table as its query table: both runs ranked exactly
  these rows, or the command exits 1.
- `--gold` is the query table with a gold concept column (`--gold-column`,
  default `concept_id`); `--split` restricts the comparison to one split.
- Output: `mcnemar.json` (per cut-off: `n`, the four cells, both hit totals,
  `p_value`, the test name, the alternative, the discordant query ids; plus
  all per-query outcomes) and `run_manifest.json` with the SHA-256 of both
  rankings files and the gold table.

### Fine-tune result

The fine-tuned BioLORD encoder ([fine-tuning.md](fine-tuning.md)) against
the untuned BioLORD-2023-M baseline, on the fresh held-out split of the
source experiments (n = 251 queries; fp32 CPU re-embedding of both encoders;
a hit is any acceptable target concept within the top `k` of the candidate
pool). Recorded in the paired held-out results of
[ludvigalden/vocabmod-se](https://github.com/ludvigalden/vocabmod-se)
(wave 6, fine-tune v1):

| Cut-off | Baseline hits | Fine-tuned hits | b (baseline only) | c (fine-tuned only) | Exact two-sided p |
|---|---|---|---|---|---|
| recall@1 | 91 | 103 | 8 | 20 | 0.036 |
| recall@5 | 138 | 150 | 3 | 15 | 0.0075 |
| recall@10 | 159 | 162 | 0 | 3 | 0.25 |

Fine-tuning moved the right concept to the top: it gained 12 held-out hits at
recall@1 and at recall@5, both significant. At recall@10 it gained 3 and lost
none, which is not significant at n = 3 discordant queries: the base encoder
already had most reachable gold concepts in its top 10, and fine-tuning
mainly reordered them. The test suite reproduces all three p-values exactly
from the discordant counts.
