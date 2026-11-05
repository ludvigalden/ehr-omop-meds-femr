# Results

Aggregate results of the source vocabulary-mapping experiments in
[ludvigalden/vocabmod-se](https://github.com/ludvigalden/vocabmod-se). This
page holds aggregate figures only: no concept text, no source terms and no
per-query rows. Each figure gives the file and JSON key it was taken from,
relative to that project's experiment workspace.

## What these runs are

- **Inputs.** Real, not synthetic: Swedish source-vocabulary terms as queries,
  and OMOP standard concepts from a pinned Athena vocabulary export as
  candidates. For the
  retrieval comparison, 68,604 queries with text were ranked against 608,207
  candidates (`gates_inputs.run_meta.queries_with_text` in
  `pilot/metrics/out/metrics.json`; the candidate count is the scope-D
  dictionary in [dictionary-eligibility.md](dictionary-eligibility.md)).
- **Reference standard.** A sample of 500 queries (100 development, 400
  held out) was reviewed by human adjudicators, then corrected by documented
  arbitration rulings, the pattern that `--overrides` implements
  ([evaluation-and-statistics.md](evaluation-and-statistics.md)). The figures
  below are the held-out 400 after correction.
- **Compute.** Embedding and retrieval ran on rented GPUs at Runpod: the
  retrieval comparison on one NVIDIA RTX 4000 Ada Generation in fp16
  (`gates_inputs.run_meta.environment.gpu`), with SapBERT pinned at revision
  `47b6bd041ba61311584bb2494edfda5c7d9b719f`
  (`gates_inputs.run_meta.model_revision_sha`).
- **Configurations.** A is lexical token containment, B is multilingual
  SapBERT, and C is reciprocal rank fusion of A and B with `k = 60` and depth
  200 (`gates_inputs.run_meta.params`). The methods are described in
  [retrieval-methods.md](retrieval-methods.md).

All intervals are 95% Wilson score intervals, as stored by the source
(4 decimals). This repository's `omopmap metrics compute` uses the same
interval; recomputed from the counts, it gives every stored bound below, and
its tests pin six of them.

## Frame of record (held-out, n = 400)

Source: `pilot/metrics/out/metrics.json`, `decisions.heldout`.

| Verdict | Count | Share | 95% interval | Key |
|---|---|---|---|---|
| accept | 315 | 78.75% | 74.48–82.48% | `post.accept`, `accept_post_ci` |
| no-target | 79 | 19.75% | 16.14–23.93% | `post.no-target`, `no_target_post_ci` |
| defer | 6 | 1.50% | — | `post.defer` |

These are the counts after the documented corrections (`flips_applied`: 40 on
the held-out set). Before correction, the accept share was 278/400 = 69.50%
(64.82–73.81%, `pre.accept`, `accept_pre_ci`). Across the whole 500-row frame,
the corrected totals are 386 accept, 105 no-target and 9 defer
(`frame_of_record.post_totals`).

The source also stores a weighted accept share of 78.9%
(`post_weighted_pct.accept`), with no interval. It is a different estimator
from the unweighted 315/400 above. The 74.48–82.48% interval belongs to the
unweighted share, not to the weighted 78.9%.

## Retrieval recall (held-out accepted queries, n = 315)

Source: `pilot/metrics/out/metrics.json`, `recall.heldout.configs.<A|B|C>`.
A hit is any acceptable target within the configuration's top `k`.

| Configuration | hit@1 | hit@5 | hit@10 |
|---|---|---|---|
| A lexical | 7 (2.2%; 1.1–4.5) | 21 (6.7%; 4.4–10.0) | 23 (7.3%; 4.9–10.7) |
| B SapBERT | 70 (22.2%; 18.0–27.1) | 110 (34.9%; 29.9–40.3) | 123 (39.1%; 33.8–44.5) |
| C RRF (A + B) | 42 (13.3%; 10.0–17.5) | 102 (32.4%; 27.5–37.7) | 117 (37.1%; 32.0–42.6) |

Paired comparisons at hit@10 (`recall.heldout.pairwise@10`, exact McNemar
on the discordant queries). The source stores a single binomial tail,
without the doubling of a two-sided test; the two-sided p below is computed
from the stored discordant counts the way `omopmap stats mcnemar` computes
it:

| Comparison | Discordant (first only / second only) | Stored p (one tail) | Exact two-sided p |
|---|---|---|---|
| C vs A | 94 / 0 | 0.0 | < 0.0001 |
| C vs B | 3 / 9 | 0.073 | 0.146 |
| B vs A | 101 / 1 | 0.0 | < 0.0001 |

The multilingual encoder finds an acceptable target in the top 10 for 39.1%
of mappable held-out queries; lexical containment finds one for 7.3%. Fusing
the two did not beat the encoder alone on this sample: at hit@10 the
difference is not significant (two-sided p = 0.146), and at hit@1 fusion is
lower (42 against 70 hits).

## Encoder fine-tuning (fresh held-out, n = 251)

Source: `pilot/w6/finetune_v1/artifacts/paired_heldout.json`,
`recall@<k>.{baseline_hits, finetuned_hits, base_only_hit, ft_only_hit,
mcnemar_exact_two_sided_p}`. The fine-tuned BioLORD-2023-M encoder
([fine-tuning.md](fine-tuning.md)) is compared with the untuned one on a
separate, fresh held-out sample.

| Cut-off | Baseline hits | Fine-tuned hits | Baseline only | Fine-tuned only | Exact two-sided p |
|---|---|---|---|---|---|
| recall@1 | 91 | 103 | 8 | 20 | 0.036 |
| recall@5 | 138 | 150 | 3 | 15 | 0.0075 |
| recall@10 | 159 | 162 | 0 | 3 | 0.25 |

`omopmap stats mcnemar` reproduces all three p-values exactly from the
discordant counts; [evaluation-and-statistics.md](evaluation-and-statistics.md)
explains the test.
