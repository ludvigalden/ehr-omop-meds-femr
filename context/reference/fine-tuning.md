# Fine-tuning the retrieval encoder

`omopmap finetune` adapts the embedding channel's encoder to the mapping
task: it pulls each Swedish source term towards the name of its gold OMOP
standard concept and away from the concepts the untuned encoder confuses it
with. This page covers the training setup and why it is built the way it is:
the objective, the hard negatives, dev-based early stopping, how the training
set is built, and the guards that keep evaluation data out of training.

The method and its parameters come from the Swedish vocabulary-mapping
experiments in [ludvigalden/vocabmod-se](https://github.com/ludvigalden/vocabmod-se),
which ran on rented GPUs at Runpod. Their result is in
[evaluation-and-statistics.md](evaluation-and-statistics.md#fine-tune-result).

## Where it sits

The embedding channel ([retrieval-methods.md](retrieval-methods.md)) ranks
concepts by cosine similarity between a source term and each concept name.
A general biomedical encoder places a Swedish term near related English
concept names, but often ranks a sibling or a broader concept above the right
one. Fine-tuning targets exactly that: the near misses. It is optional. The
retrieval pipeline runs unchanged with or without it, and the paired test in
[evaluation-and-statistics.md](evaluation-and-statistics.md) decides whether
a fine-tuned encoder is better.

## Objective: multiple-negatives ranking loss

Each training example is an anchor (the Swedish source term), a positive (the
name of its gold concept) and up to 8 hard negatives (names of concepts that
are not gold). For a batch of `B` anchors, every anchor is scored against all
`B` positives and all hard negatives in the batch. The score is cosine
similarity multiplied by a scale of 20, and the loss is the cross-entropy of
the anchor's own positive against all the others (MNRL). The scale sharpens
the softmax: cosines live in [-1, 1], and without it the loss cannot
separate a near miss from the gold concept strongly enough.

The positives of the other anchors in the batch serve as extra negatives at no
cost. That is only correct if none of them is also a true match, so the
harness enforces two rules:

- a batch never holds two anchors whose positive is the same concept (an
  example that would repeat a positive waits for the next batch);
- a candidate that is a gold concept of the anchor (a source term may have
  several gold concepts) is masked out of that anchor's softmax, unless it is
  the anchor's own positive.

## Hard negatives, and why

In-batch negatives are mostly easy: a random other concept's name is far from
the anchor already, and the loss learns little from it. The informative
negatives are the concepts the retriever ranks highly for the anchor but that
are wrong. They are mined from a retrieval run of the [B3 retrieval
core](retrieval-methods.md):

- by default, from the untuned encoder itself (the dictionary is embedded
  into a frozen, hash-checked index and each term is ranked against it);
- or from any lexical or embedding run over the same pair table
  (`--negatives RUN`), whose manifest must record the pair table's SHA-256.
  A lexical run lists only concepts that share a token with the term, so
  it can leave an anchor with no negative; the harness then stops with exit
  1 rather than train without them.

For each anchor the first 8 ranked concepts that are eligible targets and are
**not** a gold concept of the term (from any of its rows) are kept, in rank
order. Every anchor gets the same number of negatives: the smaller of the cap
and the fewest any anchor has, so the batch tensor has a fixed width and no
anchor is trained against more competition than another.

## Training contract

| Parameter | Value | Note |
|---|---|---|
| Base encoder | `FremyCompany/BioLORD-2023-M` @ `4ea2ea2c89ef63365f7fcd91406a0cd8ac36d2e2`, mean pooling | `--model sapbert` selects the SapBERT pin with CLS pooling |
| Loss | MNRL, cosine similarity, scale 20 | |
| Hard negatives | at most 8 per anchor | `--hard-negatives`, capped at 8 |
| Batch size | 16 | |
| Learning rate | 2e-5, AdamW, linear decay after 10% warmup | |
| Seed | 20261106 | `--seed`; fixes torch init and the per-epoch batch order |
| Epochs | at most 4 | `--epochs` |
| Monitor | dev recall@10, after every epoch | |
| Early stopping | patience 2, strict improvement | |
| Kept weights | the best dev epoch, not the last | epoch 0 (untuned) competes |
| Max input length | 128 tokens | |
| Precision | fp16 autocast with gradient scaling on CUDA, fp32 on CPU | |

## Early stopping on dev, not on held-out

The held-out split exists to measure the final encoder once. If it also chose
when to stop, the reported held-out number would be the best of several
looks at it, biased upward. So a separate dev split drives the stopping
decision, and held-out is never read. After every epoch the harness ranks the
dev queries against a fixed dev pool (the dev gold concepts plus their mined
hard negatives) and computes recall@10. The untuned encoder is evaluated first
as epoch 0. When recall has not strictly improved for 2 consecutive epochs,
training stops, and the weights of the best epoch are restored before
anything is saved. A fine-tune that never beats the untuned encoder on dev
therefore saves the untuned weights.

The source experiment's dev split was small (5 queries), which makes this
monitor noisy; the full per-epoch history is kept in `finetune.json` so a
reader can see how much the stopping decision rested on.

## How the dataset is built

Input is a user-supplied pair table (`--pairs`) with the columns
`source_term`, `concept_id` (the gold concept) and `split` (`train`, `dev` or
`heldout`); repeated rows give a term several gold concepts. The concept names
come from a target table or OMOP `CONCEPT` table (`--concepts`), filtered to
standard, valid concepts unless it is a built target table
([dictionary-eligibility.md](dictionary-eligibility.md)).

- Train rows become anchors. Dev rows become monitor queries. Held-out rows
  are dropped when the table is read; only their count is recorded.
- A pair whose gold concept is not an eligible target is dropped and counted
  (`gold_not_eligible`): there is no positive text to train on.
- Anchor text is the source term as written. Matching between rows (for gold
  sets and the overlap check) uses the retrieval normalisation: NFKC,
  casefold, whitespace collapse, edge punctuation stripped.

No pair table ships with this repository. The committed test fixture is a
60-row table of invented terms.

## Leakage guards

Each guard fails closed with exit 1:

| Guard | What it prevents |
|---|---|
| Held-out rows dropped at read time | held-out text reaching mining, training or monitoring |
| A normalised term in both train and dev is a `LeakageError` | dev monitoring anchors the model was trained on |
| Negatives exclude every gold concept of the term | training a true match as a negative |
| Distinct positives per batch, gold masking | the same, through in-batch negatives |
| `--negatives` run must have ranked the same pair table (SHA-256) | negatives mined for other rows than the anchors |

## Dry-run and full mode

`--dry-run` runs the whole harness without torch and without a download. The
dataset, the guards, the mining, the batching, the monitor, early stopping and
the manifest are the same code as in full mode; only the trainer differs. The
dry-run trainer is a small real encoder in pure numpy: hashed character
trigrams of the normalised text, multiplied by a learned 512×32 matrix and
L2-normalised. It optimises the same masked MNRL objective with analytic
gradients (checked against finite differences in the tests) and plain SGD at
its own step size, because a learning rate of 2e-5 suits a pretrained
transformer under Adam, not a randomly initialised linear map. On the
committed 60-row fixture it improves dev recall over its untuned start, so the
dry-run exercises a real early-stopping decision, not a stub.

Full mode loads the pinned encoder through the `embed` extra and trains it in
plain torch (`TorchTrainer`); the tests run it on a tiny, randomly initialised
BERT built offline.

```
omopmap finetune --dry-run --pairs pairs.tsv --concepts targets/targets.parquet --out ft-dry
omopmap finetune --pairs pairs.tsv --concepts targets/targets.parquet --out ft --negatives runs/embed
```

## Outputs

The output directory must be new or empty. It receives:

- `finetune.json`: the mode, the trainer, the full contract, the effective
  hard-negative count, where the negatives came from, the dataset counts
  (`train_rows`, `dev_rows`, `heldout_rows_excluded`, `gold_not_eligible`,
  `train_examples`, `dev_queries`, `dev_pool`) and every epoch's mean train
  loss and dev recall, the best epoch, and whether training stopped early;
- `checkpoint/` (full mode only): the best-dev model and tokenizer in Hugging
  Face format;
- `run_manifest.json`, written last: the seed, the encoder record (model id,
  revision, device, GPU name, precision, library versions) in full mode, the
  contract as parameters, the SHA-256 of the pair table, the concept table and
  the negatives run, and the SHA-256 of every output file including each
  checkpoint file.

## Using the fine-tuned encoder

`retrieve index --model-dir <finetune out>` and `retrieve embed --model-dir
<finetune out>` load the saved checkpoint instead of a hub model:

```
omopmap retrieve index --concepts targets/targets.parquet --model-dir ft/ --out idx-ft
omopmap retrieve embed --index idx-ft --queries terms.tsv --model-dir ft/ --out run-ft
omopmap stats mcnemar --baseline run-base --candidate run-ft --gold terms.tsv \
    --split heldout --out stats/ft-vs-base
```

- **Verified before loading.** The directory's `run_manifest.json` must be a
  full-mode `finetune` record. The files under `checkpoint/` must be exactly
  the recorded ones, none missing and none added, and each must hash to its
  recorded SHA-256. Any difference exits **4** before a weight is read, the
  same rule as a frozen index whose matrix changed; a dry-run directory or a
  missing manifest exits 1.
- **Identity.** A fine-tuned checkpoint shares its base model's id and
  revision, so those cannot tell it apart from the base or from another
  fine-tune. Its identity is `checkpoint_sha256`, the SHA-256 of the sorted
  `<path>\t<sha256>` listing of its recorded files. Run manifests record it in
  the encoder record and as the `checkpoint` input, with the model label
  `<base>+ft:<first 12 hex>`. `retrieve embed` refuses an index built by a
  different encoder, base or fine-tuned (exit 1).
- **Same settings as the base.** The checkpoint is loaded with its base
  model's pooling (mean for BioLORD, CLS for SapBERT) and the 128-token limit.
  `--model` may be given only if it names that base model.
- **Offline.** Weights and tokenizer are read from the directory with
  `local_files_only`, so this route never contacts the hub, which is what an
  enclave without outbound network needs. The tests run it end to end on a
  tiny randomly initialised BERT with `HF_HUB_OFFLINE=1`.

`omopmap stats mcnemar` then compares the fine-tuned run with the base
encoder's run on held-out queries. See [reproducibility.md](reproducibility.md)
for how runs are pinned and checked.
