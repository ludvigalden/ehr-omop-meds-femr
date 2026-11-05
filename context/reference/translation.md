# Translation (sv->en)

The embedding channel works on concept names in English. Swedish source terms
therefore reach it only after translation. `omopmap translate` is the optional
machine-translation step: it takes a table of source terms, translates each
one Swedish-to-English, and writes a table that keeps the original string per
row — the translation is additional, never a replacement. The step is
optional and sits behind the `embed` extra: the real translator loads its
model on first use, and importing the module never imports torch.

The method comes from the Swedish
vocabulary-mapping experiments in
[ludvigalden/vocabmod-se](https://github.com/ludvigalden/vocabmod-se); those
translation runs executed on rented GPUs at Runpod.

## The translation contract

- **Pinned model.** `google/translategemma-4b-it` at full commit revision
  `10042cb0e6e7fdce748996a71dc3dc432a4e0c89`, carried over from the source
  experiment. `--revision` can override it with another full 40-hex commit;
  anything else is refused.
- **The model's own request format.** The model is loaded through
  `AutoProcessor` and `AutoModelForImageTextToText`, as in the source run.
  Each input becomes one user chat message whose content is a structured
  text part carrying `source_lang_code` `sv` and `target_lang_code` `en`;
  no instruction text is composed around the term. Padding is on the left,
  so every generated continuation starts at the same position and is cut
  off cleanly.
- **Greedy decoding.** `do_sample=False`, `num_beams=1`,
  `max_new_tokens=64`. One input yields the same translation on every run,
  which is what makes the step reproducible.
- **bf16 on GPU, fp32 on CPU.**
- **Smoke check.** Before the full run, the first `--smoke-n` (default 8)
  non-empty inputs are translated and every one of them must come back
  non-empty. On an empty output the step aborts before any output directory
  is created — a model that silently emits nothing must not be able to
  produce a plausible-looking empty result set. This check is this
  implementation's; the source run's smoke step of 8 rows instead projected
  the full translation's wall time against twice its estimate and stopped
  with exit 3. That cost decision belongs to [`omopmap gate`](compute-gates.md).
- **Original preserved.** Every input row reappears, in order, with its
  original string; empty inputs pass through untranslated and never feed the
  smoke gate.

## Output and manifest

`omopmap translate run --queries TERMS.tsv --out DIR` writes:

- `translations.parquet` — `source_term` (original) and `translated_term`,
  one row per query, in input order;
- `run_manifest.json`, written last: `run_type` `"translate"`, the SHA-256 of
  the queries table and of `translations.parquet`, parameters
  (`smoke_n`, `max_new_tokens`, `query_column`, `queries`), and a translator
  record with the model id, revision, device, GPU name, precision,
  `max_new_tokens` and library versions.

Hermetic tests never load a model: a deterministic fake translator stands in
for the protocol, and the smoke-gate failure path is tested through the CLI
exit code.
