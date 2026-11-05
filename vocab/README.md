# Vocabulary KDL tree

Generated concept documents in [KDL 2](https://kdl.dev). `module.kdl` is the
tree root; `schema.kdl` describes the node conventions every document follows.

## `clmbr/`

The concept set of the pinned CLMBR-T-Base dictionary, one document per
vocabulary (`<vocabulary_id>.kdl`), written by

```
omopmap clmbr import --model-dir <checkpoint dir> --out vocab/clmbr
omopmap clmbr verify --manifest vocab/clmbr/manifest.json
```

Only `clmbr/manifest.json` is committed: the dictionary's SHA-256, entry-type
counts, per-vocabulary concept counts and the hash of the sorted concept list.
The generated `*.kdl` files are git-ignored, because the concept list belongs to
a gated model release and some of its vocabularies are licensed. `verify`
re-derives every count and the hash from the local documents and fails on any
difference. See `context/reference/clmbr-concept-set.md`.
