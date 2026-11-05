"""Loading retrieval runs and gold tables for evaluation and reproducibility checks.

A retrieval channel run directory (``retrieve lexical|embed``) holds
``rankings.tsv`` and ``run_manifest.json``. ``load_run`` reads both and fails
closed unless the rankings still hash to what the manifest recorded, and
(when asked) unless the run ranked exactly the query table the caller is
evaluating against. Query identity comes from the row index of that table,
or from an explicit id column, so every per-query result can be traced back
to its row.

Gold tables are user-supplied ``source_term / concept_id / split`` tables.
A query may carry several gold concepts (repeated rows with the same id
column value); the gold set of a query is the union.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from omopmap.artifacts.manifest import sha256_file
from omopmap.meds.convert import read_table
from omopmap.retrieval.fusion import FusionError, ScoredRanking
from omopmap.retrieval.manifest import RUN_MANIFEST_NAME, RunManifest, parse_run_manifest
from omopmap.retrieval.rankings import RANKINGS_NAME, decode_scored

SPLITS = ("train", "dev", "heldout")


class RunError(ValueError):
    """A run directory or gold table cannot be used as asked."""


@dataclass(frozen=True, slots=True)
class LoadedRun:
    """A verified run: its manifest, the rankings sha and one scored ranking per query row."""

    path: Path
    manifest: RunManifest
    rankings_sha256: str
    rankings: tuple[ScoredRanking, ...]

    def by_query_id(self, query_ids: Sequence[tuple[int, str]]) -> dict[str, tuple[int, ...]]:
        """Rankings keyed by query id, for the given ``(row_index, query_id)`` pairs."""
        out: dict[str, tuple[int, ...]] = {}
        for row, query_id in query_ids:
            if not 0 <= row < len(self.rankings):
                raise RunError(f"{self.path}: no ranking for query row {row}")
            out[query_id] = self.rankings[row].ids
        return out


def load_run(path: Path, *, queries_sha256: str | None = None) -> LoadedRun:
    """Read and verify one run directory."""
    manifest_path = path / RUN_MANIFEST_NAME
    rankings_path = path / RANKINGS_NAME
    if not manifest_path.is_file() or not rankings_path.is_file():
        raise RunError(f"{path} lacks {RUN_MANIFEST_NAME} or {RANKINGS_NAME}")
    manifest = parse_run_manifest(manifest_path.read_text(encoding="utf-8"))
    # Fused runs record their channel runs, not the query table, so they
    # cannot be tied to a gold table; compare their channel runs instead.
    if manifest.run_type not in ("lexical", "embed"):
        raise RunError(f"{path} is a {manifest.run_type} run, not a lexical or embed run")
    digest = sha256_file(rankings_path)
    if digest != manifest.output_sha256(RANKINGS_NAME):
        raise RunError(f"{path}/{RANKINGS_NAME} differs from its manifest")
    if queries_sha256 is not None and manifest.inputs_by_name().get("queries") != queries_sha256:
        raise RunError(f"{path} was not computed on the given query table")
    n = manifest.parameters.get("queries")
    if not isinstance(n, int) or isinstance(n, bool):
        raise RunError(f"{path}: the manifest records no integer `queries` parameter")
    try:
        rankings = decode_scored(
            rankings_path.read_bytes(),
            channel=manifest.channel(),
            queries=n,
        )
    except FusionError as exc:
        raise RunError(f"{path}/{RANKINGS_NAME} is malformed: {exc}") from exc
    return LoadedRun(path=path, manifest=manifest, rankings_sha256=digest, rankings=tuple(rankings))


@dataclass(frozen=True, slots=True)
class GoldQueries:
    """Gold concepts per query id, plus the ``(row_index, query_id)`` pairs they came from.

    With repeated ids (several gold rows per query), the first row's index
    is the one whose ranking is read: the runs rank one row per table row,
    and repeated rows of one query hold the same text.
    """

    gold: dict[str, frozenset[int]]
    query_ids: tuple[tuple[int, str], ...]
    texts: dict[str, str]


def gold_frame(path: Path) -> pl.DataFrame:
    """The gold table as strings, with a 0-based ``row_index`` column."""
    return read_table(path).with_row_index("row_index")


def gold_queries(
    path: Path,
    *,
    gold_column: str = "concept_id",
    split: str | None = None,
    id_column: str | None = None,
    text_column: str = "source_term",
) -> GoldQueries:
    """Group a gold table into one gold set per query, optionally for one split."""
    return gold_queries_from_frame(
        gold_frame(path),
        gold_column=gold_column,
        split=split,
        id_column=id_column,
        text_column=text_column,
    )


def gold_queries_from_frame(
    frame: pl.DataFrame,
    *,
    gold_column: str = "concept_id",
    split: str | None = None,
    id_column: str | None = None,
    text_column: str = "source_term",
) -> GoldQueries:
    """``gold_queries`` over an already-read frame that has a ``row_index`` column."""
    needed: set[str] = {gold_column}
    if id_column:
        needed.add(id_column)
    if split is not None:
        needed.add("split")
    missing = sorted(needed - set(frame.columns))
    if missing:
        raise RunError(f"the gold table lacks columns {missing}")
    if split is not None:
        if split not in SPLITS:
            raise RunError(f"--split must be one of {list(SPLITS)}, got {split!r}")
        frame = frame.filter(pl.col("split") == split)
    gold: dict[str, set[int]] = {}
    first_row: dict[str, int] = {}
    texts: dict[str, str] = {}
    has_text = text_column in frame.columns
    for record in frame.iter_rows(named=True):
        row = int(record["row_index"])
        query_id = str(record[id_column]) if id_column else str(row)
        raw = record[gold_column]
        if raw is None or str(raw).strip() == "":
            raise RunError(f"query {query_id!r} has an empty gold concept id")
        try:
            concept_id = int(str(raw))
        except ValueError as exc:
            raise RunError(f"query {query_id!r}: gold id {raw!r} is not an integer") from exc
        gold.setdefault(query_id, set()).add(concept_id)
        first_row.setdefault(query_id, row)
        if has_text:
            texts.setdefault(
                query_id, "" if record[text_column] is None else str(record[text_column])
            )
    if not gold:
        raise RunError("the gold table has no rows for the requested split")
    pairs = tuple(sorted(((first_row[q], q) for q in gold), key=lambda p: p[0]))
    return GoldQueries(
        gold={q: frozenset(ids) for q, ids in gold.items()}, query_ids=pairs, texts=texts
    )
