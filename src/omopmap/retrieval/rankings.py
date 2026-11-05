"""The ``rankings.tsv`` file passed between retrieval steps, encoded and decoded purely.

Channel runs (lexical, embed) write one row per (query, rank):
``query_index  rank  concept_id  score``. The score is kept for inspection, but
``decode_rankings`` drops it and returns ``Ranking`` values, so a ranking that
passes through a file still reaches fusion as rank order alone. Fused runs write
``query_index  rank  concept_id  rrf_score  channels``, where ``channels`` lists
``<channel>:<rank>`` pairs joined by ``+``.

Query indices are the 0-based row positions in the query table. A query with
no hits has no rows. Floats are written with ``repr``, which round-trips
exactly and gives the same text for the same float on every platform.
"""

from __future__ import annotations

from collections.abc import Sequence

from omopmap.retrieval.fusion import FusedHit, Ranking, ScoredRanking

RANKINGS_NAME = "rankings.tsv"
CHANNEL_HEADER = ("query_index", "rank", "concept_id", "score")
FUSED_HEADER = ("query_index", "rank", "concept_id", "rrf_score", "channels")


class RankingsFormatError(ValueError):
    """A rankings file does not have the expected shape."""


def _render(header: Sequence[str], rows: list[list[str]]) -> bytes:
    lines = ["\t".join(header), *("\t".join(row) for row in rows)]
    return ("\n".join(lines) + "\n").encode("utf-8")


def encode_scored(rankings: Sequence[ScoredRanking]) -> bytes:
    """One channel's rankings, query by query, rank 1 first."""
    rows = [
        [str(q), str(rank), str(cid), repr(score)]
        for q, ranking in enumerate(rankings)
        for rank, (cid, score) in enumerate(zip(ranking.ids, ranking.scores, strict=True), 1)
    ]
    return _render(CHANNEL_HEADER, rows)


def encode_fused(fused: Sequence[Sequence[FusedHit]]) -> bytes:
    """Fused rankings, query by query, rank 1 first."""
    rows = [
        [
            str(q),
            str(rank),
            str(hit.concept_id),
            repr(hit.rrf_score),
            "+".join(f"{channel}:{r}" for channel, r in hit.channel_ranks),
        ]
        for q, hits in enumerate(fused)
        for rank, hit in enumerate(hits, 1)
    ]
    return _render(FUSED_HEADER, rows)


def _rows(data: bytes, header: Sequence[str]) -> list[list[str]]:
    lines = data.decode("utf-8").splitlines()
    if not lines or tuple(lines[0].split("\t")) != tuple(header):
        raise RankingsFormatError(f"expected header {'/'.join(header)}")
    rows = [line.split("\t") for line in lines[1:]]
    if any(len(row) != len(header) for row in rows):
        raise RankingsFormatError("a row has the wrong number of columns")
    return rows


def decode_rankings(
    data: bytes, *, channel: str, queries: int, header: Sequence[str] = CHANNEL_HEADER
) -> list[Ranking]:
    """Rank order per query (scores discarded); ``queries`` sets the list length."""
    ids: list[list[int]] = [[] for _ in range(queries)]
    for row in _rows(data, header):
        try:
            q, rank, cid = int(row[0]), int(row[1]), int(row[2])
        except ValueError as exc:
            raise RankingsFormatError(f"non-integer key column: {exc}") from exc
        if not 0 <= q < queries:
            raise RankingsFormatError(f"query_index {q} is outside 0..{queries - 1}")
        if rank != len(ids[q]) + 1:
            raise RankingsFormatError(f"query {q}: rank {rank} is out of sequence")
        ids[q].append(cid)
    return [Ranking(channel=channel, ids=tuple(row)) for row in ids]


def decode_scored(
    data: bytes, *, channel: str, queries: int, fused: bool = False
) -> list[ScoredRanking]:
    """Rank order with the recorded score per query (channel score or RRF score).

    For inspection and reproducibility checks only; fusion never reads it.
    """
    header = FUSED_HEADER if fused else CHANNEL_HEADER
    ids: list[list[int]] = [[] for _ in range(queries)]
    scores: list[list[float]] = [[] for _ in range(queries)]
    for row in _rows(data, header):
        try:
            q, rank, cid, score = int(row[0]), int(row[1]), int(row[2]), float(row[3])
        except ValueError as exc:
            raise RankingsFormatError(f"non-numeric key or score column: {exc}") from exc
        if not 0 <= q < queries:
            raise RankingsFormatError(f"query_index {q} is outside 0..{queries - 1}")
        if rank != len(ids[q]) + 1:
            raise RankingsFormatError(f"query {q}: rank {rank} is out of sequence")
        ids[q].append(cid)
        scores[q].append(score)
    return [
        ScoredRanking(channel=channel, ids=tuple(i), scores=tuple(s))
        for i, s in zip(ids, scores, strict=True)
    ]
