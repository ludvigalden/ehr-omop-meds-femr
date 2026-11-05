"""Reciprocal rank fusion (RRF) over per-channel rankings.

Lexical containment scores live in [0, 1] and count shared tokens. Cosine
similarities from an encoder live in [-1, 1] and depend on the model, so the
two are not on a common scale. Fusion therefore never sees a raw score: it
accepts only ``Ranking`` values, which carry candidate ids in rank order and
nothing else. A ``ScoredRanking`` from a retriever has to be turned into a
``Ranking`` with ``ScoredRanking.ranking()`` first. That step drops the
scores, so mixing them is a type error rather than a convention.

For a candidate ``c`` the fused score is ``sum over channels of 1 / (k + rank)``,
with ranks starting at 1, over the channels whose first ``depth`` entries
contain ``c``. Channels are truncated to ``depth`` before fusion.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

RRF_K = 60
"""Damping constant; larger values flatten the gap between top ranks."""

RRF_DEPTH = 200
"""Number of entries taken from each channel's ranking."""


class FusionError(ValueError):
    """A ranking or fusion parameter is malformed."""


def _check_unique(ids: tuple[int, ...], what: str) -> None:
    if len(set(ids)) != len(ids):
        raise FusionError(f"{what} repeats a candidate id")


@dataclass(frozen=True, slots=True)
class Ranking:
    """Candidate ids of one channel for one query, best first. No scores."""

    channel: str
    ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.channel:
            raise FusionError("a ranking needs a channel name")
        _check_unique(self.ids, f"ranking {self.channel!r}")


@dataclass(frozen=True, slots=True)
class ScoredRanking:
    """A retriever's output for one query: ids best first, plus their raw scores.

    The scores are on the channel's own scale. They are kept for inspection and
    run records, and they are deliberately not accepted by ``fuse``.
    """

    channel: str
    ids: tuple[int, ...]
    scores: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.ids) != len(self.scores):
            raise FusionError(
                f"{self.channel!r}: {len(self.ids)} ids but {len(self.scores)} scores"
            )
        _check_unique(self.ids, f"ranking {self.channel!r}")
        if any(b > a for a, b in zip(self.scores, self.scores[1:], strict=False)):
            raise FusionError(f"{self.channel!r}: scores are not in descending order")

    def ranking(self) -> Ranking:
        """The rank order alone; the only form fusion accepts."""
        return Ranking(channel=self.channel, ids=self.ids)


@dataclass(frozen=True, slots=True)
class FusedHit:
    """One fused candidate: its RRF score and the 1-based rank it had per channel."""

    concept_id: int
    rrf_score: float
    channel_ranks: tuple[tuple[str, int], ...]


def fuse(
    rankings: Sequence[Ranking],
    *,
    k: int = RRF_K,
    depth: int = RRF_DEPTH,
    top_k: int | None = None,
) -> tuple[FusedHit, ...]:
    """Fuse channel rankings for one query with reciprocal rank fusion.

    The result does not depend on the order of ``rankings``: contributions are
    summed with ``math.fsum``, which is exactly rounded and so order-free, and
    equal RRF scores break on the smaller concept id.
    """
    if k < 1:
        raise FusionError(f"k must be >= 1, got {k}")
    if depth < 1:
        raise FusionError(f"depth must be >= 1, got {depth}")
    if top_k is not None and top_k < 1:
        raise FusionError(f"top_k must be >= 1, got {top_k}")
    for ranking in rankings:
        # Static typing already forbids this; the runtime check keeps untyped
        # callers from passing scored rankings and mixing scales anyway.
        if not isinstance(ranking, Ranking):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError(
                f"fuse accepts Ranking only, got {type(ranking).__name__}; "
                "call .ranking() to drop raw scores"
            )
    channels = [r.channel for r in rankings]
    if len(set(channels)) != len(channels):
        raise FusionError(f"channel names repeat: {sorted(channels)}")

    ranks: dict[int, dict[str, int]] = {}
    for ranking in rankings:
        for position, concept_id in enumerate(ranking.ids[:depth], start=1):
            ranks.setdefault(concept_id, {})[ranking.channel] = position

    hits = [
        FusedHit(
            concept_id=concept_id,
            rrf_score=math.fsum(1.0 / (k + rank) for rank in sorted(per_channel.values())),
            channel_ranks=tuple(sorted(per_channel.items())),
        )
        for concept_id, per_channel in ranks.items()
    ]
    hits.sort(key=lambda h: (-h.rrf_score, h.concept_id))
    return tuple(hits if top_k is None else hits[:top_k])
