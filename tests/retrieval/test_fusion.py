"""RRF properties: formula, order invariance, depth truncation, ties, raw scores refused."""

from __future__ import annotations

import itertools
import math
from typing import Any, cast

import pytest

from omopmap.retrieval.fusion import (
    RRF_DEPTH,
    RRF_K,
    FusionError,
    Ranking,
    ScoredRanking,
    fuse,
)


def test_defaults_are_k60_depth200() -> None:
    assert (RRF_K, RRF_DEPTH) == (60, 200)


def test_score_is_sum_of_reciprocal_ranks() -> None:
    a = Ranking("a", (10, 20, 30))
    b = Ranking("b", (20, 40))
    hits = {h.concept_id: h for h in fuse([a, b])}
    assert hits[20].rrf_score == pytest.approx(1 / 62 + 1 / 61)
    assert hits[10].rrf_score == pytest.approx(1 / 61)
    assert hits[40].rrf_score == pytest.approx(1 / 62)
    assert hits[20].channel_ranks == (("a", 2), ("b", 1))
    assert [h.concept_id for h in fuse([a, b])] == [20, 10, 40, 30]


def test_order_invariance_over_channel_permutations() -> None:
    rankings = [
        Ranking("lexical", (5, 3, 9, 1, 7)),
        Ranking("embed:sapbert", (3, 5, 2, 8)),
        Ranking("embed:biolord", (9, 2, 5, 4, 6, 3)),
    ]
    reference = fuse(rankings)
    for permutation in itertools.permutations(rankings):
        assert fuse(list(permutation)) == reference


def test_depth_truncates_each_channel_before_fusion() -> None:
    a = Ranking("a", (1, 2, 3, 4))
    b = Ranking("b", (4, 5))
    ids = [h.concept_id for h in fuse([a, b], depth=2)]
    assert sorted(ids) == [1, 2, 4, 5]
    four = next(h for h in fuse([a, b], depth=2) if h.concept_id == 4)
    assert four.channel_ranks == (("b", 1),)
    assert four.rrf_score == pytest.approx(1 / 61)


def test_depth_200_cuts_the_201st_entry() -> None:
    long = Ranking("a", tuple(range(1000, 1300)))
    ids = {h.concept_id for h in fuse([long, Ranking("b", ())])}
    assert len(ids) == 200 and 1199 in ids and 1200 not in ids


def test_ties_break_on_concept_id() -> None:
    a = Ranking("a", (7, 3))
    b = Ranking("b", (3, 7))
    hits = fuse([a, b])
    assert hits[0].rrf_score == hits[1].rrf_score
    assert [h.concept_id for h in hits] == [3, 7]
    assert [h.concept_id for h in fuse([b, a])] == [3, 7]


def test_top_k_and_k_parameter() -> None:
    a = Ranking("a", (1, 2, 3))
    assert [h.concept_id for h in fuse([a], top_k=2)] == [1, 2]
    assert fuse([a], k=1)[0].rrf_score == pytest.approx(0.5)


def test_raw_scores_are_refused() -> None:
    scored = ScoredRanking("a", (1, 2), (0.9, 0.1))
    with pytest.raises(TypeError, match="Ranking only"):
        fuse([cast(Any, scored)])
    assert fuse([scored.ranking()])[0].concept_id == 1


@pytest.mark.parametrize(
    "build",
    [
        lambda: Ranking("a", (1, 1)),
        lambda: Ranking("", (1,)),
        lambda: ScoredRanking("a", (1, 2), (0.1,)),
        lambda: ScoredRanking("a", (1, 2), (0.1, 0.9)),
        lambda: fuse([Ranking("a", (1,)), Ranking("a", (2,))]),
        lambda: fuse([Ranking("a", (1,))], k=0),
        lambda: fuse([Ranking("a", (1,))], depth=0),
    ],
)
def test_malformed_input_fails_closed(build: Any) -> None:
    with pytest.raises(FusionError):
        build()


def test_sum_is_exactly_rounded() -> None:
    hit = fuse([Ranking(f"c{i}", (1,)) for i in range(7)])[0]
    assert hit.rrf_score == math.fsum([1 / 61] * 7)
