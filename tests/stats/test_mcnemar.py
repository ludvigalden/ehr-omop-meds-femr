"""Exact McNemar against hand-computed tables.

Every expected p-value below is computed by hand from the binomial
distribution with p = 1/2 on the discordant pairs, written out as a
fraction of 2**n so the arithmetic can be checked on paper:

- b=1, c=7 (n=8): P(X<=1) = (C(8,0)+C(8,1))/256 = 9/256; two-sided 18/256.
- b=0, c=3 (n=3): P(X<=0) = 1/8; two-sided 2/8 = 0.25.
- b=3, c=15 (n=18): P(X<=3) = (1+18+153+816)/262144 = 988/262144;
  two-sided 1976/262144.
- b=8, c=20 (n=28): P(X<=8) = sum_{i<=8} C(28,i) / 2**28 = 4791323/2**28;
  two-sided 9582646/2**28.
- b=c: the whole distribution is within the observed tail; p = 1.
- b=c=0: no discordant pair; p = 1 by definition.

The last three tables are the source experiments' published held-out
discordant counts, so the published p-values are reproduced exactly.
"""

from __future__ import annotations

from math import comb

import pytest

from omopmap.stats import (
    McNemarResult,
    QueryOutcome,
    StatsError,
    contingency,
    hit_at,
    mcnemar_exact_p,
    paired_mcnemar,
    parse_ks,
)


def _two_sided(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(b, c) + 1))
    return min(1.0, 2 * tail / 2**n)


@pytest.mark.parametrize(
    ("b", "c", "expected"),
    [
        (1, 7, 18 / 256),
        (7, 1, 18 / 256),
        (0, 3, 0.25),
        (3, 15, 1976 / 262144),
        (8, 20, 9582646 / 2**28),
        (5, 5, 1.0),
        (0, 0, 1.0),
        (0, 1, 1.0),
        (0, 10, 2 / 1024),
    ],
)
def test_exact_p_matches_the_hand_computed_binomial(b: int, c: int, expected: float) -> None:
    assert mcnemar_exact_p(b, c) == pytest.approx(expected, rel=1e-12, abs=1e-15)
    assert mcnemar_exact_p(b, c) == pytest.approx(_two_sided(b, c), rel=1e-12, abs=1e-15)


def test_both_tails_are_symmetric() -> None:
    """Which system wins does not change the two-sided p-value."""
    for b, c in [(2, 9), (4, 11), (0, 6), (13, 1)]:
        assert mcnemar_exact_p(b, c) == pytest.approx(mcnemar_exact_p(c, b), rel=1e-12)


def test_published_held_out_p_values_are_reproduced() -> None:
    """The source's paired held-out tables (n=251) and their recorded p-values."""
    assert mcnemar_exact_p(8, 20) == pytest.approx(0.03569813817739487, rel=1e-12)
    assert mcnemar_exact_p(3, 15) == pytest.approx(0.007537841796875, rel=1e-12)
    assert mcnemar_exact_p(0, 3) == pytest.approx(0.25, rel=1e-12)


def test_negative_counts_fail() -> None:
    with pytest.raises(StatsError):
        mcnemar_exact_p(-1, 3)


def _outcomes(a: int, b: int, c: int, d: int) -> list[QueryOutcome]:
    cells = [(True, True)] * a + [(True, False)] * b + [(False, True)] * c + [(False, False)] * d
    return [
        QueryOutcome(query_id=f"q{i:03d}", baseline_hit=base, candidate_hit=cand)
        for i, (base, cand) in enumerate(cells)
    ]


def test_contingency_cells_margins_and_traceable_ids() -> None:
    result = contingency(5, _outcomes(a=10, b=1, c=7, d=2))
    assert (result.both_hit, result.baseline_only, result.candidate_only, result.neither_hit) == (
        10,
        1,
        7,
        2,
    )
    assert result.n == 20
    assert result.baseline_hits == 11 and result.candidate_hits == 17
    assert result.p_value == pytest.approx(18 / 256, rel=1e-12)
    assert result.baseline_only_ids == ("q010",)
    assert result.candidate_only_ids == tuple(f"q{i:03d}" for i in range(11, 18))
    assert result.test == "exact-binomial-mcnemar" and result.alternative == "two-sided"


def test_zero_discordant_pairs_give_p_one() -> None:
    result = contingency(1, _outcomes(a=4, b=0, c=0, d=3))
    assert result.p_value == 1.0
    assert result.baseline_only_ids == () and result.candidate_only_ids == ()


def test_contingency_rejects_duplicate_query_ids() -> None:
    rows = _outcomes(1, 1, 0, 0)
    with pytest.raises(StatsError, match="unique"):
        contingency(1, [rows[0], rows[0]])


def test_result_model_rejects_inconsistent_tables() -> None:
    with pytest.raises(ValueError, match="sum to n"):
        McNemarResult(
            k=1,
            n=5,
            both_hit=1,
            baseline_only=1,
            candidate_only=1,
            neither_hit=1,
            baseline_hits=2,
            candidate_hits=2,
            p_value=1.0,
            baseline_only_ids=("a",),
            candidate_only_ids=("b",),
        )


def test_hit_at_uses_any_gold_concept_within_k() -> None:
    ranking = (5, 9, 2, 7)
    assert hit_at(ranking, frozenset({2, 100}), 3)
    assert not hit_at(ranking, frozenset({2}), 2)
    assert not hit_at((), frozenset({2}), 10)
    with pytest.raises(StatsError):
        hit_at(ranking, frozenset({5}), 0)


def test_paired_mcnemar_per_k_from_rankings() -> None:
    """Five queries, hand-tabulated at k=1 and k=3."""
    gold = {
        "a": frozenset({1}),
        "b": frozenset({2}),
        "c": frozenset({3, 30}),
        "d": frozenset({4}),
        "e": frozenset({5}),
    }
    baseline = {"a": (1, 9), "b": (9, 2), "c": (9, 8, 30), "d": (9,), "e": ()}
    candidate = {"a": (9, 1), "b": (2,), "c": (3,), "d": (8, 7, 6, 4), "e": (5,)}
    result = paired_mcnemar(baseline, candidate, gold, ks=(3, 1))
    assert result.ks == (1, 3)
    at1, at3 = result.results
    # k=1: base hits {a}; cand hits {b, c, e}.
    assert (at1.both_hit, at1.baseline_only, at1.candidate_only, at1.neither_hit) == (0, 1, 3, 1)
    assert at1.baseline_only_ids == ("a",) and at1.candidate_only_ids == ("b", "c", "e")
    assert at1.p_value == pytest.approx(_two_sided(1, 3))
    # k=3: base hits {a, b, c}; cand hits {a, b, c, e}; d is in cand only at rank 4.
    assert (at3.both_hit, at3.baseline_only, at3.candidate_only, at3.neither_hit) == (3, 0, 1, 1)
    assert at3.p_value == 1.0
    assert [o.query_id for o in result.outcomes["1"]] == ["a", "b", "c", "d", "e"]


def test_paired_mcnemar_fails_closed_on_coverage_mismatch() -> None:
    gold = {"a": frozenset({1}), "b": frozenset({2})}
    with pytest.raises(StatsError, match="baseline"):
        paired_mcnemar({"a": (1,)}, {"a": (1,), "b": (2,)}, gold)
    with pytest.raises(StatsError, match="candidate"):
        paired_mcnemar({"a": (1,), "b": (2,)}, {"a": (1,), "b": (2,), "x": ()}, gold)
    with pytest.raises(StatsError, match="cut-offs"):
        paired_mcnemar({"a": (1,), "b": (2,)}, {"a": (1,), "b": (2,)}, gold, ks=(1, 1))
    with pytest.raises(StatsError, match="no gold"):
        paired_mcnemar({}, {}, {})


def test_json_is_deterministic() -> None:
    gold = {"a": frozenset({1}), "b": frozenset({2})}
    runs = ({"a": (1,), "b": (3,)}, {"a": (3,), "b": (2,)})
    first = paired_mcnemar(*runs, gold).to_json()
    assert first == paired_mcnemar(*runs, gold).to_json()
    assert first.endswith("\n")


def test_parse_ks() -> None:
    assert parse_ks("1,5,10") == (1, 5, 10)
    for bad in ("", "0,1", "1,1", "a"):
        with pytest.raises(StatsError):
            parse_ks(bad)
