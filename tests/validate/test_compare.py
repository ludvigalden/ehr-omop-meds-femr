"""Pure GPU-vs-CPU comparison against hand-computed expectations.

Every expectation below is derived on paper from the definitions in
``omopmap.validate_gpu_cpu``: overlap@k as set containment shares,
first divergence as the first differing 1-based rank, rank shift as the
largest displacement of a shared candidate, and ``tie_explained`` as
"same top-k set, every reordered pair within ``tie_eps`` in the first
run". The tolerances are the defaults (2e-3).
"""

from __future__ import annotations

import json
from collections.abc import Sequence

import pytest

from omopmap.retrieval.fusion import ScoredRanking
from omopmap.validate_gpu_cpu import (
    RunComparison,
    ValidationError,
    compare_runs,
    first_divergence,
    overlap_at,
)

KS = (1, 5, 10)


def _ranking(ids: Sequence[int], scores: Sequence[float] | None = None) -> ScoredRanking:
    """A scored ranking with descending defaults, one step of 0.1 apart."""
    if scores is None:
        scores = [1.0 - 0.1 * i for i in range(len(ids))]
    return ScoredRanking(channel="test", ids=tuple(ids), scores=tuple(scores))


def test_identical_runs_pass_with_full_agreement() -> None:
    first = [_ranking(range(1, 11))]
    result = compare_runs(first, [_ranking(range(1, 11))], KS)
    assert result.passed
    assert result.identical_top_k == result.queries == 1
    assert result.tie_explained == 1 and result.diverged_queries == ()
    row = result.per_query[0]
    assert row.first_divergence is None
    assert row.max_rank_shift == 0
    assert row.max_score_delta == 0.0
    assert row.overlap == {"1": 1.0, "5": 1.0, "10": 1.0}
    assert row.tie_explained


def test_near_tie_swap_is_explained_and_passes() -> None:
    """Ranks 1 and 2 swap; their first-run gap is 1e-4, far below tie_eps = 2e-3."""
    first = [_ranking((1, 2, 3), (0.5, 0.4999, 0.4))]
    second = [_ranking((2, 1, 3), (0.5001, 0.5, 0.4))]
    result = compare_runs(first, second, KS)
    row = result.per_query[0]
    assert row.first_divergence == 1  # the swap rank
    assert row.max_rank_shift == 1
    assert row.max_score_delta == pytest.approx(2e-4, rel=1e-12)
    # overlap@1 is 0.0: the two top-1 sets name different concepts after the swap.
    assert row.overlap == {"1": 0.0, "5": 1.0, "10": 1.0}
    assert row.tie_explained and result.passed
    assert result.diverged_queries == () and result.identical_top_k == 0


def test_real_swap_beyond_tie_eps_fails() -> None:
    """Same swap, but the first-run gap (0.4) is far above tie_eps."""
    first = [_ranking((1, 2, 3), (0.9, 0.5, 0.4))]
    second = [_ranking((2, 1, 3), (0.9, 0.5, 0.4))]
    result = compare_runs(first, second, KS)
    assert not result.passed
    assert result.diverged_queries == (0,)
    row = result.per_query[0]
    assert not row.tie_explained
    assert row.first_divergence == 1 and row.max_rank_shift == 1
    assert row.overlap == {"1": 0.0, "5": 1.0, "10": 1.0}


def test_concept_entering_the_top_k_changes_overlap_by_hand() -> None:
    """Candidate 11 enters at rank 5 of the second run; 5 falls out of the top 10."""
    first = [_ranking(range(1, 11))]
    second = [_ranking((1, 2, 3, 4, 11, 6, 7, 8, 9, 10))]
    result = compare_runs(first, second, KS)
    row = result.per_query[0]
    assert row.overlap == {"1": 1.0, "5": 0.8, "10": 0.9}
    assert result.mean_overlap["10"] == pytest.approx(0.9)
    assert result.min_overlap["10"] == pytest.approx(0.9)
    assert row.first_divergence == 5
    # Both runs place every shared candidate (1-4, 6-10) at the same position,
    # so no shared candidate changes rank; the shift is 0 despite the swap in.
    assert row.max_rank_shift == 0
    assert row.max_score_delta == 0.0  # every shared candidate kept its score
    assert not row.tie_explained and not result.passed
    assert result.diverged_queries == (0,) and result.identical_top_k == 0


def test_score_delta_above_score_tol_fails_even_in_identical_order() -> None:
    first = [_ranking((1, 2, 3), (0.9, 0.8, 0.7))]
    second = [_ranking((1, 2, 3), (0.9, 0.8, 0.4))]
    result = compare_runs(first, second, KS)
    assert not result.passed
    assert result.max_score_delta == pytest.approx(0.3, rel=1e-12)
    assert result.diverged_queries == ()  # the order agrees everywhere
    row = result.per_query[0]
    assert row.first_divergence is None and row.max_rank_shift == 0
    assert row.tie_explained and result.identical_top_k == 1


def test_overlap_at_boundaries() -> None:
    assert overlap_at((), (), 5) == 1.0  # both empty agree
    assert overlap_at((1, 2), (), 5) == 0.0
    assert overlap_at((), (1, 2), 5) == 0.0
    assert overlap_at((1, 2, 3), (1, 2, 99), 3) == pytest.approx(2 / 3)
    assert overlap_at((1, 2, 3), (1, 2, 99), 2) == 1.0  # k truncates before the miss
    assert overlap_at((1,), (5,), 1) == 0.0


def test_first_divergence_positions() -> None:
    assert first_divergence((1, 2, 3), (1, 2, 3), 5) is None
    assert first_divergence((1, 2, 3), (1, 2, 3), 2) is None
    assert first_divergence((1, 2), (1, 2, 3), 5) == 3  # prefix of the longer list
    assert first_divergence((1, 2, 3), (1, 2), 5) == 3
    assert first_divergence((1, 2, 3), (9, 2, 3), 5) == 1
    assert first_divergence((1, 2), (2, 1), 5) == 1


def test_compare_runs_fails_closed() -> None:
    one = [_ranking((1, 2))]
    with pytest.raises(ValidationError, match="queries"):
        compare_runs(one, [])
    with pytest.raises(ValidationError, match="no queries"):
        compare_runs([], [])
    with pytest.raises(ValidationError, match="cut-offs"):
        compare_runs(one, one, ())
    with pytest.raises(ValidationError, match="cut-offs"):
        compare_runs(one, one, (0, 1))
    with pytest.raises(ValidationError, match="cut-offs"):
        compare_runs(one, one, (1, 1))
    with pytest.raises(ValidationError, match="tolerances"):
        compare_runs(one, one, KS, tie_eps=-1.0)
    with pytest.raises(ValidationError, match="tolerances"):
        compare_runs(one, one, KS, score_tol=-1.0)


def test_ks_are_sorted_and_aggregates_follow() -> None:
    result = compare_runs([_ranking((1, 2))], [_ranking((1, 2))], (10, 1))
    assert result.ks == (1, 10)
    assert set(result.mean_overlap) == {"1", "10"}
    assert result.max_score_delta == 0.0


def test_to_json_is_deterministic_and_round_trips() -> None:
    first = [_ranking((1, 2, 3), (0.9, 0.5, 0.4))]
    second = [_ranking((2, 1, 3), (0.9, 0.5, 0.4))]
    text = compare_runs(first, second, KS).to_json()
    assert text.endswith("\n")
    assert text == json.dumps(json.loads(text), sort_keys=True, indent=2) + "\n"
    assert RunComparison.model_validate_json(text).to_json() == text
