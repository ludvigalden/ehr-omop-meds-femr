"""``assign_tier`` truth table and ``build_evidence`` feature extraction."""

from __future__ import annotations

import pytest

from omopmap.calibration import (
    CalibrationError,
    Evidence,
    Proposal,
    TierRule,
    assign_tier,
    build_evidence,
)
from omopmap.metrics import FrameRow, Verdict
from omopmap.retrieval.fusion import ScoredRanking
from tests.calibration.conftest import make_evidence

RULE = TierRule(s1_high=0.85, rank_medium=3, capped_paths=("truncation",))


def test_u_reasons_precede_every_other_branch() -> None:
    """Each U reason wins even when the row otherwise qualifies for H."""
    assert assign_tier(make_evidence(rank=1, s1=0.95, eligible=False), RULE) == (
        "U",
        "target-not-eligible",
    )
    assert assign_tier(make_evidence(rank=1, s1=0.95, has_text=False), RULE) == (
        "U",
        "no-source-text",
    )
    assert assign_tier(make_evidence(rank=1, s1=0.95, retrieved=False), RULE) == (
        "U",
        "not-retrieved",
    )


def test_high_tier_needs_rank_one_and_the_threshold() -> None:
    """H exactly at the threshold; just below it the row falls to M."""
    assert assign_tier(make_evidence(rank=1, s1=0.85), RULE) == ("H", "rank-1-high-score")
    assert assign_tier(make_evidence(rank=1, s1=0.8499), RULE) == ("M", "top-ranks")
    assert assign_tier(make_evidence(rank=2, s1=0.95), RULE) == ("M", "top-ranks")


def test_capped_path_is_capped_at_medium() -> None:
    """A rank-1 high-scoring proposal on a capped path lands in M."""
    assert assign_tier(make_evidence(rank=1, s1=0.95, path="truncation"), RULE) == (
        "M",
        "capped-path",
    )


def test_rank_bands_and_the_absent_rank() -> None:
    """Rank 2 and 3 are M, deeper ranks are L, and a missing rank is L too."""
    assert assign_tier(make_evidence(rank=3), RULE) == ("M", "top-ranks")
    assert assign_tier(make_evidence(rank=4), RULE) == ("L", "low-rank")
    assert assign_tier(make_evidence(rank=None, retrieved=True), RULE) == ("L", "not-in-list")


def test_rank_medium_is_taken_from_the_rule() -> None:
    """A rule with a wider M band promotes rank 4."""
    wide = TierRule(s1_high=0.85, rank_medium=5, capped_paths=("truncation",))
    assert assign_tier(make_evidence(rank=5), wide) == ("M", "top-ranks")


# --------------------------------------------------------------------------
# build_evidence.
# --------------------------------------------------------------------------


def _frame(verdicts: list[tuple[Verdict, tuple[int, ...]]]) -> tuple[FrameRow, ...]:
    return tuple(
        FrameRow(
            row=index,
            query_id=f"q{index}",
            split="train",
            verdict=verdict,
            targets=frozenset(targets),
        )
        for index, (verdict, targets) in enumerate(verdicts)
    )


def _primary() -> tuple[ScoredRanking, ...]:
    return (
        ScoredRanking(channel="embed", ids=(3, 7, 1), scores=(0.9, 0.5, 0.3)),
        ScoredRanking(channel="embed", ids=(7,), scores=(0.8,)),
    )


def test_features_are_read_off_the_runs() -> None:
    """Margin, lexical rank/agreement and external agreement per proposal."""
    frame = _frame([("accept", (3,)), ("accept", (1,))])
    texts = ["demo text one", "demo text two"]
    primary = _primary()
    lexical = (
        ScoredRanking(channel="lexical", ids=(7, 3), scores=(0.6, 0.4)),
        ScoredRanking(channel="lexical", ids=(), scores=()),
    )
    proposals = (
        Proposal("q0", 3, "committed", None),
        Proposal("q1", 7, "committed", 7),
    )
    evidence = build_evidence(
        frame, texts, proposals, primary, frozenset({3, 7, 1}), lexical=lexical
    )
    first, second = evidence
    assert first.margin == pytest.approx(0.4)
    assert first.rank == 1
    assert first.s1 == 0.9
    assert first.s2 == 0.5
    assert first.lexical_rank == 2
    assert first.lexical_agrees is False
    assert first.external_agrees is None
    assert second.margin is None  # one candidate only
    assert second.rank == 1
    assert second.s1 == 0.8
    assert second.s2 is None
    assert second.lexical_rank is None  # the lexical run returned nothing
    assert second.lexical_agrees is False  # no lexical top-1 to agree with
    assert second.external_agrees is True  # the external match names concept 7


def test_labels_come_from_the_frame_verdicts() -> None:
    """exact, other, no-target and defer per frame verdict."""
    frame = _frame([("accept", (3,)), ("accept", (5,)), ("no-target", ()), ("defer", ())])
    texts = ["demo a", "demo b", "demo c", "demo d"]
    primary = tuple(_primary()[:1]) * 4
    proposals = tuple(Proposal(f"q{i}", 3, "committed", None) for i in range(4))
    evidence = build_evidence(frame, texts, proposals, primary, frozenset({3}))
    assert [e.label for e in evidence] == ["exact", "other", "no-target", "defer"]


def test_ineligible_and_empty_text_are_flagged() -> None:
    """Dictionary membership and the source text come through as features."""
    frame = _frame([("accept", (3,)), ("no-target", ())])
    texts = ["demo a", "  "]
    primary = _primary()
    proposals = (Proposal("q0", 7, "committed", None), Proposal("q1", 3, "committed", None))
    evidence = build_evidence(frame, texts, proposals, primary, frozenset({3}))
    assert evidence[0].eligible is False
    assert evidence[0].rank == 2
    assert evidence[1].has_text is False
    assert evidence[1].retrieved is True


def test_unknown_query_and_duplicate_proposals_are_refused() -> None:
    """Unknown query id and a repeated (query, concept) pair both raise."""
    frame = _frame([("accept", (3,))])
    texts = ["demo a"]
    primary = (_primary()[0],)
    with pytest.raises(CalibrationError, match="unknown query"):
        build_evidence(
            frame, texts, (Proposal("q9", 3, "committed", None),), primary, frozenset({3})
        )
    with pytest.raises(CalibrationError, match="twice"):
        build_evidence(
            frame,
            texts,
            (Proposal("q0", 3, "committed", None), Proposal("q0", 3, "committed", None)),
            primary,
            frozenset({3}),
        )


def test_misaligned_inputs_are_refused() -> None:
    """Texts or runs that do not match the frame's length raise."""
    frame = _frame([("accept", (3,))])
    texts = ["demo a"]
    proposals = (Proposal("q0", 3, "committed", None),)
    with pytest.raises(CalibrationError, match="differ in length"):
        build_evidence(frame, ["demo a", "demo b"], proposals, _primary(), frozenset({3}))
    with pytest.raises(CalibrationError, match="lexical run ranked a different number"):
        build_evidence(
            frame,
            texts,
            proposals,
            (_primary()[0],),
            frozenset({3}),
            lexical=(
                ScoredRanking(channel="lexical", ids=(3,), scores=(0.5,)),
                ScoredRanking(channel="lexical", ids=(1,), scores=(0.4,)),
            ),
        )


def test_evidence_holds_no_unknown_attributes() -> None:
    """The Evidence dataclass is fully typed and frozen; spot-check one row."""
    row: Evidence = make_evidence()
    assert row.query_id == "demo query"
