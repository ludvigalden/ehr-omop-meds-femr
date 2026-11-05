"""Train-only fitting: refused splits, threshold learning, frozen scores, held-out invariance.

The learn_s1_high expectation below is hand-computed with
``omopmap.metrics.wilson`` (z = 1.959963984540054): at threshold 0.65 the pool
is the ten exact rank-1 proposals, 10/10 exact, whose Wilson lower bound is
0.7195 >= 0.70; the two lower candidates fail (10/12 exact -> 0.4622, then
10/11 -> 0.5154). So the smallest passing threshold is 0.65.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from omopmap.calibration import (
    SCORED_TIERS,
    Calibration,
    CalibrationError,
    CalibrationReport,
    EmptyTrainError,
    Evidence,
    FitConfig,
    InSampleApplyError,
    NonTrainRowError,
    NoThresholdError,
    apply_calibration,
    fit_calibration,
    learn_s1_high,
    tier_score,
)
from omopmap.metrics import wilson
from tests.calibration.conftest import make_evidence

LEARN_SET: tuple[Evidence, ...] = (
    *(
        make_evidence(query_id=f"t{i}", s1=s1, label="exact")
        for i, s1 in enumerate((0.95, 0.90, 0.88, 0.85, 0.82))
    ),
    *(
        make_evidence(query_id=f"t{i}", s1=s1, label="exact")
        for i, s1 in enumerate((0.78, 0.75, 0.72, 0.70, 0.65), start=5)
    ),
    make_evidence(query_id="t10", s1=0.55, label="other"),
    make_evidence(query_id="t11", s1=0.50, label="other"),
)


def test_fit_refuses_any_non_train_row() -> None:
    """Dev rows alone, and a train/dev mix, are both refused."""
    config = FitConfig(s1_high=0.85)
    dev = tuple(replace(row, split="dev") for row in LEARN_SET)
    with pytest.raises(NonTrainRowError, match="'dev'"):
        fit_calibration(dev, config, channel="embed", encoder=None)
    mixed = (LEARN_SET[0], dev[0])
    with pytest.raises(NonTrainRowError, match="'dev'"):
        fit_calibration(mixed, config, channel="embed", encoder=None)


def test_fit_on_no_rows_is_refused() -> None:
    """The empty training set raises."""
    with pytest.raises(EmptyTrainError, match="no training proposals"):
        fit_calibration((), FitConfig(s1_high=0.85), channel="embed", encoder=None)


def test_learn_s1_high_returns_the_smallest_passing_threshold() -> None:
    """The hand-computed 0.65 (see module docstring) is returned."""
    config = FitConfig(s1_high=None, target_precision=0.7, min_n=3)
    assert learn_s1_high(LEARN_SET, config) == 0.65


def test_learn_s1_high_without_a_reachable_threshold_raises() -> None:
    """No threshold lifts the Wilson lower bound to 0.8 on this set."""
    config = FitConfig(s1_high=None, target_precision=0.8, min_n=3)
    with pytest.raises(NoThresholdError, match="no s1 threshold"):
        learn_s1_high(LEARN_SET, config)


def _fit_fixed() -> Calibration:
    """A calibration with a fixed threshold over a mixed, deferred, unassessable set."""
    train = (
        *(make_evidence(query_id=f"h{i}", rank=1, s1=0.9, label="exact") for i in range(8)),
        *(make_evidence(query_id=f"h{i}", rank=1, s1=0.9, label="other") for i in range(8, 10)),
        *(make_evidence(query_id=f"d{i}", rank=2, label="exact") for i in range(4)),
        *(make_evidence(query_id=f"d{i}", rank=2, label="other") for i in range(4, 10)),
        *(make_evidence(query_id=f"l{i}", rank=4, label="exact") for i in range(1)),
        *(make_evidence(query_id=f"l{i}", rank=4, label="other") for i in range(1, 5)),
        *(make_evidence(query_id=f"h{i}", rank=1, s1=0.9, label="defer") for i in range(10, 12)),
        make_evidence(query_id="d10", rank=2, label="defer"),
        *(make_evidence(query_id=f"l{i}", rank=4, label="defer") for i in range(5, 7)),
        make_evidence(query_id="u0", eligible=False, rank=None, s1=None, s2=None, margin=None),
    )
    return fit_calibration(train, FitConfig(s1_high=0.85), channel="embed", encoder=None)


def test_tier_scores_are_rounded_wilson_lower_bounds() -> None:
    """Each score equals round(wilson lower bound, 2) of the tier's training exact rate."""
    calibration = _fit_fixed()
    for tier in SCORED_TIERS:
        stats = calibration.train[tier]
        if stats.exact.n == 0:
            assert calibration.scores[tier] is None
        else:
            lower = wilson(stats.exact.count, stats.exact.n).lower
            assert lower is not None
            assert calibration.scores[tier] == round(lower, 2)
        assert tier_score(stats) == calibration.scores[tier]


def test_deferred_rows_count_but_do_not_enter_the_denominator() -> None:
    """High tier: twelve rows, two deferred, ten decided, eight exact."""
    calibration = _fit_fixed()
    high = calibration.train["H"]
    assert (high.rows, high.deferred, high.exact.count, high.exact.n) == (12, 2, 8, 10)
    lower = wilson(8, 10).lower
    assert lower is not None
    assert calibration.scores["H"] == round(lower, 2)


def test_fixed_and_learned_thresholds_are_recorded() -> None:
    """threshold_source and its parameters distinguish fixed from learned fits."""
    fixed = _fit_fixed()
    assert fixed.threshold_source == "fixed"
    assert (fixed.target_precision, fixed.min_n) == (None, None)
    learned = fit_calibration(
        LEARN_SET,
        FitConfig(s1_high=None, target_precision=0.7, min_n=3),
        channel="embed",
        encoder=None,
    )
    assert learned.threshold_source == "learned"
    assert learned.rule.s1_high == 0.65
    assert (learned.target_precision, learned.min_n) == (0.7, 3)


# --------------------------------------------------------------------------
# The key property: held-out rows can never move a frozen calibration.
# --------------------------------------------------------------------------


def _full_evidence() -> tuple[Evidence, ...]:
    """Twelve train rows, four dev rows, four heldout rows."""
    dev = (
        make_evidence(query_id="v0", split="dev", rank=2, label="exact"),
        make_evidence(query_id="v1", split="dev", rank=2, label="other"),
        make_evidence(query_id="v2", split="dev", rank=3, label="exact"),
        make_evidence(query_id="v3", split="dev", rank=4, label="other"),
    )
    heldout = (
        make_evidence(query_id="w0", split="heldout", rank=1, s1=0.95, label="exact"),
        make_evidence(query_id="w1", split="heldout", rank=3, label="other"),
        make_evidence(query_id="w2", split="heldout", rank=None, retrieved=True, label="no-target"),
        make_evidence(
            query_id="w3",
            split="heldout",
            eligible=False,
            rank=None,
            s1=None,
            s2=None,
            margin=None,
            label="no-target",
        ),
    )
    return (*LEARN_SET, *dev, *heldout)


def test_calibration_is_a_pure_function_of_the_training_rows() -> None:
    """Perturbing every non-train row leaves the fit and its bytes unchanged."""
    config = FitConfig(s1_high=None, target_precision=0.7, min_n=3)
    full = _full_evidence()
    train = tuple(e for e in full if e.split == "train")
    first = fit_calibration(train, config, channel="embed:sapbert", encoder=None)

    # Flip labels, move ranks and scores, drop a dev row, add two new rows:
    # every held-out byte changes, the training rows do not.
    dev = [e for e in full if e.split == "dev"]
    heldout = [e for e in full if e.split == "heldout"]
    perturbed = (
        *train,
        replace(dev[0], label="other"),
        replace(dev[1], rank=3, s1=0.99, margin=0.99),
        replace(dev[2], label="defer"),
        # dev[3] removed
        replace(heldout[0], rank=2, s1=0.5, label="other"),
        replace(heldout[1], rank=1, s1=0.99, label="exact"),
        make_evidence(query_id="v9", split="dev", rank=1, s1=0.99, label="exact"),
        make_evidence(query_id="v10", split="dev", rank=4, label="other"),
    )
    second = fit_calibration(
        tuple(e for e in perturbed if e.split == "train"),
        config,
        channel="embed:sapbert",
        encoder=None,
    )
    assert first == second
    assert first.to_json() == second.to_json()


def test_apply_never_touches_the_rule_and_refuses_the_training_split() -> None:
    """The report carries the frozen rule and scores; the train split is refused."""
    config = FitConfig(s1_high=None, target_precision=0.7, min_n=3)
    full = _full_evidence()
    calibration = fit_calibration(
        tuple(e for e in full if e.split == "train"), config, channel="embed:sapbert", encoder=None
    )
    report, tiers = apply_calibration(
        calibration, full, split="heldout", calibration_sha256="a" * 64
    )
    assert isinstance(report, CalibrationReport)
    assert report.rule == calibration.rule
    assert report.scores == calibration.scores
    assert report.split == "heldout"
    assert report.rows == 4
    assert len(tiers) == 4
    with pytest.raises(InSampleApplyError, match="training split"):
        apply_calibration(calibration, full, split="train", calibration_sha256="a" * 64)


def test_calibration_error_is_a_value_error() -> None:
    """The base error keeps its ValueError lineage for CLI exit-code mapping."""
    assert issubclass(CalibrationError, ValueError)
