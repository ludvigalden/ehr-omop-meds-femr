"""Pure-function tests: the Wilson interval, frame parsing, overrides and the gates.

Each gate error is raised on purpose here and checked by its class. The CLI
tests run the same failures end to end and check the exit codes.
"""

from __future__ import annotations

import math
import statistics

import polars as pl
import pytest

from omopmap.metrics import (
    ConfigRecall,
    CorrectedRow,
    DepthExceededError,
    DuplicateCandidateError,
    DuplicateQueryIdError,
    EmptyDenominatorError,
    EvaluatedRun,
    ExpectedTotalsError,
    FrameRow,
    GoldNotInDictionaryError,
    MetricsError,
    NonFiniteScoreError,
    Override,
    OverrideError,
    Overrides,
    RunCoverageError,
    VerdictError,
    apply_overrides,
    check_dictionary,
    check_expected,
    check_run,
    check_verdict,
    compute_metrics,
    first_gold_rank,
    parse_frame,
    parse_overrides,
    scan_rankings,
    verdict_counts,
    wilson,
)
from omopmap.retrieval.fusion import ScoredRanking

# ---------------------------------------------------------------------------
# Wilson score interval.
# ---------------------------------------------------------------------------


def _hand_wilson(count: int, n: int) -> tuple[float, float, float]:
    """The Wilson interval written out from the formula, with an independently derived z."""
    z = statistics.NormalDist().inv_cdf(0.975)
    p = count / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, centre - half, centre + half


@pytest.mark.parametrize(
    ("count", "n"),
    [(0, 10), (10, 10), (1, 2), (7, 21), (23, 315), (70, 315), (102, 315), (123, 315)],
)
def test_wilson_matches_the_hand_formula(count: int, n: int) -> None:
    p, lo, hi = _hand_wilson(count, n)
    got = wilson(count, n)
    assert got.count == count and got.n == n
    assert got.estimate == pytest.approx(p, rel=1e-15)
    assert got.lower == pytest.approx(max(0.0, lo), rel=1e-12, abs=1e-15)
    assert got.upper == pytest.approx(min(1.0, hi), rel=1e-12, abs=1e-15)


def test_wilson_hand_computed_small_case() -> None:
    """1 success in 2 trials, worked by hand with z = 1.96 to 4 decimals.

    p = 0.5, z² = 3.8416, d = 1 + 3.8416/2 = 2.9208,
    centre = (0.5 + 3.8416/4) / 2.9208 = 1.4604 / 2.9208 = 0.5,
    half = 1.96 · sqrt(0.25/2 + 3.8416/16) / 2.9208
         = 1.96 · sqrt(0.365100) / 2.9208 = 1.96 · 0.604235 / 2.9208 = 0.405474,
    interval = (0.0945, 0.9055).
    """
    got = wilson(1, 2, z=1.96)
    assert got.lower is not None and got.upper is not None
    assert round(got.lower, 4) == 0.0945
    assert round(got.upper, 4) == 0.9055


def test_wilson_reproduces_the_source_experiment_numbers() -> None:
    """The intervals the source metrics reported, at its stored 4-decimal rounding."""
    for count, n, lo, hi in [
        (315, 400, 0.7448, 0.8248),  # held-out accept share, post-correction
        (79, 400, 0.1614, 0.2393),  # held-out no-target share
        (23, 315, 0.0491, 0.1072),  # lexical hit@10
        (123, 315, 0.3382, 0.4454),  # SapBERT hit@10
        (117, 315, 0.3199, 0.4260),  # RRF fusion hit@10
        (106, 315, 0.2866, 0.3904),  # RRF complete-set@10
    ]:
        got = wilson(count, n)
        assert got.lower is not None and got.upper is not None
        assert round(got.lower, 4) == lo, (count, n)
        assert round(got.upper, 4) == hi, (count, n)


def test_wilson_edges() -> None:
    zero = wilson(0, 40)
    assert zero.estimate == 0.0 and zero.lower == 0.0
    assert zero.upper is not None and zero.upper > 0.0
    full = wilson(40, 40)
    assert full.estimate == 1.0 and full.upper == 1.0
    assert full.lower is not None and full.lower < 1.0
    empty = wilson(0, 0)
    assert empty.estimate is None and empty.lower is None and empty.upper is None
    with pytest.raises(MetricsError):
        wilson(5, 4)


# ---------------------------------------------------------------------------
# Frame parsing and its gates.
# ---------------------------------------------------------------------------

COLUMNS = ("query_id", "split", "verdict", "targets")


def _frame(*rows: dict[str, str]) -> pl.DataFrame:
    return pl.DataFrame(
        [[row[c] for c in COLUMNS] for row in rows],
        schema=dict.fromkeys(COLUMNS, pl.String),
        orient="row",
    )


def _row(
    query_id: str = "q0", verdict: str = "accept", targets: str = "11", split: str = "dev"
) -> dict[str, str]:
    return {"query_id": query_id, "verdict": verdict, "targets": targets, "split": split}


def test_frame_parses_verdicts_and_split() -> None:
    rows = parse_frame(
        _frame(_row(), _row("q1", "no-target", "", "train"), _row("q2", "defer", "", ""))
    )
    assert [(r.row, r.query_id, r.verdict, r.targets, r.split) for r in rows] == [
        (0, "q0", "accept", frozenset({11}), "dev"),
        (1, "q1", "no-target", frozenset(), "train"),
        (2, "q2", "defer", frozenset(), ""),
    ]


def test_multi_target_accept_uses_the_union() -> None:
    (row,) = parse_frame(_frame(_row(targets="11; 22;33")))
    assert row.targets == frozenset({11, 22, 33})


def test_missing_columns_are_named() -> None:
    with pytest.raises(MetricsError, match="lacks columns"):
        parse_frame(pl.DataFrame({"query_id": ["q0"]}))


def test_duplicate_query_id_fails() -> None:
    with pytest.raises(DuplicateQueryIdError, match="appears more than once"):
        parse_frame(_frame(_row("q0"), _row("q0", "no-target", "")))
    with pytest.raises(DuplicateQueryIdError, match="empty query id"):
        parse_frame(_frame(_row("  ")))


@pytest.mark.parametrize(
    ("verdict", "targets", "message"),
    [
        ("accepted", "11", "not one of"),
        ("accept", "", "accepted but lists no target"),
        ("no-target", "11", "no-target but lists"),
        ("defer", "11", "defer but lists"),
        ("accept", "eleven", "not integer ids"),
    ],
)
def test_verdict_gate(verdict: str, targets: str, message: str) -> None:
    with pytest.raises(VerdictError, match=message):
        parse_frame(_frame(_row(verdict=verdict, targets=targets)))


def test_check_verdict_is_reused_for_overrides() -> None:
    with pytest.raises(VerdictError, match="accepted but lists no target"):
        check_verdict("q9", "accept", frozenset())


# ---------------------------------------------------------------------------
# Overrides.
# ---------------------------------------------------------------------------


def _toml(query_id: str, start: str, body: str) -> str:
    return f'[[override]]\nquery_id = "{query_id}"\nfrom = "{start}"\n{body}'


def _frames() -> tuple[FrameRow, ...]:
    return (
        FrameRow(0, "q0", "dev", "accept", frozenset({11})),
        FrameRow(1, "q1", "dev", "no-target", frozenset()),
        FrameRow(2, "q2", "train", "defer", frozenset()),
    )


def test_overrides_change_verdict_targets_and_exclusion_and_are_counted() -> None:
    overrides = parse_overrides(
        _toml("q1", "no-target", 'verdict = "accept"\ntargets = [12]\nreason = "arbitration"\n')
        + _toml("q2", "defer", 'exclude = true\nreason = "sampled twice"\n')
        + _toml("q0", "accept", 'verdict = "accept"\ntargets = [13]\nreason = "reviewer fix"\n')
    )
    corrected, applied = apply_overrides(_frames(), overrides)
    assert [(r.verdict, sorted(r.targets), r.excluded, r.override) for r in corrected] == [
        ("accept", [13], False, "targets"),
        ("accept", [12], False, "verdict"),
        ("defer", [], True, "exclude"),
    ]
    # Applied in frame order, each with its reason; exclusions carry no new verdict.
    assert [(a.query_id, a.kind, a.reason) for a in applied] == [
        ("q0", "targets", "reviewer fix"),
        ("q1", "verdict", "arbitration"),
        ("q2", "exclude", "sampled twice"),
    ]
    assert applied[2].to_verdict is None and applied[1].to_verdict == "accept"


@pytest.mark.parametrize(
    "body",
    [
        'verdict = "accept"\ntargets = [1]\n',  # no reason
        'verdict = "accept"\ntargets = [1]\nreason = "  "\n',  # blank reason
        'reason = "r"\n',  # neither a verdict nor an exclusion
        'exclude = true\nverdict = "accept"\ntargets = [1]\nreason = "r"\n',  # both
        'exclude = true\ntargets = [1]\nreason = "r"\n',  # exclusion with targets
        'verdict = "maybe"\nreason = "r"\n',  # unknown verdict
        'verdict = "accept"\ntargets = ["12"]\nreason = "r"\n',  # id given as a string
        'exclude = true\nreason = "r"\nnote = "x"\n',  # unknown key
    ],
)
def test_malformed_override_is_refused(body: str) -> None:
    with pytest.raises(OverrideError, match="invalid"):
        parse_overrides(_toml("q1", "no-target", body))


def test_well_formed_exclusion_parses() -> None:
    (item,) = parse_overrides(_toml("q1", "no-target", 'exclude = true\nreason = "r"\n')).override
    assert isinstance(item, Override) and item.exclude and item.expect_from == "no-target"


def test_override_gates_fail_closed() -> None:
    frames = _frames()
    exclude = 'exclude = true\nreason = "r"\n'
    with pytest.raises(OverrideError, match="unknown query"):
        apply_overrides(frames, parse_overrides(_toml("zz", "accept", exclude)))
    with pytest.raises(OverrideError, match="expects verdict accept"):
        apply_overrides(frames, parse_overrides(_toml("q1", "accept", exclude)))
    twice = _toml("q1", "no-target", exclude)
    with pytest.raises(OverrideError, match="more than one override"):
        apply_overrides(frames, parse_overrides(twice + twice))
    same = 'verdict = "accept"\ntargets = [11]\nreason = "r"\n'
    with pytest.raises(OverrideError, match="changes nothing"):
        apply_overrides(frames, parse_overrides(_toml("q0", "accept", same)))
    no_targets = 'verdict = "accept"\nreason = "r"\n'
    with pytest.raises(VerdictError, match="accepted but lists no target"):
        apply_overrides(frames, parse_overrides(_toml("q1", "no-target", no_targets)))
    with pytest.raises(OverrideError, match="invalid"):
        parse_overrides("this is not toml = = =")


def test_expected_totals_are_asserted() -> None:
    overrides = parse_overrides(
        _toml("q1", "no-target", 'verdict = "accept"\ntargets = [12]\nreason = "r"\n')
    )
    corrected, _ = apply_overrides(_frames(), overrides)
    check_expected(corrected, overrides.expect)  # nothing expected: nothing asserted
    check_expected(
        corrected,
        {
            "dev": {"accept": 2, "no-target": 0, "defer": 0},
            "all": {"accept": 2, "no-target": 0, "defer": 1},
        },
    )
    with pytest.raises(ExpectedTotalsError, match="dev"):
        check_expected(corrected, {"dev": {"accept": 1, "no-target": 1, "defer": 0}})


def test_expected_totals_keys_are_validated() -> None:
    with pytest.raises(OverrideError):
        parse_overrides("[expect.dev]\naccept = 1\n")
    with pytest.raises(OverrideError):
        parse_overrides('[expect.all]\naccept = 1\n"no-target" = 0\ndefer = 0\nextra = 0\n')
    parsed = parse_overrides('[expect.all]\naccept = 1\n"no-target" = 0\ndefer = 0\n')
    assert parsed.expect == {"all": {"accept": 1, "no-target": 0, "defer": 0}}


# ---------------------------------------------------------------------------
# Dictionary, run and denominator gates.
# ---------------------------------------------------------------------------


def _corrected(*rows: FrameRow, excluded: bool = False) -> list[CorrectedRow]:
    return [CorrectedRow(row, row.verdict, row.targets, excluded, None) for row in rows]


def test_dictionary_gate_flags_absent_targets() -> None:
    ok = FrameRow(0, "q0", "dev", "accept", frozenset({11}))
    bad = FrameRow(1, "q1", "dev", "accept", frozenset({12, 999}))
    with pytest.raises(GoldNotInDictionaryError, match="999"):
        check_dictionary(_corrected(ok, bad), frozenset({11, 12}))
    no_target = FrameRow(2, "q2", "dev", "no-target", frozenset())
    check_dictionary(_corrected(ok, no_target), frozenset({11}))
    # An excluded query is not counted, so its gold is not checked either.
    check_dictionary(_corrected(bad, excluded=True), frozenset({11}))


def _run(
    *, ids: tuple[int, ...] = (11,), scores: tuple[float, ...] = (0.5,), depth: int = 50
) -> EvaluatedRun:
    return EvaluatedRun(
        name="emb",
        run_type="embed",
        channel="embed:sapbert",
        rankings_sha256="a" * 64,
        depth=depth,
        rankings=tuple([ScoredRanking(channel="x", ids=ids, scores=scores)] * 3),
    )


def test_run_gates() -> None:
    check_run(_run(), n_queries=3, ks=(1, 5, 10))
    with pytest.raises(RunCoverageError, match="ranked 3 queries, the frame has 4"):
        check_run(_run(), n_queries=4, ks=(1,))
    with pytest.raises(DepthExceededError, match="needs more"):
        check_run(_run(depth=10), n_queries=3, ks=(1, 5, 10, 50))
    with pytest.raises(NonFiniteScoreError):
        check_run(_run(ids=(11, 12), scores=(float("inf"), 0.5)), n_queries=3, ks=(1, 5))


def test_raw_scan_catches_duplicates_and_nan() -> None:
    header = b"query_index\trank\tconcept_id\tscore\n"
    scan_rankings(header + b"0\t1\t11\t0.5\n0\t2\t12\t0.4\n1\t1\t11\t0.3\n", "ok")
    with pytest.raises(DuplicateCandidateError, match="ranked twice"):
        scan_rankings(header + b"0\t1\t11\t0.5\n0\t2\t11\t0.4\n", "dup")
    with pytest.raises(NonFiniteScoreError, match="nan"):
        scan_rankings(header + b"0\t1\t11\tnan\n", "nan")


def test_first_gold_rank() -> None:
    assert first_gold_rank((5, 11, 12), frozenset({12})) == 3
    assert first_gold_rank((5, 11), frozenset({12})) is None
    assert first_gold_rank((), frozenset({12})) is None


def test_empty_denominator_gate() -> None:
    frames = tuple(FrameRow(i, f"q{i}", "dev", "no-target", frozenset()) for i in range(3))
    with pytest.raises(EmptyDenominatorError):
        compute_metrics(frames, Overrides(), [_run()], frozenset({11}), (1, 5, 10))


def test_compute_metrics_denominator_and_recall() -> None:
    frames = (
        FrameRow(0, "q0", "dev", "accept", frozenset({11})),
        FrameRow(1, "q1", "dev", "accept", frozenset({12})),
        FrameRow(2, "q2", "train", "no-target", frozenset()),
        FrameRow(3, "q3", "train", "accept", frozenset({11})),
    )
    run = EvaluatedRun(
        name="emb",
        run_type="embed",
        channel="embed:sapbert",
        rankings_sha256="b" * 64,
        depth=50,
        rankings=(
            ScoredRanking(channel="x", ids=(11, 99), scores=(0.9, 0.1)),
            ScoredRanking(channel="x", ids=(99, 12), scores=(0.8, 0.2)),
            ScoredRanking(channel="x", ids=(), scores=()),
            ScoredRanking(channel="x", ids=(99,), scores=(0.5,)),
        ),
    )
    dictionary = frozenset({11, 12, 99})
    evaluation = compute_metrics(frames, Overrides(), [run], dictionary, (10, 1, 5))
    report = evaluation.report
    assert report.ks == (1, 5, 10)
    # Recall denominator: the three accepts. q1's gold is at rank 2; q3's gold is absent.
    assert report.recall_denominator == 3
    config = report.recall["emb"]
    assert isinstance(config, ConfigRecall)
    at = {c.k: c.recall for c in config.at}
    assert (at[1].count, at[1].n) == (1, 3)
    assert (at[5].count, at[5].n) == (2, 3)
    assert at[10] == wilson(2, 3)
    assert config.no_retrieval == 0  # q2 returned nothing, but q2 is not accepted
    # Verdict shares count every row (no split chosen).
    assert report.frame.n == 4
    assert report.frame.verdicts == {"accept": 3, "no-target": 1, "defer": 0}
    assert report.frame.no_target == wilson(1, 4)
    assert report.frame.by_split["dev"]["after"] == {"accept": 2, "no-target": 0, "defer": 0}
    assert evaluation.ranks["emb"] == {0: 1, 1: 2, 3: None}
    # Choosing a split moves the denominator.
    dev = compute_metrics(frames, Overrides(), [run], dictionary, (1, 5), split="dev")
    assert dev.report.recall_denominator == 2
    assert dev.report.recall["emb"].at[0].recall.count == 1
    # Byte-identical JSON on recompute.
    again = compute_metrics(frames, Overrides(), [run], dictionary, (1, 5, 10))
    assert again.report.to_json() == report.to_json()


def test_bad_cut_offs_and_run_names() -> None:
    frames = (FrameRow(0, "q0", "dev", "accept", frozenset({11})),)
    single = EvaluatedRun(
        "a", "embed", "embed:x", "c" * 64, 5, (ScoredRanking("x", (11,), (1.0,)),)
    )
    with pytest.raises(MetricsError, match="distinct positive"):
        compute_metrics(frames, Overrides(), [single], frozenset({11}), (1, 1))
    with pytest.raises(MetricsError, match="repeat"):
        compute_metrics(frames, Overrides(), [single, single], frozenset({11}), (1,))


def test_verdict_counts_cover_every_verdict() -> None:
    assert verdict_counts([]) == {"accept": 0, "no-target": 0, "defer": 0}
    counts = verdict_counts(["accept", "accept", "defer"])
    assert counts == {"accept": 2, "no-target": 0, "defer": 1}
