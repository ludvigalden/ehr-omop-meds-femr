"""Retrieval metrics over an adjudicated frame of record, with Wilson intervals and hard gates.

**Frame of record.** Recall means nothing until its denominator is fixed. The
frame of record is the adjudicated query table the runs ranked: one row per
query with an id, a verdict and, for accepted queries, the set of acceptable
target concepts.

- ``accept``: at least one target concept is an acceptable mapping
  (``targets`` holds them, ``;``-separated).
- ``no-target``: reviewed, and no concept in the target dictionary fits.
- ``defer``: reviewed, and no decision could be made.

Every row counts in the verdict shares. Only accepted queries enter the
recall denominator, because a query with no acceptable target cannot be hit;
counting it as a miss would mix "the retriever failed" with "nothing exists to
find". Unmappable queries therefore stay visible, as the ``no-target`` share
with its own interval, instead of silently deflating or inflating recall.

**Documented overrides.** Corrections after adjudication (an arbitration
ruling, a re-review, a query that should never have been sampled) are applied
from a separate typed overrides file, never by editing the frame. Each
override names the query, asserts the verdict it starts from, gives the new
verdict and targets or excludes the query, and carries a reason. All of them
are applied, each is counted by kind and listed with its reason in the
output, and the verdict totals before and after correction are both reported.
Optional expected totals in the same file are asserted after correction.

**Intervals.** Every proportion carries a 95% Wilson score interval (no
continuity correction), the method of the source experiments.

**Gates.** The computation stops before writing anything when an input is
inconsistent. Each gate raises its own ``MetricsGateError`` subclass; the CLI
exits 6 and prints the error's name. A run whose rankings no longer hash to
their manifest exits 4, unreadable input exits 1.

The functions are pure; ``omopmap metrics compute`` does the file I/O.
"""

from __future__ import annotations

import json
import math
import tomllib
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, Self

import polars as pl
import typer
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from omopmap.retrieval.fusion import ScoredRanking

METRICS_NAME = "metrics.json"
FRAME_NAME = "frame_of_record.tsv"
METRICS_SCHEMA = "omopmap-metrics/1"
GATE_EXIT = 6
RUN_HASH_EXIT = 4

Z_95 = 1.959963984540054
"""Two-sided 95% standard-normal quantile, ``NormalDist().inv_cdf(0.975)``."""

Verdict = Literal["accept", "no-target", "defer"]
VERDICTS: tuple[Verdict, ...] = ("accept", "no-target", "defer")


# --------------------------------------------------------------------------
# Errors: one class per gate, so a failure names the exact check.
# --------------------------------------------------------------------------


class MetricsError(ValueError):
    """Input that cannot be read or interpreted (exit 1)."""


class MetricsGateError(MetricsError):
    """An evaluation gate failed (exit 6)."""


class DuplicateQueryIdError(MetricsGateError):
    """A query id is empty or appears more than once in the frame."""


class VerdictError(MetricsGateError):
    """A verdict is unknown, or targets and verdict disagree."""


class OverrideError(MetricsGateError):
    """An override is malformed, duplicated, unknown, contradicted, or changes nothing."""


class ExpectedTotalsError(MetricsGateError):
    """Post-correction verdict totals differ from the totals the overrides file expects."""


class GoldNotInDictionaryError(MetricsGateError):
    """An accepted target concept is not in the target dictionary."""


class RunCoverageError(MetricsGateError):
    """A run did not rank exactly the frame's queries."""


class DepthExceededError(MetricsGateError):
    """A cut-off ``k`` is larger than the depth a run kept."""


class NonFiniteScoreError(MetricsGateError):
    """A run holds a NaN or infinite score."""


class DuplicateCandidateError(MetricsGateError):
    """A run lists the same concept twice for one query."""


class RankingOrderError(MetricsGateError):
    """A run's ranks are out of sequence or its scores are not in descending order."""


class EmptyDenominatorError(MetricsGateError):
    """No accepted query is left, so recall is undefined."""


class RunHashError(MetricsError):
    """A run's ``rankings.tsv`` differs from the digest its manifest recorded (exit 4)."""


# --------------------------------------------------------------------------
# Wilson score interval.
# --------------------------------------------------------------------------


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class Proportion(_Frozen):
    """``count / n`` with its Wilson interval; all three are null when ``n == 0``."""

    count: int = Field(ge=0)
    n: int = Field(ge=0)
    estimate: float | None
    lower: float | None
    upper: float | None


def wilson(count: int, n: int, z: float = Z_95) -> Proportion:
    """Wilson score interval for ``count`` successes in ``n`` trials, no continuity correction.

    With ``p = count / n``::

        centre     = (p + z²/(2n)) / (1 + z²/n)
        half-width = z · sqrt(p(1 − p)/n + z²/(4n²)) / (1 + z²/n)

    The bounds are clamped to [0, 1] only to absorb floating-point rounding
    at ``p = 0`` and ``p = 1``, where they are 0 and 1 exactly in real arithmetic.
    """
    if n < 0 or not 0 <= count <= n:
        raise MetricsError(f"a proportion needs 0 <= count <= n, got {count}/{n}")
    if n == 0:
        return Proportion(count=0, n=0, estimate=None, lower=None, upper=None)
    p = count / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4 * n * n)) / denominator
    return Proportion(
        count=count,
        n=n,
        estimate=p,
        lower=max(0.0, centre - half),
        upper=min(1.0, centre + half),
    )


# --------------------------------------------------------------------------
# Frame of record and overrides.
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FrameRow:
    """One adjudicated query; ``row`` is its position in the table the runs ranked."""

    row: int
    query_id: str
    split: str
    verdict: Verdict
    targets: frozenset[int]


def _verdict(raw: str, query_id: str) -> Verdict:
    value = raw.strip().lower()
    for verdict in VERDICTS:
        if value == verdict:
            return verdict
    raise VerdictError(f"query {query_id!r}: verdict {raw!r} is not one of {list(VERDICTS)}")


def _targets(raw: str, query_id: str) -> frozenset[int]:
    parts = [part.strip() for part in raw.split(";") if part.strip()]
    try:
        return frozenset(int(part) for part in parts)
    except ValueError as exc:
        raise VerdictError(f"query {query_id!r}: targets {raw!r} are not integer ids") from exc


def check_verdict(query_id: str, verdict: Verdict, targets: frozenset[int]) -> None:
    """Accept needs targets; the other verdicts must have none."""
    if verdict == "accept" and not targets:
        raise VerdictError(f"query {query_id!r} is accepted but lists no target concept")
    if verdict != "accept" and targets:
        raise VerdictError(f"query {query_id!r} is {verdict} but lists target concepts")


def parse_frame(frame: pl.DataFrame, *, id_column: str = "query_id") -> tuple[FrameRow, ...]:
    """Typed frame rows from a string table with ``id``, ``verdict``, ``targets`` (+ ``split``)."""
    missing = sorted({id_column, "verdict", "targets"} - set(frame.columns))
    if missing:
        raise MetricsError(f"the frame of record lacks columns {missing}")
    has_split = "split" in frame.columns
    rows: list[FrameRow] = []
    seen: set[str] = set()
    for index, record in enumerate(frame.iter_rows(named=True)):
        query_id = "" if record[id_column] is None else str(record[id_column]).strip()
        if not query_id:
            raise DuplicateQueryIdError(f"frame row {index} has an empty query id")
        if query_id in seen:
            raise DuplicateQueryIdError(f"query id {query_id!r} appears more than once")
        seen.add(query_id)
        verdict = _verdict("" if record["verdict"] is None else str(record["verdict"]), query_id)
        targets = _targets("" if record["targets"] is None else str(record["targets"]), query_id)
        check_verdict(query_id, verdict, targets)
        split = str(record["split"] or "") if has_split else ""
        rows.append(FrameRow(index, query_id, split, verdict, targets))
    if not rows:
        raise MetricsError("the frame of record has no rows")
    return tuple(rows)


class Override(_Frozen):
    """One documented correction to the frame of record."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True, populate_by_name=True)

    query_id: str = Field(min_length=1)
    expect_from: Verdict = Field(alias="from", description="The verdict the query must have now.")
    verdict: Verdict | None = None
    targets: tuple[int, ...] = ()
    exclude: bool = False
    reason: str = Field(min_length=1, pattern=r"\S")

    @model_validator(mode="after")
    def _one_action(self) -> Self:
        if self.exclude == (self.verdict is not None):
            raise ValueError("an override either sets `verdict` or sets `exclude = true`")
        if self.exclude and self.targets:
            raise ValueError("an exclusion takes no targets")
        return self


class Overrides(_Frozen):
    """The overrides file: corrections plus optional expected post-correction totals."""

    override: tuple[Override, ...] = ()
    expect: dict[str, dict[str, int]] = Field(
        default_factory=dict[str, dict[str, int]],
        description="Split name (or `all`) to expected verdict counts after correction.",
    )

    @model_validator(mode="after")
    def _expect_keys(self) -> Self:
        for split, counts in self.expect.items():
            if set(counts) != set(VERDICTS):
                raise ValueError(f"expect.{split} must give exactly {list(VERDICTS)}")
            if any(value < 0 for value in counts.values()):
                raise ValueError(f"expect.{split} counts must be non-negative")
        return self


def parse_overrides(text: str) -> Overrides:
    """Parse the TOML overrides file; any defect is an ``OverrideError``."""
    try:
        # Validate in JSON mode: TOML arrays map to tuples, while strict mode
        # still refuses every lax coercion (e.g. the string "12" as an id).
        return Overrides.model_validate_json(json.dumps(tomllib.loads(text)))
    except (tomllib.TOMLDecodeError, ValidationError) as exc:
        raise OverrideError(f"the overrides file is invalid: {exc}") from exc


OverrideKind = Literal["exclude", "verdict", "targets"]


class AppliedOverride(_Frozen):
    """An override as applied, for the output."""

    query_id: str
    kind: OverrideKind
    from_verdict: Verdict
    to_verdict: Verdict | None
    targets: tuple[int, ...]
    reason: str


@dataclass(frozen=True, slots=True)
class CorrectedRow:
    """A frame row after overrides; ``excluded`` rows stay listed but are not counted."""

    source: FrameRow
    verdict: Verdict
    targets: frozenset[int]
    excluded: bool
    override: OverrideKind | None


def apply_overrides(
    rows: Sequence[FrameRow], overrides: Overrides
) -> tuple[tuple[CorrectedRow, ...], tuple[AppliedOverride, ...]]:
    """Apply every override exactly once; fail closed on any that cannot apply cleanly."""
    by_id = {row.query_id: row for row in rows}
    pending: dict[str, Override] = {}
    for item in overrides.override:
        if item.query_id in pending:
            raise OverrideError(f"query {item.query_id!r} has more than one override")
        if item.query_id not in by_id:
            raise OverrideError(f"override for unknown query {item.query_id!r}")
        if by_id[item.query_id].verdict != item.expect_from:
            raise OverrideError(
                f"override for {item.query_id!r} expects verdict {item.expect_from}, "
                f"the frame has {by_id[item.query_id].verdict}"
            )
        pending[item.query_id] = item
    corrected: list[CorrectedRow] = []
    applied: list[AppliedOverride] = []
    for row in rows:
        item = pending.get(row.query_id)
        if item is None:
            corrected.append(CorrectedRow(row, row.verdict, row.targets, False, None))
            continue
        if item.exclude:
            kind: OverrideKind = "exclude"
            verdict, targets = row.verdict, row.targets
        else:
            assert item.verdict is not None
            verdict, targets = item.verdict, frozenset(item.targets)
            check_verdict(row.query_id, verdict, targets)
            if (verdict, targets) == (row.verdict, row.targets):
                raise OverrideError(f"override for {row.query_id!r} changes nothing")
            kind = "verdict" if verdict != row.verdict else "targets"
        corrected.append(CorrectedRow(row, verdict, targets, item.exclude, kind))
        applied.append(
            AppliedOverride(
                query_id=row.query_id,
                kind=kind,
                from_verdict=row.verdict,
                to_verdict=None if item.exclude else verdict,
                targets=tuple(sorted(targets)) if not item.exclude else (),
                reason=item.reason,
            )
        )
    if len(applied) != len(overrides.override):
        raise OverrideError("not every override was applied")
    return tuple(corrected), tuple(applied)


def verdict_counts(verdicts: Sequence[Verdict]) -> dict[str, int]:
    """Counts for every verdict, zeros included."""
    counter = Counter(verdicts)
    return {verdict: counter[verdict] for verdict in VERDICTS}


def check_expected(rows: Sequence[CorrectedRow], expect: Mapping[str, Mapping[str, int]]) -> None:
    """Assert post-correction totals (excluded rows not counted) per split and for ``all``."""
    for split, expected in sorted(expect.items()):
        selected = [
            r for r in rows if not r.excluded and (split == "all" or r.source.split == split)
        ]
        got = verdict_counts([r.verdict for r in selected])
        if got != dict(expected):
            raise ExpectedTotalsError(
                f"{split}: post-correction totals {got} != expected {dict(expected)}"
            )


def check_dictionary(rows: Sequence[CorrectedRow], dictionary: frozenset[int]) -> None:
    """Every target of a counted accepted query must be a candidate the runs could return."""
    for row in rows:
        if row.excluded or row.verdict != "accept":
            continue
        absent = sorted(row.targets - dictionary)
        if absent:
            raise GoldNotInDictionaryError(
                f"query {row.source.query_id!r}: target ids {absent[:5]} are not in the "
                "target dictionary; fix the gold, or exclude the query with a documented override"
            )


# --------------------------------------------------------------------------
# Runs and recall.
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EvaluatedRun:
    """A verified run, ready to score: one ranking per frame row."""

    name: str
    run_type: str
    channel: str
    rankings_sha256: str
    depth: int
    rankings: tuple[ScoredRanking, ...]


def check_run(run: EvaluatedRun, *, n_queries: int, ks: Sequence[int]) -> None:
    """Coverage, depth and finite scores for one run (unique ids hold by construction)."""
    if len(run.rankings) != n_queries:
        raise RunCoverageError(
            f"run {run.name!r} ranked {len(run.rankings)} queries, the frame has {n_queries}"
        )
    if max(ks) > run.depth:
        raise DepthExceededError(
            f"run {run.name!r} kept {run.depth} candidates per query; k={max(ks)} needs more"
        )
    for index, ranking in enumerate(run.rankings):
        if not all(math.isfinite(score) for score in ranking.scores):
            raise NonFiniteScoreError(f"run {run.name!r}, query row {index}: non-finite score")


def scan_rankings(data: bytes, name: str) -> None:
    """Row-level gates on raw ``rankings.tsv`` bytes, before decoding.

    The decoder would reject a repeated candidate as a malformed file; this
    names it as the gate it is. NaN scores decode silently (NaN compares false
    with everything, so a descending-order check cannot see it), so they are
    caught here too.
    """
    seen: set[tuple[str, str]] = set()
    for number, line in enumerate(data.decode("utf-8").splitlines()[1:], start=2):
        cells = line.split("\t")
        if len(cells) < 4:
            continue  # the decoder reports the malformed row
        key = (cells[0], cells[2])
        if key in seen:
            raise DuplicateCandidateError(
                f"run {name!r}, line {number}: concept {cells[2]} ranked twice for query {cells[0]}"
            )
        seen.add(key)
        try:
            score = float(cells[3])
        except ValueError:
            continue  # the decoder reports the malformed score
        if not math.isfinite(score):
            raise NonFiniteScoreError(f"run {name!r}, line {number}: score {cells[3]}")


def first_gold_rank(ids: Sequence[int], gold: frozenset[int]) -> int | None:
    """1-based rank of the first acceptable target, or ``None``."""
    for rank, concept_id in enumerate(ids, 1):
        if concept_id in gold:
            return rank
    return None


class CutOff(_Frozen):
    """Recall at one cut-off: hits over the accepted denominator."""

    k: int = Field(ge=1)
    recall: Proportion


class ConfigRecall(_Frozen):
    """One run's recall at every cut-off."""

    run_type: str
    channel: str
    rankings_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    depth: int = Field(ge=1)
    no_retrieval: int = Field(ge=0, description="Accepted queries the run returned nothing for.")
    at: tuple[CutOff, ...]


class FrameSummary(_Frozen):
    """The denominator: what was counted, what was corrected, what was excluded."""

    rows: int = Field(ge=1, description="Rows in the frame-of-record table.")
    in_scope: int = Field(ge=0, description="Rows in the evaluated split (all rows if none).")
    excluded: int = Field(ge=0)
    n: int = Field(ge=0, description="Counted queries: in scope minus excluded.")
    verdicts_before: dict[str, int] = Field(description="In-scope counts before any override.")
    verdicts: dict[str, int] = Field(description="Counted queries after overrides.")
    by_split: dict[str, dict[str, dict[str, int]]]
    accept: Proportion
    no_target: Proportion
    defer: Proportion
    accept_before: Proportion


class OverrideSummary(_Frozen):
    applied: int = Field(ge=0)
    by_kind: dict[str, int]
    records: tuple[AppliedOverride, ...]


class IntervalMethod(_Frozen):
    method: Literal["wilson-score"] = "wilson-score"
    confidence: float = 0.95
    z: float = Z_95
    continuity_correction: bool = False


class MetricsReport(_Frozen):
    """``metrics.json``."""

    schema_: str = Field(alias="schema", default=METRICS_SCHEMA)
    split: str | None
    ks: tuple[int, ...]
    interval: IntervalMethod
    frame: FrameSummary
    overrides: OverrideSummary
    recall_denominator: int = Field(ge=1)
    recall: dict[str, ConfigRecall]

    def to_json(self) -> str:
        """Deterministic JSON: sorted keys, two-space indent, trailing newline."""
        payload = self.model_dump(mode="json", by_alias=True)
        return json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"


@dataclass(frozen=True, slots=True)
class Evaluation:
    """The report plus the per-query frame, for ``frame_of_record.tsv``."""

    report: MetricsReport
    rows: tuple[CorrectedRow, ...]
    ranks: dict[str, dict[int, int | None]]


def compute_metrics(
    frame: Sequence[FrameRow],
    overrides: Overrides,
    runs: Sequence[EvaluatedRun],
    dictionary: frozenset[int],
    ks: Sequence[int],
    *,
    split: str | None = None,
) -> Evaluation:
    """Apply overrides, run every gate, and compute shares and recall@k with Wilson intervals."""
    if not ks or any(k < 1 for k in ks) or len(set(ks)) != len(ks):
        raise MetricsError(f"cut-offs must be distinct positive integers, got {list(ks)}")
    if not runs:
        raise MetricsError("give at least one run")
    names = [run.name for run in runs]
    if len(set(names)) != len(names):
        raise MetricsError(f"run names repeat: {sorted(names)}")
    ordered = tuple(sorted(ks))
    corrected, applied = apply_overrides(frame, overrides)
    check_expected(corrected, overrides.expect)
    for run in runs:
        check_run(run, n_queries=len(frame), ks=ordered)
    scope = [r for r in corrected if split is None or r.source.split == split]
    if split is not None and not scope:
        raise MetricsError(f"the frame has no rows in split {split!r}")
    counted = [r for r in scope if not r.excluded]
    check_dictionary(counted, dictionary)
    accepted = [r for r in counted if r.verdict == "accept"]
    if not accepted:
        raise EmptyDenominatorError("no accepted query is counted, so recall is undefined")

    after = verdict_counts([r.verdict for r in counted])
    before = verdict_counts([r.source.verdict for r in scope])
    by_split: dict[str, dict[str, dict[str, int]]] = {}
    for name in sorted({r.source.split for r in corrected}):
        rows = [r for r in corrected if r.source.split == name]
        by_split[name or "-"] = {
            "before": verdict_counts([r.source.verdict for r in rows]),
            "after": verdict_counts([r.verdict for r in rows if not r.excluded]),
            "excluded": {"count": sum(1 for r in rows if r.excluded)},
        }
    n = len(counted)
    summary = FrameSummary(
        rows=len(frame),
        in_scope=len(scope),
        excluded=len(scope) - n,
        n=n,
        verdicts_before=before,
        verdicts=after,
        by_split=by_split,
        accept=wilson(after["accept"], n),
        no_target=wilson(after["no-target"], n),
        defer=wilson(after["defer"], n),
        accept_before=wilson(before["accept"], len(scope)),
    )
    kinds = Counter(item.kind for item in applied)
    override_summary = OverrideSummary(
        applied=len(applied),
        by_kind={kind: kinds[kind] for kind in ("exclude", "verdict", "targets")},
        records=applied,
    )

    recall: dict[str, ConfigRecall] = {}
    ranks: dict[str, dict[int, int | None]] = {}
    denominator = len(accepted)
    for run in runs:
        run_ranks = {
            r.source.row: first_gold_rank(run.rankings[r.source.row].ids, r.targets)
            for r in accepted
        }
        ranks[run.name] = run_ranks
        cut_offs = tuple(
            CutOff(
                k=k,
                recall=wilson(
                    sum(1 for rank in run_ranks.values() if rank is not None and rank <= k),
                    denominator,
                ),
            )
            for k in ordered
        )
        recall[run.name] = ConfigRecall(
            run_type=run.run_type,
            channel=run.channel,
            rankings_sha256=run.rankings_sha256,
            depth=run.depth,
            no_retrieval=sum(1 for r in accepted if not run.rankings[r.source.row].ids),
            at=cut_offs,
        )
    report = MetricsReport(
        split=split,
        ks=ordered,
        interval=IntervalMethod(),
        frame=summary,
        overrides=override_summary,
        recall_denominator=denominator,
        recall=recall,
    )
    return Evaluation(report=report, rows=tuple(scope), ranks=ranks)


def encode_frame(evaluation: Evaluation) -> bytes:
    """``frame_of_record.tsv``: one row per in-scope query, with the first gold rank per run."""
    names = sorted(evaluation.ranks)
    header = [
        "query_id",
        "split",
        "verdict_before",
        "verdict",
        "targets",
        "override",
        "excluded",
        *(f"rank:{name}" for name in names),
    ]
    lines = ["\t".join(header)]
    for row in evaluation.rows:
        cells = [
            row.source.query_id,
            row.source.split,
            row.source.verdict,
            row.verdict,
            ";".join(str(t) for t in sorted(row.targets)),
            row.override or "",
            "1" if row.excluded else "0",
        ]
        for name in names:
            rank = evaluation.ranks[name].get(row.source.row)
            cells.append("" if rank is None else str(rank))
        lines.append("\t".join(cells))
    return ("\n".join(lines) + "\n").encode("utf-8")


def parse_ks(value: str) -> tuple[int, ...]:
    """Parse ``--k 1,5,10``."""
    try:
        ks = tuple(int(part) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise MetricsError(f"--k must be comma-separated integers, got {value!r}") from exc
    if not ks or any(k < 1 for k in ks) or len(set(ks)) != len(ks):
        raise MetricsError(f"--k must list distinct positive integers, got {value!r}")
    return ks


# --------------------------------------------------------------------------
# CLI: all file I/O lives here; the functions above are pure.
# --------------------------------------------------------------------------


def named_runs(values: Sequence[str]) -> list[tuple[str | None, Path]]:
    out: list[tuple[str | None, Path]] = []
    for value in values:
        name, sep, path = value.partition("=")
        out.append((name, Path(path)) if sep and name else (None, Path(value)))
    return out


def load_runs(
    specs: Sequence[tuple[str | None, Path]], *, frame_sha256: str, n_queries: int
) -> list[EvaluatedRun]:
    """Verify and decode channel and fused runs against the frame table.

    A channel run (lexical, embed) must record the frame table's SHA-256 as
    its query table. A fused run records only its channel runs' rankings
    digests, so it is accepted only when every one of those channel runs is
    also given (and therefore verified against the frame table).
    """
    from omopmap.artifacts.manifest import sha256_file
    from omopmap.retrieval.fusion import FusionError
    from omopmap.retrieval.manifest import (
        RUN_MANIFEST_NAME,
        RunManifest,
        RunManifestError,
        parse_run_manifest,
    )
    from omopmap.retrieval.rankings import RANKINGS_NAME, RankingsFormatError, decode_scored

    loaded: list[tuple[str, Path, str, RunManifest]] = []
    for name, path in specs:
        manifest_path, rankings_path = path / RUN_MANIFEST_NAME, path / RANKINGS_NAME
        if not manifest_path.is_file() or not rankings_path.is_file():
            raise MetricsError(f"{path} lacks {RUN_MANIFEST_NAME} or {RANKINGS_NAME}")
        try:
            manifest = parse_run_manifest(manifest_path.read_text(encoding="utf-8"))
            recorded = manifest.output_sha256(RANKINGS_NAME)
        except RunManifestError as exc:
            raise MetricsError(str(exc)) from exc
        if manifest.run_type not in ("lexical", "embed", "fuse"):
            raise MetricsError(f"{path} is a {manifest.run_type} run, not a ranking run")
        digest = sha256_file(rankings_path)
        if digest != recorded:
            raise RunHashError(f"{rankings_path}: SHA-256 {digest} differs from its manifest")
        loaded.append((name or manifest.channel(), path, digest, manifest))

    channel_digests = set[str]()
    runs: list[EvaluatedRun] = []
    for name, path, digest, manifest in loaded:
        if manifest.run_type == "fuse":
            continue
        if manifest.inputs_by_name().get("queries") != frame_sha256:
            raise RunCoverageError(f"run {name!r} ({path}) did not rank this frame table")
        if manifest.parameters.get("queries") != n_queries:
            raise RunCoverageError(f"run {name!r} ranked a different number of queries")
        channel_digests.add(digest)
    for name, path, digest, manifest in loaded:
        fused = manifest.run_type == "fuse"
        if fused:
            needed = {
                sha for key, sha in manifest.inputs_by_name().items() if key.startswith("run:")
            }
            if not needed or not needed <= channel_digests:
                raise RunCoverageError(
                    f"fused run {name!r} ({path}): give each of its channel runs as --run too, "
                    "so they can be checked against the frame table"
                )
        depth = manifest.parameters.get("top_k" if fused else "depth")
        if not isinstance(depth, int) or isinstance(depth, bool) or depth < 1:
            raise MetricsError(f"{path}: the manifest records no positive depth")
        data = (path / RANKINGS_NAME).read_bytes()
        scan_rankings(data, name)
        try:
            rankings = decode_scored(
                data, channel=manifest.channel(), queries=n_queries, fused=fused
            )
        except RankingsFormatError as exc:
            if "out of sequence" in str(exc):
                raise RankingOrderError(f"run {name!r}: {exc}") from exc
            raise MetricsError(f"{path}/{RANKINGS_NAME} is malformed: {exc}") from exc
        except FusionError as exc:
            raise RankingOrderError(f"run {name!r}: {exc}") from exc
        runs.append(
            EvaluatedRun(
                name=name,
                run_type=manifest.run_type,
                channel=manifest.channel(),
                rankings_sha256=digest,
                depth=depth,
                rankings=tuple(rankings),
            )
        )
    return runs


app = typer.Typer(
    name="metrics", help="Recall@k over an adjudicated frame of record.", no_args_is_help=True
)


@app.command("compute")
def compute(
    run: Annotated[
        list[str],
        typer.Option("--run", help="NAME=DIR (or DIR) of a lexical, embed or fused run; repeat."),
    ],
    gold: Annotated[
        Path,
        typer.Option(
            "--gold", help="Frame of record: the query table the runs ranked, with verdicts."
        ),
    ],
    concepts: Annotated[
        Path, typer.Option("--concepts", help="The target dictionary the runs ranked from.")
    ],
    out: Annotated[Path, typer.Option("--out", help="New output directory.")],
    overrides: Annotated[
        Path | None, typer.Option("--overrides", help="TOML file of documented overrides.")
    ] = None,
    k: Annotated[str, typer.Option("--k", help="Cut-offs, comma-separated.")] = "1,5,10",
    split: Annotated[
        str | None, typer.Option("--split", help="Evaluate only rows of this split.")
    ] = None,
    id_column: Annotated[str, typer.Option("--id-column", help="Query id column.")] = "query_id",
    created_at: Annotated[
        str | None, typer.Option("--created-at", help="Fixed manifest timestamp.")
    ] = None,
) -> None:
    """Verdict shares and recall@k with Wilson intervals; writes ``metrics.json``.

    Exit 0 on success; 1 unreadable input or occupied output; 4 a run's
    rankings differ from its manifest; 6 an evaluation gate failed. Nothing
    is written unless every gate passes.
    """
    from omopmap.artifacts.manifest import sha256_file
    from omopmap.artifacts.models import utc_timestamp
    from omopmap.meds.convert import read_table
    from omopmap.meds.writer import OutputExistsError, prepare_output
    from omopmap.retrieval.cli import eligible_labels
    from omopmap.retrieval.manifest import RUN_MANIFEST_NAME, build_run_manifest

    try:
        ks = parse_ks(k)
        frame = parse_frame(read_table(gold), id_column=id_column)
        corrections = (
            parse_overrides(overrides.read_text(encoding="utf-8")) if overrides else Overrides()
        )
        frame_sha = sha256_file(gold)
        runs = load_runs(named_runs(run), frame_sha256=frame_sha, n_queries=len(frame))
        dictionary = frozenset(label.concept_id for label in eligible_labels(concepts, False))
        evaluation = compute_metrics(frame, corrections, runs, dictionary, ks, split=split)
        prepare_output(out)
        files = {
            METRICS_NAME: evaluation.report.to_json().encode("utf-8"),
            FRAME_NAME: encode_frame(evaluation),
        }
        for name, data in files.items():
            (out / name).write_bytes(data)
        inputs = {"concepts": sha256_file(concepts), "gold": frame_sha}
        if overrides is not None:
            inputs["overrides"] = sha256_file(overrides)
        inputs.update({f"run:{r.name}": r.rankings_sha256 for r in runs})
        manifest = build_run_manifest(
            "metrics",
            created_at=created_at or utc_timestamp(),
            parameters={
                "id_column": id_column,
                "interval": "wilson-score-95",
                "k": ",".join(str(x) for x in evaluation.report.ks),
                "overrides_applied": evaluation.report.overrides.applied,
                "queries": len(frame),
                "split": split or "",
            },
            inputs=inputs,
            outputs={name: sha256_file(out / name) for name in files},
        )
        (out / RUN_MANIFEST_NAME).write_text(manifest.to_json(), encoding="utf-8")
    except MetricsGateError as exc:
        typer.echo(f"GATE FAILED {type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(GATE_EXIT) from exc
    except RunHashError as exc:
        typer.echo(f"{type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(RUN_HASH_EXIT) from exc
    except (MetricsError, OutputExistsError, OSError, ValueError) as exc:
        typer.echo(f"{type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(1) from exc
    report = evaluation.report
    typer.echo(
        f"frame n={report.frame.n} (excluded {report.frame.excluded}, "
        f"overrides {report.overrides.applied}); recall denominator {report.recall_denominator}"
    )
    for name, config in sorted(report.recall.items()):
        cells = " ".join(f"@{c.k}={c.recall.count}/{c.recall.n}" for c in config.at)
        typer.echo(f"{name}: {cells}")
