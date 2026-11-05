"""Confidence tiers for proposed mappings: evidence features, the H/M/L rule, train-only fitting.

A mapping pipeline proposes one or more target concepts per source term. Not
every proposal deserves the same trust, and a reviewer's time is best spent
where trust is lowest. This module turns the retrieval evidence behind each
proposal into features, assigns a confidence tier, and measures how often
each tier is right.

**Evidence features** (one row per proposal, :class:`Evidence`):

- ``rank``: 1-based rank of the proposed concept in the primary (embedding)
  run; ``None`` when the run returned candidates but not this one.
- ``s1``, ``s2``, ``margin``: the run's top-1 and top-2 scores for the query,
  and ``s1 - s2`` (``None`` with fewer than two candidates).
- ``lexical_rank`` / ``lexical_agrees``: rank of the proposal in an optional
  lexical run, and whether it is that run's top-1 (cross-channel agreement).
- ``external_agrees``: whether an externally supplied match for the query
  (another tool's suggestion, given as a column of the proposals file) names
  the same concept.
- ``eligible`` (the target is in the target dictionary), ``has_text`` (the
  query has a source term to retrieve with), ``path`` (how the mapping was
  produced, free text, e.g. ``committed`` or ``truncation``).

**The tier rule** (:func:`assign_tier`), evaluated top-down:

1. ``U`` (unassessable) when the target is not eligible, the query has no
   text, or the primary run returned nothing for it.
2. ``H`` when ``rank == 1`` and ``s1 >= s1_high`` (0.85 by default); a
   proposal whose ``path`` is in ``capped_paths`` is capped at ``M``.
3. ``M`` when ``rank <= rank_medium`` (3 by default).
4. ``L`` otherwise, including a proposal absent from the run's list.

**Fitting uses the training split only.** :func:`fit_calibration` refuses
any row whose split is not ``train``. It freezes the rule (either the given
``s1_high`` or one learned from the training rows) and each tier's score, the
95% Wilson lower bound of its training exact-match rate. Applying a frozen
calibration to dev or held-out rows never refits; it reports each tier's
observed rates next to the frozen scores. Fitting and judging on the same rows
would make every tier look better than it is.

Labels come from the adjudicated frame of record (``omopmap.metrics``): a
proposal is ``exact`` when its query was accepted and the concept is among the
acceptable targets, ``other`` when the query was accepted with other targets,
``no-target`` or ``defer`` otherwise. Deferred rows are counted but left out
of the rate denominators, because no decision was made on them.

The functions are pure; ``omopmap calibrate fit|apply`` does the file I/O.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, Self

import polars as pl
import typer
from pydantic import BaseModel, ConfigDict, Field, model_validator

from omopmap.metrics import FrameRow, Proportion, parse_frame, wilson
from omopmap.retrieval.fusion import ScoredRanking
from omopmap.retrieval.manifest import EncoderRecord

CALIBRATION_NAME = "calibration.json"
REPORT_NAME = "calibration_report.json"
TIERS_NAME = "tiers.tsv"
CALIBRATION_SCHEMA = "omopmap-calibration/1"
REPORT_SCHEMA = "omopmap-calibration-report/1"
TRAIN_SPLIT = "train"
GATE_EXIT = 6
HASH_EXIT = 4

S1_HIGH = 0.85
"""Default top-1 score threshold for the H tier, the value of the source experiments."""

RANK_MEDIUM = 3
"""Default deepest rank that still counts as M."""

Tier = Literal["H", "M", "L", "U"]
TIERS: tuple[Tier, ...] = ("H", "M", "L", "U")
SCORED_TIERS: tuple[Tier, ...] = ("H", "M", "L")
Label = Literal["exact", "other", "no-target", "defer"]


# --------------------------------------------------------------------------
# Errors.
# --------------------------------------------------------------------------


class CalibrationError(ValueError):
    """Input that cannot be read or interpreted (exit 1)."""


class CalibrationGateError(CalibrationError):
    """A calibration gate failed (exit 6)."""


class NonTrainRowError(CalibrationGateError):
    """Fitting was handed a row outside the training split."""


class EmptyTrainError(CalibrationGateError):
    """No training row is left to fit on."""


class NoThresholdError(CalibrationGateError):
    """No ``s1`` threshold reaches the target precision on enough training rows."""


class InSampleApplyError(CalibrationGateError):
    """A calibration was applied to its own training split."""


class EncoderMismatchError(CalibrationGateError):
    """The applied run's encoder differs from the one the calibration was fit on."""


class CalibrationHashError(CalibrationError):
    """``calibration.json`` differs from the digest its fit manifest recorded (exit 4)."""


# --------------------------------------------------------------------------
# Evidence features.
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Proposal:
    """One proposed mapping: a query, a target concept, and how it was produced."""

    query_id: str
    concept_id: int
    path: str
    external_concept_id: int | None


@dataclass(frozen=True, slots=True)
class Evidence:
    """The evidence features of one proposal, plus its adjudicated label."""

    query_id: str
    split: str
    concept_id: int
    eligible: bool
    has_text: bool
    path: str
    retrieved: bool
    rank: int | None
    s1: float | None
    s2: float | None
    margin: float | None
    lexical_rank: int | None
    lexical_agrees: bool | None
    external_agrees: bool | None
    label: Label


def _rank_of(concept_id: int, ids: Sequence[int]) -> int | None:
    for rank, candidate in enumerate(ids, 1):
        if candidate == concept_id:
            return rank
    return None


def _label(row: FrameRow, concept_id: int) -> Label:
    if row.verdict == "accept":
        return "exact" if concept_id in row.targets else "other"
    return row.verdict


def build_evidence(
    frame: Sequence[FrameRow],
    texts: Sequence[str],
    proposals: Sequence[Proposal],
    primary: Sequence[ScoredRanking],
    dictionary: frozenset[int],
    *,
    lexical: Sequence[ScoredRanking] | None = None,
) -> tuple[Evidence, ...]:
    """Features for every proposal, in proposal order.

    ``frame``, ``texts``, ``primary`` and ``lexical`` are aligned by frame row:
    the runs must have ranked exactly the frame table.
    """
    if len(texts) != len(frame) or len(primary) != len(frame):
        raise CalibrationError("the frame, its texts and the primary run differ in length")
    if lexical is not None and len(lexical) != len(frame):
        raise CalibrationError("the lexical run ranked a different number of queries")
    by_id = {row.query_id: row for row in frame}
    seen: set[tuple[str, int]] = set()
    out: list[Evidence] = []
    for proposal in proposals:
        row = by_id.get(proposal.query_id)
        if row is None:
            raise CalibrationError(f"proposal for unknown query {proposal.query_id!r}")
        key = (proposal.query_id, proposal.concept_id)
        if key in seen:
            raise CalibrationError(
                f"query {proposal.query_id!r} proposes concept {proposal.concept_id} twice"
            )
        seen.add(key)
        ranking = primary[row.row]
        scores = ranking.scores
        s1 = scores[0] if scores else None
        s2 = scores[1] if len(scores) > 1 else None
        lexical_rank: int | None = None
        lexical_agrees: bool | None = None
        if lexical is not None:
            lexical_ids = lexical[row.row].ids
            lexical_rank = _rank_of(proposal.concept_id, lexical_ids)
            lexical_agrees = bool(lexical_ids) and lexical_ids[0] == proposal.concept_id
        out.append(
            Evidence(
                query_id=proposal.query_id,
                split=row.split,
                concept_id=proposal.concept_id,
                eligible=proposal.concept_id in dictionary,
                has_text=bool(texts[row.row].strip()),
                path=proposal.path,
                retrieved=bool(ranking.ids),
                rank=_rank_of(proposal.concept_id, ranking.ids),
                s1=s1,
                s2=s2,
                margin=None if s1 is None or s2 is None else s1 - s2,
                lexical_rank=lexical_rank,
                lexical_agrees=lexical_agrees,
                external_agrees=None
                if proposal.external_concept_id is None
                else proposal.external_concept_id == proposal.concept_id,
                label=_label(row, proposal.concept_id),
            )
        )
    return tuple(out)


# --------------------------------------------------------------------------
# The tier rule.
# --------------------------------------------------------------------------


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class TierRule(_Frozen):
    """The frozen rule: H needs rank 1 and ``s1 >= s1_high``; M needs ``rank <= rank_medium``."""

    s1_high: float = Field(ge=-1.0, le=1.0)
    rank_medium: int = Field(ge=1)
    capped_paths: tuple[str, ...] = Field(
        description="Mapping paths that can never be H; such proposals are capped at M."
    )


def assign_tier(evidence: Evidence, rule: TierRule) -> tuple[Tier, str]:
    """The tier of one proposal and the reason for it."""
    if not evidence.eligible:
        return "U", "target-not-eligible"
    if not evidence.has_text:
        return "U", "no-source-text"
    if not evidence.retrieved:
        return "U", "not-retrieved"
    rank = evidence.rank
    if rank == 1 and evidence.s1 is not None and evidence.s1 >= rule.s1_high:
        if evidence.path in rule.capped_paths:
            return "M", "capped-path"
        return "H", "rank-1-high-score"
    if rank is not None and rank <= rule.rank_medium:
        return "M", "top-ranks"
    return "L", "low-rank" if rank is not None else "not-in-list"


# --------------------------------------------------------------------------
# Tier statistics and fitting.
# --------------------------------------------------------------------------


class TierStats(_Frozen):
    """Rates for one tier; ``n`` excludes deferred rows."""

    rows: int = Field(ge=0, description="Proposals in the tier, deferred ones included.")
    deferred: int = Field(ge=0)
    exact: Proportion = Field(description="Proposal is an acceptable target of its query.")
    any_accept: Proportion = Field(description="The query was accepted with some target.")


def tier_stats(rows: Sequence[Evidence]) -> TierStats:
    decided = [r for r in rows if r.label != "defer"]
    n = len(decided)
    return TierStats(
        rows=len(rows),
        deferred=len(rows) - n,
        exact=wilson(sum(1 for r in decided if r.label == "exact"), n),
        any_accept=wilson(sum(1 for r in decided if r.label in ("exact", "other")), n),
    )


def stats_by_tier(
    evidence: Sequence[Evidence], rule: TierRule
) -> tuple[dict[str, TierStats], dict[str, int], tuple[tuple[Tier, str], ...]]:
    """Per-tier statistics, unassessable counts by reason, and each row's tier."""
    tiers = tuple(assign_tier(e, rule) for e in evidence)
    stats = {
        tier: tier_stats([e for e, (t, _) in zip(evidence, tiers, strict=True) if t == tier])
        for tier in TIERS
    }
    reasons = Counter(reason for tier, reason in tiers if tier == "U")
    return stats, dict(sorted(reasons.items())), tiers


def tier_score(stats: TierStats) -> float | None:
    """A tier's frozen score: the training Wilson lower bound, rounded to two decimals."""
    lower = stats.exact.lower
    return None if lower is None else round(lower, 2)


class FitConfig(_Frozen):
    """How to fit: a fixed ``s1_high``, or ``None`` to learn it from the training rows."""

    s1_high: float | None = Field(default=S1_HIGH, ge=-1.0, le=1.0)
    target_precision: float = Field(default=0.9, gt=0.0, lt=1.0)
    min_n: int = Field(default=10, ge=1)
    rank_medium: int = Field(default=RANK_MEDIUM, ge=1)
    capped_paths: tuple[str, ...] = ("truncation",)


def learn_s1_high(train: Sequence[Evidence], config: FitConfig) -> float:
    """The smallest ``s1`` threshold whose H tier reaches the target on training rows.

    Candidates are the distinct ``s1`` values of assessable rank-1 training
    proposals (capped paths excluded, since they can never be H). For each,
    ascending, the H tier is every such proposal at or above it; the first
    threshold with at least ``min_n`` decided rows and a Wilson lower bound of
    the exact rate at or above ``target_precision`` wins. The smallest passing
    threshold keeps the most proposals in H. None passing is an error, never a
    silent fallback.
    """
    pool = [
        e
        for e in train
        if e.eligible
        and e.has_text
        and e.rank == 1
        and e.s1 is not None
        and e.path not in config.capped_paths
    ]
    for threshold in sorted({e.s1 for e in pool if e.s1 is not None}):
        selected = [e for e in pool if e.s1 is not None and e.s1 >= threshold]
        stats = tier_stats(selected)
        lower = stats.exact.lower
        if stats.exact.n >= config.min_n and lower is not None and lower >= config.target_precision:
            return threshold
    raise NoThresholdError(
        f"no s1 threshold gives {config.min_n}+ decided rank-1 training proposals with a "
        f"Wilson lower bound >= {config.target_precision}; lower the target, or fix s1_high"
    )


class Calibration(_Frozen):
    """``calibration.json``: the frozen rule and tier scores, fit on training rows only."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True, populate_by_name=True)

    schema_: str = Field(alias="schema", default=CALIBRATION_SCHEMA)
    rule: TierRule
    threshold_source: Literal["fixed", "learned"]
    target_precision: float | None
    min_n: int | None
    scores: dict[str, float | None] = Field(description="Tier to training Wilson lower bound.")
    train_rows: int = Field(ge=1)
    train: dict[str, TierStats]
    unassessable: dict[str, int]
    channel: str = Field(description="Channel of the primary run the features were read from.")
    encoder: EncoderRecord | None

    @model_validator(mode="after")
    def _tiers(self) -> Self:
        if set(self.scores) != set(SCORED_TIERS) or set(self.train) != set(TIERS):
            raise ValueError("scores need tiers H, M, L and train stats need H, M, L, U")
        return self

    def to_json(self) -> str:
        return _dump(self)


def fit_calibration(
    train: Sequence[Evidence],
    config: FitConfig,
    *,
    channel: str,
    encoder: EncoderRecord | None,
) -> Calibration:
    """Freeze the rule and tier scores from training rows; any other row is refused."""
    if not train:
        raise EmptyTrainError("no training proposals to fit on")
    foreign = sorted({e.split for e in train if e.split != TRAIN_SPLIT})
    if foreign:
        raise NonTrainRowError(
            f"fitting was handed rows of split(s) {foreign}; calibration is fit on "
            f"{TRAIN_SPLIT!r} rows only"
        )
    learned = config.s1_high is None
    s1_high = learn_s1_high(train, config) if config.s1_high is None else config.s1_high
    rule = TierRule(
        s1_high=s1_high, rank_medium=config.rank_medium, capped_paths=config.capped_paths
    )
    stats, reasons, _ = stats_by_tier(train, rule)
    return Calibration(
        rule=rule,
        threshold_source="learned" if learned else "fixed",
        target_precision=config.target_precision if learned else None,
        min_n=config.min_n if learned else None,
        scores={tier: tier_score(stats[tier]) for tier in SCORED_TIERS},
        train_rows=len(train),
        train=stats,
        unassessable=reasons,
        channel=channel,
        encoder=encoder,
    )


class CalibrationReport(_Frozen):
    """``calibration_report.json``: a frozen calibration applied to one evaluation split."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True, populate_by_name=True)

    schema_: str = Field(alias="schema", default=REPORT_SCHEMA)
    split: str
    calibration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rule: TierRule
    scores: dict[str, float | None]
    rows: int = Field(ge=0)
    observed: dict[str, TierStats]
    unassessable: dict[str, int]

    def to_json(self) -> str:
        return _dump(self)


def apply_calibration(
    calibration: Calibration,
    evidence: Sequence[Evidence],
    *,
    split: str,
    calibration_sha256: str,
) -> tuple[CalibrationReport, tuple[tuple[Tier, str], ...]]:
    """Tier ``split`` rows with the frozen rule; never refits, refuses the training split."""
    if split == TRAIN_SPLIT:
        raise InSampleApplyError(
            "applying a calibration to its training split measures nothing; choose dev or heldout"
        )
    rows = [e for e in evidence if e.split == split]
    if not rows:
        raise CalibrationError(f"no proposals in split {split!r}")
    stats, reasons, tiers = stats_by_tier(rows, calibration.rule)
    report = CalibrationReport(
        split=split,
        calibration_sha256=calibration_sha256,
        rule=calibration.rule,
        scores=dict(calibration.scores),
        rows=len(rows),
        observed=stats,
        unassessable=reasons,
    )
    return report, tiers


def _dump(model: BaseModel) -> str:
    payload = model.model_dump(mode="json", by_alias=True)
    return json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"


def parse_calibration(text: str) -> Calibration:
    try:
        return Calibration.model_validate_json(text)
    except ValueError as exc:
        raise CalibrationError(f"{CALIBRATION_NAME} is invalid: {exc}") from exc


TIER_COLUMNS = (
    "query_id",
    "split",
    "concept_id",
    "tier",
    "reason",
    "label",
    "rank",
    "s1",
    "s2",
    "margin",
    "lexical_rank",
    "lexical_agrees",
    "external_agrees",
    "eligible",
    "has_text",
    "path",
)


def _cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def encode_tiers(evidence: Sequence[Evidence], tiers: Sequence[tuple[Tier, str]]) -> bytes:
    """``tiers.tsv``: the features, tier and reason of every proposal."""
    lines = ["\t".join(TIER_COLUMNS)]
    for e, (tier, reason) in zip(evidence, tiers, strict=True):
        values: list[object] = [
            e.query_id,
            e.split,
            e.concept_id,
            tier,
            reason,
            e.label,
            e.rank,
            e.s1,
            e.s2,
            e.margin,
            e.lexical_rank,
            e.lexical_agrees,
            e.external_agrees,
            e.eligible,
            e.has_text,
            e.path,
        ]
        lines.append("\t".join(_cell(v) for v in values))
    return ("\n".join(lines) + "\n").encode("utf-8")


# --------------------------------------------------------------------------
# Input parsing.
# --------------------------------------------------------------------------


def _optional_int(raw: object, what: str) -> int | None:
    text = "" if raw is None else str(raw).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError as exc:
        raise CalibrationError(f"{what} {text!r} is not an integer id") from exc


def parse_proposals(frame: pl.DataFrame) -> tuple[Proposal, ...]:
    """Proposals from a ``query_id, concept_id [, path] [, external_concept_id]`` table."""
    missing = sorted({"query_id", "concept_id"} - set(frame.columns))
    if missing:
        raise CalibrationError(f"the proposals table lacks columns {missing}")
    has_path = "path" in frame.columns
    has_external = "external_concept_id" in frame.columns
    out: list[Proposal] = []
    for record in frame.iter_rows(named=True):
        query_id = "" if record["query_id"] is None else str(record["query_id"]).strip()
        if not query_id:
            raise CalibrationError("a proposal has an empty query id")
        concept_id = _optional_int(record["concept_id"], f"query {query_id!r}: concept_id")
        if concept_id is None:
            raise CalibrationError(f"query {query_id!r}: the proposal names no concept")
        path = str(record["path"] or "").strip() if has_path else ""
        external = (
            _optional_int(record["external_concept_id"], f"query {query_id!r}: external match")
            if has_external
            else None
        )
        out.append(Proposal(query_id, concept_id, path or "committed", external))
    if not out:
        raise CalibrationError("the proposals table has no rows")
    return tuple(out)


def frame_texts(frame: pl.DataFrame, text_column: str) -> tuple[str, ...]:
    if text_column not in frame.columns:
        raise CalibrationError(f"the frame of record lacks the text column {text_column!r}")
    return tuple("" if v is None else str(v) for v in frame.get_column(text_column).to_list())


# --------------------------------------------------------------------------
# CLI: all file I/O lives here.
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Inputs:
    evidence: tuple[Evidence, ...]
    digests: dict[str, str]
    channel: str
    encoder: EncoderRecord | None
    queries: int


def _load_inputs(
    gold: Path,
    proposals: Path,
    run: Path,
    lexical: Path | None,
    concepts: Path,
    *,
    id_column: str,
    text_column: str,
) -> _Inputs:
    from omopmap.artifacts.manifest import sha256_file
    from omopmap.meds.convert import read_table
    from omopmap.retrieval.cli import eligible_labels
    from omopmap.runs import load_run

    table = read_table(gold)
    if "split" not in table.columns:
        raise CalibrationError("the frame of record needs a split column")
    frame = parse_frame(table, id_column=id_column)
    texts = frame_texts(table, text_column)
    frame_sha = sha256_file(gold)
    primary = load_run(run, queries_sha256=frame_sha)
    if primary.manifest.run_type != "embed":
        raise CalibrationError(
            f"{run} is a {primary.manifest.run_type} run; --run takes an embed run"
        )
    lexical_run = None
    if lexical is not None:
        lexical_run = load_run(lexical, queries_sha256=frame_sha)
        if lexical_run.manifest.run_type != "lexical":
            raise CalibrationError(f"{lexical} is not a lexical run")
    dictionary = frozenset(label.concept_id for label in eligible_labels(concepts, False))
    evidence = build_evidence(
        frame,
        texts,
        parse_proposals(read_table(proposals)),
        primary.rankings,
        dictionary,
        lexical=None if lexical_run is None else lexical_run.rankings,
    )
    digests = {
        "concepts": sha256_file(concepts),
        "gold": frame_sha,
        "proposals": sha256_file(proposals),
        "run:primary": primary.rankings_sha256,
    }
    if lexical_run is not None:
        digests["run:lexical"] = lexical_run.rankings_sha256
    return _Inputs(
        evidence=evidence,
        digests=digests,
        channel=primary.manifest.channel(),
        encoder=primary.manifest.encoder,
        queries=len(frame),
    )


def _write(
    out: Path,
    files: dict[str, bytes],
    inputs: dict[str, str],
    params: dict[str, object],
    created_at: str | None,
) -> None:
    from omopmap.artifacts.manifest import sha256_file
    from omopmap.artifacts.models import utc_timestamp
    from omopmap.meds.writer import prepare_output
    from omopmap.retrieval.manifest import RUN_MANIFEST_NAME, ParamValue, build_run_manifest

    prepare_output(out)
    for name, data in files.items():
        (out / name).write_bytes(data)
    parameters: dict[str, ParamValue] = {}
    for key, value in params.items():
        if isinstance(value, bool | int | float | str):
            parameters[key] = value
    manifest = build_run_manifest(
        "calibrate",
        created_at=created_at or utc_timestamp(),
        parameters=parameters,
        inputs=inputs,
        outputs={name: sha256_file(out / name) for name in files},
    )
    (out / RUN_MANIFEST_NAME).write_text(manifest.to_json(), encoding="utf-8")


def _fail(exc: Exception) -> typer.Exit:
    from omopmap.meds.writer import OutputExistsError
    from omopmap.runs import RunError

    typer.echo(f"{type(exc).__name__}: {exc}", err=True)
    if isinstance(exc, CalibrationGateError):
        return typer.Exit(GATE_EXIT)
    if isinstance(exc, CalibrationHashError):
        return typer.Exit(HASH_EXIT)
    if isinstance(exc, CalibrationError | RunError | OutputExistsError | OSError | ValueError):
        return typer.Exit(1)
    raise exc


app = typer.Typer(
    name="calibrate",
    help="Confidence tiers for proposed mappings, fit on the training split only.",
    no_args_is_help=True,
)

GoldOption = Annotated[
    Path, typer.Option("--gold", help="Frame of record the runs ranked, with a split column.")
]
ProposalsOption = Annotated[
    Path,
    typer.Option(
        "--proposals",
        help="query_id, concept_id [, path] [, external_concept_id] of the mappings to tier.",
    ),
]
RunOption = Annotated[Path, typer.Option("--run", help="Primary embed run over the frame.")]
LexicalOption = Annotated[
    Path | None, typer.Option("--lexical", help="Optional lexical run over the frame.")
]
ConceptsOption = Annotated[
    Path, typer.Option("--concepts", help="The target dictionary the runs ranked from.")
]
OutOption = Annotated[Path, typer.Option("--out", help="New output directory.")]
IdOption = Annotated[str, typer.Option("--id-column", help="Query id column.")]
TextOption = Annotated[str, typer.Option("--text-column", help="Source term column.")]
CreatedOption = Annotated[
    str | None, typer.Option("--created-at", help="Fixed manifest timestamp.")
]


@app.command("fit")
def fit(
    gold: GoldOption,
    proposals: ProposalsOption,
    run: RunOption,
    concepts: ConceptsOption,
    out: OutOption,
    lexical: LexicalOption = None,
    s1_high: Annotated[
        float, typer.Option("--s1-high", help="Fixed H threshold on the top-1 score.")
    ] = S1_HIGH,
    learn_s1: Annotated[
        bool, typer.Option("--learn-s1", help="Learn the H threshold from training rows.")
    ] = False,
    target_precision: Annotated[
        float, typer.Option("--target-precision", help="With --learn-s1: H Wilson lower bound.")
    ] = 0.9,
    min_n: Annotated[
        int, typer.Option("--min-n", help="With --learn-s1: fewest decided H rows.")
    ] = 10,
    rank_medium: Annotated[
        int, typer.Option("--rank-medium", help="Deepest rank that counts as M.")
    ] = RANK_MEDIUM,
    cap_path: Annotated[
        list[str] | None,
        typer.Option("--cap-path", help="Mapping path capped at M; repeat. Default: truncation."),
    ] = None,
    id_column: IdOption = "query_id",
    text_column: TextOption = "source_term",
    created_at: CreatedOption = None,
) -> None:
    """Fit the tier rule and scores on the training split; writes ``calibration.json``.

    Only rows whose split is ``train`` are used; dev and held-out rows in the
    same files are ignored. Exit 0 on success; 1 unreadable input or occupied
    output; 6 a calibration gate failed (no training rows, no threshold).
    """
    try:
        loaded = _load_inputs(
            gold, proposals, run, lexical, concepts, id_column=id_column, text_column=text_column
        )
        config = FitConfig(
            s1_high=None if learn_s1 else s1_high,
            target_precision=target_precision,
            min_n=min_n,
            rank_medium=rank_medium,
            capped_paths=tuple(cap_path) if cap_path else ("truncation",),
        )
        train = tuple(e for e in loaded.evidence if e.split == TRAIN_SPLIT)
        calibration = fit_calibration(train, config, channel=loaded.channel, encoder=loaded.encoder)
        _, _, tiers = stats_by_tier(train, calibration.rule)
        _write(
            out,
            {
                CALIBRATION_NAME: calibration.to_json().encode("utf-8"),
                TIERS_NAME: encode_tiers(train, tiers),
            },
            loaded.digests,
            {
                "step": "fit",
                "split": TRAIN_SPLIT,
                "threshold_source": calibration.threshold_source,
                "s1_high": calibration.rule.s1_high,
                "rank_medium": rank_medium,
                "train_rows": len(train),
                "queries": loaded.queries,
            },
            created_at,
        )
    except Exception as exc:
        raise _fail(exc) from exc
    rule = calibration.rule
    typer.echo(
        f"fit on {calibration.train_rows} training proposals: s1_high={rule.s1_high!r} "
        f"({calibration.threshold_source}), rank_medium={rule.rank_medium}"
    )
    for tier in TIERS:
        stats = calibration.train[tier]
        typer.echo(
            f"{tier}: {stats.exact.count}/{stats.exact.n} exact, "
            f"score {calibration.scores.get(tier)}"
        )


def load_calibration(directory: Path) -> tuple[Calibration, str]:
    """Read a fit directory's ``calibration.json``; fail closed unless it hashes to its manifest."""
    from omopmap.artifacts.manifest import sha256_file
    from omopmap.retrieval.manifest import RUN_MANIFEST_NAME, RunManifestError, parse_run_manifest

    path, manifest_path = directory / CALIBRATION_NAME, directory / RUN_MANIFEST_NAME
    if not path.is_file() or not manifest_path.is_file():
        raise CalibrationError(f"{directory} lacks {CALIBRATION_NAME} or {RUN_MANIFEST_NAME}")
    try:
        manifest = parse_run_manifest(manifest_path.read_text(encoding="utf-8"))
        recorded = manifest.output_sha256(CALIBRATION_NAME)
    except RunManifestError as exc:
        raise CalibrationError(str(exc)) from exc
    if manifest.run_type != "calibrate" or manifest.parameters.get("step") != "fit":
        raise CalibrationError(f"{directory} is not a calibration fit")
    digest = sha256_file(path)
    if digest != recorded:
        raise CalibrationHashError(f"{path}: SHA-256 {digest} differs from its manifest")
    return parse_calibration(path.read_text(encoding="utf-8")), digest


@app.command("apply")
def apply(
    calibration_dir: Annotated[
        Path, typer.Option("--calibration", help="Output directory of `calibrate fit`.")
    ],
    gold: GoldOption,
    proposals: ProposalsOption,
    run: RunOption,
    concepts: ConceptsOption,
    split: Annotated[str, typer.Option("--split", help="Evaluation split (not train).")],
    out: OutOption,
    lexical: LexicalOption = None,
    id_column: IdOption = "query_id",
    text_column: TextOption = "source_term",
    created_at: CreatedOption = None,
) -> None:
    """Tier one evaluation split with a frozen calibration; never refits.

    Exit 0 on success; 1 unreadable input or occupied output; 4
    ``calibration.json`` differs from its fit manifest; 6 a gate failed (the
    training split, or a run from a different encoder).
    """
    try:
        calibration, digest = load_calibration(calibration_dir)
        if split == TRAIN_SPLIT:
            raise InSampleApplyError("apply evaluates a held-back split; train was used to fit")
        loaded = _load_inputs(
            gold, proposals, run, lexical, concepts, id_column=id_column, text_column=text_column
        )
        if (loaded.channel, loaded.encoder) != (calibration.channel, calibration.encoder):
            raise EncoderMismatchError(
                f"the calibration was fit on {calibration.channel!r} scores; this run is "
                f"{loaded.channel!r} or comes from a different encoder build"
            )
        report, tiers = apply_calibration(
            calibration, loaded.evidence, split=split, calibration_sha256=digest
        )
        rows = tuple(e for e in loaded.evidence if e.split == split)
        _write(
            out,
            {REPORT_NAME: report.to_json().encode("utf-8"), TIERS_NAME: encode_tiers(rows, tiers)},
            {**loaded.digests, "calibration": digest},
            {"step": "apply", "split": split, "rows": len(rows), "queries": loaded.queries},
            created_at,
        )
    except Exception as exc:
        raise _fail(exc) from exc
    for tier in TIERS:
        stats = report.observed[tier]
        typer.echo(
            f"{tier}: {stats.exact.count}/{stats.exact.n} exact "
            f"(frozen score {report.scores.get(tier)})"
        )
