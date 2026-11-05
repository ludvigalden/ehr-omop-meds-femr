"""Exact paired McNemar test for two retrieval systems on the same queries.

Two systems ranked the same query set. For a cut-off ``k`` each query is a
hit for a system when one of its gold concepts appears in that system's top
``k``. The paired comparison only needs the 2x2 table of per-query
outcomes:

=================  ===============  ===============
                   candidate hit    candidate miss
=================  ===============  ===============
baseline hit       ``a`` (both)     ``b``
baseline miss      ``c``            ``d`` (neither)
=================  ===============  ===============

Concordant queries (``a``, ``d``) carry no information about which system
is better. Under the null hypothesis of equal hit rates, each of the
``n = b + c`` discordant queries is equally likely to fall either way, so
``c ~ Binomial(n, 1/2)``. The exact two-sided p-value is the binomial test
of ``c`` against ``p = 1/2`` (``scipy.stats.binomtest``); for this
symmetric null it equals twice the smaller tail, capped at 1. With no
discordant query (``n = 0``) the systems are indistinguishable on this
data and ``p = 1``. No continuity correction and no mid-p are applied: the
exact test needs neither, and both would change the reported numbers.

The table is built from per-query outcomes that keep each query's id, so
every discordant cell can be traced back to the queries that produced it.
The functions here are pure; ``omopmap stats mcnemar`` does the file I/O.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Literal

import typer
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.stats import binomtest

MCNEMAR_NAME = "mcnemar.json"
DEFAULT_KS: tuple[int, ...] = (1, 5, 10)

Alternative = Literal["two-sided"]


class StatsError(ValueError):
    """The inputs cannot be compared."""


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


def hit_at(ranking: Sequence[int], gold: frozenset[int], k: int) -> bool:
    """Whether any gold concept is among the first ``k`` ranked ids."""
    if k < 1:
        raise StatsError(f"k must be >= 1, got {k}")
    return any(concept_id in gold for concept_id in ranking[:k])


def mcnemar_exact_p(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value from the discordant counts ``b`` and ``c``."""
    if b < 0 or c < 0:
        raise StatsError(f"discordant counts must be non-negative, got b={b}, c={c}")
    n = b + c
    if n == 0:
        return 1.0
    return float(binomtest(c, n, 0.5, alternative="two-sided").pvalue)


class QueryOutcome(_Frozen):
    """One query's paired outcome at one cut-off."""

    query_id: str = Field(min_length=1)
    baseline_hit: bool
    candidate_hit: bool


class McNemarResult(_Frozen):
    """The paired comparison at one cut-off ``k``."""

    k: int = Field(ge=1)
    n: int = Field(ge=0, description="Queries compared.")
    both_hit: int = Field(ge=0, description="a: hit in both systems.")
    baseline_only: int = Field(ge=0, description="b: hit by the baseline only.")
    candidate_only: int = Field(ge=0, description="c: hit by the candidate only.")
    neither_hit: int = Field(ge=0, description="d: hit by neither.")
    baseline_hits: int = Field(ge=0)
    candidate_hits: int = Field(ge=0)
    p_value: float = Field(ge=0.0, le=1.0)
    test: Literal["exact-binomial-mcnemar"] = "exact-binomial-mcnemar"
    alternative: Alternative = "two-sided"
    baseline_only_ids: tuple[str, ...]
    candidate_only_ids: tuple[str, ...]

    @model_validator(mode="after")
    def _consistent(self) -> McNemarResult:
        a, b, c, d = self.both_hit, self.baseline_only, self.candidate_only, self.neither_hit
        if a + b + c + d != self.n:
            raise ValueError("the four cells must sum to n")
        if self.baseline_hits != a + b or self.candidate_hits != a + c:
            raise ValueError("hit totals must equal the table margins")
        if len(self.baseline_only_ids) != b or len(self.candidate_only_ids) != c:
            raise ValueError("discordant id lists must match their cells")
        return self


def contingency(k: int, outcomes: Sequence[QueryOutcome]) -> McNemarResult:
    """The 2x2 table and exact p-value of per-query outcomes at cut-off ``k``."""
    ids = [o.query_id for o in outcomes]
    if len(set(ids)) != len(ids):
        raise StatsError("query ids must be unique")
    a = sum(1 for o in outcomes if o.baseline_hit and o.candidate_hit)
    b_ids = tuple(o.query_id for o in outcomes if o.baseline_hit and not o.candidate_hit)
    c_ids = tuple(o.query_id for o in outcomes if o.candidate_hit and not o.baseline_hit)
    d = sum(1 for o in outcomes if not o.baseline_hit and not o.candidate_hit)
    b, c = len(b_ids), len(c_ids)
    return McNemarResult(
        k=k,
        n=len(outcomes),
        both_hit=a,
        baseline_only=b,
        candidate_only=c,
        neither_hit=d,
        baseline_hits=a + b,
        candidate_hits=a + c,
        p_value=mcnemar_exact_p(b, c),
        baseline_only_ids=b_ids,
        candidate_only_ids=c_ids,
    )


class PairedComparison(_Frozen):
    """McNemar results for every requested cut-off, plus the per-query outcomes."""

    ks: tuple[int, ...]
    results: tuple[McNemarResult, ...]
    outcomes: dict[str, tuple[QueryOutcome, ...]] = Field(
        description="Per-query outcomes keyed by the cut-off, as a string."
    )

    def to_json(self) -> str:
        """Deterministic JSON: sorted keys, two-space indent, trailing newline."""
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, indent=2) + "\n"


def paired_mcnemar(
    baseline: Mapping[str, Sequence[int]],
    candidate: Mapping[str, Sequence[int]],
    gold: Mapping[str, frozenset[int]],
    ks: Sequence[int] = DEFAULT_KS,
) -> PairedComparison:
    """Compare two systems' rankings on the gold queries, at each cut-off.

    Every gold query must have a gold set (possibly reachable by neither
    system) and both systems must cover exactly the gold queries -- a
    ranking keyed by a query that has no gold, or a missing ranking, fails
    closed. A query with an empty ranking is a miss.
    """
    if not ks or any(k < 1 for k in ks) or len(set(ks)) != len(ks):
        raise StatsError(f"cut-offs must be distinct positive integers, got {list(ks)}")
    queries = sorted(gold)
    if not queries:
        raise StatsError("there are no gold queries to compare")
    for name, system in (("baseline", baseline), ("candidate", candidate)):
        if set(system) != set(queries):
            missing = sorted(set(queries) - set(system))[:5]
            extra = sorted(set(system) - set(queries))[:5]
            raise StatsError(
                f"the {name} rankings do not cover exactly the gold queries "
                f"(missing {missing}, extra {extra})"
            )
    if any(not gold[q] for q in queries):
        raise StatsError("every gold query needs at least one gold concept")
    ordered = tuple(sorted(ks))
    results: list[McNemarResult] = []
    outcomes: dict[str, tuple[QueryOutcome, ...]] = {}
    for k in ordered:
        rows = tuple(
            QueryOutcome(
                query_id=q,
                baseline_hit=hit_at(baseline[q], gold[q], k),
                candidate_hit=hit_at(candidate[q], gold[q], k),
            )
            for q in queries
        )
        results.append(contingency(k, rows))
        outcomes[str(k)] = rows
    return PairedComparison(ks=ordered, results=tuple(results), outcomes=outcomes)


def parse_ks(value: str) -> tuple[int, ...]:
    """Parse ``--k 1,5,10``."""
    try:
        ks = tuple(int(part) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise StatsError(f"--k must be comma-separated integers, got {value!r}") from exc
    if not ks or any(k < 1 for k in ks) or len(set(ks)) != len(ks):
        raise StatsError(f"--k must list distinct positive integers, got {value!r}")
    return ks


# --------------------------------------------------------------------------
# CLI: all file I/O lives here; the functions above are pure.
# --------------------------------------------------------------------------

app = typer.Typer(name="stats", help="Paired statistics over retrieval runs.", no_args_is_help=True)


@app.command("mcnemar")
def mcnemar(
    baseline: Annotated[Path, typer.Option("--baseline", help="Baseline run directory.")],
    candidate: Annotated[Path, typer.Option("--candidate", help="Candidate run directory.")],
    gold: Annotated[
        Path, typer.Option("--gold", help="The query table both runs ranked (with gold ids).")
    ],
    out: Annotated[Path, typer.Option("--out", help="New output directory.")],
    k: Annotated[str, typer.Option("--k", help="Cut-offs, comma-separated.")] = "1,5,10",
    gold_column: Annotated[
        str, typer.Option("--gold-column", help="Column holding the gold concept id.")
    ] = "concept_id",
    split: Annotated[
        str | None,
        typer.Option("--split", help="Compare only rows whose `split` column has this value."),
    ] = None,
    id_column: Annotated[
        str | None,
        typer.Option("--id-column", help="Column with query ids; default: the row index."),
    ] = None,
    created_at: Annotated[
        str | None, typer.Option("--created-at", help="Fixed manifest timestamp.")
    ] = None,
) -> None:
    """Exact paired McNemar test of two runs at each cut-off; writes ``mcnemar.json``.

    Both runs must have ranked exactly the ``--gold`` table (their manifests
    record its SHA-256). Exit 1 on any input or consistency failure.
    """
    from omopmap.artifacts.manifest import sha256_file
    from omopmap.artifacts.models import utc_timestamp
    from omopmap.meds.writer import OutputExistsError, prepare_output
    from omopmap.retrieval.manifest import RUN_MANIFEST_NAME, RunManifestError, build_run_manifest
    from omopmap.retrieval.rankings import RankingsFormatError
    from omopmap.runs import RunError, gold_queries, load_run

    try:
        ks = parse_ks(k)
        gold_sha = sha256_file(gold)
        base_run = load_run(baseline, queries_sha256=gold_sha)
        cand_run = load_run(candidate, queries_sha256=gold_sha)
        gold_sets = gold_queries(gold, gold_column=gold_column, split=split, id_column=id_column)
        result = paired_mcnemar(
            base_run.by_query_id(gold_sets.query_ids),
            cand_run.by_query_id(gold_sets.query_ids),
            gold_sets.gold,
            ks,
        )
        prepare_output(out)
        (out / MCNEMAR_NAME).write_text(result.to_json(), encoding="utf-8")
        manifest = build_run_manifest(
            "mcnemar",
            created_at=created_at or utc_timestamp(),
            parameters={
                "gold_column": gold_column,
                "id_column": id_column or "",
                "k": ",".join(str(x) for x in result.ks),
                "queries": len(gold_sets.gold),
                "split": split or "",
            },
            inputs={
                "baseline": base_run.rankings_sha256,
                "candidate": cand_run.rankings_sha256,
                "gold": gold_sha,
            },
            outputs={MCNEMAR_NAME: sha256_file(out / MCNEMAR_NAME)},
        )
        (out / RUN_MANIFEST_NAME).write_text(manifest.to_json(), encoding="utf-8")
        for row in result.results:
            typer.echo(
                f"@{row.k}: n={row.n} baseline={row.baseline_hits} "
                f"candidate={row.candidate_hits} b={row.baseline_only} "
                f"c={row.candidate_only} p={row.p_value:.4g}"
            )
    except (
        StatsError,
        RunError,
        RunManifestError,
        RankingsFormatError,
        OutputExistsError,
        OSError,
    ) as exc:
        typer.echo(f"{type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(1) from exc
