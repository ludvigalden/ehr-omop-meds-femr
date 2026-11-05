"""Reproducibility check: do two runs of the same retrieval agree?

The same frozen dictionary and queries, embedded by the same pinned model on
a GPU (fp16) and on a CPU (fp32), should rank the same concepts. Floating
point makes byte equality the wrong expectation: half-precision changes the
last digits of every score, and two candidates whose scores differ by less
than that noise can swap places. So the comparison is on the top-``k``
lists, per query:

- ``overlap@k``: the share of the first run's top ``k`` that is also in the
  second run's top ``k``;
- ``first_divergence``: the first rank (1-based) at which the two lists name
  different concepts, or ``None`` when the shorter list is a prefix of the
  longer;
- ``max_rank_shift``: the largest change in rank of a concept present in
  both top ``k`` lists;
- ``max_score_delta``: the largest absolute score difference over the
  concepts both runs scored;
- ``tie_explained``: the top ``k`` sets are equal, and every pair of
  concepts the two runs order differently scored within ``tie_eps`` of each
  other in the first run -- the swap is a near-tie, not a ranking change.

The verdict passes when every query is ``tie_explained`` at the largest
``k`` and the largest score difference is within ``score_tol``. The default
tolerances (``2e-3`` both) are the source experiments' CPU-regression gate.
The functions here are pure; ``omopmap validate runs`` does the file I/O and
exits 5 when the runs diverge.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import typer
from pydantic import BaseModel, ConfigDict, Field

from omopmap.retrieval.fusion import ScoredRanking

VALIDATION_NAME = "validation.json"
DIVERGED_EXIT = 5
TIE_EPS = 2e-3
SCORE_TOL = 2e-3
DEFAULT_KS: tuple[int, ...] = (1, 5, 10)


class ValidationError(ValueError):
    """The runs cannot be compared."""


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class QueryComparison(_Frozen):
    """One query's agreement between the two runs."""

    query_index: int = Field(ge=0)
    overlap: dict[str, float] = Field(description="overlap@k keyed by k, as a string.")
    first_divergence: int | None
    max_rank_shift: int = Field(ge=0)
    max_score_delta: float | None
    tie_explained: bool


class RunComparison(_Frozen):
    """The aggregate verdict over all queries."""

    queries: int = Field(ge=0)
    ks: tuple[int, ...]
    tie_eps: float
    score_tol: float
    mean_overlap: dict[str, float]
    min_overlap: dict[str, float]
    identical_top_k: int = Field(ge=0, description="Queries with identical top-k at the largest k.")
    tie_explained: int = Field(ge=0)
    diverged_queries: tuple[int, ...]
    max_score_delta: float | None
    passed: bool
    per_query: tuple[QueryComparison, ...]

    def to_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, indent=2) + "\n"


def overlap_at(a: Sequence[int], b: Sequence[int], k: int) -> float:
    """Share of ``a``'s top ``k`` that is in ``b``'s top ``k`` (1.0 if ``a`` has none)."""
    top_a = a[:k]
    if not top_a:
        return 1.0 if not b[:k] else 0.0
    return len(set(top_a) & set(b[:k])) / len(top_a)


def first_divergence(a: Sequence[int], b: Sequence[int], k: int) -> int | None:
    """First 1-based rank within ``k`` where the lists differ; ``None`` if they agree."""
    top_a, top_b = a[:k], b[:k]
    for rank, (x, y) in enumerate(zip(top_a, top_b, strict=False), start=1):
        if x != y:
            return rank
    if len(top_a) != len(top_b):
        return min(len(top_a), len(top_b)) + 1
    return None


def compare_query(
    index: int, a: ScoredRanking, b: ScoredRanking, ks: Sequence[int], *, tie_eps: float
) -> QueryComparison:
    """Compare one query's two rankings."""
    k_max = max(ks)
    top_a, top_b = a.ids[:k_max], b.ids[:k_max]
    rank_a = {cid: r for r, cid in enumerate(top_a)}
    rank_b = {cid: r for r, cid in enumerate(top_b)}
    shared = set(rank_a) & set(rank_b)
    shift = max((abs(rank_a[c] - rank_b[c]) for c in shared), default=0)
    score_a = dict(zip(a.ids, a.scores, strict=True))
    score_b = dict(zip(b.ids, b.scores, strict=True))
    common = set(score_a) & set(score_b)
    delta = max((abs(score_a[c] - score_b[c]) for c in common), default=None)
    explained = set(top_a) == set(top_b)
    if explained:
        ordered = list(top_a)
        for i, x in enumerate(ordered):
            for y in ordered[i + 1 :]:
                if rank_b[x] > rank_b[y] and abs(score_a[x] - score_a[y]) >= tie_eps:
                    explained = False
                    break
            if not explained:
                break
    return QueryComparison(
        query_index=index,
        overlap={str(k): overlap_at(a.ids, b.ids, k) for k in ks},
        first_divergence=first_divergence(a.ids, b.ids, k_max),
        max_rank_shift=shift,
        max_score_delta=delta,
        tie_explained=explained,
    )


def compare_runs(
    first: Sequence[ScoredRanking],
    second: Sequence[ScoredRanking],
    ks: Sequence[int] = DEFAULT_KS,
    *,
    tie_eps: float = TIE_EPS,
    score_tol: float = SCORE_TOL,
) -> RunComparison:
    """Compare two runs query by query; both must rank the same number of queries."""
    if len(first) != len(second):
        raise ValidationError(f"the runs rank {len(first)} and {len(second)} queries")
    if not first:
        raise ValidationError("the runs rank no queries")
    if not ks or any(k < 1 for k in ks) or len(set(ks)) != len(ks):
        raise ValidationError(f"cut-offs must be distinct positive integers, got {list(ks)}")
    if tie_eps < 0 or score_tol < 0:
        raise ValidationError("tolerances must be non-negative")
    ordered = tuple(sorted(ks))
    rows = tuple(
        compare_query(i, a, b, ordered, tie_eps=tie_eps)
        for i, (a, b) in enumerate(zip(first, second, strict=True))
    )
    identical = sum(1 for row in rows if row.first_divergence is None)
    diverged = tuple(row.query_index for row in rows if not row.tie_explained)
    deltas = [row.max_score_delta for row in rows if row.max_score_delta is not None]
    max_delta = max(deltas) if deltas else None
    passed = not diverged and (max_delta is None or max_delta <= score_tol)
    return RunComparison(
        queries=len(rows),
        ks=ordered,
        tie_eps=tie_eps,
        score_tol=score_tol,
        mean_overlap={
            str(k): sum(row.overlap[str(k)] for row in rows) / len(rows) for k in ordered
        },
        min_overlap={str(k): min(row.overlap[str(k)] for row in rows) for k in ordered},
        identical_top_k=identical,
        tie_explained=len(rows) - len(diverged),
        diverged_queries=diverged,
        max_score_delta=max_delta,
        passed=passed,
        per_query=rows,
    )


# --------------------------------------------------------------------------
# CLI: all file I/O lives here.
# --------------------------------------------------------------------------

app = typer.Typer(name="validate", help="Reproducibility checks.", no_args_is_help=True)


@app.command("runs")
def runs(
    first: Annotated[Path, typer.Option("--first", help="Reference run (e.g. the GPU run).")],
    second: Annotated[Path, typer.Option("--second", help="Run to check (e.g. the CPU run).")],
    out: Annotated[Path, typer.Option("--out", help="New output directory.")],
    k: Annotated[str, typer.Option("--k", help="Cut-offs, comma-separated.")] = "1,5,10",
    tie_eps: Annotated[
        float, typer.Option("--tie-eps", help="Score gap below which a swap is a near-tie.")
    ] = TIE_EPS,
    score_tol: Annotated[
        float, typer.Option("--score-tol", help="Largest allowed score difference.")
    ] = SCORE_TOL,
    created_at: Annotated[
        str | None, typer.Option("--created-at", help="Fixed manifest timestamp.")
    ] = None,
) -> None:
    """Compare two runs of the same model and queries; exit 5 if they diverge.

    Both runs must be the same channel (same run type and model, same model
    revision) over the same query table; their devices may differ. Writes
    ``validation.json`` and ``run_manifest.json``. Exit 1 on unusable input.
    """
    from omopmap.artifacts.manifest import sha256_file
    from omopmap.artifacts.models import utc_timestamp
    from omopmap.meds.writer import OutputExistsError, prepare_output
    from omopmap.retrieval.manifest import RUN_MANIFEST_NAME, RunManifestError, build_run_manifest
    from omopmap.retrieval.rankings import RankingsFormatError
    from omopmap.runs import RunError, load_run
    from omopmap.stats import StatsError, parse_ks

    try:
        ks = parse_ks(k)
        run_a, run_b = load_run(first), load_run(second)
        ma, mb = run_a.manifest, run_b.manifest
        if ma.channel() != mb.channel():
            raise ValidationError(f"different channels: {ma.channel()} vs {mb.channel()}")
        if ma.inputs_by_name().get("queries") != mb.inputs_by_name().get("queries"):
            raise ValidationError("the runs ranked different query tables")
        ea, eb = ma.encoder, mb.encoder
        if (ea is None) != (eb is None) or (
            ea is not None
            and eb is not None
            and (ea.model_id, ea.revision) != (eb.model_id, eb.revision)
        ):
            raise ValidationError("the runs used different encoders")
        result = compare_runs(
            run_a.rankings, run_b.rankings, ks, tie_eps=tie_eps, score_tol=score_tol
        )
        prepare_output(out)
        (out / VALIDATION_NAME).write_text(result.to_json(), encoding="utf-8")
        devices = {
            "first_device": "" if ea is None else ea.device,
            "second_device": "" if eb is None else eb.device,
        }
        manifest = build_run_manifest(
            "validate",
            created_at=created_at or utc_timestamp(),
            parameters={
                "k": ",".join(str(x) for x in result.ks),
                "passed": result.passed,
                "queries": result.queries,
                "score_tol": score_tol,
                "tie_eps": tie_eps,
                **devices,
            },
            inputs={"first": run_a.rankings_sha256, "second": run_b.rankings_sha256},
            outputs={VALIDATION_NAME: sha256_file(out / VALIDATION_NAME)},
        )
        (out / RUN_MANIFEST_NAME).write_text(manifest.to_json(), encoding="utf-8")
    except (
        ValidationError,
        StatsError,
        RunError,
        RunManifestError,
        RankingsFormatError,
        OutputExistsError,
        OSError,
    ) as exc:
        typer.echo(f"{type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(1) from exc
    k_max = str(result.ks[-1])
    typer.echo(
        f"{result.queries} queries; identical top-{k_max}: {result.identical_top_k}; "
        f"tie-explained: {result.tie_explained}; mean overlap@{k_max} "
        f"{result.mean_overlap[k_max]:.4f}; max score delta {result.max_score_delta}"
    )
    if not result.passed:
        typer.echo(
            f"DIVERGED: {len(result.diverged_queries)} query(ies) beyond tolerance", err=True
        )
        raise typer.Exit(DIVERGED_EXIT)
