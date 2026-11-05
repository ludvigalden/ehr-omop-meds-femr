"""``omopmap labels``: build label sets over a verified MEDS extract."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from omopmap.artifacts.errors import ArtifactError
from omopmap.labels.windows import WindowTask, build_labels
from omopmap.labels.writer import write_label_set
from omopmap.meds.read import read_extract

app = typer.Typer(name="labels", help="Build prediction times and labels.", no_args_is_help=True)


@app.command("build")
def build(
    extract: Annotated[Path, typer.Option("--extract", help="Verified MEDS extract directory.")],
    task: Annotated[str, typer.Option("--task", help="Task name; also the output directory name.")],
    index: Annotated[
        list[str], typer.Option("--index", help="Index event code or `prefix*`; repeatable.")
    ],
    outcome: Annotated[
        list[str], typer.Option("--outcome", help="Outcome code or `prefix*`; repeatable.")
    ],
    window_days: Annotated[
        int, typer.Option("--window-days", help="Outcome window after the index event.")
    ],
    out: Annotated[
        Path, typer.Option("--out", help="New label-set directory named after the task.")
    ],
    first_index_only: Annotated[bool, typer.Option("--first-index-only")] = False,
    exclude_prior_outcome: Annotated[
        bool, typer.Option("--exclude-prior-outcome/--keep-prior-outcome")
    ] = True,
    min_age_years: Annotated[int, typer.Option("--min-age-years")] = 0,
) -> None:
    """Binary task: does an outcome event follow an index event within the window?"""
    try:
        spec = WindowTask(
            name=task,
            index_codes=tuple(index),
            outcome_codes=tuple(outcome),
            window_days=window_days,
            first_index_only=first_index_only,
            exclude_prior_outcome=exclude_prior_outcome,
            min_age_years=min_age_years,
        )
        extract_manifest, events = read_extract(extract)
        manifest = write_label_set(
            out,
            build_labels(events, spec),
            spec.definition(),
            extract_root=extract,
            created_at=extract_manifest.created_at,
        )
    except (ArtifactError, ValueError) as exc:
        typer.echo(f"{type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"wrote {out}: {manifest.summary.model_dump_json()}")
