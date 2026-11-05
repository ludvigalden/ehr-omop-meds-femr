"""``omopmap artifacts``: verify artifacts against the contract."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from omopmap.artifacts.errors import ArtifactError
from omopmap.artifacts.validate import validate_artifact

app = typer.Typer(
    name="artifacts",
    help="Verify model-ready artifacts against the artifact contract.",
    no_args_is_help=True,
)


@app.command("verify")
def verify(
    root: Annotated[
        Path, typer.Argument(help="Artifact directory holding artifact_manifest.json.")
    ],
    extract: Annotated[
        Path | None,
        typer.Option("--extract", help="The labelled MEDS extract (required for label sets)."),
    ] = None,
) -> None:
    """Check file hashes, layout, schemas and content; exit 1 on any violation."""
    try:
        manifest = validate_artifact(root, extract_root=extract)
    except ArtifactError as exc:
        typer.echo(f"{type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(
        f"ok: {manifest.artifact_type} contract {manifest.contract_version}, "
        f"{len(manifest.files)} files verified"
    )
