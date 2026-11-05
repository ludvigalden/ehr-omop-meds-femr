"""Command-line interface for omopmap.

All subprocess-facing I/O lives here; the library modules stay pure.
Each command group is a Typer sub-app registered below.
"""

from __future__ import annotations

from typing import Annotated

import typer

from omopmap import __version__
from omopmap.adjudication import app as adjudicate_app
from omopmap.artifacts.cli import app as artifacts_app
from omopmap.calibration import app as calibrate_app
from omopmap.clmbr.cli import app as clmbr_app
from omopmap.dictionary import app as dictionary_app
from omopmap.finetune import finetune as finetune_command
from omopmap.labels.cli import app as labels_app
from omopmap.meds.cli import app as meds_app
from omopmap.metrics import app as metrics_app
from omopmap.retrieval.cli import app as retrieve_app
from omopmap.retrieval.cli import gate as gate_command
from omopmap.stats import app as stats_app
from omopmap.translate import app as translate_app
from omopmap.validate_gpu_cpu import app as validate_app

app = typer.Typer(
    name="omopmap",
    help="Map source terms to OMOP standard concepts and build model-ready artifacts.",
    no_args_is_help=True,
)
app.add_typer(artifacts_app)
app.add_typer(meds_app)
app.add_typer(labels_app)
app.add_typer(clmbr_app)
app.add_typer(retrieve_app)
app.add_typer(dictionary_app)
app.add_typer(translate_app)
app.add_typer(stats_app)
app.add_typer(metrics_app)
app.add_typer(validate_app)
app.add_typer(calibrate_app)
app.add_typer(adjudicate_app)
app.command("gate")(gate_command)
app.command("finetune")(finetune_command)


def _print_version(value: bool) -> None:
    if value:
        typer.echo(f"omopmap {__version__}")
        raise typer.Exit


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            callback=_print_version,
            is_eager=True,
            help="Print the omopmap version and exit.",
        ),
    ] = None,
) -> None:
    """Map source terms to OMOP standard concepts."""


if __name__ == "__main__":
    app()
