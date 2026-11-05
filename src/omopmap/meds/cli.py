"""``omopmap meds``: produce MEDS extracts (synthetic or from OMOP CDM tables)."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Annotated

import typer

from omopmap.artifacts.errors import ArtifactError
from omopmap.artifacts.manifest import sha256_file
from omopmap.artifacts.models import ArtifactManifest, InputRecord, utc_timestamp
from omopmap.meds.convert import (
    ConceptTable,
    ConvertConfig,
    convert_tables,
    read_omop_dir,
    read_table,
)
from omopmap.meds.splits import SplitConfig
from omopmap.meds.synth import SynthConfig, generate_events, synth_created_at
from omopmap.meds.writer import write_extract

app = typer.Typer(name="meds", help="Produce MEDS extracts.", no_args_is_help=True)


def _done(manifest: ArtifactManifest, out: Path) -> None:
    summary = manifest.summary
    typer.echo(f"wrote {out}: {summary.model_dump_json()}")


def _fail(exc: Exception) -> typer.Exit:
    typer.echo(f"{type(exc).__name__}: {exc}", err=True)
    return typer.Exit(1)


def _parse_kv(values: list[str], what: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in values:
        key, sep, value = item.partition("=")
        if not sep or not key or not value:
            raise typer.BadParameter(f"{what} must be TABLE=VOCABULARY, got {item!r}")
        result[key] = value
    return result


@app.command("synth")
def synth(
    out: Annotated[
        Path, typer.Option("--out", help="New extract directory (its name is the dataset name).")
    ],
    subjects: Annotated[int, typer.Option("--subjects", "--n", help="Number of subjects.")] = 1000,
    seed: Annotated[int, typer.Option("--seed", help="Seed for the cohort and the splits.")] = 0,
    as_of: Annotated[
        str, typer.Option("--as-of", help="Last event date, YYYY-MM-DD.")
    ] = "2024-12-31",
    subjects_per_shard: Annotated[int, typer.Option("--subjects-per-shard")] = 10_000,
) -> None:
    """Write a seeded synthetic cohort; same arguments, same bytes."""
    try:
        config = SynthConfig(subjects=subjects, seed=seed, as_of=date.fromisoformat(as_of))
        manifest = write_extract(
            out,
            generate_events(config),
            split_config=SplitConfig(seed=seed, subjects_per_shard=subjects_per_shard),
            dataset_version=f"synthetic-seed-{seed}",
            created_at=synth_created_at(config),
            inputs=[InputRecord(name=f"synthetic:{seed}", sha256=config.sha256())],
        )
    except (ArtifactError, ValueError) as exc:
        raise _fail(exc) from exc
    _done(manifest, out)


@app.command("convert")
def convert(
    omop: Annotated[
        Path, typer.Option("--omop", help="Directory of OMOP CDM tables (.parquet/.csv/.tsv).")
    ],
    concept: Annotated[
        Path, typer.Option("--concept", help="Athena CONCEPT table (.csv/.tsv/.parquet).")
    ],
    out: Annotated[
        Path, typer.Option("--out", help="New extract directory (its name is the dataset name).")
    ],
    concept_ancestor: Annotated[
        Path | None,
        typer.Option(
            "--concept-ancestor", help="Optional Athena CONCEPT_ANCESTOR for parent codes."
        ),
    ] = None,
    source_vocabulary: Annotated[
        list[str] | None,
        typer.Option(
            "--source-vocabulary",
            help="TABLE=VOCABULARY for source-value fallback codes; repeatable.",
        ),
    ] = None,
    timezone: Annotated[
        str, typer.Option("--timezone", help="Zone aware input timestamps are converted to.")
    ] = "UTC",
    seed: Annotated[int, typer.Option("--seed", help="Split seed.")] = 0,
    train: Annotated[float, typer.Option("--train")] = 0.8,
    tuning: Annotated[float, typer.Option("--tuning")] = 0.1,
    held_out: Annotated[float, typer.Option("--held-out")] = 0.1,
    subjects_per_shard: Annotated[int, typer.Option("--subjects-per-shard")] = 10_000,
    dataset_version: Annotated[str, typer.Option("--dataset-version")] = "1",
    created_at: Annotated[
        str | None, typer.Option("--created-at", help="UTC YYYY-MM-DDTHH:MM:SSZ; default now.")
    ] = None,
    descriptions: Annotated[
        bool,
        typer.Option(
            "--descriptions/--no-descriptions", help="Copy concept names into codes.parquet."
        ),
    ] = False,
) -> None:
    """Convert OMOP CDM tables to a MEDS extract with seeded subject splits."""
    try:
        config = ConvertConfig(
            source_vocabularies=_parse_kv(source_vocabulary or [], "--source-vocabulary"),
            timezone=timezone,
        )
        split_config = SplitConfig(
            train=train,
            tuning=tuning,
            held_out=held_out,
            seed=seed,
            subjects_per_shard=subjects_per_shard,
        )
        found = read_omop_dir(omop)
        concepts = ConceptTable.from_frame(read_table(concept))
        events, report = convert_tables({t: f for t, (_, f) in found.items()}, concepts, config)
        codes = set(events.get_column("code").unique().to_list())
        inputs = [
            InputRecord(name=f"omop:{t}", sha256=sha256_file(p)) for t, (p, _) in found.items()
        ]
        inputs.append(InputRecord(name=f"vocabulary:{concept.name}", sha256=sha256_file(concept)))
        parents = None
        if concept_ancestor is not None:
            parents = concepts.parents(read_table(concept_ancestor), codes)
            inputs.append(
                InputRecord(
                    name=f"vocabulary:{concept_ancestor.name}", sha256=sha256_file(concept_ancestor)
                )
            )
        manifest = write_extract(
            out,
            events,
            split_config=split_config,
            dataset_version=dataset_version,
            created_at=created_at or utc_timestamp(),
            inputs=inputs,
            descriptions=concepts.descriptions(codes) if descriptions else None,
            parents=parents,
        )
    except (ArtifactError, ValueError) as exc:
        raise _fail(exc) from exc
    typer.echo(f"conversion: {report.model_dump_json()}")
    _done(manifest, out)
