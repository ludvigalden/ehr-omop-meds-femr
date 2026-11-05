"""``omopmap clmbr``: import, verify and measure coverage of the CLMBR dictionary.

Exit codes: 0 success; 1 an artifact or contract check failed; 2 the pinned
dictionary could not be obtained or did not match its pin.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import polars as pl
import typer

from omopmap.artifacts.contract import CONCEPT_SET_PARQUET, MANIFEST_NAME
from omopmap.artifacts.errors import ArtifactError
from omopmap.artifacts.io import read_contract_parquet, write_parquet
from omopmap.artifacts.manifest import build_manifest, sha256_file, write_manifest
from omopmap.artifacts.models import ConceptSetSummary, InputRecord
from omopmap.artifacts.schemas import CONCEPT_SET
from omopmap.artifacts.validate import check_concept_set, validate_artifact
from omopmap.clmbr import kdl
from omopmap.clmbr.coverage import DEFAULT_MIN_EVENTS, DEFAULT_TOP_N, coverage_report
from omopmap.clmbr.importer import ClmbrConceptSet, build_concept_set, concept_set_frame, enrich
from omopmap.clmbr.model import load_dictionary
from omopmap.clmbr.source import (
    Pins,
    ResolvedDictionary,
    SourceError,
    load_pins,
    resolve_hf,
    resolve_local,
)
from omopmap.meds.convert import read_table
from omopmap.meds.read import read_extract
from omopmap.meds.writer import prepare_output

app = typer.Typer(
    name="clmbr", help="Import and measure the CLMBR dictionary.", no_args_is_help=True
)

SOURCE_EXIT = 2


def _fail(exc: Exception, code: int = 1) -> typer.Exit:
    typer.echo(f"{type(exc).__name__}: {exc}", err=True)
    return typer.Exit(code)


def _resolve(
    model_dir: Path | None, hf_repo: str | None, revision: str | None
) -> ResolvedDictionary:
    if (model_dir is None) == (hf_repo is None):
        raise typer.BadParameter("give exactly one of --model-dir or --hf-repo")
    if revision is not None and hf_repo is None:
        raise typer.BadParameter("--revision only applies to --hf-repo")
    pins = load_pins()
    if model_dir is not None:
        return resolve_local(model_dir, pins)
    assert hf_repo is not None
    return resolve_hf(hf_repo, revision, pins)


def _check_concept_list_pin(concept_set: ClmbrConceptSet, pins: Pins) -> None:
    """The second, content-level pin: both routes must reproduce this concept set."""
    digest = concept_set.concept_list_sha256()
    if digest != pins.concept_list_sha256:
        raise SourceError(
            f"concept-list hash {digest} differs from the pinned "
            f"{pins.concept_list_sha256}: the dictionary contents or the import "
            "rules changed; review and re-pin together"
        )


def _write_concept_set_artifact(
    root: Path,
    frame: pl.DataFrame,
    source: ResolvedDictionary,
    athena: Path | None,
) -> None:
    prepare_output(root)
    write_parquet(frame, root / CONCEPT_SET_PARQUET, CONCEPT_SET)
    summary = check_concept_set(root, source.sha256)
    inputs = [InputRecord(name="dictionary", sha256=source.sha256)]
    if athena is not None:
        inputs.append(InputRecord(name=f"vocabulary:{athena.name}", sha256=sha256_file(athena)))
    write_manifest(root, build_manifest(root, summary, inputs))


@app.command("import")
def import_(
    out: Annotated[Path, typer.Option("--out", help="KDL directory, e.g. vocab/clmbr.")],
    model_dir: Annotated[
        Path | None,
        typer.Option("--model-dir", help="Local checkpoint directory (offline; hash-checked)."),
    ] = None,
    hf_repo: Annotated[
        str | None,
        typer.Option("--hf-repo", help="Hugging Face repo id (pinned revision; needs HF_TOKEN)."),
    ] = None,
    revision: Annotated[
        str | None, typer.Option("--revision", help="Must equal the pinned revision.")
    ] = None,
    athena: Annotated[
        Path | None,
        typer.Option("--athena", help="Optional Athena CONCEPT table for OMOP concept ids."),
    ] = None,
    concept_set: Annotated[
        Path | None,
        typer.Option("--concept-set", help="Optional new directory for concept_set.parquet."),
    ] = None,
) -> None:
    """Import the pinned dictionary: KDL per vocabulary plus counts-and-hashes manifest."""
    try:
        source = _resolve(model_dir, hf_repo, revision)
        concepts = build_concept_set(load_dictionary(source.path))
        _check_concept_list_pin(concepts, load_pins())
    except SourceError as exc:
        raise _fail(exc, SOURCE_EXIT) from exc
    except ArtifactError as exc:
        raise _fail(exc) from exc
    try:
        athena_sha = None
        if athena is not None:
            concepts = enrich(concepts, read_table(athena))
            athena_sha = sha256_file(athena)
        manifest = kdl.build_manifest(concepts, source)
        kdl.write_kdl_dir(out, kdl.render_files(concepts, source, athena_sha), manifest)
        kdl.verify_kdl(out / kdl.MANIFEST_JSON)
        if concept_set is not None:
            _write_concept_set_artifact(concept_set, concept_set_frame(concepts), source, athena)
    except (ArtifactError, ValueError) as exc:
        raise _fail(exc) from exc
    unresolved = concepts.unresolved()
    typer.echo(
        f"wrote {out}: {manifest.total} concepts in {len(manifest.vocabulary_counts)} "
        f"vocabularies, {manifest.care_site_excluded} CARE_SITE refs excluded"
    )
    if concepts.enriched:
        typer.echo(f"unresolved in the Athena table: {len(unresolved)} of {manifest.total}")


@app.command("verify")
def verify(
    manifest: Annotated[Path, typer.Option("--manifest", help="vocab/clmbr/manifest.json")],
) -> None:
    """Re-derive counts and the concept-list hash from the KDL files; compare with the manifest."""
    try:
        checked = kdl.verify_kdl(manifest)
    except ArtifactError as exc:
        raise _fail(exc) from exc
    typer.echo(f"verified {manifest}: {checked.total} concepts, {checked.concept_list_sha256}")


@app.command("coverage")
def coverage(
    extract: Annotated[Path, typer.Option("--extract", help="Verified MEDS extract directory.")],
    concept_set: Annotated[
        Path, typer.Option("--concept-set", help="Concept-set artifact directory.")
    ],
    out: Annotated[Path, typer.Option("--out", help="New coverage report JSON file.")],
    top_n: Annotated[int, typer.Option("--top-n", min=0)] = DEFAULT_TOP_N,
    min_events: Annotated[int, typer.Option("--min-events", min=1)] = DEFAULT_MIN_EVENTS,
) -> None:
    """Share of the extract's distinct codes present in the CLMBR concept set."""
    try:
        if out.exists():
            raise ValueError(f"{out} exists; coverage reports are never overwritten")
        manifest = validate_artifact(concept_set)
        if not isinstance(manifest.summary, ConceptSetSummary):
            raise ValueError(f"{concept_set} is not a CLMBR concept-set artifact")
        codes = read_contract_parquet(
            concept_set / CONCEPT_SET_PARQUET, CONCEPT_SET, CONCEPT_SET_PARQUET
        ).get_column("code")
        extract_manifest, events = read_extract(extract)
        report = coverage_report(
            events,
            codes.to_list(),
            extract_manifest_sha256=sha256_file(extract / MANIFEST_NAME),
            concept_list_sha256=manifest.summary.concept_list_sha256,
            created_at=extract_manifest.created_at,
            top_n=top_n,
            min_events=min_events,
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report.to_json(), encoding="utf-8")
    except (ArtifactError, ValueError) as exc:
        raise _fail(exc) from exc
    typer.echo(
        f"wrote {out}: {report.recognised_codes} of {report.distinct_codes} distinct codes "
        f"recognised (share {report.share:.4f})"
    )
