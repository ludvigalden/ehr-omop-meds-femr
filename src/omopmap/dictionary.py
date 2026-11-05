"""Eligible-target dictionary builder.

The retrieval target space is the subset of an OMOP vocabulary's CONCEPT
table that a source term may be mapped to: the standard, currently valid
concepts of the vocabularies and domains in scope. Building it once as a
frozen target table gives every downstream step the same candidates, and
``retrieve index`` can consume the table directly instead of re-filtering
a raw CONCEPT export.

The eligibility contract, as a pure function of the table:

- ``standard_concept`` is ``S``: only standard concepts are mapping
  targets. OMOP reserves the standard space as the interoperability
  layer -- source terms map to one standard concept per clinical meaning,
  never to a vocabulary-local classification concept.
- ``invalid_reason`` is null or empty and ``valid_end_date`` is not in
  the past: the concept is still current. Retired concepts would map
  source terms to codes that no longer resolve.
- the vocabulary/domain scope: a configurable mapping from vocabulary id
  to an optional domain restriction. The default scope is the experiment
  contract: the clinical vocabularies (SNOMED, RxNorm, LOINC, UCUM)
  restricted to the clinical metadata domains, plus the OpenStreetMap
  geography vocabulary.

The CLI filters inside a lazy polars plan collected with the streaming
engine, so only the eligible rows are ever materialised, never the full
CONCEPT export. Athena's ``YYYYMMDD`` end dates and ISO ``YYYY-MM-DD``
end dates are both accepted.

The source experiment's real-data counts (811,546 eligible targets out of
6,587,793 streamed rows) are recorded in
``context/reference/dictionary-eligibility.md`` with attribution; they are
facts about that run, never defaults of this code.
"""

from __future__ import annotations

from datetime import UTC, date
from datetime import datetime as dt
from pathlib import Path
from typing import Annotated, Self

import polars as pl
import typer
from pydantic import BaseModel, ConfigDict, model_validator

TARGET_COLUMNS: tuple[str, ...] = (
    "concept_id",
    "concept_name",
    "vocabulary_id",
    "domain_id",
    "concept_code",
)
TARGETS_NAME = "targets.parquet"

REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {
        "concept_id",
        "concept_name",
        "vocabulary_id",
        "domain_id",
        "concept_code",
        "standard_concept",
        "invalid_reason",
        "valid_end_date",
    }
)

#: The experiment's domain scope for the clinical vocabularies.
SCOPE_D_DOMAINS = frozenset(
    {
        "Condition",
        "Device",
        "Drug",
        "Measurement",
        "Metadata",
        "Observation",
        "Procedure",
        "Provider",
        "Spec Anatomic Site",
        "Unit",
    }
)


class DictionaryError(ValueError):
    """The concept table cannot be turned into a target table."""


class DictionaryConfig(BaseModel):
    """Vocabulary-to-domain scope for eligibility.

    A vocabulary maps to ``None`` (every domain) or to a set of domain
    ids. An empty scope admits nothing.
    """

    model_config = ConfigDict(frozen=True)

    scope: dict[str, frozenset[str] | None]

    @model_validator(mode="after")
    def _vocabularies_non_empty(self) -> Self:
        if not self.scope:
            raise ValueError("the vocabulary scope must name at least one vocabulary")
        return self

    def admits(self, vocabulary_id: str, domain_id: str) -> bool:
        """Is a concept of this vocabulary and domain in scope?"""
        domains = self.scope.get(vocabulary_id)
        return domains is None or domain_id in domains

    def scope_parameter(self) -> str:
        """The scope as a manifest-parameter string."""
        return ";".join(
            f"{vocab}={','.join(sorted(domains)) if domains is not None else '*'}"
            for vocab, domains in sorted(self.scope.items())
        )


def clinical_scope() -> DictionaryConfig:
    """The default experiment scope: clinical vocabularies plus OSM geography.

    SNOMED, RxNorm, LOINC and UCUM are admitted over the clinical metadata
    domains; the OpenStreetMap vocabulary is admitted over Geography.
    """
    return DictionaryConfig(
        scope={
            "SNOMED": SCOPE_D_DOMAINS,
            "RxNorm": SCOPE_D_DOMAINS,
            "LOINC": SCOPE_D_DOMAINS,
            "UCUM": SCOPE_D_DOMAINS,
            "OSM": frozenset({"Geography"}),
        }
    )


def config_from_options(vocabularies: list[str]) -> DictionaryConfig:
    """Parse ``--vocabulary VOC`` / ``--vocabulary VOC=DOM1,DOM2`` options."""
    scope: dict[str, frozenset[str] | None] = {}
    for entry in vocabularies:
        vocab, sep, domains = entry.partition("=")
        vocab = vocab.strip()
        if not vocab:
            raise DictionaryError(f"empty vocabulary id in --vocabulary {entry!r}")
        if sep:
            parsed = frozenset(d.strip() for d in domains.split(",") if d.strip())
            if not parsed:
                raise DictionaryError(f"--vocabulary {entry!r} names no domain")
            scope[vocab] = parsed
        else:
            scope[vocab] = None
    return DictionaryConfig(scope=scope)


def require_columns(frame: pl.DataFrame) -> None:
    """Fail closed unless every eligibility column is present."""
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise DictionaryError(f"the concept table lacks columns {missing}")


def end_date_expression(as_of: date) -> pl.Expr:
    """The currency test as a polars expression over ``valid_end_date``.

    Athena exports concept dates as ``YYYYMMDD``; other exports use ISO
    ``YYYY-MM-DD``. Both compare correctly as strings once the separators
    are stripped, because the remaining eight digits sort chronologically.
    A null or empty end date is never current; a value that is empty after
    separator stripping likewise fails the comparison.
    """
    return pl.col("valid_end_date").fill_null("").str.replace_all("[^0-9]", "").str.slice(
        0, 8
    ) >= pl.lit(as_of.strftime("%Y%m%d"))


def scope_expression(config: DictionaryConfig) -> pl.Expr:
    """The vocabulary/domain scope as a polars expression."""
    in_scope = pl.lit(False, dtype=pl.Boolean)
    for vocab, domains in sorted(config.scope.items()):
        condition = pl.col("vocabulary_id") == pl.lit(vocab)
        if domains is not None:
            condition = condition & pl.col("domain_id").is_in(sorted(domains))
        in_scope = in_scope | condition
    return in_scope


def eligibility_expression(config: DictionaryConfig, *, as_of: date) -> pl.Expr:
    """The whole eligibility contract as one expression: standard, current, in scope.

    The same expression drives the eager ``filter_eligible`` and the lazy,
    streamed pipeline of ``dictionary build``, so both apply one contract.
    """
    return (
        (pl.col("standard_concept").fill_null("") == "S")
        & (pl.col("invalid_reason").fill_null("") == "")
        & end_date_expression(as_of)
        & scope_expression(config)
    )


def filter_eligible(frame: pl.DataFrame, config: DictionaryConfig, *, as_of: date) -> pl.DataFrame:
    """The eligible rows: standard, current, in scope. Pure."""
    require_columns(frame)
    return frame.filter(eligibility_expression(config, as_of=as_of))


def finalise_targets(eligible: pl.DataFrame) -> pl.DataFrame:
    """Type, sort and deduplicate already-filtered eligible rows.

    A duplicate concept_id fails closed: OMOP concept ids are unique, and a
    table that violates that is not a usable target space. A concept_id that
    is not a decimal integer fails the cast loudly.
    """
    try:
        targets = eligible.select(
            pl.col("concept_id").cast(pl.Int64),
            *[pl.col(name) for name in TARGET_COLUMNS[1:]],
        ).sort("concept_id")
    except pl.exceptions.PolarsError as exc:
        raise DictionaryError(f"the concept table cannot be cast to target columns: {exc}") from exc
    ids = targets.get_column("concept_id")
    if ids.is_duplicated().any():
        raise DictionaryError("the concept table has duplicate concept_id values")
    return targets


def build_targets(frame: pl.DataFrame, config: DictionaryConfig, *, as_of: date) -> pl.DataFrame:
    """The target table: eligible rows, typed, deduplicated, sorted by concept_id."""
    return finalise_targets(filter_eligible(frame, config, as_of=as_of))


def counts_by_vocabulary(targets: pl.DataFrame) -> dict[str, int]:
    """Eligible-target count per vocabulary, sorted by vocabulary id."""
    counts = targets.group_by("vocabulary_id").agg(pl.len().alias("n")).sort("vocabulary_id")
    return {str(vocab): int(n) for vocab, n in counts.iter_rows()}


# --------------------------------------------------------------------------
# CLI: all file I/O lives here; the functions above are pure.
# --------------------------------------------------------------------------

app = typer.Typer(
    name="dictionary", help="Build the eligible-target dictionary.", no_args_is_help=True
)

VocabularyOpt = Annotated[
    list[str] | None,
    typer.Option(
        "--vocabulary",
        help="Scope entry: VOC (all domains) or VOC=DOM1,DOM2. Default: the clinical scope.",
    ),
]
AsOfOpt = Annotated[
    str | None,
    typer.Option(
        "--as-of",
        help="Validity reference date (ISO). Default: today (UTC). Pin it for byte-stable reruns.",
    ),
]
CreatedAtOpt = Annotated[
    str | None,
    typer.Option("--created-at", help="Fixed manifest timestamp, for byte identity."),
]


def parse_date(value: str | None, option: str) -> date:
    """Parse an ISO date option; the default is today (UTC)."""
    if value is None:
        return dt.now(tz=UTC).date()
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise DictionaryError(f"{option} must be an ISO date, got {value!r}") from exc


def scan_concept_table(path: Path) -> pl.LazyFrame:
    """Lazily scan a CSV/TSV concept table (delimiter sniffed from the header)."""
    with path.open(encoding="utf-8") as handle:
        header = handle.readline()
    if not header:
        raise DictionaryError(f"{path.name} is empty")
    separator = "\t" if "\t" in header else ","
    lazy = pl.scan_csv(
        path,
        separator=separator,
        infer_schema=False,
        quote_char=None if separator == "\t" else '"',
    )
    return lazy.rename({name: name.lower() for name in lazy.collect_schema().names()})


@app.command("build")
def build(
    concepts: Annotated[Path, typer.Option("--concepts", help="OMOP CONCEPT table.")],
    out: Annotated[Path, typer.Option("--out", help="New output directory.")],
    vocabulary: VocabularyOpt = None,
    as_of: AsOfOpt = None,
    created_at: CreatedAtOpt = None,
) -> None:
    """Filter a CONCEPT table down to the eligible targets and write them.

    The output directory receives ``targets.parquet`` and ``run_manifest.json``
    (written last). ``retrieve index`` consumes the parquet table directly.
    """
    from omopmap.artifacts.manifest import sha256_file
    from omopmap.artifacts.models import utc_timestamp
    from omopmap.meds.writer import OutputExistsError, prepare_output
    from omopmap.retrieval.manifest import (
        RUN_MANIFEST_NAME,
        ParamValue,
        build_run_manifest,
    )

    try:
        config = clinical_scope() if vocabulary is None else config_from_options(vocabulary)
        as_of_date = parse_date(as_of, "--as-of")
        prepare_output(out)
        lazy = scan_concept_table(concepts)
        missing = sorted(REQUIRED_COLUMNS - set(lazy.collect_schema().names()))
        if missing:
            raise DictionaryError(f"the concept table lacks columns {missing}")
        # Filter and project inside the lazy plan, so the streaming engine
        # never materialises the full CONCEPT export -- only eligible rows.
        eligible = (
            lazy.filter(eligibility_expression(config, as_of=as_of_date))
            .select(*TARGET_COLUMNS)
            .collect(engine="streaming")
        )
        targets = finalise_targets(eligible)
        counts = counts_by_vocabulary(targets)
        targets.write_parquet(out / TARGETS_NAME)
        parameters: dict[str, ParamValue] = {
            "as_of": as_of_date.isoformat(),
            "eligible": len(targets),
            "scope": config.scope_parameter(),
            **{f"vocabulary:{name}": count for name, count in counts.items()},
        }
        manifest = build_run_manifest(
            "dictionary",
            created_at=created_at or utc_timestamp(),
            parameters=parameters,
            inputs={"concepts": sha256_file(concepts)},
            outputs={TARGETS_NAME: sha256_file(out / TARGETS_NAME)},
        )
        (out / RUN_MANIFEST_NAME).write_text(manifest.to_json(), encoding="utf-8")
        typer.echo(f"wrote {len(targets)} targets to {out}")
    except (DictionaryError, OutputExistsError, OSError) as exc:
        typer.echo(f"{type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(1) from exc
