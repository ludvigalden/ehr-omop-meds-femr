"""OMOP CDM to MEDS conversion.

Reads OMOP CDM tables (CSV or Parquet, one file per table) and an Athena
``CONCEPT`` table, and produces an event frame for :mod:`omopmap.meds.writer`.

Code resolution per event, first match wins:

1. the standard concept column (``<table>_concept_id``), when non-zero and
   present in ``CONCEPT``, as ``<vocabulary_id>/<concept_code>``;
2. the source concept column (``<table>_source_concept_id``), same rule;
3. the source value column, as ``<source_vocabulary>/<value>``, where the
   source vocabulary is a per-table parameter (for example a local Swedish
   code system).

Concept ``0`` ("no matching concept") never becomes a code. Rows that resolve
to no code, rows without a usable time, rows of persons absent from ``person``
and timed rows before the person's birth are dropped and counted in the
:class:`ConversionReport`; nothing is dropped silently.

Time: the ``*_datetime`` column when set, else the ``*_date`` column at
midnight. Timezone-aware inputs are converted to ``timezone`` (UTC unless
given) and made naive. Birth comes from ``birth_datetime``, else from
``year_of_birth`` with ``month_of_birth`` / ``day_of_birth`` defaulting to 1.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from omopmap.artifacts.contract import BIRTH_CODE, DEATH_CODE
from omopmap.artifacts.errors import ArtifactError, MissingFileError
from omopmap.meds.events import EVENT_SCHEMA, as_event_frame, empty_events


class ConversionError(ArtifactError):
    """An OMOP input cannot be converted (missing column, unparsable value)."""


@dataclass(frozen=True, slots=True)
class EventSpec:
    """How one OMOP table (or column group) becomes events."""

    table: str
    concept: str
    source_concept: str | None
    source_value: str | None
    datetime: str | None
    date: str | None
    numeric: str | None = None
    text: str | None = None
    value_concept: str | None = None


#: Person columns that become static (untimed) demographic events.
PERSON_SPECS: Final = (
    EventSpec("person", "gender_concept_id", "gender_source_concept_id", None, None, None),
    EventSpec("person", "race_concept_id", "race_source_concept_id", None, None, None),
    EventSpec("person", "ethnicity_concept_id", "ethnicity_source_concept_id", None, None, None),
)

#: Timed event tables.
EVENT_SPECS: Final = (
    EventSpec(
        "visit_occurrence",
        "visit_concept_id",
        "visit_source_concept_id",
        "visit_source_value",
        "visit_start_datetime",
        "visit_start_date",
    ),
    EventSpec(
        "condition_occurrence",
        "condition_concept_id",
        "condition_source_concept_id",
        "condition_source_value",
        "condition_start_datetime",
        "condition_start_date",
    ),
    EventSpec(
        "drug_exposure",
        "drug_concept_id",
        "drug_source_concept_id",
        "drug_source_value",
        "drug_exposure_start_datetime",
        "drug_exposure_start_date",
    ),
    EventSpec(
        "procedure_occurrence",
        "procedure_concept_id",
        "procedure_source_concept_id",
        "procedure_source_value",
        "procedure_datetime",
        "procedure_date",
    ),
    EventSpec(
        "measurement",
        "measurement_concept_id",
        "measurement_source_concept_id",
        "measurement_source_value",
        "measurement_datetime",
        "measurement_date",
        numeric="value_as_number",
        value_concept="value_as_concept_id",
    ),
    EventSpec(
        "observation",
        "observation_concept_id",
        "observation_source_concept_id",
        "observation_source_value",
        "observation_datetime",
        "observation_date",
        numeric="value_as_number",
        text="value_as_string",
        value_concept="value_as_concept_id",
    ),
    EventSpec(
        "death",
        "cause_concept_id",
        "cause_source_concept_id",
        "cause_source_value",
        "death_datetime",
        "death_date",
    ),
)

TABLES: Final = ("person", *(s.table for s in EVENT_SPECS))


class ConvertConfig(BaseModel):
    """Conversion parameters; site-specific assumptions are explicit here."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_vocabularies: dict[str, str] = Field(
        default_factory=dict,
        description="Vocabulary id used for source-value fallback codes, per table; "
        "tables without an entry use `<table>_source`.",
    )
    timezone: str = Field(
        default="UTC",
        description="Zone that timezone-aware input timestamps are converted to before "
        "they are made naive.",
    )

    def source_vocabulary(self, table: str) -> str:
        return self.source_vocabularies.get(table, f"{table}_source")


class ConversionReport(BaseModel):
    """Counts of what the conversion kept and dropped."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    subjects: int = 0
    events_written: int = 0
    by_standard_concept: int = 0
    by_source_concept: int = 0
    by_source_value: int = 0
    dropped_no_code: int = 0
    dropped_no_time: int = 0
    dropped_unknown_person: int = 0
    dropped_before_birth: int = 0
    dropped_no_birth: int = 0


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def find_table(directory: Path, table: str) -> Path | None:
    """The file for ``table`` in ``directory`` (``.parquet``, ``.csv`` or ``.tsv``)."""
    found = [
        directory / f"{name}{suffix}"
        for name in (table, table.upper())
        for suffix in (".parquet", ".csv", ".tsv")
        if (directory / f"{name}{suffix}").is_file()
    ]
    if len(found) > 1:
        raise ConversionError(f"several files for table {table}: {[p.name for p in found]}")
    return found[0] if found else None


def read_table(path: Path) -> pl.DataFrame:
    """Read a CSV/TSV (all columns as strings) or Parquet file; column names lowercased.

    The CSV delimiter is sniffed from the header line (tab or comma); Athena
    exports are tab-separated without quoting.
    """
    if path.suffix == ".parquet":
        frame = pl.read_parquet(path)
    else:
        with path.open(encoding="utf-8") as handle:
            header = handle.readline()
        separator = "\t" if "\t" in header else ","
        frame = pl.read_csv(
            path,
            separator=separator,
            infer_schema=False,
            quote_char=None if separator == "\t" else '"',
        )
    return frame.rename({c: c.lower() for c in frame.columns})


def _require(frame: pl.DataFrame, columns: list[str], what: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ConversionError(f"{what} lacks columns {missing}")


def _blank_to_null(column: str) -> pl.Expr:
    col = pl.col(column)
    return pl.when(col.cast(pl.String).str.strip_chars() == "").then(None).otherwise(col)


def _int_column(frame: pl.DataFrame, column: str | None, what: str) -> pl.Series:
    if column is None or column not in frame.columns:
        return pl.Series(column or "none", [None] * frame.height, dtype=pl.Int64)
    try:
        return frame.select(_blank_to_null(column).cast(pl.Int64, strict=True)).to_series()
    except pl.exceptions.PolarsError as exc:
        raise ConversionError(f"{what}.{column} is not an integer column: {exc}") from exc


def _float_column(frame: pl.DataFrame, column: str | None, what: str) -> pl.Series:
    if column is None or column not in frame.columns:
        return pl.Series("value", [None] * frame.height, dtype=pl.Float64)
    try:
        return frame.select(_blank_to_null(column).cast(pl.Float64, strict=True)).to_series()
    except pl.exceptions.PolarsError as exc:
        raise ConversionError(f"{what}.{column} is not numeric: {exc}") from exc


def _text_column(frame: pl.DataFrame, column: str | None) -> pl.Series:
    if column is None or column not in frame.columns:
        return pl.Series("text", [None] * frame.height, dtype=pl.String)
    return frame.select(_blank_to_null(column).cast(pl.String)).to_series()


def _time_column(frame: pl.DataFrame, column: str | None, timezone: str, what: str) -> pl.Series:
    """A naive ``Datetime("us")`` series; dates become midnight."""
    if column is None or column not in frame.columns:
        return pl.Series("time", [None] * frame.height, dtype=pl.Datetime("us"))
    series = frame.get_column(column)
    try:
        if series.dtype == pl.String:
            cleaned = frame.select(_blank_to_null(column)).to_series()
            if cleaned.drop_nulls().str.contains("[T ]").any():
                parsed = cleaned.str.to_datetime(time_unit="us", strict=True)
            else:
                parsed = cleaned.str.to_date(strict=True).cast(pl.Datetime("us"))
            series = parsed
        if series.dtype == pl.Date:
            return series.cast(pl.Datetime("us"))
        if isinstance(series.dtype, pl.Datetime):
            if series.dtype.time_zone is not None:
                series = series.dt.convert_time_zone(timezone).dt.replace_time_zone(None)
            return series.cast(pl.Datetime("us"))
    except pl.exceptions.PolarsError as exc:
        raise ConversionError(f"{what}.{column} is not a date/datetime column: {exc}") from exc
    raise ConversionError(f"{what}.{column} has unsupported type {series.dtype}")


# ---------------------------------------------------------------------------
# Concepts
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ConceptTable:
    """``concept_id`` to code string and name, concept 0 excluded."""

    codes: pl.DataFrame  # concept_id: Int64, concept_code_string: String, concept_name: String

    @classmethod
    def from_frame(cls, frame: pl.DataFrame) -> ConceptTable:
        _require(frame, ["concept_id", "vocabulary_id", "concept_code"], "CONCEPT")
        name = pl.col("concept_name") if "concept_name" in frame.columns else pl.lit(None)
        codes = (
            frame.select(
                pl.col("concept_id").cast(pl.Int64, strict=True),
                (
                    pl.col("vocabulary_id").cast(pl.String).str.strip_chars()
                    + pl.lit("/")
                    + pl.col("concept_code").cast(pl.String).str.strip_chars()
                ).alias("concept_code_string"),
                name.cast(pl.String).alias("concept_name"),
            )
            .filter(pl.col("concept_id") != 0)
            .unique("concept_id", keep="first")
        )
        return cls(codes)

    def lookup(self, ids: pl.Series) -> pl.Series:
        """Code strings for ``ids`` (null where absent or 0), in input order."""
        frame = pl.DataFrame({"concept_id": ids.cast(pl.Int64)}).with_row_index("_row")
        joined = frame.join(self.codes, on="concept_id", how="left").sort("_row")
        return joined.get_column("concept_code_string")

    def descriptions(self, codes: set[str]) -> dict[str, str]:
        rows = self.codes.filter(
            pl.col("concept_code_string").is_in(sorted(codes))
            & pl.col("concept_name").is_not_null()
        )
        return dict(
            zip(
                rows.get_column("concept_code_string").to_list(),
                rows.get_column("concept_name").to_list(),
                strict=True,
            )
        )

    def parents(self, ancestor: pl.DataFrame, codes: set[str]) -> dict[str, list[str]]:
        """Direct parents (``min_levels_of_separation == 1``) as code strings."""
        _require(
            ancestor,
            ["ancestor_concept_id", "descendant_concept_id", "min_levels_of_separation"],
            "CONCEPT_ANCESTOR",
        )
        ids = self.codes.select("concept_id", code="concept_code_string")
        direct = (
            ancestor.select(
                pl.col("ancestor_concept_id").cast(pl.Int64, strict=True),
                pl.col("descendant_concept_id").cast(pl.Int64, strict=True),
                pl.col("min_levels_of_separation").cast(pl.Int64, strict=True),
            )
            .filter(pl.col("min_levels_of_separation") == 1)
            .join(ids.rename({"concept_id": "descendant_concept_id"}), on="descendant_concept_id")
            .filter(pl.col("code").is_in(sorted(codes)))
            .join(
                ids.rename({"concept_id": "ancestor_concept_id", "code": "parent"}),
                on="ancestor_concept_id",
            )
            .group_by("code")
            .agg(pl.col("parent").unique().sort())
        )
        return {code: list(parents) for code, parents in direct.iter_rows()}


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _Counts:
    by_standard: int = 0
    by_source_concept: int = 0
    by_source_value: int = 0
    no_code: int = 0
    no_time: int = 0


def _resolve(
    frame: pl.DataFrame, spec: EventSpec, concepts: ConceptTable, config: ConvertConfig
) -> tuple[pl.Series, _Counts]:
    """The code per row, and how each was resolved."""
    what = spec.table
    standard = concepts.lookup(_int_column(frame, spec.concept, what))
    source = concepts.lookup(_int_column(frame, spec.source_concept, what))
    value = _text_column(frame, spec.source_value).str.strip_chars()
    value = pl.Series(
        "value",
        [
            None if v is None or v == "" else f"{config.source_vocabulary(spec.table)}/{v}"
            for v in value.to_list()
        ],
        dtype=pl.String,
    )
    code = (
        pl.DataFrame({"a": standard, "b": source, "c": value})
        .select(pl.coalesce("a", "b", "c"))
        .to_series()
    )
    counts = _Counts(
        by_standard=int(standard.is_not_null().sum()),
        by_source_concept=int((standard.is_null() & source.is_not_null()).sum()),
        by_source_value=int((standard.is_null() & source.is_null() & value.is_not_null()).sum()),
        no_code=int(code.is_null().sum()),
    )
    return code, counts


def _table_events(
    frame: pl.DataFrame, spec: EventSpec, concepts: ConceptTable, config: ConvertConfig
) -> tuple[pl.DataFrame, _Counts]:
    _require(frame, ["person_id"], spec.table)
    if spec.concept not in frame.columns and spec.table != "death":
        raise ConversionError(f"{spec.table} lacks column {spec.concept}")
    code, counts = _resolve(frame, spec, concepts, config)
    timed = spec.datetime is not None or spec.date is not None
    if timed:
        dt = _time_column(frame, spec.datetime, config.timezone, spec.table)
        d = _time_column(frame, spec.date, config.timezone, spec.table)
        time = pl.DataFrame({"a": dt, "b": d}).select(pl.coalesce("a", "b")).to_series()
    else:
        time = pl.Series("time", [None] * frame.height, dtype=pl.Datetime("us"))
    text = _text_column(frame, spec.text)
    if spec.value_concept is not None and spec.value_concept in frame.columns:
        value_codes = concepts.lookup(_int_column(frame, spec.value_concept, spec.table))
        text = pl.DataFrame({"a": text, "b": value_codes}).select(pl.coalesce("a", "b")).to_series()
    events = pl.DataFrame(
        {
            "subject_id": _int_column(frame, "person_id", spec.table),
            "time": time,
            "code": code,
            "numeric_value": _float_column(frame, spec.numeric, spec.table),
            "text_value": text,
        }
    )
    events = events.filter(pl.col("code").is_not_null())
    if timed:
        counts.no_time = int(events.get_column("time").is_null().sum())
        events = events.filter(pl.col("time").is_not_null())
    return as_event_frame(events), counts


def birth_events(person: pl.DataFrame, config: ConvertConfig) -> pl.DataFrame:
    """One ``MEDS_BIRTH`` row per person whose birth can be derived."""
    _require(person, ["person_id", "year_of_birth"], "person")
    year = _int_column(person, "year_of_birth", "person")
    month = _int_column(person, "month_of_birth", "person").fill_null(1)
    day = _int_column(person, "day_of_birth", "person").fill_null(1)
    from_parts = (
        pl.DataFrame({"y": year, "m": month, "d": day})
        .select(pl.datetime(pl.col("y"), pl.col("m"), pl.col("d"), time_unit="us"))
        .to_series()
    )
    exact = _time_column(person, "birth_datetime", config.timezone, "person")
    birth = pl.DataFrame({"a": exact, "b": from_parts}).select(pl.coalesce("a", "b")).to_series()
    return as_event_frame(
        pl.DataFrame(
            {
                "subject_id": _int_column(person, "person_id", "person"),
                "time": birth,
                "code": pl.Series([BIRTH_CODE] * person.height, dtype=pl.String),
                "numeric_value": pl.Series([None] * person.height, dtype=pl.Float32),
                "text_value": pl.Series([None] * person.height, dtype=pl.String),
            }
        )
    )


def death_events(death: pl.DataFrame, config: ConvertConfig) -> pl.DataFrame:
    """One ``MEDS_DEATH`` row per person with a death time (the earliest)."""
    spec = next(s for s in EVENT_SPECS if s.table == "death")
    _require(death, ["person_id"], "death")
    dt = _time_column(death, spec.datetime, config.timezone, "death")
    d = _time_column(death, spec.date, config.timezone, "death")
    frame = pl.DataFrame(
        {
            "subject_id": _int_column(death, "person_id", "death"),
            "time": pl.DataFrame({"a": dt, "b": d}).select(pl.coalesce("a", "b")).to_series(),
        }
    ).filter(pl.col("time").is_not_null())
    first = frame.group_by("subject_id").agg(pl.col("time").min())
    return as_event_frame(
        first.with_columns(
            pl.lit(DEATH_CODE).alias("code"),
            pl.lit(None, dtype=pl.Float32).alias("numeric_value"),
            pl.lit(None, dtype=pl.String).alias("text_value"),
        )
    )


def convert_tables(
    tables: Mapping[str, pl.DataFrame], concepts: ConceptTable, config: ConvertConfig
) -> tuple[pl.DataFrame, ConversionReport]:
    """Convert OMOP tables (``person`` required) to an event frame and a report."""
    if "person" not in tables:
        raise MissingFileError("the OMOP input has no person table")
    person = tables["person"]
    births = birth_events(person, config)
    no_birth = int(births.get_column("time").is_null().sum())
    births = births.filter(pl.col("time").is_not_null()).unique("subject_id", keep="first")

    parts: list[pl.DataFrame] = [births]
    totals = _Counts()
    specs = [*PERSON_SPECS, *EVENT_SPECS]
    for spec in specs:
        frame = tables.get(spec.table)
        if frame is None:
            continue
        if spec.table == "person" and spec.concept not in frame.columns:
            continue
        events, counts = _table_events(frame, spec, concepts, config)
        parts.append(events)
        totals.by_standard += counts.by_standard
        totals.by_source_concept += counts.by_source_concept
        totals.by_source_value += counts.by_source_value
        totals.no_code += counts.no_code
        totals.no_time += counts.no_time
    if "death" in tables:
        parts.append(death_events(tables["death"], config))

    events = pl.concat(parts) if parts else empty_events()
    known = births.select("subject_id", birth=pl.col("time"))
    joined = events.join(known, on="subject_id", how="left")
    unknown = int(joined.get_column("birth").is_null().sum())
    joined = joined.filter(pl.col("birth").is_not_null())
    early_mask = pl.col("time").is_not_null() & (pl.col("time") < pl.col("birth"))
    before_birth = int(joined.select(early_mask.sum()).item())
    kept = joined.filter(~early_mask).select(EVENT_SCHEMA.names())
    report = ConversionReport(
        subjects=kept.get_column("subject_id").n_unique(),
        events_written=kept.unique().height,
        by_standard_concept=totals.by_standard,
        by_source_concept=totals.by_source_concept,
        by_source_value=totals.by_source_value,
        dropped_no_code=totals.no_code,
        dropped_no_time=totals.no_time,
        dropped_unknown_person=unknown,
        dropped_before_birth=before_birth,
        dropped_no_birth=no_birth,
    )
    return kept, report


def read_omop_dir(directory: Path) -> dict[str, tuple[Path, pl.DataFrame]]:
    """Every known OMOP table present in ``directory``, with its file path."""
    if not directory.is_dir():
        raise MissingFileError(f"OMOP directory {directory} does not exist")
    found: dict[str, tuple[Path, pl.DataFrame]] = {}
    for table in TABLES:
        path = find_table(directory, table)
        if path is not None:
            found[table] = (path, read_table(path))
    if "person" not in found:
        raise MissingFileError(f"{directory} has no person table (person.parquet / person.csv)")
    return found
