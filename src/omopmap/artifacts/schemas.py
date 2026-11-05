"""Arrow schemas for every Parquet artifact: the single source of truth.

Each ``Column`` carries its Arrow type, nullability and semantics, so the
contract tables in ``context/reference/artifacts.md`` are rendered from these
definitions (see ``omopmap.artifacts.render``) and a drift test keeps the two
equal. The MEDS schemas follow MEDS 0.3.3 (``meds.schema``) and add the
``text_value`` column that MEDS allows as a custom property.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import pyarrow as pa


@dataclass(frozen=True, slots=True)
class Column:
    """One contract column."""

    name: str
    dtype: pa.DataType
    nullable: bool
    semantics: str

    def field(self) -> pa.Field[pa.DataType]:
        return pa.field(self.name, self.dtype, nullable=self.nullable)


@dataclass(frozen=True, slots=True)
class TableSpec:
    """A named Parquet artifact schema."""

    name: str
    columns: tuple[Column, ...]

    @property
    def schema(self) -> pa.Schema:
        return pa.schema([c.field() for c in self.columns])

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.columns)

    def column(self, name: str) -> Column:
        for c in self.columns:
            if c.name == name:
                return c
        raise KeyError(name)


MEDS_DATA: Final = TableSpec(
    "meds_data",
    (
        Column(
            "subject_id",
            pa.int64(),
            False,
            "Pseudonymous subject id; every row of a subject lives in exactly one shard.",
        ),
        Column(
            "time",
            pa.timestamp("us"),
            True,
            "Event time, timezone-naive, microseconds. Null only for static events, "
            "which sort first within the subject.",
        ),
        Column(
            "code",
            pa.string(),
            False,
            "`<vocabulary_id>/<concept_code>` (OMOP vocabulary id and concept code, "
            "split on the first `/`), or a reserved code `MEDS_BIRTH` / `MEDS_DEATH`.",
        ),
        Column(
            "numeric_value",
            pa.float32(),
            True,
            "Numeric measurement value; null when the event carries none.",
        ),
        Column(
            "text_value",
            pa.string(),
            True,
            "Text value; null when the event carries none.",
        ),
    ),
)

MEDS_CODES: Final = TableSpec(
    "meds_codes",
    (
        Column(
            "code",
            pa.string(),
            False,
            "One row per distinct code in `data/`; unique.",
        ),
        Column(
            "description",
            pa.string(),
            True,
            "Human-readable description, or null. Generated from a local vocabulary; "
            "never committed.",
        ),
        Column(
            "parent_codes",
            pa.list_(pa.string()),
            True,
            "Direct parents as code strings in the same format, or null / empty.",
        ),
    ),
)

SUBJECT_SPLITS: Final = TableSpec(
    "subject_splits",
    (
        Column(
            "subject_id",
            pa.int64(),
            False,
            "Every subject present in `data/`, exactly once.",
        ),
        Column(
            "split",
            pa.string(),
            False,
            "`train`, `tuning` or `held_out`; equals the `data/<split>/` directory "
            "that holds the subject.",
        ),
    ),
)

LABELS: Final = TableSpec(
    "labels",
    (
        Column(
            "subject_id",
            pa.int64(),
            False,
            "A subject of the referenced extract, listed in its `subject_splits.parquet`.",
        ),
        Column(
            "prediction_time",
            pa.timestamp("us"),
            False,
            "Timezone-naive; the model may use events with `time <= prediction_time`.",
        ),
        Column(
            "boolean_value",
            pa.bool_(),
            True,
            "Set (non-null) exactly when the task's `label_type` is `boolean`.",
        ),
        Column(
            "integer_value",
            pa.int64(),
            True,
            "Set exactly when `label_type` is `integer`.",
        ),
        Column(
            "float_value",
            pa.float64(),
            True,
            "Set exactly when `label_type` is `float`.",
        ),
        Column(
            "categorical_value",
            pa.string(),
            True,
            "Set exactly when `label_type` is `categorical`.",
        ),
    ),
)

CONCEPT_SET: Final = TableSpec(
    "concept_set",
    (
        Column(
            "vocabulary_id",
            pa.string(),
            False,
            "OMOP vocabulary id as written in the CLMBR dictionary (for example `LOINC`).",
        ),
        Column(
            "concept_code",
            pa.string(),
            False,
            "Concept code within the vocabulary.",
        ),
        Column(
            "code",
            pa.string(),
            False,
            "`<vocabulary_id>/<concept_code>`: the MEDS code string; unique; rows sorted by it.",
        ),
        Column(
            "entry_types",
            pa.list_(pa.string()),
            False,
            "Sorted, distinct dictionary entry types for the code (`code`, `numeric`, `text`).",
        ),
    ),
)

MEDS_DATA_SCHEMA: Final = MEDS_DATA.schema
MEDS_CODES_SCHEMA: Final = MEDS_CODES.schema
SUBJECT_SPLITS_SCHEMA: Final = SUBJECT_SPLITS.schema
LABEL_SCHEMA: Final = LABELS.schema
CONCEPT_SET_SCHEMA: Final = CONCEPT_SET.schema

#: Value columns of a label set by task ``label_type``.
LABEL_VALUE_COLUMNS: Final = {
    "boolean": "boolean_value",
    "integer": "integer_value",
    "float": "float_value",
    "categorical": "categorical_value",
}

ALL_TABLES: Final = (MEDS_DATA, MEDS_CODES, SUBJECT_SPLITS, LABELS, CONCEPT_SET)
