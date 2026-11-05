"""Drift guards: the contract document equals the code, and the code equals MEDS 0.3.3."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa

from omopmap.artifacts import (
    LABEL_SCHEMA,
    MEDS_CODES_SCHEMA,
    MEDS_DATA_SCHEMA,
    SUBJECT_SPLITS_SCHEMA,
    is_valid_data_code,
    is_vocabulary_code,
    split_code,
)
from omopmap.artifacts.io import schema_columns
from omopmap.artifacts.render import document_blocks, render_blocks

DOC = Path(__file__).resolve().parents[2] / "context" / "reference" / "artifacts.md"


def test_document_tables_equal_code() -> None:
    expected = render_blocks()
    actual = document_blocks(DOC.read_text(encoding="utf-8"))
    assert sorted(actual) == sorted(expected), "generated block set differs"
    for name, body in expected.items():
        assert actual[name] == body, (
            f"block {name} in {DOC.name} is stale; regenerate with "
            "omopmap.artifacts.render.rewrite_document"
        )


# The MEDS 0.3.3 schemas (``meds.schema``), written out independently of the
# contract code so a change on either side is caught.
_MEDS_0_3_3_DATA = pa.schema(
    [
        ("subject_id", pa.int64()),
        ("time", pa.timestamp("us")),
        ("code", pa.string()),
        ("numeric_value", pa.float32()),
    ]
)
_MEDS_0_3_3_LABEL = pa.schema(
    [
        ("subject_id", pa.int64()),
        ("prediction_time", pa.timestamp("us")),
        ("boolean_value", pa.bool_()),
        ("integer_value", pa.int64()),
        ("float_value", pa.float64()),
        ("categorical_value", pa.string()),
    ]
)
_MEDS_0_3_3_SPLITS = pa.schema([("subject_id", pa.int64()), ("split", pa.string())])
_MEDS_0_3_3_CODES = pa.schema(
    [
        ("code", pa.string()),
        ("description", pa.string()),
        ("parent_codes", pa.list_(pa.string())),
    ]
)


def _types(schema: pa.Schema) -> list[tuple[str, pa.DataType]]:
    return [(name, dtype) for name, dtype, _ in schema_columns(schema)]


def test_meds_core_columns_match_meds_0_3_3() -> None:
    contract = _types(MEDS_DATA_SCHEMA)
    assert contract[: len(_MEDS_0_3_3_DATA)] == _types(_MEDS_0_3_3_DATA)
    assert contract[len(_MEDS_0_3_3_DATA) :] == [("text_value", pa.string())]
    assert _types(LABEL_SCHEMA) == _types(_MEDS_0_3_3_LABEL)
    assert _types(SUBJECT_SPLITS_SCHEMA) == _types(_MEDS_0_3_3_SPLITS)
    assert _types(MEDS_CODES_SCHEMA) == _types(_MEDS_0_3_3_CODES)


def test_code_string_rules() -> None:
    assert split_code("DEMOLAB/1234-5") == ("DEMOLAB", "1234-5")
    assert split_code("DEMO VOCAB/a/b") == ("DEMO VOCAB", "a/b")
    assert is_valid_data_code("MEDS_BIRTH")
    assert not is_vocabulary_code("MEDS_BIRTH")
    for bad in ("", "X", "X/", "/Y", " X/Y", "X/Y ", "X/\nY"):
        assert not is_valid_data_code(bad), bad
