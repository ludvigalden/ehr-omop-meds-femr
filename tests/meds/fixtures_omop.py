"""Invented OMOP CDM fixture tables for converter and reader-precondition tests.

Every concept id, code and value here is invented; the only codes that carry
meaning outside this module are the OMOP-generated ``Gender/*`` and ``Visit/*``
codes. No real vocabulary content appears. Person 99 is deliberately absent
from ``person`` to exercise the unknown-person drop.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl
from typer.testing import CliRunner, Result

from omopmap.cli import app

CREATED_AT = "2026-01-01T00:00:00Z"

#: Person 1 carries the only exact ``birth_datetime``; persons 2 and 3 have a
#: ``year_of_birth`` only, so their birth becomes January 1 at midnight.
PERSON1_BIRTH = datetime(1980, 5, 10, 6, 30)
PERSON2_BIRTH = datetime(1975, 1, 1)
PERSON3_BIRTH = datetime(1990, 1, 1)
DEATH_TIME = datetime(2023, 6, 15, 18, 45)

#: Vocabulary id used for the source-value fallback condition (``K99``).
LOCAL_VOCABULARY = "DEMOLOCAL"
#: The code the source-value fallback row resolves to under ``run_convert``.
DEMOLOCAL_K99 = f"{LOCAL_VOCABULARY}/K99"

CONCEPT_ROWS: tuple[tuple[int, str, str, str], ...] = (
    (0, "None", "No matching concept", ""),
    (1001, "Gender", "F", "Female"),
    (1002, "Gender", "M", "Male"),
    (2001, "Visit", "ER", "Emergency Room Visit"),
    (2002, "Visit", "IP", "Inpatient Visit"),
    (3001, "DEMOCOND", "C001", "Invented condition one"),
    (3002, "DEMOCOND", "C002", "Invented condition two"),
    (3010, "DEMOCOND", "C010", "Invented condition parent"),
    (4001, "DEMODRUG", "D001", "Invented drug one"),
    (5001, "DEMOLAB", "L001", "Invented lab one"),
    (6001, "DEMOPROC", "P001", "Invented procedure one"),
    (7001, "DEMOOBS", "O001", "Invented observation one"),
    (7100, "DEMOVAL", "POS", "Invented positive"),
    (9001, "LOCALSRC", "X1", "Invented source concept"),
)

ANCESTOR_ROWS: tuple[tuple[int, int, int], ...] = (
    (3010, 3001, 1),
    (3010, 3002, 1),
)


def _comma(header: tuple[str, ...], rows: tuple[tuple[object, ...], ...]) -> str:
    return "\n".join(
        [
            ",".join(header),
            *(",".join("" if v is None else str(v) for v in row) for row in rows),
        ]
    )


def _tsv(header: tuple[str, ...], rows: tuple[tuple[object, ...], ...]) -> str:
    return "\n".join(
        [
            "\t".join(header),
            *("\t".join("" if v is None else str(v) for v in row) for row in rows),
        ]
    )


def build_omop_dir(root: Path, *, with_person: bool = True) -> Path:
    """Write the tiny OMOP CDM directory (plus CONCEPT tables) under ``root``."""
    root.mkdir(parents=True, exist_ok=True)
    person_rows: tuple[tuple[object, ...], ...] = (
        (1, 1001, 1980, 5, 10, "1980-05-10 06:30:00"),
        (2, 1001, 1975, None, None, None),
        (3, 1002, 1990, None, None, None),
    )
    if with_person:
        (root / "person.csv").write_text(
            _comma(
                (
                    "person_id",
                    "gender_concept_id",
                    "year_of_birth",
                    "month_of_birth",
                    "day_of_birth",
                    "birth_datetime",
                ),
                person_rows,
            ),
            encoding="utf-8",
        )
    (root / "visit_occurrence.csv").write_text(
        _comma(
            (
                "visit_occurrence_id",
                "person_id",
                "visit_concept_id",
                "visit_start_datetime",
                "visit_start_date",
            ),
            (
                (1, 1, 2001, "2024-03-01 08:00:00", None),
                (2, 2, 2001, None, "2024-03-02"),
            ),
        ),
        encoding="utf-8",
    )
    (root / "condition_occurrence.csv").write_text(
        _comma(
            (
                "condition_occurrence_id",
                "person_id",
                "condition_concept_id",
                "condition_source_concept_id",
                "condition_source_value",
                "condition_start_datetime",
            ),
            (
                (1, 1, 3001, None, None, "2024-03-01 09:00:00"),
                (2, 1, 0, 9001, None, "2024-03-01 10:00:00"),
                (3, 1, 0, None, "K99", "2024-03-01 11:00:00"),
                (4, 1, None, None, None, "2024-03-01 12:00:00"),
                (5, 99, 3001, None, None, "2024-03-01 13:00:00"),
                (6, 1, 3001, None, None, "1970-01-01 00:00:00"),
            ),
        ),
        encoding="utf-8",
    )
    (root / "drug_exposure.csv").write_text(
        _comma(
            ("drug_exposure_id", "person_id", "drug_concept_id", "drug_exposure_start_datetime"),
            ((1, 1, 4001, "2024-03-01 08:30:00"),),
        ),
        encoding="utf-8",
    )
    (root / "measurement.csv").write_text(
        _comma(
            (
                "measurement_id",
                "person_id",
                "measurement_concept_id",
                "value_as_number",
                "measurement_datetime",
            ),
            ((1, 1, 5001, "4.2", "2024-03-01 09:45:00"),),
        ),
        encoding="utf-8",
    )
    (root / "observation.csv").write_text(
        _comma(
            (
                "observation_id",
                "person_id",
                "observation_concept_id",
                "value_as_concept_id",
                "value_as_string",
                "observation_datetime",
            ),
            (
                (1, 1, 7001, 7100, None, "2024-03-01 09:15:00"),
                (2, 1, 7001, 0, "free text", "2024-03-01 09:20:00"),
            ),
        ),
        encoding="utf-8",
    )
    (root / "procedure_occurrence.csv").write_text(
        _comma(
            ("procedure_occurrence_id", "person_id", "procedure_concept_id", "procedure_datetime"),
            ((1, 2, 6001, "2024-03-01 10:30:00"),),
        ),
        encoding="utf-8",
    )
    (root / "death.csv").write_text(
        _comma(
            ("person_id", "cause_concept_id", "death_datetime"),
            ((3, 3002, "2023-06-15 18:45:00"),),
        ),
        encoding="utf-8",
    )
    (root / "CONCEPT.tsv").write_text(
        _tsv(
            ("concept_id", "vocabulary_id", "concept_code", "concept_name"),
            CONCEPT_ROWS,
        ),
        encoding="utf-8",
    )
    (root / "CONCEPT_ANCESTOR.tsv").write_text(
        _tsv(
            ("ancestor_concept_id", "descendant_concept_id", "min_levels_of_separation"),
            ANCESTOR_ROWS,
        ),
        encoding="utf-8",
    )
    return root


def run_convert(omop_dir: Path, out: Path) -> Result:
    """Run ``meds convert`` with the fixture's canonical arguments."""
    return CliRunner().invoke(
        app,
        [
            "meds",
            "convert",
            "--omop",
            str(omop_dir),
            "--concept",
            str(omop_dir / "CONCEPT.tsv"),
            "--concept-ancestor",
            str(omop_dir / "CONCEPT_ANCESTOR.tsv"),
            "--source-vocabulary",
            f"condition_occurrence={LOCAL_VOCABULARY}",
            "--seed",
            "3",
            "--out",
            str(out),
            "--created-at",
            CREATED_AT,
            "--descriptions",
        ],
    )


def read_data(root: Path) -> pl.DataFrame:
    """Every data shard of an extract, concatenated (reading is unchecked)."""
    shards = sorted((root / "data").rglob("*.parquet"))
    return pl.concat([pl.read_parquet(path) for path in shards])


def tree_bytes(root: Path) -> dict[str, bytes]:
    """Relative path -> file bytes for every file under ``root``."""
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def assert_trees_identical(left: Path, right: Path) -> None:
    """Both trees hold the same relative file set with byte-identical content."""
    left_tree = tree_bytes(left)
    right_tree = tree_bytes(right)
    assert set(left_tree) == set(right_tree), "trees hold different files"
    for rel, content in left_tree.items():
        assert right_tree[rel] == content, f"{rel} differs between the trees"
