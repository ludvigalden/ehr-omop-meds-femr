"""OMOP-to-MEDS conversion through the CLI: codes, times, counts, determinism."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

from omopmap.artifacts import MedsExtractSummary, validate_artifact
from omopmap.cli import app
from omopmap.meds.convert import ConceptTable, ConvertConfig, convert_tables, read_table
from tests.meds.fixtures_omop import (
    DEMOLOCAL_K99,
    LOCAL_VOCABULARY,
    PERSON1_BIRTH,
    PERSON2_BIRTH,
    PERSON3_BIRTH,
    assert_trees_identical,
    build_omop_dir,
    read_data,
    run_convert,
)

CREATED_AT = "2026-01-01T00:00:00Z"


@pytest.fixture
def omop_dir(tmp_path: Path) -> Path:
    return build_omop_dir(tmp_path / "omop_input")


@pytest.fixture
def extract(tmp_path: Path, omop_dir: Path) -> Path:
    out = tmp_path / "omop_demo"
    result = run_convert(omop_dir, out)
    assert result.exit_code == 0, result.stderr
    return out


def test_convert_cli_succeeds_and_validates(extract: Path) -> None:
    manifest = validate_artifact(extract)
    summary = manifest.summary
    assert isinstance(summary, MedsExtractSummary)
    assert summary.subjects == 3


def test_expected_codes_are_present(extract: Path) -> None:
    events = read_data(extract)
    codes = set(events.get_column("code").to_list())
    expected = {
        "Gender/F",
        "Gender/M",
        "Visit/ER",
        "DEMOCOND/C001",
        "LOCALSRC/X1",
        DEMOLOCAL_K99,
        "DEMODRUG/D001",
        "DEMOLAB/L001",
        "DEMOOBS/O001",
        "DEMOPROC/P001",
        "DEMOCOND/C002",
        "MEDS_BIRTH",
        "MEDS_DEATH",
    }
    assert expected <= codes
    assert not any(code.endswith("/0") for code in codes)
    assert not any(code.startswith("None/") for code in codes)


def test_event_counts_and_values(extract: Path) -> None:
    events = read_data(extract)
    births = events.filter(pl.col("code") == "MEDS_BIRTH")
    assert births.height == 3
    deaths = events.filter(pl.col("code") == "MEDS_DEATH")
    assert deaths.height == 1
    assert deaths.get_column("subject_id").to_list() == [3]
    lab = events.filter(pl.col("code") == "DEMOLAB/L001")
    assert lab.height == 1
    value = lab.get_column("numeric_value").item()
    assert value is not None and abs(value - 4.2) < 1e-5
    observations = events.filter(pl.col("code") == "DEMOOBS/O001").sort("time")
    assert observations.get_column("text_value").to_list() == ["DEMOVAL/POS", "free text"]


def test_birth_times(extract: Path) -> None:
    births = read_data(extract).filter(pl.col("code") == "MEDS_BIRTH")
    times = {
        sid: time
        for sid, time in zip(
            births.get_column("subject_id").to_list(),
            births.get_column("time").to_list(),
            strict=True,
        )
    }
    assert times[1] == PERSON1_BIRTH
    assert times[2] == PERSON2_BIRTH
    assert times[3] == PERSON3_BIRTH


def test_date_only_row_gets_midnight(extract: Path) -> None:
    events = read_data(extract)
    visit = events.filter((pl.col("subject_id") == 2) & (pl.col("code") == "Visit/ER"))
    assert visit.height == 1
    time = visit.get_column("time").item()
    assert time == datetime(2024, 3, 2)


def test_conversion_report_counts(tmp_path: Path, omop_dir: Path) -> None:
    out = tmp_path / "omop_demo"
    result = run_convert(omop_dir, out)
    assert result.exit_code == 0, result.stderr
    report_line = next(
        line for line in result.output.splitlines() if line.startswith("conversion:")
    )
    assert '"dropped_no_code":1' in report_line
    assert '"dropped_unknown_person":1' in report_line
    assert '"dropped_before_birth":1' in report_line
    assert '"by_source_concept":1' in report_line
    assert '"by_source_value":1' in report_line


def test_descriptions_and_parents(omop_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "omop_demo"
    result = run_convert(omop_dir, out)
    assert result.exit_code == 0, result.stderr
    codes = pl.read_parquet(out / "metadata" / "codes.parquet")
    row = codes.filter(pl.col("code") == "DEMOCOND/C001")
    assert row.get_column("description").to_list() == ["Invented condition one"]
    assert row.get_column("parent_codes").to_list() == [["DEMOCOND/C010"]]


def test_two_runs_are_byte_identical(tmp_path: Path, omop_dir: Path) -> None:
    """Same arguments into same-named directories must reproduce exact bytes."""
    first = tmp_path / "a" / "omop_demo"
    second = tmp_path / "b" / "omop_demo"
    assert run_convert(omop_dir, first).exit_code == 0
    assert run_convert(omop_dir, second).exit_code == 0
    assert_trees_identical(first, second)


def test_missing_person_table_fails(tmp_path: Path) -> None:
    omop_dir = build_omop_dir(tmp_path / "no_person", with_person=False)
    result = run_convert(omop_dir, tmp_path / "omop_demo")
    assert result.exit_code == 1
    assert "MissingFileError" in result.stderr


def test_source_value_vocabulary_parameter(tmp_path: Path) -> None:
    """``ConvertConfig.source_vocabularies`` decides the fallback vocabulary."""
    omop_dir = build_omop_dir(tmp_path / "omop_input")
    concepts = ConceptTable.from_frame(read_table(omop_dir / "CONCEPT.tsv"))
    config = ConvertConfig(source_vocabularies={"condition_occurrence": LOCAL_VOCABULARY})
    tables = {path.stem.lower(): read_table(path) for path in sorted(omop_dir.glob("*.csv"))}
    events, report = convert_tables(tables, concepts, config)
    codes = set(events.get_column("code").to_list())
    assert DEMOLOCAL_K99 in codes
    assert report.by_source_value >= 1


def test_concept_zero_and_empty_row_drops_no_code(tmp_path: Path) -> None:
    """A row with concept 0 and no source at all resolves to no code, not ``/0``."""
    omop_dir = build_omop_dir(tmp_path / "omop_input")
    concepts = ConceptTable.from_frame(read_table(omop_dir / "CONCEPT.tsv"))
    config = ConvertConfig(source_vocabularies={"condition_occurrence": LOCAL_VOCABULARY})
    tables = {path.stem.lower(): read_table(path) for path in sorted(omop_dir.glob("*.csv"))}
    events, report = convert_tables(tables, concepts, config)
    codes = set(events.get_column("code").to_list())
    assert not any(code.endswith("/0") for code in codes)
    assert not any(code.startswith("None/") for code in codes)
    assert report.dropped_no_code == 1


def test_timezone_aware_parquet_visits(tmp_path: Path) -> None:
    """Aware input timestamps are converted to ``timezone`` and made naive."""
    omop_dir = build_omop_dir(tmp_path / "omop_input")
    (omop_dir / "visit_occurrence.csv").unlink()
    visits = pl.DataFrame(
        {
            "visit_occurrence_id": [1, 2],
            "person_id": [1, 2],
            "visit_concept_id": [2001, 2001],
            "visit_start_datetime": [datetime(2024, 3, 1, 8, 0), None],
            "visit_start_date": [None, datetime(2024, 3, 2).date()],
        },
        schema={
            "visit_occurrence_id": pl.Int64,
            "person_id": pl.Int64,
            "visit_concept_id": pl.Int64,
            "visit_start_datetime": pl.Datetime("us"),
            "visit_start_date": pl.Date,
        },
    )
    aware = visits.with_columns(
        pl.col("visit_start_datetime").dt.replace_time_zone("Europe/Stockholm")
    )
    aware.write_parquet(omop_dir / "visit_occurrence.parquet")
    out = tmp_path / "omop_demo"
    result = CliRunner().invoke(
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
            "--timezone",
            "Europe/Stockholm",
            "--seed",
            "3",
            "--out",
            str(out),
            "--created-at",
            CREATED_AT,
        ],
    )
    assert result.exit_code == 0, result.stderr
    events = read_data(out)
    visit_times = sorted(events.filter(pl.col("code") == "Visit/ER").get_column("time").to_list())
    assert visit_times == [datetime(2024, 3, 1, 8, 0), datetime(2024, 3, 2)]
