"""The synthetic generator: byte determinism, code inventory and subject structure."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl
import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from omopmap.artifacts import (
    BIRTH_CODE,
    DEATH_CODE,
    MedsExtractSummary,
    validate_artifact,
)
from omopmap.artifacts.contract import RESERVED_CODES
from omopmap.cli import app
from omopmap.meds.synth import SynthConfig, generate_events, synthetic_dictionary_codes
from tests.meds.fixtures_omop import assert_trees_identical

CODE_PREFIXES = (
    "Gender/",
    "Visit/",
    "DEMOCOND/",
    "DEMODRUG/",
    "DEMOLAB/",
    "DEMOPROC/",
    "DEMOSRC/",
    "DEMOOBS/",
)


def _synth_via_cli(out: Path, *, subjects: int, seed: int) -> None:
    result = CliRunner().invoke(
        app,
        ["meds", "synth", "--n", str(subjects), "--seed", str(seed), "--out", str(out)],
    )
    assert result.exit_code == 0, result.output


def test_same_seed_writes_identical_bytes(tmp_path: Path) -> None:
    """Same arguments into same-named directories must reproduce exact bytes.

    The directories share a name (under different parents) because the
    extract records its dataset name.
    """
    _synth_via_cli(tmp_path / "a" / "demo", subjects=50, seed=7)
    _synth_via_cli(tmp_path / "b" / "demo", subjects=50, seed=7)
    assert_trees_identical(tmp_path / "a" / "demo", tmp_path / "b" / "demo")


def test_different_seed_writes_different_data(tmp_path: Path) -> None:
    _synth_via_cli(tmp_path / "first", subjects=50, seed=7)
    _synth_via_cli(tmp_path / "second", subjects=50, seed=8)
    first = (tmp_path / "first" / "data" / "train" / "0.parquet").read_bytes()
    second = (tmp_path / "second" / "data" / "train" / "0.parquet").read_bytes()
    assert first != second


def test_extract_validates_with_expected_summary(tmp_path: Path) -> None:
    _synth_via_cli(tmp_path / "demo", subjects=50, seed=7)
    manifest = validate_artifact(tmp_path / "demo")
    summary = manifest.summary
    assert isinstance(summary, MedsExtractSummary)
    assert summary.subjects == 50
    assert set(summary.subjects_by_split) == {"train", "tuning", "held_out"}
    assert all(count >= 0 for count in summary.subjects_by_split.values())
    assert sum(summary.subjects_by_split.values()) == 50


def _all_codes(events: pl.DataFrame) -> set[str]:
    return set(events.get_column("code").to_list())


def test_every_code_is_invented_or_omop_generated() -> None:
    events = generate_events(SynthConfig(subjects=200, seed=1))
    for code in _all_codes(events):
        assert code in RESERVED_CODES or any(code.startswith(prefix) for prefix in CODE_PREFIXES)


def test_dictionary_coverage_is_partial() -> None:
    events = generate_events(SynthConfig(subjects=200, seed=1))
    dictionary = synthetic_dictionary_codes()
    codes = _all_codes(events)
    vocabulary = {code for code in codes if code not in RESERVED_CODES}
    share = len(vocabulary & dictionary) / len(vocabulary)
    assert 0.0 < share < 1.0
    source_codes = {code for code in codes if code.startswith("DEMOSRC/")}
    assert source_codes
    assert not (source_codes & dictionary)


def test_birth_and_death_structure() -> None:
    events = generate_events(SynthConfig(subjects=200, seed=1))
    for subject in events.get_column("subject_id").unique().to_list():
        rows = events.filter(pl.col("subject_id") == subject)
        births = rows.filter(pl.col("code") == BIRTH_CODE)
        assert births.height == 1
        deaths = rows.filter(pl.col("code") == DEATH_CODE)
        assert deaths.height <= 1
        if deaths.height == 1:
            death_time = deaths.get_column("time").item()
            assert isinstance(death_time, datetime)
            other = rows.filter(pl.col("code") != DEATH_CODE).get_column("time").drop_nulls()
            assert death_time >= max(other.to_list())


def test_subject_generation_is_independent() -> None:
    small = generate_events(SynthConfig(subjects=5, seed=3))
    large = generate_events(SynthConfig(subjects=10, seed=3))
    columns = ["subject_id", "time", "code", "numeric_value", "text_value"]
    head = small.filter(pl.col("subject_id") <= 5).sort(columns)
    tail = large.filter(pl.col("subject_id") <= 5).sort(columns)
    assert head.equals(tail)


def test_nonempty_output_directory_fails(tmp_path: Path) -> None:
    """An existing non-empty output fails closed with ``OutputExistsError``."""
    out = tmp_path / "demo"
    out.mkdir()
    (out / "occupied.txt").write_text("occupied", encoding="utf-8")
    result = CliRunner().invoke(
        app,
        ["meds", "synth", "--n", "10", "--seed", "0", "--out", str(out)],
    )
    assert result.exit_code == 1
    assert "OutputExistsError" in result.stderr


def test_pydantic_rejects_bad_subject_count() -> None:
    with pytest.raises(ValidationError):
        SynthConfig(subjects=0, seed=0)
