"""The ``labels build`` CLI: end-to-end labels, determinism, fail-closed errors."""

from __future__ import annotations

import shutil
from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner, Result

from omopmap.artifacts import validate_artifact
from omopmap.cli import app
from tests.meds.fixtures_omop import assert_trees_identical

TASK = "er_to_ip_30d"
LABEL_ARGS = [
    "labels",
    "build",
    "--task",
    TASK,
    "--index",
    "Visit/ER",
    "--outcome",
    "Visit/IP",
    "--window-days",
    "30",
]


def _synth_extract(tmp_path: Path) -> Path:
    extract = tmp_path / "extract_demo"
    result = CliRunner().invoke(
        app,
        ["meds", "synth", "--n", "80", "--seed", "4", "--out", str(extract)],
    )
    assert result.exit_code == 0, result.stderr
    return extract


def _build(extract: Path, out: Path) -> Result:
    return CliRunner().invoke(app, [*LABEL_ARGS, "--extract", str(extract), "--out", str(out)])


@pytest.fixture
def extract(tmp_path: Path) -> Path:
    return _synth_extract(tmp_path)


def test_build_writes_a_valid_label_set(tmp_path: Path, extract: Path) -> None:
    out = tmp_path / TASK
    result = _build(extract, out)
    assert result.exit_code == 0, result.stderr
    validate_artifact(out, extract_root=extract)
    labels = pl.read_parquet(out / "labels.parquet")
    assert labels.height > 0
    assert set(labels.get_column("boolean_value").to_list()) == {True, False}


def test_two_runs_are_byte_identical(tmp_path: Path, extract: Path) -> None:
    first = tmp_path / "first" / TASK
    second = tmp_path / "second" / TASK
    assert _build(extract, first).exit_code == 0
    assert _build(extract, second).exit_code == 0
    assert_trees_identical(first, second)


def test_output_directory_must_be_named_after_the_task(tmp_path: Path, extract: Path) -> None:
    result = _build(extract, tmp_path / "wrong_name")
    assert result.exit_code == 1
    assert "LabelError" in result.stderr


def test_tampered_extract_fails_closed(tmp_path: Path, extract: Path) -> None:
    tampered = tmp_path / "tampered_demo"
    shutil.copytree(extract, tampered)
    shard = tampered / "data" / "train" / "0.parquet"
    shard.write_bytes(shard.read_bytes() + b"x")
    result = _build(tampered, tmp_path / TASK)
    assert result.exit_code == 1
    # The hash is checked before the corrupted footer is parsed: a typed error.
    assert "HashMismatchError" in result.stderr
