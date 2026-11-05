"""Contract preconditions a MEDS consumer (``meds_reader_convert``) relies on.

Each precondition is asserted on a synthetic extract and on a converted OMOP
extract, so both producers are held to the same reader-visible contract.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import cast

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from omopmap.artifacts import (
    BIRTH_CODE,
    MANIFEST_NAME,
    MEDS_DATA_SCHEMA,
    validate_artifact,
)
from omopmap.cli import app
from tests.meds.fixtures_omop import build_omop_dir, run_convert

SPLITS = ("train", "tuning", "held_out")
METADATA_FILES = (
    "metadata/dataset.json",
    "metadata/codes.parquet",
    "metadata/subject_splits.parquet",
)

# The pyarrow stubs leave ``read_schema``'s parameter types unparameterised;
# narrow it here exactly as ``omopmap.artifacts.io`` does.
_read_schema = cast(
    "Callable[[Path], pa.Schema]",
    pq.read_schema,  # pyright: ignore[reportUnknownMemberType]
)


def synth_extract(tmp_path: Path) -> Path:
    out = tmp_path / "synth_demo"
    result = CliRunner().invoke(
        app,
        ["meds", "synth", "--n", "60", "--seed", "2", "--out", str(out)],
    )
    assert result.exit_code == 0, result.stderr
    return out


def omop_extract(tmp_path: Path) -> Path:
    omop_dir = build_omop_dir(tmp_path / "omop_input")
    out = tmp_path / "omop_demo"
    result = run_convert(omop_dir, out)
    assert result.exit_code == 0, result.stderr
    return out


@pytest.fixture(params=["synth", "omop"])
def extract(request: pytest.FixtureRequest, tmp_path: Path) -> Path:
    return synth_extract(tmp_path) if request.param == "synth" else omop_extract(tmp_path)


def _shard_paths(root: Path) -> list[Path]:
    return sorted((root / "data").rglob("*.parquet"))


def _rel_paths(root: Path) -> set[str]:
    return {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}


def test_layout_is_exactly_the_contract_layout(extract: Path) -> None:
    validate_artifact(extract)
    expected = {MANIFEST_NAME, *METADATA_FILES}
    for split in SPLITS:
        expected.add(f"data/{split}/0.parquet")
    assert _rel_paths(extract) == expected


def test_shard_schemas_are_identical_and_equal_the_contract(extract: Path) -> None:
    schemas = [_read_schema(path) for path in _shard_paths(extract)]
    assert schemas
    first = schemas[0]
    assert first.equals(MEDS_DATA_SCHEMA, check_metadata=False)
    assert all(schema.equals(first, check_metadata=False) for schema in schemas[1:])


def test_time_column_is_naive_microseconds(extract: Path) -> None:
    for path in _shard_paths(extract):
        dtype = pl.read_parquet_schema(path)["time"]
        assert dtype == pl.Datetime("us")


def test_rows_are_sorted_by_subject_and_time(extract: Path) -> None:
    for path in _shard_paths(extract):
        frame = pl.read_parquet(path)
        ordered = frame.sort(["subject_id", "time"], nulls_last=False)
        assert frame.equals(ordered)


def test_exactly_one_timed_birth_and_no_earlier_event(extract: Path) -> None:
    for path in _shard_paths(extract):
        frame = pl.read_parquet(path)
        for subject in frame.get_column("subject_id").unique().to_list():
            rows = frame.filter(pl.col("subject_id") == subject)
            births = rows.filter(pl.col("code") == BIRTH_CODE)
            assert births.height == 1
            birth_time = births.get_column("time").item()
            assert birth_time is not None
            timed = rows.filter(pl.col("time").is_not_null())
            assert timed.get_column("time").min() == birth_time


def test_subjects_are_partitioned_exactly_once(extract: Path) -> None:
    owners: dict[int, str] = {}
    for path in _shard_paths(extract):
        split = path.parent.name
        for subject in pl.read_parquet(path).get_column("subject_id").unique().to_list():
            assert subject not in owners, f"subject {subject} appears in several shards"
            owners[subject] = split
    splits = pl.read_parquet(extract / "metadata" / "subject_splits.parquet")
    listed = dict(
        zip(
            splits.get_column("subject_id").to_list(),
            splits.get_column("split").to_list(),
            strict=True,
        )
    )
    assert owners == listed


def test_numeric_value_is_float32_everywhere(extract: Path) -> None:
    dtypes = {pl.read_parquet_schema(path)["numeric_value"] for path in _shard_paths(extract)}
    assert dtypes == {pl.Float32}


def test_codes_metadata_lists_exactly_the_data_codes(extract: Path) -> None:
    data_codes: set[str] = set()
    for path in _shard_paths(extract):
        data_codes.update(pl.read_parquet(path).get_column("code").to_list())
    codes = pl.read_parquet(extract / "metadata" / "codes.parquet")
    assert codes.get_column("code").to_list() == sorted(data_codes)


@pytest.mark.skipif(
    shutil.which("meds_reader_convert") is None,
    reason="meds_reader_convert not installed (consumer-side tool)",
)
def test_meds_reader_convert_accepts_extract(extract: Path, tmp_path: Path) -> None:
    subprocess.run(
        ["meds_reader_convert", str(extract), str(tmp_path / "meds_reader_db")],
        check=True,
    )
