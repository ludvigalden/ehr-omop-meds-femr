"""The validator accepts a conformant extract and rejects each kind of violation."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import polars as pl
import pyarrow as pa
import pytest

from omopmap.artifacts import (
    MEDS_DATA,
    SUBJECT_SPLITS,
    ArtifactError,
    CodeFormatError,
    CodeMetadataError,
    HashMismatchError,
    LabelError,
    ManifestError,
    MedsExtractSummary,
    MissingFileError,
    NullViolationError,
    SchemaMismatchError,
    SortOrderError,
    SubjectPartitionError,
    UnexpectedFileError,
    check_meds_extract,
    validate_artifact,
    write_parquet,
)
from omopmap.artifacts.io import read_table, write_table

SHARD = Path("data/train/0.parquet")


def _rewrite_shard(root: Path, edit: Callable[[pl.DataFrame], pl.DataFrame]) -> None:
    path = root / SHARD
    write_parquet(edit(pl.read_parquet(path)), path, MEDS_DATA)


def test_valid_extract_passes(extract: Path) -> None:
    manifest = validate_artifact(extract)
    summary = manifest.summary
    assert isinstance(summary, MedsExtractSummary)
    assert summary.subjects == 5
    assert summary.events == 30
    assert summary.subjects_by_split == {"train": 3, "tuning": 1, "held_out": 1}
    assert summary.distinct_codes == 7


def test_valid_label_set_and_concept_set_pass(
    label_set: Path, extract: Path, concept_set: Path
) -> None:
    validate_artifact(label_set, extract_root=extract)
    validate_artifact(concept_set)


def test_wrong_dtype_fails(bare_extract: Path) -> None:
    path = bare_extract / SHARD
    table = read_table(path)
    index = table.schema.get_field_index("numeric_value")
    write_table(
        table.set_column(index, "numeric_value", table.column(index).cast(pa.float64())), path
    )
    with pytest.raises(SchemaMismatchError, match="numeric_value"):
        check_meds_extract(bare_extract)


def test_polars_large_string_is_not_the_contract_type(bare_extract: Path) -> None:
    path = bare_extract / SHARD
    pl.read_parquet(path).write_parquet(path)  # polars writes large_string
    with pytest.raises(SchemaMismatchError, match="large_string"):
        check_meds_extract(bare_extract)


def test_missing_file_fails(extract: Path) -> None:
    (extract / "metadata" / "codes.parquet").unlink()
    with pytest.raises(MissingFileError, match="codes.parquet"):
        validate_artifact(extract)


def test_missing_split_fails(bare_extract: Path) -> None:
    shutil.rmtree(bare_extract / "data" / "tuning")
    with pytest.raises(MissingFileError, match="tuning"):
        check_meds_extract(bare_extract)


def test_unexpected_file_fails(bare_extract: Path) -> None:
    (bare_extract / "data" / "train" / "notes.txt").write_text("x", encoding="utf-8")
    with pytest.raises(UnexpectedFileError, match="notes.txt"):
        check_meds_extract(bare_extract)


def test_hash_mismatch_fails(extract: Path) -> None:
    dataset = extract / "metadata" / "dataset.json"
    payload = json.loads(dataset.read_text(encoding="utf-8"))
    payload["dataset_version"] = "2"
    dataset.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(HashMismatchError, match="dataset.json"):
        validate_artifact(extract)


def test_unsorted_subject_fails(bare_extract: Path) -> None:
    _rewrite_shard(
        bare_extract, lambda f: f.sort("subject_id", descending=True, maintain_order=True)
    )
    with pytest.raises(SortOrderError, match="subject_id, time"):
        check_meds_extract(bare_extract)


def test_unsorted_time_fails(bare_extract: Path) -> None:
    _rewrite_shard(bare_extract, lambda f: f.reverse())
    with pytest.raises(SortOrderError):
        check_meds_extract(bare_extract)


def test_null_code_fails(bare_extract: Path) -> None:
    # A foreign writer that declares ``code`` nullable and stores a null.
    path = bare_extract / SHARD
    table = read_table(path)
    index = table.schema.get_field_index("code")
    codes = pa.array([None, *table.column(index).to_pylist()[1:]], type=pa.string())
    write_table(table.set_column(index, pa.field("code", pa.string()), codes), path)
    with pytest.raises(SchemaMismatchError, match="'code', 'string', True"):
        check_meds_extract(bare_extract)


def test_writer_refuses_null_in_non_nullable_column(tmp_path: Path) -> None:
    frame = pl.DataFrame(
        {"subject_id": [1], "split": [None]}, schema={"subject_id": pl.Int64, "split": pl.String}
    )
    with pytest.raises(SchemaMismatchError, match="null"):
        write_parquet(frame, tmp_path / "s.parquet", SUBJECT_SPLITS)


@pytest.mark.parametrize("bad", ["NOVOCAB", "/C1", "DEMOCOND/", " DEMOCOND/C1", "DEMOCOND/C1 "])
def test_malformed_code_fails(bare_extract: Path, bad: str) -> None:
    _rewrite_shard(
        bare_extract,
        lambda f: f.with_columns(
            pl.when(pl.col("code") == "DEMOLAB/L1")
            .then(pl.lit(bad))
            .otherwise(pl.col("code"))
            .alias("code")
        ),
    )
    with pytest.raises(CodeFormatError):
        check_meds_extract(bare_extract)


def test_concept_code_may_contain_slash(bare_extract: Path) -> None:
    _rewrite_shard(
        bare_extract,
        lambda f: f.with_columns(pl.col("code").str.replace("DEMOLAB/L1", "DEMOLAB/L1/x")),
    )
    with pytest.raises(CodeMetadataError, match="DEMOLAB/L1/x"):
        check_meds_extract(bare_extract)  # the code is well-formed; only codes.parquet lags


def test_subject_spanning_shards_fails(bare_extract: Path) -> None:
    second = bare_extract / "data" / "train" / "1.parquet"
    frame = pl.concat(
        [
            pl.read_parquet(second),
            pl.read_parquet(bare_extract / SHARD).filter(pl.col("subject_id") == 2),
        ]
    )
    write_parquet(frame.sort("subject_id", "time", nulls_last=False), second, MEDS_DATA)
    with pytest.raises(SubjectPartitionError, match="span"):
        check_meds_extract(bare_extract)


def test_split_mismatch_fails(bare_extract: Path) -> None:
    path = bare_extract / "metadata" / "subject_splits.parquet"
    frame = pl.read_parquet(path).with_columns(
        pl.when(pl.col("subject_id") == 4)
        .then(pl.lit("train"))
        .otherwise(pl.col("split"))
        .alias("split")
    )
    write_parquet(frame, path, SUBJECT_SPLITS)
    with pytest.raises(SubjectPartitionError, match="subject_splits"):
        check_meds_extract(bare_extract)


def test_missing_birth_fails(bare_extract: Path) -> None:
    _rewrite_shard(
        bare_extract,
        lambda f: f.filter(~((pl.col("subject_id") == 1) & (pl.col("code") == "MEDS_BIRTH"))),
    )
    with pytest.raises(SubjectPartitionError, match="MEDS_BIRTH"):
        check_meds_extract(bare_extract)


def test_event_before_birth_fails(bare_extract: Path) -> None:
    _rewrite_shard(
        bare_extract,
        lambda f: f.with_columns(
            pl.when((pl.col("subject_id") == 1) & (pl.col("code") == "DEMOVISIT/ED"))
            .then(pl.lit(datetime(1900, 1, 1)))
            .otherwise(pl.col("time"))
            .alias("time")
        ).sort("subject_id", "time", nulls_last=False, maintain_order=True),
    )
    with pytest.raises(SortOrderError, match="precede"):
        check_meds_extract(bare_extract)


def test_codes_metadata_must_cover_data(bare_extract: Path) -> None:
    _rewrite_shard(
        bare_extract,
        lambda f: f.with_columns(pl.col("code").str.replace("DEMOCOND/C0", "DEMOCOND/C9")),
    )
    with pytest.raises(CodeMetadataError):
        check_meds_extract(bare_extract)


def test_summary_mismatch_fails(extract: Path) -> None:
    path = extract / "artifact_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["summary"]["distinct_codes"] = 8
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ManifestError, match="summary"):
        validate_artifact(extract)


def test_label_value_column_mismatch_fails(label_set: Path, extract: Path) -> None:
    task = label_set / "task.json"
    payload = json.loads(task.read_text(encoding="utf-8"))
    payload["label_type"] = "integer"
    task.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(LabelError):
        from omopmap.artifacts import check_label_set

        check_label_set(label_set, extract)


def test_label_for_unknown_subject_fails(label_set: Path, extract: Path, tmp_path: Path) -> None:
    from omopmap.artifacts import LABELS, check_label_set

    path = label_set / "labels.parquet"
    frame = pl.read_parquet(path).with_columns(
        pl.when(pl.col("subject_id") == 5)
        .then(99)
        .otherwise(pl.col("subject_id"))
        .alias("subject_id")
    )
    write_parquet(frame, path, LABELS)
    with pytest.raises(LabelError, match="not in the extract"):
        check_label_set(label_set, extract)


def test_every_error_is_an_artifact_error() -> None:
    for error in (
        CodeFormatError,
        CodeMetadataError,
        HashMismatchError,
        LabelError,
        ManifestError,
        MissingFileError,
        NullViolationError,
        SchemaMismatchError,
        SortOrderError,
        SubjectPartitionError,
        UnexpectedFileError,
    ):
        assert issubclass(error, ArtifactError)
    assert issubclass(HashMismatchError, ManifestError)


def test_cli_verify(extract: Path, label_set: Path) -> None:
    from typer.testing import CliRunner

    from omopmap.cli import app

    runner = CliRunner()
    ok = runner.invoke(app, ["artifacts", "verify", str(extract)])
    assert ok.exit_code == 0, ok.output
    assert "meds_extract" in ok.output
    labels_ok = runner.invoke(
        app, ["artifacts", "verify", str(label_set), "--extract", str(extract)]
    )
    assert labels_ok.exit_code == 0, labels_ok.output
    (extract / "metadata" / "dataset.json").write_text("{}\n", encoding="utf-8")
    bad = runner.invoke(app, ["artifacts", "verify", str(extract)])
    assert bad.exit_code == 1
    assert "HashMismatchError" in bad.output
