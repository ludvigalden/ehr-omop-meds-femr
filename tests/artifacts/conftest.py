"""Fixtures: a tiny synthetic MEDS extract, label set and concept set.

Every code is invented (``DEMOCOND/...``, ``DEMOLAB/...``); no real vocabulary
content appears.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from omopmap.artifacts import (
    CONCEPT_SET,
    LABELS,
    MEDS_CODES,
    MEDS_DATA,
    SUBJECT_SPLITS,
    ConceptSetSummary,
    InputRecord,
    build_manifest,
    check_concept_set,
    check_label_set,
    check_meds_extract,
    write_manifest,
    write_parquet,
)

CREATED_AT = "2026-01-01T00:00:00Z"

# subject_id -> split; train has two shards (subjects 1,2 | 3), the others one.
SUBJECTS: dict[int, tuple[str, int]] = {
    1: ("train", 0),
    2: ("train", 0),
    3: ("train", 1),
    4: ("tuning", 0),
    5: ("held_out", 0),
}


def _events(subject_id: int) -> list[dict[str, object]]:
    born = datetime(1960 + subject_id, 1, 1)
    visit = datetime(2024, 3, subject_id, 8, 0)
    return [
        {"code": "DEMOSEX/F", "time": None, "numeric_value": None, "text_value": None},
        {"code": "MEDS_BIRTH", "time": born, "numeric_value": None, "text_value": None},
        {"code": "DEMOVISIT/ED", "time": visit, "numeric_value": None, "text_value": None},
        {
            "code": f"DEMOCOND/C{subject_id % 2}",
            "time": visit,
            "numeric_value": None,
            "text_value": None,
        },
        {
            "code": "DEMOLAB/L1",
            "time": datetime(2024, 3, subject_id, 9, 30),
            "numeric_value": 4.5 + subject_id,
            "text_value": None,
        },
        {
            "code": "DEMONOTE/N1",
            "time": datetime(2024, 3, subject_id, 10, 0),
            "numeric_value": None,
            "text_value": "invented",
        },
    ]


def subject_frame(subject_ids: list[int]) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for sid in subject_ids:
        rows.extend({"subject_id": sid, **event} for event in _events(sid))
    return pl.DataFrame(
        rows,
        schema={
            "subject_id": pl.Int64,
            "time": pl.Datetime("us"),
            "code": pl.String,
            "numeric_value": pl.Float32,
            "text_value": pl.String,
        },
    )


def build_extract(root: Path, *, with_manifest: bool = True) -> Path:
    """Write a valid five-subject MEDS extract (and its manifest) under ``root``."""
    shards: dict[tuple[str, int], list[int]] = {}
    for sid, key in SUBJECTS.items():
        shards.setdefault(key, []).append(sid)
    codes: set[str] = set()
    for (split, index), sids in sorted(shards.items()):
        frame = subject_frame(sorted(sids))
        codes.update(frame.get_column("code").to_list())
        write_parquet(frame, root / "data" / split / f"{index}.parquet", MEDS_DATA)

    write_parquet(
        pl.DataFrame(
            {
                "code": sorted(codes),
                "description": [None] * len(codes),
                "parent_codes": [None] * len(codes),
            },
            schema={
                "code": pl.String,
                "description": pl.String,
                "parent_codes": pl.List(pl.String),
            },
        ),
        root / "metadata" / "codes.parquet",
        MEDS_CODES,
    )
    write_parquet(
        pl.DataFrame(
            {
                "subject_id": sorted(SUBJECTS),
                "split": [SUBJECTS[s][0] for s in sorted(SUBJECTS)],
            },
            schema={"subject_id": pl.Int64, "split": pl.String},
        ),
        root / "metadata" / "subject_splits.parquet",
        SUBJECT_SPLITS,
    )
    (root / "metadata" / "dataset.json").write_text(
        json.dumps(
            {
                "dataset_name": root.name,
                "dataset_version": "1",
                "etl_name": "omopmap",
                "etl_version": "0.1.0",
                "meds_version": "0.3.3",
                "created_at": CREATED_AT,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if with_manifest:
        manifest = build_manifest(
            root,
            check_meds_extract(root),
            inputs=[InputRecord(name="omop:person", sha256="0" * 64)],
            created_at=CREATED_AT,
        )
        write_manifest(root, manifest)
    return root


def build_label_set(root: Path, extract_root: Path) -> Path:
    """Write a valid boolean label set for the extract under ``root``."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "task.json").write_text(
        json.dumps(
            {
                "name": root.name,
                "label_type": "boolean",
                "description": "Invented outcome for tests.",
                "categories": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    sids = sorted(SUBJECTS)
    write_parquet(
        pl.DataFrame(
            {
                "subject_id": sids,
                "prediction_time": [datetime(2024, 3, s, 9, 0) for s in sids],
                "boolean_value": [s % 2 == 0 for s in sids],
                "integer_value": [None] * len(sids),
                "float_value": [None] * len(sids),
                "categorical_value": [None] * len(sids),
            },
            schema={
                "subject_id": pl.Int64,
                "prediction_time": pl.Datetime("us"),
                "boolean_value": pl.Boolean,
                "integer_value": pl.Int64,
                "float_value": pl.Float64,
                "categorical_value": pl.String,
            },
        ),
        root / "labels.parquet",
        LABELS,
    )
    write_manifest(
        root,
        build_manifest(root, check_label_set(root, extract_root), created_at=CREATED_AT),
    )
    return root


def build_concept_set(root: Path) -> tuple[Path, ConceptSetSummary]:
    """Write a valid invented concept-set export under ``root``."""
    rows = [
        ("DEMOCOND", "C0", ["code"]),
        ("DEMOLAB", "L1", ["code", "numeric"]),
        ("DEMOVISIT", "ED", ["code"]),
    ]
    write_parquet(
        pl.DataFrame(
            {
                "vocabulary_id": [r[0] for r in rows],
                "concept_code": [r[1] for r in rows],
                "code": [f"{r[0]}/{r[1]}" for r in rows],
                "entry_types": [r[2] for r in rows],
            },
            schema={
                "vocabulary_id": pl.String,
                "concept_code": pl.String,
                "code": pl.String,
                "entry_types": pl.List(pl.String),
            },
        ),
        root / "concept_set.parquet",
        CONCEPT_SET,
    )
    summary = check_concept_set(root, "1" * 64)
    write_manifest(root, build_manifest(root, summary, created_at=CREATED_AT))
    return root, summary


@pytest.fixture
def extract(tmp_path: Path) -> Path:
    """A valid MEDS extract with its manifest."""
    return build_extract(tmp_path / "demo_extract")


@pytest.fixture
def bare_extract(tmp_path: Path) -> Path:
    """A valid MEDS extract without a manifest (for content-only checks)."""
    return build_extract(tmp_path / "demo_extract", with_manifest=False)


@pytest.fixture
def label_set(tmp_path: Path, extract: Path) -> Path:
    """A valid boolean label set for ``extract``."""
    return build_label_set(tmp_path / "labels" / "demo_outcome", extract)


@pytest.fixture
def concept_set(tmp_path: Path) -> Path:
    """A valid invented concept-set export."""
    return build_concept_set(tmp_path / "clmbr")[0]
