"""Write a MEDS extract from an event frame: shards, metadata, validation, manifest.

The writer is the only place a MEDS extract is created, so every producer (the
synthetic generator and the OMOP converter) gets the same layout, the same
contract casts and the same post-write validation. The manifest is written
last, after :func:`omopmap.artifacts.check_meds_extract` has accepted the
files, so a directory without ``artifact_manifest.json`` is incomplete.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import polars as pl

from omopmap import __version__
from omopmap.artifacts.contract import (
    CODES_PARQUET,
    DATA_DIR,
    DATASET_JSON,
    MEDS_VERSION,
    SUBJECT_SPLITS_PARQUET,
)
from omopmap.artifacts.errors import ArtifactError
from omopmap.artifacts.io import write_parquet
from omopmap.artifacts.manifest import build_manifest, write_manifest
from omopmap.artifacts.models import ArtifactManifest, DatasetMetadata, InputRecord
from omopmap.artifacts.schemas import MEDS_CODES, MEDS_DATA, SUBJECT_SPLITS
from omopmap.artifacts.validate import check_meds_extract
from omopmap.meds.events import codes_frame, normalise_events
from omopmap.meds.splits import SplitConfig, assign_splits, shard_plan


class OutputExistsError(ArtifactError):
    """The output directory exists and is not empty; artifacts are never overwritten."""


def prepare_output(root: Path) -> None:
    """Create ``root``; fail closed if it already holds anything."""
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise OutputExistsError(f"{root} exists and is not empty; choose a new directory")
    root.mkdir(parents=True, exist_ok=True)


def write_extract(
    root: Path,
    events: pl.DataFrame,
    *,
    split_config: SplitConfig,
    dataset_version: str,
    created_at: str,
    inputs: Sequence[InputRecord] = (),
    descriptions: dict[str, str] | None = None,
    parents: dict[str, list[str]] | None = None,
) -> ArtifactManifest:
    """Write ``events`` as a contract MEDS extract under ``root`` and return its manifest.

    ``root`` must not exist or be empty. The directory name is the dataset name.
    """
    prepare_output(root)
    ordered = normalise_events(events)
    splits = assign_splits(ordered.get_column("subject_id").unique().to_list(), split_config)

    for split, shards in shard_plan(splits, split_config.subjects_per_shard).items():
        for index, subject_ids in enumerate(shards):
            shard = ordered.filter(pl.col("subject_id").is_in(subject_ids))
            write_parquet(shard, root / DATA_DIR / split / f"{index}.parquet", MEDS_DATA)

    write_parquet(codes_frame(ordered, descriptions, parents), root / CODES_PARQUET, MEDS_CODES)
    write_parquet(splits, root / SUBJECT_SPLITS_PARQUET, SUBJECT_SPLITS)
    metadata = DatasetMetadata(
        dataset_name=root.name,
        dataset_version=dataset_version,
        etl_name="omopmap",
        etl_version=__version__,
        meds_version=MEDS_VERSION,
        created_at=created_at,
    )
    (root / DATASET_JSON).write_text(metadata.to_json(), encoding="utf-8")

    summary = check_meds_extract(root)
    manifest = build_manifest(root, summary, inputs=inputs, created_at=created_at)
    write_manifest(root, manifest)
    return manifest
