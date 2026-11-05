"""Write a contract label set: ``task.json``, ``labels.parquet``, validation, manifest."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from omopmap.artifacts.contract import LABELS_PARQUET, TASK_JSON
from omopmap.artifacts.errors import LabelError
from omopmap.artifacts.io import write_parquet
from omopmap.artifacts.manifest import build_manifest, write_manifest
from omopmap.artifacts.models import ArtifactManifest, TaskDefinition
from omopmap.artifacts.schemas import LABELS
from omopmap.artifacts.validate import check_label_set
from omopmap.meds.writer import prepare_output


def write_label_set(
    root: Path,
    labels: pl.DataFrame,
    task: TaskDefinition,
    *,
    extract_root: Path,
    created_at: str,
) -> ArtifactManifest:
    """Write a label set under ``root`` (whose name must equal the task name).

    The set is validated against the extract before its manifest is written.
    """
    if root.name != task.name:
        raise LabelError(
            f"output directory {root.name!r} must be named after the task {task.name!r}"
        )
    prepare_output(root)
    (root / TASK_JSON).write_text(task.to_json(), encoding="utf-8")
    write_parquet(labels, root / LABELS_PARQUET, LABELS)
    summary = check_label_set(root, extract_root)
    manifest = build_manifest(root, summary, created_at=created_at)
    write_manifest(root, manifest)
    return manifest
