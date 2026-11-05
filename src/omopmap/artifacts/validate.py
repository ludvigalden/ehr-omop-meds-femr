"""Fail-closed validators: check an artifact directory against the contract.

``check_*`` functions validate content and layout and return the computed
summary; ``validate_artifact`` additionally verifies the manifest's file
hashes and that its recorded summary equals the computed one. Every
violation raises a typed ``ArtifactError``; nothing degrades to a warning.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

import polars as pl
from pydantic import ValidationError

from omopmap.artifacts.contract import (
    BIRTH_CODE,
    CARE_SITE_VOCABULARY,
    CODES_PARQUET,
    CONCEPT_SET_PARQUET,
    DATA_DIR,
    DATASET_JSON,
    LABELS_PARQUET,
    MANIFEST_NAME,
    SHARD_NAME_PATTERN,
    SPLITS,
    SUBJECT_SPLITS_PARQUET,
    TASK_JSON,
    Split,
    is_valid_data_code,
    is_vocabulary_code,
)
from omopmap.artifacts.errors import (
    CodeFormatError,
    CodeMetadataError,
    LabelError,
    ManifestError,
    MissingFileError,
    NullViolationError,
    SortOrderError,
    SubjectPartitionError,
    UnexpectedFileError,
)
from omopmap.artifacts.io import read_contract_parquet
from omopmap.artifacts.manifest import artifact_files, concept_list_sha256, sha256_file
from omopmap.artifacts.manifest import verify_manifest as _verify_manifest
from omopmap.artifacts.models import (
    ArtifactManifest,
    ConceptSetSummary,
    DatasetMetadata,
    LabelSetSummary,
    MedsExtractSummary,
    TaskDefinition,
)
from omopmap.artifacts.schemas import (
    CONCEPT_SET,
    LABEL_VALUE_COLUMNS,
    LABELS,
    MEDS_CODES,
    MEDS_DATA,
    SUBJECT_SPLITS,
)

_SHARD_RE = re.compile(SHARD_NAME_PATTERN)
_SPLIT_BY_NAME: dict[str, Split] = {s: s for s in SPLITS}
_ENTRY_TYPES = frozenset({"code", "numeric", "text"})


def _require_no_nulls(frame: pl.DataFrame, columns: tuple[str, ...], label: str) -> None:
    for column in columns:
        nulls = frame.get_column(column).null_count()
        if nulls:
            raise NullViolationError(f"{label}: {nulls} null value(s) in non-nullable {column}")


def _bad_codes(codes: pl.Series, valid: Callable[[str], bool]) -> list[str]:
    values: list[str] = codes.unique().sort().to_list()
    return [c for c in values if not valid(c)][:5]


# ---------------------------------------------------------------------------
# MEDS extract
# ---------------------------------------------------------------------------


def _extract_layout(root: Path) -> dict[Split, list[str]]:
    """Return shard paths per split; fail on missing or unexpected files."""
    files = artifact_files(root)
    required = {DATASET_JSON, CODES_PARQUET, SUBJECT_SPLITS_PARQUET}
    shards: dict[Split, list[tuple[int, str]]] = {s: [] for s in SPLITS}
    for rel in files:
        if rel in required:
            continue
        parts = rel.split("/")
        split = _SPLIT_BY_NAME.get(parts[1]) if len(parts) == 3 else None
        if split is not None and parts[0] == DATA_DIR:
            if _SHARD_RE.match(parts[2]) is not None:
                shards[split].append((int(parts[2].removesuffix(".parquet")), rel))
                continue
        raise UnexpectedFileError(f"{rel} is not part of the MEDS extract layout")
    missing = sorted(required - set(files))
    if missing:
        raise MissingFileError(f"MEDS extract is missing {missing}")
    result: dict[Split, list[str]] = {}
    for split in SPLITS:
        indices = sorted(shards[split])
        if not indices:
            raise MissingFileError(f"{DATA_DIR}/{split}/ has no shards (need 0.parquet)")
        if [i for i, _ in indices] != list(range(len(indices))):
            raise MissingFileError(
                f"{DATA_DIR}/{split}/ shards must be numbered 0..{len(indices) - 1} without gaps"
            )
        result[split] = [rel for _, rel in indices]
    return result


def _check_shard(frame: pl.DataFrame, label: str) -> None:
    _require_no_nulls(frame, ("subject_id", "code"), label)
    bad = _bad_codes(frame.get_column("code"), is_valid_data_code)
    if bad:
        raise CodeFormatError(f"{label}: malformed code strings, for example {bad}")
    keys = frame.select("subject_id", "time")
    if not keys.equals(keys.sort(["subject_id", "time"], nulls_last=False)):
        raise SortOrderError(f"{label}: rows are not sorted by (subject_id, time), nulls first")
    births = (
        frame.filter(pl.col("code") == BIRTH_CODE)
        .group_by("subject_id")
        .agg(pl.len().alias("n"), pl.col("time").first().alias("birth"))
    )
    if births.filter(pl.col("n") != 1).height or births.get_column("birth").null_count():
        raise SubjectPartitionError(f"{label}: a subject has several or untimed {BIRTH_CODE} rows")
    subjects = frame.get_column("subject_id").unique()
    if births.height != subjects.len():
        raise SubjectPartitionError(f"{label}: every subject needs exactly one {BIRTH_CODE} row")
    early = (
        frame.join(births, on="subject_id")
        .filter(pl.col("time").is_not_null() & (pl.col("time") < pl.col("birth")))
        .height
    )
    if early:
        raise SortOrderError(f"{label}: {early} event(s) precede the subject's {BIRTH_CODE}")


def check_meds_extract(root: Path) -> MedsExtractSummary:
    """Validate a MEDS extract's layout and content; return its summary."""
    layout = _extract_layout(root)
    try:
        DatasetMetadata.model_validate_json((root / DATASET_JSON).read_text(encoding="utf-8"))
    except ValidationError as exc:
        raise ManifestError(f"{DATASET_JSON} is invalid: {exc}") from exc

    splits = read_contract_parquet(
        root / SUBJECT_SPLITS_PARQUET, SUBJECT_SPLITS, SUBJECT_SPLITS_PARQUET
    )
    _require_no_nulls(splits, SUBJECT_SPLITS.names, SUBJECT_SPLITS_PARQUET)
    unknown = sorted(set(splits.get_column("split").to_list()) - set(SPLITS))
    if unknown:
        raise SubjectPartitionError(f"{SUBJECT_SPLITS_PARQUET}: unknown split names {unknown}")
    ids = splits.get_column("subject_id")
    if ids.n_unique() != ids.len():
        raise SubjectPartitionError(f"{SUBJECT_SPLITS_PARQUET}: a subject is listed twice")
    if not ids.equals(ids.sort()):
        raise SortOrderError(f"{SUBJECT_SPLITS_PARQUET}: rows are not sorted by subject_id")

    owners: list[pl.DataFrame] = []
    codes: list[pl.DataFrame] = []
    events_by_split: dict[Split, int] = {s: 0 for s in SPLITS}
    for split, rels in layout.items():
        for rel in rels:
            frame = read_contract_parquet(root / rel, MEDS_DATA, rel)
            _check_shard(frame, rel)
            events_by_split[split] += frame.height
            owners.append(
                frame.select("subject_id")
                .unique()
                .with_columns(pl.lit(split).alias("split"), pl.lit(rel).alias("shard"))
            )
            codes.append(frame.group_by("code").agg(pl.len().alias("events")))

    owner = pl.concat(owners)
    repeated = owner.group_by("subject_id").len().filter(pl.col("len") > 1)
    if repeated.height:
        sample = repeated.get_column("subject_id").sort().head(5).to_list()
        raise SubjectPartitionError(f"subjects span several shards, for example {sample}")
    joined = owner.select("subject_id", "split").sort("subject_id")
    if not joined.equals(splits):
        raise SubjectPartitionError(
            f"{SUBJECT_SPLITS_PARQUET} does not list exactly the subjects in {DATA_DIR}/ "
            "with the split of the directory that holds them"
        )

    data_codes = sorted(pl.concat(codes).get_column("code").unique().to_list())
    meta = read_contract_parquet(root / CODES_PARQUET, MEDS_CODES, CODES_PARQUET)
    _require_no_nulls(meta, ("code",), CODES_PARQUET)
    meta_codes = meta.get_column("code").to_list()
    if meta_codes != sorted(set(meta_codes)):
        raise CodeMetadataError(f"{CODES_PARQUET}: codes must be unique and sorted")
    if meta_codes != data_codes:
        missing = sorted(set(data_codes) - set(meta_codes))[:5]
        extra = sorted(set(meta_codes) - set(data_codes))[:5]
        raise CodeMetadataError(
            f"{CODES_PARQUET} must list exactly the codes in {DATA_DIR}/ "
            f"(missing e.g. {missing}, extra e.g. {extra})"
        )

    subjects_by_split: dict[Split, int] = {s: 0 for s in SPLITS}
    for split, count in joined.group_by("split").len().iter_rows():
        subjects_by_split[split] = count
    return MedsExtractSummary(
        artifact_type="meds_extract",
        meds_version="0.3.3",
        subjects=joined.height,
        events=sum(events_by_split.values()),
        distinct_codes=len(data_codes),
        subjects_by_split=subjects_by_split,
        events_by_split=events_by_split,
    )


def read_subject_splits(extract_root: Path) -> pl.DataFrame:
    """Read an extract's ``subject_splits.parquet`` (schema-checked)."""
    return read_contract_parquet(
        extract_root / SUBJECT_SPLITS_PARQUET, SUBJECT_SPLITS, SUBJECT_SPLITS_PARQUET
    )


# ---------------------------------------------------------------------------
# Label set
# ---------------------------------------------------------------------------


def _exact_files(root: Path, expected: set[str], what: str) -> None:
    present = set(artifact_files(root))
    missing = sorted(expected - present)
    if missing:
        raise MissingFileError(f"{what} is missing {missing}")
    extra = sorted(present - expected)
    if extra:
        raise UnexpectedFileError(f"{what} has unexpected files {extra}")


def check_label_set(root: Path, extract_root: Path) -> LabelSetSummary:
    """Validate a label set against the extract it labels; return its summary."""
    _exact_files(root, {LABELS_PARQUET, TASK_JSON}, "label set")
    try:
        task = TaskDefinition.model_validate_json((root / TASK_JSON).read_text(encoding="utf-8"))
    except ValidationError as exc:
        raise LabelError(f"{TASK_JSON} is invalid: {exc}") from exc
    if task.name != root.name:
        raise LabelError(f"task name {task.name!r} differs from directory {root.name!r}")

    extract_manifest = extract_root / MANIFEST_NAME
    if not extract_manifest.is_file():
        raise MissingFileError(f"labelled extract has no {MANIFEST_NAME}: {extract_root}")

    labels = read_contract_parquet(root / LABELS_PARQUET, LABELS, LABELS_PARQUET)
    _require_no_nulls(labels, ("subject_id", "prediction_time"), LABELS_PARQUET)
    value_column = LABEL_VALUE_COLUMNS[task.label_type]
    if labels.get_column(value_column).null_count():
        raise LabelError(
            f"{LABELS_PARQUET}: {value_column} must be set for a {task.label_type} task"
        )
    for other in LABEL_VALUE_COLUMNS.values():
        if other != value_column and labels.get_column(other).null_count() != labels.height:
            raise LabelError(f"{LABELS_PARQUET}: {other} must be null for a {task.label_type} task")
    if task.categories is not None:
        unknown = sorted(set(labels.get_column(value_column).to_list()) - set(task.categories))
        if unknown:
            raise LabelError(f"{LABELS_PARQUET}: categories {unknown} are not in {TASK_JSON}")

    keys = labels.select("subject_id", "prediction_time")
    if not keys.equals(keys.sort(["subject_id", "prediction_time"])):
        raise SortOrderError(
            f"{LABELS_PARQUET}: rows are not sorted by (subject_id, prediction_time)"
        )
    if keys.is_duplicated().any():
        raise LabelError(f"{LABELS_PARQUET}: (subject_id, prediction_time) pairs must be unique")

    splits = read_subject_splits(extract_root)
    with_split = labels.select("subject_id").join(splits, on="subject_id", how="left")
    orphans = with_split.filter(pl.col("split").is_null()).height
    if orphans:
        raise LabelError(f"{LABELS_PARQUET}: {orphans} label(s) for subjects not in the extract")
    labels_by_split: dict[Split, int] = {s: 0 for s in SPLITS}
    for split, count in with_split.group_by("split").len().iter_rows():
        labels_by_split[split] = count
    return LabelSetSummary(
        artifact_type="label_set",
        task_name=task.name,
        label_type=task.label_type,
        extract_manifest_sha256=sha256_file(extract_manifest),
        labels=labels.height,
        subjects=labels.get_column("subject_id").n_unique(),
        labels_by_split=labels_by_split,
    )


# ---------------------------------------------------------------------------
# CLMBR concept set
# ---------------------------------------------------------------------------


def check_concept_set(root: Path, dictionary_sha256: str) -> ConceptSetSummary:
    """Validate a CLMBR concept-set export; return its summary."""
    _exact_files(root, {CONCEPT_SET_PARQUET}, "concept set")
    frame = read_contract_parquet(root / CONCEPT_SET_PARQUET, CONCEPT_SET, CONCEPT_SET_PARQUET)
    _require_no_nulls(frame, CONCEPT_SET.names, CONCEPT_SET_PARQUET)
    bad = _bad_codes(frame.get_column("code"), is_vocabulary_code)
    if bad:
        raise CodeFormatError(f"{CONCEPT_SET_PARQUET}: malformed codes, for example {bad}")
    rebuilt = frame.select(
        (pl.col("vocabulary_id") + pl.lit("/") + pl.col("concept_code")).alias("code")
    ).get_column("code")
    if not rebuilt.equals(frame.get_column("code")):
        raise CodeFormatError(
            f"{CONCEPT_SET_PARQUET}: code differs from vocabulary_id/concept_code"
        )
    codes: list[str] = frame.get_column("code").to_list()
    if codes != sorted(set(codes)):
        raise SortOrderError(f"{CONCEPT_SET_PARQUET}: codes must be unique and sorted")
    if (frame.get_column("vocabulary_id") == CARE_SITE_VOCABULARY).any():
        raise CodeFormatError(f"{CONCEPT_SET_PARQUET}: {CARE_SITE_VOCABULARY} rows are excluded")
    for types in frame.get_column("entry_types").to_list():
        values = list(types)
        if not values or values != sorted(set(values)) or not set(values) <= _ENTRY_TYPES:
            raise CodeFormatError(
                f"{CONCEPT_SET_PARQUET}: entry_types must be a sorted, non-empty subset of "
                f"{sorted(_ENTRY_TYPES)}, got {values}"
            )
    counts: dict[str, int] = {
        vocabulary: count
        for vocabulary, count in frame.group_by("vocabulary_id")
        .len()
        .sort("vocabulary_id")
        .iter_rows()
    }
    return ConceptSetSummary(
        artifact_type="clmbr_concept_set",
        dictionary_sha256=dictionary_sha256,
        concepts=frame.height,
        concept_list_sha256=concept_list_sha256(codes),
        vocabulary_counts=counts,
    )


# ---------------------------------------------------------------------------
# Manifest-level entry point
# ---------------------------------------------------------------------------


def validate_artifact(root: Path, extract_root: Path | None = None) -> ArtifactManifest:
    """Verify an artifact's manifest and content; return the manifest.

    ``extract_root`` is required for label sets (the extract they label).
    """
    manifest = _verify_manifest(root)
    summary = manifest.summary
    match summary:
        case MedsExtractSummary():
            computed = check_meds_extract(root)
        case LabelSetSummary():
            if extract_root is None:
                raise MissingFileError("validating a label set needs the labelled extract")
            computed = check_label_set(root, extract_root)
        case ConceptSetSummary():
            computed = check_concept_set(root, summary.dictionary_sha256)
    if computed != summary:
        raise ManifestError(
            f"manifest summary differs from the files: recorded {summary.model_dump()}, "
            f"computed {computed.model_dump()}"
        )
    return manifest
