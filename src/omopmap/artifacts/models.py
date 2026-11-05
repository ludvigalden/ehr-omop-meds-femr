"""Pydantic models for the JSON artifacts: manifests, task definitions, coverage reports.

All models are frozen and reject unknown keys. Serialisation is deterministic:
``to_json`` sorts keys, indents by two spaces, keeps non-ASCII text and ends with
a newline, and every list a model holds is sorted by the field its validator
names, so the same content always produces the same bytes.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Annotated, Literal, Self

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

from omopmap.artifacts.contract import (
    CARE_SITE_VOCABULARY,
    CODE_PATTERN,
    CONTRACT_VERSION,
    MEDS_VERSION,
    SPLITS,
    Split,
)

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Timestamp = Annotated[str, StringConstraints(pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")]
SemVer = Annotated[str, StringConstraints(pattern=r"^\d+\.\d+\.\d+$")]


def _relative_posix(value: str) -> str:
    parts = value.split("/")
    if not value or "\\" in value or "\n" in value or any(p in ("", ".", "..") for p in parts):
        raise ValueError(f"{value!r} is not a normalised relative POSIX path")
    return value


RelPath = Annotated[str, AfterValidator(_relative_posix)]
TaskName = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9_]*$")]
VocabularyCode = Annotated[str, StringConstraints(pattern=CODE_PATTERN)]
LabelType = Literal["boolean", "integer", "float", "categorical"]
EntryType = Literal["code", "numeric", "text", "unused"]


def utc_timestamp(moment: datetime | None = None) -> str:
    """Format a UTC timestamp the way every contract JSON file records it."""
    value = (moment or datetime.now(UTC)).astimezone(UTC)
    return value.strftime("%Y-%m-%dT%H:%M:%SZ")


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    def to_json(self) -> str:
        """Deterministic JSON: sorted keys, two-space indent, trailing newline."""
        payload = self.model_dump(mode="json")
        return json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def _require_sorted_unique(values: list[str], what: str) -> None:
    if values != sorted(set(values)):
        raise ValueError(f"{what} must be sorted and unique")


def _require_all_splits(counts: dict[Split, int], what: str) -> None:
    if set(counts) != set(SPLITS):
        raise ValueError(f"{what} must have exactly the keys {list(SPLITS)}")
    if any(v < 0 for v in counts.values()):
        raise ValueError(f"{what} values must be non-negative")


class Producer(_Frozen):
    """The tool and version that wrote an artifact."""

    name: Literal["omopmap"] = Field(description="Producing package; always `omopmap`.")
    version: SemVer = Field(description="Producing package version.")


class InputRecord(_Frozen):
    """A hashed input the artifact was derived from."""

    name: str = Field(
        min_length=1,
        description="Role of the input, for example `omop:person` or `dictionary`.",
    )
    sha256: Sha256 = Field(description="SHA-256 of the input file, or of its own manifest.")


class FileRecord(_Frozen):
    """One file of an artifact directory."""

    path: RelPath = Field(description="POSIX path relative to the artifact root.")
    sha256: Sha256 = Field(description="SHA-256 of the file bytes.")
    bytes: int = Field(ge=0, description="File size in bytes.")
    rows: int | None = Field(
        ge=0,
        description="Row count for Parquet files (from the footer); null for other files.",
    )


class MedsExtractSummary(_Frozen):
    """Counts recorded for a MEDS extract."""

    artifact_type: Literal["meds_extract"] = Field(description="Discriminator.")
    meds_version: Literal["0.3.3"] = Field(description="MEDS release the data follows.")
    subjects: int = Field(ge=0, description="Distinct subjects in `data/`.")
    events: int = Field(ge=0, description="Rows in `data/`.")
    distinct_codes: int = Field(ge=0, description="Distinct `code` values in `data/`.")
    subjects_by_split: dict[Split, int] = Field(description="Subjects per split; all splits.")
    events_by_split: dict[Split, int] = Field(description="Rows per split; all splits.")

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        _require_all_splits(self.subjects_by_split, "subjects_by_split")
        _require_all_splits(self.events_by_split, "events_by_split")
        if sum(self.subjects_by_split.values()) != self.subjects:
            raise ValueError("subjects_by_split does not sum to subjects")
        if sum(self.events_by_split.values()) != self.events:
            raise ValueError("events_by_split does not sum to events")
        return self


class LabelSetSummary(_Frozen):
    """Counts recorded for a label set."""

    artifact_type: Literal["label_set"] = Field(description="Discriminator.")
    task_name: TaskName = Field(description="Equals `task.json` `name` and the directory name.")
    label_type: LabelType = Field(description="Equals `task.json` `label_type`.")
    extract_manifest_sha256: Sha256 = Field(
        description="SHA-256 of the labelled extract's `artifact_manifest.json`.",
    )
    labels: int = Field(ge=0, description="Rows in `labels.parquet`.")
    subjects: int = Field(ge=0, description="Distinct labelled subjects.")
    labels_by_split: dict[Split, int] = Field(
        description="Rows per split of the labelled subject; all splits.",
    )

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        _require_all_splits(self.labels_by_split, "labels_by_split")
        if sum(self.labels_by_split.values()) != self.labels:
            raise ValueError("labels_by_split does not sum to labels")
        return self


class ConceptSetSummary(_Frozen):
    """Counts recorded for a CLMBR concept-set export."""

    artifact_type: Literal["clmbr_concept_set"] = Field(description="Discriminator.")
    dictionary_sha256: Sha256 = Field(description="SHA-256 of the source dictionary file.")
    concepts: int = Field(ge=0, description="Rows in `concept_set.parquet`.")
    concept_list_sha256: Sha256 = Field(
        description="Concept-list hash; equals the CLMBR manifest's `concept_list_sha256`.",
    )
    vocabulary_counts: dict[str, int] = Field(description="Concepts per vocabulary id.")

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if sum(self.vocabulary_counts.values()) != self.concepts:
            raise ValueError("vocabulary_counts does not sum to concepts")
        if CARE_SITE_VOCABULARY in self.vocabulary_counts:
            raise ValueError(f"{CARE_SITE_VOCABULARY} entries are excluded from the concept set")
        return self


ArtifactSummary = Annotated[
    MedsExtractSummary | LabelSetSummary | ConceptSetSummary,
    Field(discriminator="artifact_type"),
]
ArtifactType = Literal["meds_extract", "label_set", "clmbr_concept_set"]


class ArtifactManifest(_Frozen):
    """``artifact_manifest.json``: the integrity record at the root of every artifact."""

    contract_version: SemVer = Field(description="Artifact contract version the files follow.")
    artifact_type: ArtifactType = Field(description="Kind of artifact; equals `summary`'s.")
    producer: Producer = Field(description="Producing tool and version.")
    created_at: Timestamp = Field(description="UTC creation time, `YYYY-MM-DDTHH:MM:SSZ`.")
    inputs: tuple[InputRecord, ...] = Field(
        description="Hashed inputs, sorted by `name`; names unique.",
    )
    files: tuple[FileRecord, ...] = Field(
        description="Every file under the root except this manifest, sorted by `path`.",
    )
    summary: ArtifactSummary = Field(description="Type-specific counts.")

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.summary.artifact_type != self.artifact_type:
            raise ValueError("artifact_type differs from summary.artifact_type")
        _require_sorted_unique([i.name for i in self.inputs], "inputs (by name)")
        _require_sorted_unique([f.path for f in self.files], "files (by path)")
        return self


class TaskDefinition(_Frozen):
    """``task.json``: what a label set predicts."""

    name: TaskName = Field(description="Task name; equals the label-set directory name.")
    label_type: LabelType = Field(description="Which label value column is set.")
    description: str = Field(min_length=1, description="Plain-language definition of the label.")
    categories: tuple[str, ...] | None = Field(
        description="Allowed `categorical_value` values, sorted; null unless categorical.",
    )

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.label_type == "categorical":
            if not self.categories:
                raise ValueError("categorical tasks must list their categories")
            _require_sorted_unique(list(self.categories), "categories")
        elif self.categories is not None:
            raise ValueError("categories are only allowed for categorical tasks")
        return self


class DatasetMetadata(_Frozen):
    """``metadata/dataset.json`` as written by this producer (MEDS 0.3.3 fields)."""

    dataset_name: str = Field(min_length=1, description="Extract name; the directory name.")
    dataset_version: str = Field(min_length=1, description="Extract version label.")
    etl_name: Literal["omopmap"] = Field(description="Always `omopmap`.")
    etl_version: SemVer = Field(description="Producing package version.")
    meds_version: Literal["0.3.3"] = Field(description="Always `0.3.3`.")
    created_at: Timestamp = Field(description="UTC creation time.")


class ClmbrManifest(_Frozen):
    """``vocab/clmbr/manifest.json``: committed counts and hashes of the import."""

    contract_version: SemVer = Field(description="Artifact contract version.")
    source_repo: str = Field(min_length=1, description="Hugging Face repo id of the release.")
    revision: str = Field(
        pattern=r"^[0-9a-f]{40}$",
        description="Pinned commit of that repo, which both acquisition routes are pinned to.",
    )
    filename: str = Field(min_length=1, description="Dictionary file name within the release.")
    file_sha256: Sha256 = Field(description="SHA-256 of the dictionary file.")
    entry_type_counts: dict[EntryType, int] = Field(
        description="Dictionary entries per entry type, all four types.",
    )
    vocabulary_counts: dict[str, int] = Field(
        description="Distinct concepts per vocabulary id, `CARE_SITE` excluded.",
    )
    total: int = Field(ge=0, description="Distinct concepts, `CARE_SITE` excluded.")
    care_site_excluded: int = Field(ge=0, description="Distinct `CARE_SITE` refs excluded.")
    concept_list_sha256: Sha256 = Field(description="Hash of the concept list (see below).")

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if set(self.entry_type_counts) != {"code", "numeric", "text", "unused"}:
            raise ValueError("entry_type_counts must list all four entry types")
        if sum(self.vocabulary_counts.values()) != self.total:
            raise ValueError("vocabulary_counts does not sum to total")
        if CARE_SITE_VOCABULARY in self.vocabulary_counts:
            raise ValueError(f"{CARE_SITE_VOCABULARY} must not appear in vocabulary_counts")
        return self


class VocabularyCoverage(_Frozen):
    """Coverage of one vocabulary id."""

    vocabulary_id: str = Field(min_length=1, description="Vocabulary id from the code prefix.")
    distinct_codes: int = Field(ge=0, description="Distinct extract codes in the vocabulary.")
    recognised_codes: int = Field(ge=0, description="Of those, codes in the concept set.")
    share: float = Field(ge=0.0, le=1.0, description="`recognised_codes / distinct_codes`.")


class UnrecognisedCode(_Frozen):
    """An extract code absent from the concept set (code string only)."""

    code: VocabularyCode = Field(description="Code string; no description or other text.")
    events: int = Field(ge=1, description="Rows in the extract carrying the code.")


class CoverageReport(_Frozen):
    """``omopmap clmbr coverage`` output: share of extract codes the dictionary knows."""

    contract_version: SemVer = Field(description="Artifact contract version.")
    producer: Producer = Field(description="Producing tool and version.")
    created_at: Timestamp = Field(description="UTC creation time.")
    extract_manifest_sha256: Sha256 = Field(description="SHA-256 of the extract's manifest.")
    concept_list_sha256: Sha256 = Field(description="Concept-list hash of the dictionary used.")
    distinct_codes: int = Field(
        ge=0,
        description="Distinct vocabulary codes in the extract (reserved `MEDS_*` codes excluded).",
    )
    recognised_codes: int = Field(ge=0, description="Of those, codes present in the concept set.")
    share: float = Field(
        ge=0.0, le=1.0, description="`recognised_codes / distinct_codes` (0 when empty)."
    )
    by_vocabulary: tuple[VocabularyCoverage, ...] = Field(
        description="Per vocabulary id, sorted by `vocabulary_id`.",
    )
    unrecognised_top_n: int = Field(ge=0, description="Maximum length of `unrecognised`.")
    unrecognised_min_events: int = Field(
        ge=1, description="Codes with fewer events are never listed."
    )
    unrecognised: tuple[UnrecognisedCode, ...] = Field(
        description="Most frequent unrecognised codes: by `events` descending, then `code`.",
    )

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.recognised_codes > self.distinct_codes:
            raise ValueError("recognised_codes exceeds distinct_codes")
        if abs(self.share - _ratio(self.recognised_codes, self.distinct_codes)) > 1e-12:
            raise ValueError("share differs from recognised_codes / distinct_codes")
        _require_sorted_unique([v.vocabulary_id for v in self.by_vocabulary], "by_vocabulary")
        if sum(v.distinct_codes for v in self.by_vocabulary) != self.distinct_codes:
            raise ValueError("by_vocabulary distinct_codes do not sum to distinct_codes")
        if sum(v.recognised_codes for v in self.by_vocabulary) != self.recognised_codes:
            raise ValueError("by_vocabulary recognised_codes do not sum to recognised_codes")
        for v in self.by_vocabulary:
            if abs(v.share - _ratio(v.recognised_codes, v.distinct_codes)) > 1e-12:
                raise ValueError(f"share of {v.vocabulary_id} is inconsistent")
        if len(self.unrecognised) > self.unrecognised_top_n:
            raise ValueError("unrecognised exceeds unrecognised_top_n")
        order = sorted(self.unrecognised, key=lambda u: (-u.events, u.code))
        if list(self.unrecognised) != order or len({u.code for u in order}) != len(order):
            raise ValueError("unrecognised must be unique and ordered by events desc, code")
        if any(u.events < self.unrecognised_min_events for u in self.unrecognised):
            raise ValueError("unrecognised lists a code below unrecognised_min_events")
        return self


def _ratio(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def producer() -> Producer:
    """The ``Producer`` record for this installation."""
    from omopmap import __version__

    return Producer(name="omopmap", version=__version__)


__all__ = [
    "CONTRACT_VERSION",
    "MEDS_VERSION",
    "ArtifactManifest",
    "ArtifactSummary",
    "ArtifactType",
    "ClmbrManifest",
    "ConceptSetSummary",
    "CoverageReport",
    "DatasetMetadata",
    "FileRecord",
    "InputRecord",
    "LabelSetSummary",
    "MedsExtractSummary",
    "Producer",
    "TaskDefinition",
    "UnrecognisedCode",
    "VocabularyCoverage",
    "producer",
    "utc_timestamp",
]
