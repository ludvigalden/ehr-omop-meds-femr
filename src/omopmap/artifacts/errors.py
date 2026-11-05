"""Typed, fail-closed errors raised when an artifact does not satisfy the contract."""

from __future__ import annotations


class ArtifactError(Exception):
    """Base class: an artifact directory or file does not satisfy the contract."""


class MissingFileError(ArtifactError):
    """A file or directory the contract requires is absent."""


class UnexpectedFileError(ArtifactError):
    """A file the contract does not allow is present."""


class SchemaMismatchError(ArtifactError):
    """A Parquet file's columns or Arrow types differ from the contract schema."""


class NullViolationError(ArtifactError):
    """A non-nullable column holds a null value."""


class CodeFormatError(ArtifactError):
    """A code string is not ``<vocabulary_id>/<concept_code>`` or a reserved MEDS code."""


class SortOrderError(ArtifactError):
    """Rows are not sorted by ``(subject_id, time)`` within a shard."""


class SubjectPartitionError(ArtifactError):
    """A subject spans shards or splits, or its shard disagrees with ``subject_splits``."""


class CodeMetadataError(ArtifactError):
    """``codes.parquet`` does not cover the codes in the data, or repeats a code."""


class LabelError(ArtifactError):
    """A label set violates the label contract (value columns, splits, references)."""


class ManifestError(ArtifactError):
    """A manifest is missing, malformed, incompatible, or contradicts the files."""


class HashMismatchError(ManifestError):
    """A file's SHA-256 differs from the manifest record."""
