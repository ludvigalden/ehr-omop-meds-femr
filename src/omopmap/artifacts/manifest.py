"""Manifest helpers: per-file SHA-256, deterministic manifest write, fail-closed verify."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from pathlib import Path

from pydantic import ValidationError

from omopmap.artifacts.contract import MANIFEST_NAME, check_compatible
from omopmap.artifacts.errors import (
    HashMismatchError,
    ManifestError,
    MissingFileError,
    UnexpectedFileError,
)
from omopmap.artifacts.io import parquet_rows
from omopmap.artifacts.models import (
    ArtifactManifest,
    ArtifactSummary,
    FileRecord,
    InputRecord,
    producer,
    utc_timestamp,
)

_CHUNK = 1 << 20


def sha256_file(path: Path) -> str:
    """SHA-256 of a file's bytes, hex-encoded."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def concept_list_sha256(codes: Iterable[str]) -> str:
    """Hash of a concept list: SHA-256 of the UTF-8 bytes of the distinct
    ``<vocabulary_id>/<concept_code>`` strings, sorted by code point and joined
    with ``\\n`` (no trailing newline)."""
    joined = "\n".join(sorted(set(codes)))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def artifact_files(root: Path) -> list[str]:
    """Relative POSIX paths of every file under ``root`` except the manifest, sorted.

    Fails closed on symlinks and on anything that is neither a file nor a directory.
    """
    if not root.is_dir():
        raise MissingFileError(f"artifact directory {root} does not exist")
    paths: list[str] = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise UnexpectedFileError(f"{rel}: symlinks are not allowed in an artifact")
        if path.is_dir():
            continue
        if not path.is_file():
            raise UnexpectedFileError(f"{rel}: not a regular file")
        if rel != MANIFEST_NAME:
            paths.append(rel)
    return sorted(paths)


def file_record(root: Path, rel: str) -> FileRecord:
    """Build the manifest record for one file."""
    path = root / rel
    rows = parquet_rows(path) if rel.endswith(".parquet") else None
    return FileRecord(path=rel, sha256=sha256_file(path), bytes=path.stat().st_size, rows=rows)


def build_manifest(
    root: Path,
    summary: ArtifactSummary,
    inputs: Sequence[InputRecord] = (),
    created_at: str | None = None,
) -> ArtifactManifest:
    """Hash every file under ``root`` and assemble the manifest (not written)."""
    from omopmap.artifacts.contract import CONTRACT_VERSION

    return ArtifactManifest(
        contract_version=CONTRACT_VERSION,
        artifact_type=summary.artifact_type,
        producer=producer(),
        created_at=created_at or utc_timestamp(),
        inputs=tuple(sorted(inputs, key=lambda i: i.name)),
        files=tuple(file_record(root, rel) for rel in artifact_files(root)),
        summary=summary,
    )


def write_manifest(root: Path, manifest: ArtifactManifest) -> Path:
    """Write ``artifact_manifest.json`` deterministically; return its path."""
    path = root / MANIFEST_NAME
    path.write_text(manifest.to_json(), encoding="utf-8")
    return path


def read_manifest(root: Path) -> ArtifactManifest:
    """Parse and contract-check ``artifact_manifest.json`` under ``root``."""
    path = root / MANIFEST_NAME
    if not path.is_file():
        raise MissingFileError(f"{MANIFEST_NAME} is missing under {root}")
    text = path.read_text(encoding="utf-8")
    try:
        version = json.loads(text).get("contract_version")
    except (json.JSONDecodeError, AttributeError) as exc:
        raise ManifestError(f"{MANIFEST_NAME} is not a JSON object: {exc}") from exc
    if not isinstance(version, str):
        raise ManifestError(f"{MANIFEST_NAME} has no contract_version string")
    check_compatible(version)
    try:
        return ArtifactManifest.model_validate_json(text)
    except ValidationError as exc:
        raise ManifestError(f"{MANIFEST_NAME} is invalid: {exc}") from exc


def verify_manifest(root: Path) -> ArtifactManifest:
    """Check the files under ``root`` against its manifest; raise on any difference.

    The file set must match exactly (no missing and no unlisted files), and every
    file's size, SHA-256 and (for Parquet) row count must equal the record.
    """
    manifest = read_manifest(root)
    listed = {f.path: f for f in manifest.files}
    present = set(artifact_files(root))
    missing = sorted(set(listed) - present)
    if missing:
        raise MissingFileError(f"files listed in the manifest are missing: {missing}")
    unlisted = sorted(present - set(listed))
    if unlisted:
        raise UnexpectedFileError(f"files not listed in the manifest: {unlisted}")
    for rel, record in sorted(listed.items()):
        # Bytes are compared before any Parquet footer is parsed, so a corrupted
        # file fails as a typed hash mismatch rather than a reader error.
        path = root / rel
        digest, size = sha256_file(path), path.stat().st_size
        if digest != record.sha256 or size != record.bytes:
            raise HashMismatchError(
                f"{rel}: sha256 {digest} ({size} bytes) differs from "
                f"the manifest's {record.sha256} ({record.bytes} bytes)"
            )
        rows = parquet_rows(path) if rel.endswith(".parquet") else None
        if rows != record.rows:
            raise ManifestError(f"{rel}: {rows} rows, manifest says {record.rows}")
    return manifest
