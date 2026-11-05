"""Manifests are deterministic, round-trip exactly, and fail closed."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from omopmap.artifacts import (
    CONTRACT_VERSION,
    ArtifactManifest,
    ClmbrManifest,
    CoverageReport,
    ManifestError,
    MissingFileError,
    UnexpectedFileError,
    build_manifest,
    check_compatible,
    check_meds_extract,
    concept_list_sha256,
    read_manifest,
    verify_manifest,
    write_manifest,
)

from .conftest import CREATED_AT, build_extract


def test_manifest_round_trip_is_deterministic(tmp_path: Path) -> None:
    first = build_extract(tmp_path / "a" / "demo_extract")
    second = build_extract(tmp_path / "b" / "demo_extract")
    a = (first / "artifact_manifest.json").read_bytes()
    b = (second / "artifact_manifest.json").read_bytes()
    assert a == b
    manifest = read_manifest(first)
    assert manifest.to_json().encode("utf-8") == a
    assert ArtifactManifest.model_validate_json(manifest.to_json()) == manifest
    rebuilt = build_manifest(
        first, check_meds_extract(first), inputs=manifest.inputs, created_at=CREATED_AT
    )
    assert rebuilt == manifest


def test_manifest_lists_every_file_sorted_with_rows(extract: Path) -> None:
    manifest = read_manifest(extract)
    paths = [f.path for f in manifest.files]
    assert paths == sorted(paths)
    assert "artifact_manifest.json" not in paths
    assert set(paths) == {
        "data/held_out/0.parquet",
        "data/train/0.parquet",
        "data/train/1.parquet",
        "data/tuning/0.parquet",
        "metadata/codes.parquet",
        "metadata/dataset.json",
        "metadata/subject_splits.parquet",
    }
    by_path = {f.path: f for f in manifest.files}
    assert by_path["metadata/dataset.json"].rows is None
    assert by_path["data/train/0.parquet"].rows == 12
    raw = (extract / "metadata" / "dataset.json").read_bytes()
    assert by_path["metadata/dataset.json"].sha256 == hashlib.sha256(raw).hexdigest()
    text = (extract / "artifact_manifest.json").read_text(encoding="utf-8")
    assert text.endswith("}\n")
    assert list(json.loads(text)) == sorted(json.loads(text))


def test_unlisted_file_fails(extract: Path) -> None:
    (extract / "metadata" / "extra.json").write_text("{}", encoding="utf-8")
    with pytest.raises(UnexpectedFileError, match="extra.json"):
        verify_manifest(extract)


def test_missing_manifest_fails(extract: Path) -> None:
    (extract / "artifact_manifest.json").unlink()
    with pytest.raises(MissingFileError, match="artifact_manifest.json"):
        verify_manifest(extract)


@pytest.mark.parametrize(
    ("version", "ok"),
    [
        (CONTRACT_VERSION, True),
        ("1.0.9", True),
        ("1.1.0", False),
        ("2.0.0", False),
        ("0.9.0", False),
    ],
)
def test_contract_compatibility(version: str, ok: bool) -> None:
    if ok:
        check_compatible(version)
    else:
        with pytest.raises(ManifestError):
            check_compatible(version)


def test_incompatible_manifest_version_fails(extract: Path) -> None:
    path = extract / "artifact_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["contract_version"] = "2.0.0"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ManifestError, match="major"):
        read_manifest(extract)


def test_unknown_manifest_key_fails(extract: Path) -> None:
    path = extract / "artifact_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["comment"] = "x"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ManifestError):
        read_manifest(extract)


def test_write_manifest_is_byte_stable(extract: Path) -> None:
    manifest = read_manifest(extract)
    before = (extract / "artifact_manifest.json").read_bytes()
    write_manifest(extract, manifest)
    assert (extract / "artifact_manifest.json").read_bytes() == before


def test_concept_list_hash_definition() -> None:
    codes = ["DEMOLAB/L1", "DEMOCOND/C0", "DEMOLAB/L1"]
    expected = hashlib.sha256(b"DEMOCOND/C0\nDEMOLAB/L1").hexdigest()
    assert concept_list_sha256(codes) == expected


def _clmbr_manifest(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "contract_version": CONTRACT_VERSION,
        "source_repo": "example/clmbr-model",
        "revision": "0" * 40,
        "filename": "dictionary.msgpack",
        "file_sha256": "2" * 64,
        "entry_type_counts": {"code": 3, "numeric": 1, "text": 0, "unused": 4},
        "vocabulary_counts": {"DEMOCOND": 1, "DEMOLAB": 1, "DEMOVISIT": 1},
        "total": 3,
        "care_site_excluded": 1,
        "concept_list_sha256": "3" * 64,
    }
    payload.update(overrides)
    return payload


def test_clmbr_manifest_round_trip() -> None:
    manifest = ClmbrManifest.model_validate_json(json.dumps(_clmbr_manifest()))
    assert ClmbrManifest.model_validate_json(manifest.to_json()) == manifest


@pytest.mark.parametrize(
    "overrides",
    [
        {"total": 4},
        {"vocabulary_counts": {"CARE_SITE": 1, "DEMOCOND": 1, "DEMOLAB": 1}},
        {"entry_type_counts": {"code": 3}},
        {"file_sha256": "XYZ"},
    ],
)
def test_clmbr_manifest_rejects_inconsistency(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        ClmbrManifest.model_validate_json(json.dumps(_clmbr_manifest(**overrides)))


def _coverage(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "contract_version": CONTRACT_VERSION,
        "producer": {"name": "omopmap", "version": "0.1.0"},
        "created_at": CREATED_AT,
        "extract_manifest_sha256": "4" * 64,
        "concept_list_sha256": "5" * 64,
        "distinct_codes": 4,
        "recognised_codes": 1,
        "share": 0.25,
        "by_vocabulary": [
            {"vocabulary_id": "DEMOCOND", "distinct_codes": 2, "recognised_codes": 1, "share": 0.5},
            {"vocabulary_id": "DEMOLAB", "distinct_codes": 2, "recognised_codes": 0, "share": 0.0},
        ],
        "unrecognised_top_n": 10,
        "unrecognised_min_events": 1,
        "unrecognised": [
            {"code": "DEMOLAB/L2", "events": 9},
            {"code": "DEMOCOND/C1", "events": 3},
            {"code": "DEMOLAB/L1", "events": 3},
        ],
    }
    payload.update(overrides)
    return payload


def test_coverage_report_round_trip() -> None:
    report = CoverageReport.model_validate_json(json.dumps(_coverage()))
    assert CoverageReport.model_validate_json(report.to_json()) == report


@pytest.mark.parametrize(
    "overrides",
    [
        {"share": 0.3},
        {"recognised_codes": 5},
        {"unrecognised_top_n": 2},
        {
            "unrecognised": [
                {"code": "DEMOLAB/L1", "events": 3},
                {"code": "DEMOLAB/L2", "events": 9},
            ]
        },
        {"unrecognised": [{"code": "not a code", "events": 3}]},
        {"unrecognised_min_events": 5},
    ],
)
def test_coverage_report_rejects_inconsistency(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        CoverageReport.model_validate_json(json.dumps(_coverage(**overrides)))
