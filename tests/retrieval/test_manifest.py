"""Run manifests: deterministic JSON, encoder record, fail-closed lookups."""

from __future__ import annotations

import pytest

from omopmap import __version__
from omopmap.retrieval.manifest import (
    RunManifestError,
    build_run_manifest,
    parse_run_manifest,
)
from tests.retrieval.conftest import CREATED_AT, FakeEncoder


def _manifest():  # type: ignore[no-untyped-def]
    return build_run_manifest(
        "embed",
        created_at=CREATED_AT,
        parameters={"model": "sapbert", "depth": 200},
        inputs={"queries": "b" * 64, "index:matrix.npy": "a" * 64},
        outputs={"rankings.tsv": "c" * 64},
        encoder=FakeEncoder().info(),
        seed=7,
        truncated=0,
    )


def test_round_trip_and_sorted_records() -> None:
    manifest = _manifest()
    text = manifest.to_json()
    assert parse_run_manifest(text) == manifest
    assert [r.name for r in manifest.inputs] == ["index:matrix.npy", "queries"]
    assert manifest.producer.version == __version__
    assert manifest.encoder is not None and manifest.encoder.gpu is None
    assert manifest.channel() == "embed:sapbert"
    assert manifest.output_sha256("rankings.tsv") == "c" * 64


def test_missing_output_and_bad_json_fail_closed() -> None:
    with pytest.raises(RunManifestError, match="no output"):
        _manifest().output_sha256("matrix.npy")
    with pytest.raises(RunManifestError):
        parse_run_manifest('{"run_type": "embed"}')


def test_encoder_revision_must_be_a_full_commit() -> None:
    with pytest.raises(ValueError):
        build_run_manifest(
            "index",
            created_at=CREATED_AT,
            parameters={},
            inputs={},
            outputs={},
            encoder=FakeEncoder(revision="main").info(),
        )
