"""B2 smoke tests: package imports, fixture byte-stability, fake encoder."""

from __future__ import annotations

import importlib.metadata
import importlib.util
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

import omopmap
from omopmap.cli import app

FIXTURES_DIR = Path(__file__).parent / "fixtures"
FIXTURE_FILES = ("concepts.csv", "source_terms.csv")


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_package_version_matches_distribution() -> None:
    assert omopmap.__version__ == importlib.metadata.version("omopmap")


def test_cli_version_flag() -> None:
    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0
    assert f"omopmap {omopmap.__version__}" in result.output


def test_fixtures_rebuild_byte_stable(tmp_path: Path) -> None:
    """Rebuilding the fixtures from the committed builder must reproduce
    the committed files byte for byte."""
    builder = _load_module("build_fixtures", FIXTURES_DIR / "build_fixtures.py")
    builder.main(tmp_path)
    for name in FIXTURE_FILES:
        committed = (FIXTURES_DIR / name).read_bytes()
        rebuilt = (tmp_path / name).read_bytes()
        assert rebuilt == committed, f"{name} drifted from the committed bytes"
    # A second rebuild into another dir must match the first (rerun
    # stability, not just committed-vs-rebuild).
    second = tmp_path / "second"
    builder.main(second)
    for name in FIXTURE_FILES:
        assert (second / name).read_bytes() == (tmp_path / name).read_bytes()


def test_fixtures_shape_and_eligibility_mix() -> None:
    lines = (FIXTURES_DIR / "concepts.csv").read_text(encoding="utf-8").splitlines()
    header = lines[0].split("\t")
    assert header == [
        "concept_id",
        "concept_name",
        "domain_id",
        "vocabulary_id",
        "concept_class_id",
        "standard_concept",
        "concept_code",
        "valid_start_date",
        "valid_end_date",
        "invalid_reason",
    ]
    rows = [line.split("\t") for line in lines[1:]]
    assert len(rows) == 200
    assert len({row[0] for row in rows}) == 200
    # Standard / classification / non-standard mix present.
    standards = {row[5] for row in rows}
    assert {"S", "C", ""} <= standards
    # Valid / invalid mix present, and eligibility-testable real
    # vocabulary names with fake codes are present.
    assert any(row[9] == "D" for row in rows)
    vocabularies = {row[3] for row in rows}
    assert {"SNOMED", "LOINC", "RxNorm"} <= vocabularies

    term_lines = (FIXTURES_DIR / "source_terms.csv").read_text(encoding="utf-8").splitlines()[1:]
    terms = [line.split("\t") for line in term_lines]
    assert len(terms) == 60
    concept_ids = {row[0] for row in rows}
    assert all(row[1] in concept_ids for row in terms)
    assert {row[2] for row in terms} == {"train", "dev", "heldout"}


def test_fake_encoder_deterministic_and_normalised() -> None:
    encoder = _load_module("fake_encoder", FIXTURES_DIR / "fake_encoder.py")
    texts = ["smärta i bröstet", "demo finding alpha", "", "feber"]
    first: np.ndarray[tuple[int, int], np.dtype[np.float32]] = encoder.encode(texts)
    second: np.ndarray[tuple[int, int], np.dtype[np.float32]] = encoder.encode(texts)
    assert first.dtype == np.float32
    assert first.shape == (4, 20)
    assert np.array_equal(first, second)
    squared = np.sum(first.astype(np.float64) ** 2, axis=1)
    assert np.allclose(squared, 1.0)
    # Different texts must not collapse to the same vector.
    assert not np.array_equal(first[0], first[1])
