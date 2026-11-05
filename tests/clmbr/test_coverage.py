"""``omopmap clmbr coverage`` on a ``meds synth`` extract against a synthetic dictionary."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

import omopmap.clmbr.cli as clmbr_cli
from omopmap.artifacts import BIRTH_CODE, DEATH_CODE, CoverageReport
from omopmap.cli import app
from omopmap.clmbr.coverage import coverage_report
from omopmap.meds.read import read_extract
from omopmap.meds.synth import synthetic_dictionary_codes
from tests.clmbr.dictionaries import Entry, pins_for, write_layouts

SUBJECTS, SEED = 120, 7
# Pinned outcome of `meds synth --n 120 --seed 7` against the synthetic dictionary
# (cross-checked below by a plain set computation over the same extract).
EXPECTED_DISTINCT, EXPECTED_RECOGNISED = 130, 65


@pytest.fixture
def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    runner = CliRunner()
    extract = tmp_path / "meds" / "synth"
    synth = runner.invoke(
        app, ["meds", "synth", "--n", str(SUBJECTS), "--seed", str(SEED), "--out", str(extract)]
    )
    assert synth.exit_code == 0, synth.stderr
    entries: list[Entry] = [(c, "code", None, "") for c in sorted(synthetic_dictionary_codes())]
    entries.append(("CARE_SITE/900001", "code", None, ""))
    path = write_layouts(tmp_path / "model", entries)["flat"]
    pins = pins_for(path)
    monkeypatch.setattr(clmbr_cli, "load_pins", lambda: pins)
    concept_set = tmp_path / "clmbr"
    imported = runner.invoke(
        app,
        [
            "clmbr",
            "import",
            "--model-dir",
            str(tmp_path / "model"),
            "--out",
            str(tmp_path / "kdl"),
            "--concept-set",
            str(concept_set),
        ],
    )
    assert imported.exit_code == 0, imported.stderr
    return extract, concept_set


def test_coverage_share_is_exact_and_strictly_between_0_and_1(
    setup: tuple[Path, Path], tmp_path: Path
) -> None:
    extract, concept_set = setup
    out = tmp_path / "coverage" / "synth.json"
    result = CliRunner().invoke(
        app,
        [
            "clmbr",
            "coverage",
            "--extract",
            str(extract),
            "--concept-set",
            str(concept_set),
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.stderr
    report = CoverageReport.model_validate_json(out.read_text())

    _, events = read_extract(extract)
    codes = set(events.get_column("code").to_list()) - {BIRTH_CODE, DEATH_CODE}
    recognised = codes & synthetic_dictionary_codes()
    assert (report.distinct_codes, report.recognised_codes) == (len(codes), len(recognised))
    assert (report.distinct_codes, report.recognised_codes) == (
        EXPECTED_DISTINCT,
        EXPECTED_RECOGNISED,
    )
    assert report.share == EXPECTED_RECOGNISED / EXPECTED_DISTINCT
    assert 0.0 < report.share < 1.0
    assert report.unrecognised_min_events == 10 and report.unrecognised_top_n == 20
    by_vocabulary = {v.vocabulary_id: v for v in report.by_vocabulary}
    assert by_vocabulary["DEMOSRC"].recognised_codes == 0
    assert by_vocabulary["Gender"].share == 1.0
    assert all(u.events >= 10 for u in report.unrecognised)
    assert all(u.code not in synthetic_dictionary_codes() for u in report.unrecognised)
    assert out.read_text() == report.to_json()
    again = CliRunner().invoke(
        app,
        [
            "clmbr",
            "coverage",
            "--extract",
            str(extract),
            "--concept-set",
            str(concept_set),
            "--out",
            str(out),
        ],
    )
    assert again.exit_code == 1 and "never overwritten" in again.stderr


def test_small_count_suppression() -> None:
    events = pl.DataFrame(
        {
            "code": ["MEDS_BIRTH"] + ["DEMOA/1"] * 12 + ["DEMOA/2"] * 9 + ["DEMOB/1"] * 3,
        }
    )
    report = coverage_report(
        events,
        ["DEMOB/1"],
        extract_manifest_sha256="0" * 64,
        concept_list_sha256="1" * 64,
        created_at="2024-01-01T00:00:00Z",
    )
    assert (report.distinct_codes, report.recognised_codes) == (3, 1)
    assert [(u.code, u.events) for u in report.unrecognised] == [("DEMOA/1", 12)]
    lowered = coverage_report(
        events,
        ["DEMOB/1"],
        extract_manifest_sha256="0" * 64,
        concept_list_sha256="1" * 64,
        created_at="2024-01-01T00:00:00Z",
        min_events=5,
        top_n=1,
    )
    assert [u.code for u in lowered.unrecognised] == ["DEMOA/1"]


def test_empty_extract_codes_give_zero_share() -> None:
    events = pl.DataFrame({"code": ["MEDS_BIRTH"]})
    report = coverage_report(
        events,
        ["DEMOB/1"],
        extract_manifest_sha256="0" * 64,
        concept_list_sha256="1" * 64,
        created_at="2024-01-01T00:00:00Z",
    )
    assert report.distinct_codes == 0 and report.share == 0.0 and report.by_vocabulary == ()


def test_coverage_refuses_non_concept_set(setup: tuple[Path, Path], tmp_path: Path) -> None:
    extract, _ = setup
    result = CliRunner().invoke(
        app,
        [
            "clmbr",
            "coverage",
            "--extract",
            str(extract),
            "--concept-set",
            str(extract),
            "--out",
            str(tmp_path / "c.json"),
        ],
    )
    assert result.exit_code == 1
