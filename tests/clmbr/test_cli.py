"""``omopmap clmbr import`` / ``verify``: KDL emit, manifest, round trip, exit codes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner, Result

import omopmap.clmbr.cli as clmbr_cli
import omopmap.clmbr.source as source_module
from omopmap.artifacts import validate_artifact
from omopmap.artifacts.manifest import concept_list_sha256
from omopmap.cli import app
from omopmap.clmbr.kdl import read_kdl_codes
from omopmap.clmbr.source import Pins
from tests.clmbr.dictionaries import EXPECTED, pins_for, sha256, write_layouts


def _run(*args: str) -> Result:
    return CliRunner().invoke(app, ["clmbr", *args])


@pytest.fixture
def model_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    directory = tmp_path / "model"
    path = write_layouts(directory)["flat"]
    pins = pins_for(path)
    monkeypatch.setattr(clmbr_cli, "load_pins", lambda: pins)
    return directory


def _import(model_dir: Path, out: Path, *extra: str) -> Result:
    return _run("import", "--model-dir", str(model_dir), "--out", str(out), *extra)


def test_import_writes_kdl_and_counts_only_manifest(model_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "kdl"
    result = _import(model_dir, out, "--concept-set", str(tmp_path / "cs"))
    assert result.exit_code == 0, result.stderr
    assert sorted(p.name for p in out.iterdir()) == [
        "DEMO Extension.kdl",
        "DEMOCOND.kdl",
        "DEMOLAB.kdl",
        "manifest.json",
    ]
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["total"] == len(EXPECTED) == 5
    assert manifest["care_site_excluded"] == 2
    assert manifest["vocabulary_counts"] == {"DEMO Extension": 1, "DEMOCOND": 2, "DEMOLAB": 2}
    assert manifest["entry_type_counts"] == {"code": 6, "numeric": 2, "text": 2, "unused": 2}
    assert manifest["concept_list_sha256"] == concept_list_sha256(EXPECTED)
    assert manifest["file_sha256"] == sha256(model_dir / "dictionary.msgpack")
    text = (out / "manifest.json").read_text()
    assert not any(code.split("/", 1)[1] in text for code in EXPECTED), "manifest lists a code"
    validate_artifact(tmp_path / "cs")


def test_kdl_carries_codes_and_types_only(model_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "kdl"
    assert _import(model_dir, out).exit_code == 0
    lab = (out / "DEMOLAB.kdl").read_text()
    for forbidden in ("high", "low", "weight", "val_start", "age", "40.0"):
        assert forbidden not in lab
    system, context, codes = read_kdl_codes(out / "DEMOLAB.kdl")
    assert system == "DEMOLAB" and codes == ["DEMOLAB/L01", "DEMOLAB/L02"]
    assert 'clmbr-entry-types "numeric" "text"' in lab
    assert context["dictionary-sha256"] == sha256(model_dir / "dictionary.msgpack")
    _, _, ext = read_kdl_codes(out / "DEMO Extension.kdl")
    assert ext == ["DEMO Extension/X/9"]


def test_verify_round_trips_and_detects_tampering(model_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "kdl"
    assert _import(model_dir, out).exit_code == 0
    manifest = str(out / "manifest.json")
    ok = _run("verify", "--manifest", manifest)
    assert ok.exit_code == 0, ok.stderr
    cond = out / "DEMOCOND.kdl"
    original = cond.read_text()
    cond.write_text(original.replace('"C002"', '"C003"'))
    bad = _run("verify", "--manifest", manifest)
    assert bad.exit_code == 1 and "concept-list hash" in bad.stderr
    cond.unlink()
    missing = _run("verify", "--manifest", manifest)
    assert missing.exit_code == 1 and "missing" in missing.stderr
    cond.write_text(original)
    (out / "EXTRA.kdl").write_text(original)
    extra = _run("verify", "--manifest", manifest)
    assert extra.exit_code == 1 and "not in the manifest" in extra.stderr


def test_verify_without_kdl_files_fails(model_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "kdl"
    assert _import(model_dir, out).exit_code == 0
    for path in out.glob("*.kdl"):
        path.unlink()
    result = _run("verify", "--manifest", str(out / "manifest.json"))
    assert result.exit_code == 1 and "run `omopmap clmbr import` first" in result.stderr


def test_reimport_replaces_manifest_but_refuses_leftover_kdl(
    model_dir: Path, tmp_path: Path
) -> None:
    out = tmp_path / "kdl"
    assert _import(model_dir, out).exit_code == 0
    again = _import(model_dir, out)
    assert again.exit_code == 1 and "remove the previous import" in again.stderr
    for path in out.glob("*.kdl"):
        path.unlink()
    assert _import(model_dir, out).exit_code == 0


def test_sha_mismatch_exits_2(model_dir: Path, tmp_path: Path) -> None:
    dictionary = model_dir / "dictionary.msgpack"
    dictionary.write_bytes(dictionary.read_bytes() + b"\x00")
    out = tmp_path / "kdl"
    result = _import(model_dir, out)
    assert result.exit_code == 2
    assert "differs from the pinned" in result.stderr
    assert not out.exists()


def test_packaged_pins_refuse_a_file_that_is_not_the_release_file(tmp_path: Path) -> None:
    other = tmp_path / "model"
    write_layouts(other)
    result = _run("import", "--model-dir", str(other), "--out", str(tmp_path / "kdl"))
    assert result.exit_code == 2
    assert "differs from the pinned" in result.stderr


def test_concept_list_pin_mismatch_exits_2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    directory = tmp_path / "model"
    path = write_layouts(directory)["flat"]
    pins = pins_for(path, concept_list="e" * 64)
    monkeypatch.setattr(clmbr_cli, "load_pins", lambda: pins)
    out = tmp_path / "kdl"
    result = _import(directory, out)
    assert result.exit_code == 2
    assert "concept-list hash" in result.stderr
    assert not out.exists()


def test_unreachable_hf_exits_2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pins: Pins = pins_for(write_layouts(tmp_path / "m")["flat"])
    monkeypatch.setattr(clmbr_cli, "load_pins", lambda: pins)

    def offline(*, repo_id: str, filename: str, revision: str, token: str | bool) -> str:
        raise OSError("connection refused")

    monkeypatch.setattr(source_module, "_hf_downloader", lambda: offline)
    result = _run("import", "--hf-repo", pins.repo, "--out", str(tmp_path / "kdl"))
    assert result.exit_code == 2
    assert "cannot download" in result.stderr


def test_source_options_are_exclusive(model_dir: Path, tmp_path: Path) -> None:
    neither = _run("import", "--out", str(tmp_path / "a"))
    both = _import(model_dir, tmp_path / "b", "--hf-repo", "demo-org/demo-model")
    assert neither.exit_code != 0 and both.exit_code != 0


def test_athena_enrichment_reports_unresolved(model_dir: Path, tmp_path: Path) -> None:
    athena = tmp_path / "CONCEPT.csv"
    athena.write_text(
        "concept_id\tconcept_name\tdomain_id\tvocabulary_id\tconcept_class_id\t"
        "standard_concept\tconcept_code\n"
        "2000000001\tinvented condition\tCondition\tDEMOCOND\tClinical Finding\tS\tC001\n"
        "2000000002\tinvented lab\tMeasurement\tDEMOLAB\tLab Test\t\tL01\n"
    )
    out = tmp_path / "kdl"
    result = _import(model_dir, out, "--athena", str(athena), "--concept-set", str(tmp_path / "cs"))
    assert result.exit_code == 0, result.stderr
    assert "unresolved in the Athena table: 3 of 5" in result.output
    cond = (out / "DEMOCOND.kdl").read_text()
    assert 'omop-concept-id 2000000001 domain="Condition" standard="S"' in cond
    assert "invented condition" not in cond
    lab = (out / "DEMOLAB.kdl").read_text()
    assert 'omop-concept-id 2000000002 domain="Measurement"' in lab and "standard=" not in lab
    assert _run("verify", "--manifest", str(out / "manifest.json")).exit_code == 0
    manifest = validate_artifact(tmp_path / "cs")
    assert [i.name for i in manifest.inputs] == ["dictionary", "vocabulary:CONCEPT.csv"]
