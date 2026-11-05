"""``adjudicate build``: determinism, run-order irrelevance, and the blinding scan."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from omopmap.adjudication import SHEET_COLUMNS, SHEET_NAME
from tests.adjudication.conftest import EMB_NAME, LEX_NAME, AdjudicationWorkspace
from tests.retrieval.conftest import CONCEPTS, CREATED_AT


def _sheet_rows(data: bytes) -> list[list[str]]:
    return list(csv.reader(data.decode("utf-8").splitlines()))[1:]


def _case_order(data: bytes, case_id: str) -> list[str]:
    rows = [row for row in _sheet_rows(data) if row[0] == case_id]
    return [row[3] for row in rows]


def test_rebuilding_is_byte_identical(tmp_path: Path, workspace: AdjudicationWorkspace) -> None:
    """A second build with the same seed reproduces sheet and key exactly."""
    sheet2, key2 = tmp_path / "sheet2", tmp_path / "key2"
    assert workspace.build(sheet2, key2, "--seed", "12345") == 0
    assert (sheet2 / SHEET_NAME).read_bytes() == workspace.sheet_bytes
    assert (key2 / "adjudication_key.json").read_bytes() == (
        workspace.key / "adjudication_key.json"
    ).read_bytes()


def test_a_different_seed_changes_only_the_order(
    tmp_path: Path, workspace: AdjudicationWorkspace
) -> None:
    """Another seed reshuffles rows per case; each case keeps its row multiset."""
    sheet2, key2 = tmp_path / "sheet2", tmp_path / "key2"
    assert workspace.build(sheet2, key2, "--seed", "54321") == 0
    other = (sheet2 / SHEET_NAME).read_bytes()
    assert other != workspace.sheet_bytes
    key = workspace.key_model
    for case in key.cases:
        before = _case_order(workspace.sheet_bytes, case.case_id)
        after = _case_order(other, case.case_id)
        assert Counter(before) == Counter(after), case.case_id
    assert any(
        _case_order(workspace.sheet_bytes, c.case_id) != _case_order(other, c.case_id)
        for c in key.cases
    )


def test_run_order_does_not_matter(tmp_path: Path, workspace: AdjudicationWorkspace) -> None:
    """Giving the two runs in the opposite order builds the same sheet."""
    sheet2, key2 = tmp_path / "sheet2", tmp_path / "key2"
    result = _invoke_build_reversed(workspace, sheet2, key2)
    assert result == 0
    assert (sheet2 / SHEET_NAME).read_bytes() == workspace.sheet_bytes


def _invoke_build_reversed(workspace: AdjudicationWorkspace, sheet_out: Path, key_out: Path) -> int:
    """``adjudicate build`` with the run flags in reverse order."""
    from typer.testing import CliRunner

    from omopmap.cli import app

    return (
        CliRunner()
        .invoke(
            app,
            [
                "adjudicate",
                "build",
                "--run",
                f"{EMB_NAME}={workspace.emb}",
                "--run",
                f"{LEX_NAME}={workspace.lex}",
                "--cases",
                str(workspace.cases),
                "--concepts",
                str(CONCEPTS),
                "--sheet-out",
                str(sheet_out),
                "--key-out",
                str(key_out),
                "--seed",
                "12345",
                "--created-at",
                CREATED_AT,
            ],
        )
        .exit_code
    )


def test_sheet_carries_no_blinding_leak(workspace: AdjudicationWorkspace) -> None:
    """No run name, channel, encoder identity, query id, score or column leak."""
    data = workspace.sheet_bytes
    header = next(csv.reader(data.decode("utf-8").splitlines()))
    assert tuple(header) == SHEET_COLUMNS
    leaks: list[str] = [LEX_NAME, EMB_NAME]
    for run in (workspace.lex, workspace.emb):
        manifest: dict[str, Any] = json.loads(
            (run / "run_manifest.json").read_text(encoding="utf-8")
        )
        channel = manifest["run_type"]
        model = manifest["parameters"].get("model")
        if model is not None:
            channel = f"{channel}:{model}"
        leaks.append(channel)
        encoder: dict[str, Any] = manifest.get("encoder") or {}
        leaks.extend(str(encoder.get(field)) for field in ("model_id", "revision"))
    leaks.extend(f"q{index}" for index in range(workspace.key_model.n_cases))
    for run in (workspace.lex, workspace.emb):
        with (run / "rankings.tsv").open(encoding="utf-8") as handle:
            for row in csv.reader(handle, delimiter="\t"):
                if len(row) >= 4 and len(row[3]) > 4:
                    leaks.append(row[3])
    leaks.extend(("rank", "score", "model", "run", "system"))
    for leak in leaks:
        assert leak.encode("utf-8") not in data, leak


def test_sheet_directory_holds_only_the_sheet(workspace: AdjudicationWorkspace) -> None:
    """Reviewers get one file."""
    assert sorted(p.name for p in workspace.sheet.iterdir()) == [SHEET_NAME]


def test_nested_output_directories_are_refused(
    tmp_path: Path, workspace: AdjudicationWorkspace
) -> None:
    """Sheet and key directories must be separate."""
    assert workspace.build(tmp_path / "shared", tmp_path / "shared") == 1
    assert workspace.build(tmp_path / "outer", tmp_path / "outer" / "key") == 1
    assert workspace.build(tmp_path / "sheet-inside", tmp_path / "sheet-inside" / "..") == 1
