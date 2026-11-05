"""``adjudicate ingest``: the key round-trip and its fail-closed tamper gates.

The fill plan: cases 0, 1, 2, 4, 6 and 7 accept their gold concept from the
sheet (marked ``y``); case 3 is accepted by search only (its gold is a concept
the sheet never showed, put in ``other_targets``); case 5 defers; case 8
records ``no-target``.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from omopmap.adjudication import (
    FRAME_NAME,
    GATE_EXIT,
    HASH_EXIT,
    KEY_NAME,
    SHEET_COLUMNS,
)
from omopmap.cli import app
from omopmap.meds.convert import read_table
from omopmap.metrics import parse_frame
from tests.adjudication.conftest import AdjudicationWorkspace

SEARCH_ONLY_CASE = 3
DEFER_CASE = 5
NO_TARGET_CASE = 8


def _sheet_records(workspace: AdjudicationWorkspace) -> list[list[str]]:
    return list(csv.reader(workspace.sheet_bytes.decode("utf-8").splitlines()))


def _filled_bytes(
    workspace: AdjudicationWorkspace,
    *,
    drop_row: int | None = None,
    duplicate_row: int | None = None,
    swap: tuple[int, int] | None = None,
    edit_cell: tuple[int, int] | None = None,
    edit_header: bool = False,
    undecided_case: int | None = None,
    contradict_case: int | None = None,
    maybe_row: int | None = None,
    bad_other: int | None = None,
    quote_all: bool = False,
) -> bytes:
    """The completed sheet, optionally with one tamper applied."""
    records = _sheet_records(workspace)
    header, rows = records[0], records[1:]
    key = workspace.key_model
    acceptable = SHEET_COLUMNS.index("acceptable")
    other = SHEET_COLUMNS.index("other_targets")
    case_column = SHEET_COLUMNS.index("case_verdict")
    position = 0
    for case in key.cases:
        block = rows[position : position + len(case.rows)]
        gold = workspace.gold_of(int(case.case_id[1:]) - 1)
        for offset, record in enumerate(block):
            if record[3] == str(gold) and case.case_id not in (
                f"C{SEARCH_ONLY_CASE + 1:04d}",
                f"C{DEFER_CASE + 1:04d}",
                f"C{NO_TARGET_CASE + 1:04d}",
            ):
                record[acceptable] = "y"
            if case.case_id == f"C{SEARCH_ONLY_CASE + 1:04d}" and offset == 0:
                presented = {
                    key_row.concept_id for key_row in case.rows if key_row.concept_id is not None
                }
                absent = next(c for c in range(1, 200) if c % 5 != 4 and c not in presented)
                record[other] = str(absent)
            if case.case_id == f"C{DEFER_CASE + 1:04d}":
                record[case_column] = "defer"
            if case.case_id == f"C{NO_TARGET_CASE + 1:04d}":
                record[case_column] = "no-target"
            if undecided_case is not None and case.case_id == f"C{undecided_case + 1:04d}":
                record[acceptable] = ""
                record[case_column] = ""
                record[other] = ""
            if contradict_case is not None and case.case_id == f"C{contradict_case + 1:04d}":
                record[case_column] = "no-target"
            if maybe_row is not None and record is rows[maybe_row]:
                record[acceptable] = "maybe"
            if bad_other is not None and record is rows[bad_other]:
                record[other] = "demo;not-ids"
        position += len(case.rows)
    if drop_row is not None:
        del rows[drop_row]
    if duplicate_row is not None:
        rows.insert(duplicate_row, list(rows[duplicate_row]))
    if swap is not None:
        first, second = swap
        rows[first], rows[second] = rows[second], rows[first]
    if edit_cell is not None:
        row_index, column = edit_cell
        rows[row_index][column] = "demo tampered value"
    if edit_header:
        header = list(header)
        header[acceptable] = "verdict"
    buffer = io.StringIO()
    writer = csv.writer(
        buffer,
        lineterminator="\n",
        quoting=csv.QUOTE_ALL if quote_all else csv.QUOTE_MINIMAL,
    )
    writer.writerows([header, *rows])
    return buffer.getvalue().encode("utf-8")


def _ingest(
    out: Path, workspace: AdjudicationWorkspace, sheet_bytes: bytes, key_dir: Path | None = None
) -> Any:
    """Ingest a sheet's bytes against the key."""
    sheet = out.parent / "filled-sheet.csv"
    sheet.parent.mkdir(parents=True, exist_ok=True)
    sheet.write_bytes(sheet_bytes)
    return CliRunner().invoke(
        app,
        [
            "adjudicate",
            "ingest",
            "--sheet",
            str(sheet),
            "--key",
            str(key_dir or workspace.key),
            "--out",
            str(out),
            "--created-at",
            "2026-01-01T00:00:00Z",
        ],
    )


def _payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_ingest_round_trips_the_completed_sheet(
    tmp_path: Path, workspace: AdjudicationWorkspace
) -> None:
    """A honestly filled sheet becomes a frame of record in case-table order."""
    out = tmp_path / "ingest"
    result = _ingest(out, workspace, _filled_bytes(workspace))
    assert result.exit_code == 0, result.output
    frame = parse_frame(read_table(out / FRAME_NAME))
    assert [row.query_id for row in frame] == [f"q{index}" for index in range(9)]
    expected: dict[int, tuple[str, set[int]]] = {
        0: ("accept", {workspace.gold_of(0)}),
        1: ("accept", {workspace.gold_of(1)}),
        2: ("accept", {workspace.gold_of(2)}),
        3: ("accept", _search_target(workspace)),
        4: ("accept", {workspace.gold_of(4)}),
        5: ("defer", set()),
        6: ("accept", {workspace.gold_of(6)}),
        7: ("accept", {workspace.gold_of(7)}),
        8: ("no-target", set()),
    }
    for index, (verdict, targets) in expected.items():
        assert frame[index].verdict == verdict, index
        assert frame[index].targets == frozenset(targets), index
    report = _payload(out / "ingest_report.json")
    assert report["cases"] == 9
    assert report["verdicts"] == {"accept": 7, "no-target": 1, "defer": 1}
    assert report["accepted_from_list"] == 6
    assert report["accepted_by_search_only"] == 1
    assert report["presented_targets"] == 6
    assert report["other_targets"] == 1


def _search_target(workspace: AdjudicationWorkspace) -> set[int]:
    """The search-only case's other_targets id, as the fill computes it."""
    key_case = workspace.key_model.cases[SEARCH_ONLY_CASE]
    presented = {row.concept_id for row in key_case.rows if row.concept_id is not None}
    return {next(c for c in range(1, 200) if c % 5 != 4 and c not in presented)}


def test_key_proposals_point_at_the_real_rankings(workspace: AdjudicationWorkspace) -> None:
    """Every (run, rank, score) in the key matches the run's rankings.tsv entry."""
    rankings = {"SYSALPHA": workspace.lexical_rankings(), "SYSBETA": workspace.embed_rankings()}
    for case in workspace.key_model.cases:
        for key_row in case.rows:
            if key_row.concept_id is None:
                assert key_row.proposals == ()
                continue
            assert key_row.proposals, case.case_id
            for proposal in key_row.proposals:
                ranking = rankings[proposal.run][case.query_row]
                assert ranking.ids[proposal.rank - 1] == key_row.concept_id
                assert ranking.scores[proposal.rank - 1] == proposal.score


def _tampered(
    tmp_path: Path,
    workspace: AdjudicationWorkspace,
    sheet_bytes: bytes,
    *,
    expected_exit: int,
    key_dir: Path | None = None,
) -> Any:
    """Ingest a tampered sheet and require the given exit code and no output."""
    out = tmp_path / "out"
    result = _ingest(out, workspace, sheet_bytes, key_dir=key_dir)
    assert result.exit_code == expected_exit, result.output
    assert not out.exists()
    return result


def test_structural_tampering_exits_four(tmp_path: Path, workspace: AdjudicationWorkspace) -> None:
    """Deleted, added, swapped or edited rows no longer match what was built."""
    _tampered(
        tmp_path / "drop", workspace, _filled_bytes(workspace, drop_row=2), expected_exit=HASH_EXIT
    )
    _tampered(
        tmp_path / "dup",
        workspace,
        _filled_bytes(workspace, duplicate_row=2),
        expected_exit=HASH_EXIT,
    )
    _tampered(
        tmp_path / "swap", workspace, _filled_bytes(workspace, swap=(2, 3)), expected_exit=HASH_EXIT
    )
    _tampered(
        tmp_path / "cell",
        workspace,
        _filled_bytes(workspace, edit_cell=(2, SHEET_COLUMNS.index("concept_name"))),
        expected_exit=HASH_EXIT,
    )
    _tampered(
        tmp_path / "cell2",
        workspace,
        _filled_bytes(workspace, edit_cell=(2, SHEET_COLUMNS.index("concept_id"))),
        expected_exit=HASH_EXIT,
    )
    _tampered(
        tmp_path / "head",
        workspace,
        _filled_bytes(workspace, edit_header=True),
        expected_exit=HASH_EXIT,
    )


def test_tampered_key_exits_four(tmp_path: Path, workspace: AdjudicationWorkspace) -> None:
    """A key that differs from its build manifest is refused."""
    import shutil

    key2 = tmp_path / "key2"
    shutil.copytree(workspace.key, key2)
    original = (key2 / KEY_NAME).read_text(encoding="utf-8")
    (key2 / KEY_NAME).write_text(original.replace('"seed": 12345', '"seed": 999'), encoding="utf-8")
    _tampered(
        tmp_path / "out",
        workspace,
        _filled_bytes(workspace),
        expected_exit=HASH_EXIT,
        key_dir=key2,
    )


def test_requoting_the_whole_sheet_is_accepted(
    tmp_path: Path, workspace: AdjudicationWorkspace
) -> None:
    """A spreadsheet that re-quotes every cell still round-trips."""
    out = tmp_path / "out"
    result = _ingest(out, workspace, _filled_bytes(workspace, quote_all=True))
    assert result.exit_code == 0, result.output
    assert (out / FRAME_NAME).exists()


def test_contradictions_and_garbage_exit_six(
    tmp_path: Path, workspace: AdjudicationWorkspace
) -> None:
    """Undecided, contradictory or malformed marks are gate failures."""
    _tampered(
        tmp_path / "undecided",
        workspace,
        _filled_bytes(workspace, undecided_case=0),
        expected_exit=GATE_EXIT,
    )
    _tampered(
        tmp_path / "contradict",
        workspace,
        _filled_bytes(workspace, contradict_case=1),
        expected_exit=GATE_EXIT,
    )
    _tampered(
        tmp_path / "maybe",
        workspace,
        _filled_bytes(workspace, maybe_row=0),
        expected_exit=GATE_EXIT,
    )
    _tampered(
        tmp_path / "badother",
        workspace,
        _filled_bytes(workspace, bad_other=0),
        expected_exit=GATE_EXIT,
    )


def test_ingest_into_an_occupied_directory_exits_one(
    tmp_path: Path, workspace: AdjudicationWorkspace
) -> None:
    """An occupied --out is refused."""
    out = tmp_path / "out"
    assert _ingest(out, workspace, _filled_bytes(workspace)).exit_code == 0
    assert _ingest(out, workspace, _filled_bytes(workspace)).exit_code == 1
