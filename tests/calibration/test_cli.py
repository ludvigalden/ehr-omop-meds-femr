"""End-to-end ``omopmap calibrate fit`` / ``apply`` over the shared workspace.

The fit learns its threshold from the twelve training rows only; the key test
perturbs every non-train row of the frame (source terms, verdicts, targets,
proposals), re-runs both channels over the perturbed table, and requires the
second fit's ``calibration.json`` and ``tiers.tsv`` to be byte-identical to
the first.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from omopmap.calibration import CALIBRATION_NAME, GATE_EXIT, HASH_EXIT, REPORT_NAME, TIERS_NAME
from omopmap.cli import app
from omopmap.retrieval.cli import eligible_labels
from omopmap.retrieval.manifest import RUN_MANIFEST_NAME
from tests.calibration.conftest import (
    N_QUERIES,
    PROPOSAL_KINDS,
    PROPOSAL_PATHS,
    Workspace,
    embed_rankings,
    proposal_concept,
    source_term,
    split_of,
    verdict_of,
    write_cases,
    write_proposals,
)
from tests.retrieval.conftest import CONCEPTS, CREATED_AT


def _fit(
    out: Path,
    workspace: Workspace,
    *,
    cases: Path | None = None,
    proposals: Path | None = None,
    run: Path | None = None,
    lexical: Path | None = None,
) -> Any:
    """Invoke ``calibrate fit`` with the learned threshold and return the result."""
    result = CliRunner().invoke(
        app,
        [
            "calibrate",
            "fit",
            "--gold",
            str(cases or workspace.cases),
            "--proposals",
            str(proposals or workspace.proposals),
            "--run",
            str(run or workspace.emb),
            "--lexical",
            str(lexical or workspace.lex),
            "--concepts",
            str(CONCEPTS),
            "--learn-s1",
            "--target-precision",
            "0.6",
            "--min-n",
            "3",
            "--out",
            str(out),
            "--created-at",
            CREATED_AT,
        ],
    )
    return result


def _payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _apply(
    out: Path, workspace: Workspace, fit_dir: Path, split: str, run: Path | None = None
) -> Any:
    """Invoke ``calibrate apply`` for one split."""
    return CliRunner().invoke(
        app,
        [
            "calibrate",
            "apply",
            "--calibration",
            str(fit_dir),
            "--gold",
            str(workspace.cases),
            "--proposals",
            str(workspace.proposals),
            "--run",
            str(run or workspace.emb),
            "--lexical",
            str(workspace.lex),
            "--concepts",
            str(CONCEPTS),
            "--split",
            split,
            "--out",
            str(out),
            "--created-at",
            CREATED_AT,
        ],
    )


PERTURBED_KINDS: dict[str, str] = {
    "rank1": "absent",
    "rank2": "rank3",
    "rank3": "rank2",
    "absent": "rank1",
    "invalid": "rank1",
}


def _perturbed_table(root: Path, workspace: Workspace) -> tuple[Path, Path, Path, Path]:
    """Non-train rows changed: new texts, new verdicts, new proposals, new runs."""
    cases_rows: list[dict[str, str]] = []
    for row in range(N_QUERIES):
        verdict, targets = verdict_of(row)
        if row >= 12:  # every non-train row changes
            term = f"demo symptom text {row} v2"
            if row == 12:
                verdict, targets = "no-target", ""
            elif row == 13:
                verdict, targets = "accept", "22"
            elif row == 21:
                verdict, targets = "accept", "23"
        else:
            term = source_term(row)
        cases_rows.append(
            {
                "query_id": f"q{row:02d}",
                "system": "DEMOSYS",
                "code": f"S{row:03d}",
                "source_term": term,
                "split": split_of(row),
                "verdict": verdict,
                "targets": targets,
            }
        )
    cases = write_cases(root / "cases-v2.tsv", cases_rows)
    emb, lex = root / "emb-v2", root / "lex-v2"
    workspace.run_embed_lexical(cases, emb, lex)
    eligible = frozenset(label.concept_id for label in eligible_labels(CONCEPTS, False))
    rankings = embed_rankings(emb, N_QUERIES)
    proposals = write_proposals(
        root / "proposals-v2.tsv",
        [
            {
                "query_id": f"q{row:02d}",
                "concept_id": str(
                    proposal_concept(
                        PROPOSAL_KINDS[row] if row < 12 else PERTURBED_KINDS[PROPOSAL_KINDS[row]],
                        rankings[row],
                        eligible,
                    )
                ),
                "path": PROPOSAL_PATHS[row],
            }
            for row in range(N_QUERIES)
        ],
    )
    return cases, proposals, emb, lex


def test_fit_learns_its_threshold_from_the_training_rows(
    tmp_path: Path, workspace: Workspace
) -> None:
    """The learned fit passes and freezes a threshold near 1.0 on the exact matches."""
    fit = tmp_path / "fit"
    result = _fit(fit, workspace)
    assert result.exit_code == 0, result.output
    assert sorted(p.name for p in fit.iterdir()) == [
        CALIBRATION_NAME,
        RUN_MANIFEST_NAME,
        TIERS_NAME,
    ]
    payload = _payload(fit / CALIBRATION_NAME)
    assert payload["schema"] == "omopmap-calibration/1"
    assert payload["threshold_source"] == "learned"
    assert payload["rule"]["s1_high"] > 0.99
    assert payload["train_rows"] == 12


def test_fit_is_invariant_to_perturbed_non_train_rows(tmp_path: Path, workspace: Workspace) -> None:
    """Only the training rows move a calibration: the refit bytes are identical."""
    first = tmp_path / "fit-a"
    assert _fit(first, workspace).exit_code == 0
    second = tmp_path / "fit-b"
    cases2, proposals2, emb2, lex2 = _perturbed_table(tmp_path, workspace)
    result = _fit(second, workspace, cases=cases2, proposals=proposals2, run=emb2, lexical=lex2)
    assert result.exit_code == 0, result.output
    assert (first / CALIBRATION_NAME).read_bytes() == (second / CALIBRATION_NAME).read_bytes()
    assert (first / TIERS_NAME).read_bytes() == (second / TIERS_NAME).read_bytes()


def test_apply_reports_the_frozen_rule_on_heldout(tmp_path: Path, workspace: Workspace) -> None:
    """Apply tiers the heldout split and records the calibration digest."""
    fit = tmp_path / "fit"
    assert _fit(fit, workspace).exit_code == 0
    rule = _payload(fit / CALIBRATION_NAME)["rule"]
    out = tmp_path / "apply"
    result = _apply(out, workspace, fit, "heldout")
    assert result.exit_code == 0, result.output
    assert sorted(p.name for p in out.iterdir()) == [
        REPORT_NAME,
        RUN_MANIFEST_NAME,
        TIERS_NAME,
    ]
    report = _payload(out / REPORT_NAME)
    assert report["schema"] == "omopmap-calibration-report/1"
    assert report["split"] == "heldout"
    assert report["rule"] == rule
    assert report["calibration_sha256"] == _sha256(fit / CALIBRATION_NAME)
    assert report["rows"] == 6
    with (out / TIERS_NAME).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert {row["split"] for row in rows} == {"heldout"}
    assert {row["tier"] for row in rows} <= {"H", "M", "L", "U"}
    rerun = tmp_path / "apply-again"
    assert _apply(rerun, workspace, fit, "heldout").exit_code == 0
    assert (out / REPORT_NAME).read_bytes() == (rerun / REPORT_NAME).read_bytes()


def test_apply_to_the_training_split_is_refused(tmp_path: Path, workspace: Workspace) -> None:
    """Judging the fit on its own rows exits 6."""
    fit = tmp_path / "fit"
    assert _fit(fit, workspace).exit_code == 0
    result = _apply(tmp_path / "out", workspace, fit, "train")
    assert result.exit_code == GATE_EXIT
    assert "InSampleApplyError" in result.output
    assert not (tmp_path / "out").exists()


def test_tampered_calibration_is_refused(tmp_path: Path, workspace: Workspace) -> None:
    """A calibration.json that differs from its fit manifest exits 4."""
    fit = tmp_path / "fit"
    assert _fit(fit, workspace).exit_code == 0
    tampered = tmp_path / "tampered"
    shutil.copytree(fit, tampered)
    original = (tampered / CALIBRATION_NAME).read_text(encoding="utf-8")
    (tampered / CALIBRATION_NAME).write_text(
        original.replace("omopmap-calibration/1", "omopmap-calibration/0"), encoding="utf-8"
    )
    result = _apply(tmp_path / "out", workspace, tampered, "heldout")
    assert result.exit_code == HASH_EXIT
    assert "CalibrationHashError" in result.output
    assert not (tmp_path / "out").exists()


def test_apply_rejects_a_run_from_another_encoder(tmp_path: Path, workspace: Workspace) -> None:
    """The embed run's encoder must match the one the calibration was fit on."""
    fit = tmp_path / "fit"
    assert _fit(fit, workspace).exit_code == 0
    result = _apply(tmp_path / "out", workspace, fit, "heldout", run=workspace.emb_other)
    assert result.exit_code == GATE_EXIT
    assert "EncoderMismatchError" in result.output
    assert not (tmp_path / "out").exists()


def test_fit_into_an_occupied_directory_exits_one(tmp_path: Path, workspace: Workspace) -> None:
    """The second fit into the same directory is refused."""
    fit = tmp_path / "fit"
    assert _fit(fit, workspace).exit_code == 0
    result = _fit(fit, workspace)
    assert result.exit_code == 1
    assert "OutputExistsError" in result.output


def test_reruns_with_a_fixed_timestamp_are_byte_identical(
    tmp_path: Path, workspace: Workspace
) -> None:
    """Two fits at the same --created-at produce the same bytes."""
    first, second = tmp_path / "fit-a", tmp_path / "fit-b"
    assert _fit(first, workspace).exit_code == 0
    assert _fit(second, workspace).exit_code == 0
    assert (first / CALIBRATION_NAME).read_bytes() == (second / CALIBRATION_NAME).read_bytes()
    assert (first / TIERS_NAME).read_bytes() == (second / TIERS_NAME).read_bytes()
