"""End-to-end ``omopmap metrics compute`` on the committed fixtures.

Golden numbers: the frame derived from the B2 term fixture (60 queries:
48 accept, 11 no-target, 1 defer), a lexical run with train-split Swedish
surfaces, a fake-encoder embed run and their RRF fusion. Each golden is also
recomputed independently from the raw ``rankings.tsv`` files with plain
``csv``. Every gate is then triggered through the CLI on a corrupted copy of
an input and must exit with its code (6 for a gate, 4 for a run whose
rankings no longer match its manifest) and name its error, writing nothing.
"""

from __future__ import annotations

import csv
import json
import shutil
import time
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner, Result

from omopmap.cli import app
from omopmap.metrics import GATE_EXIT, RUN_HASH_EXIT, wilson
from tests.metrics.conftest import read_frame_rows, write_frame
from tests.retrieval.conftest import CONCEPTS, CREATED_AT

GOLDEN = {
    # run: hits at k = 1, 5, 10 over the 48 accepted queries
    "lexical": (31, 31, 31),
    "embed": (0, 3, 5),
    "rrf": (19, 28, 34),
}


def _compute(out: Path, frame: Path, runs: dict[str, Path], *extra: str) -> Result:
    return CliRunner().invoke(
        app,
        [
            "metrics",
            "compute",
            "--run",
            f"lexical={runs['lex']}",
            "--run",
            f"embed={runs['emb']}",
            "--run",
            f"rrf={runs['fused']}",
            "--gold",
            str(frame),
            "--concepts",
            str(CONCEPTS),
            "--k",
            "1,5,10",
            "--out",
            str(out),
            "--created-at",
            CREATED_AT,
            *extra,
        ],
    )


def _ranked(run: Path, n: int) -> list[list[int]]:
    with (run / "rankings.tsv").open(encoding="utf-8") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))[1:]
    ranked: list[list[int]] = [[] for _ in range(n)]
    for row in rows:
        ranked[int(row[0])].append(int(row[2]))
    return ranked


def _payload(out: Path) -> dict[str, Any]:
    return json.loads((out / "metrics.json").read_text(encoding="utf-8"))


def _hits(payload: dict[str, Any], run: str) -> tuple[int, ...]:
    return tuple(c["recall"]["count"] for c in payload["recall"][run]["at"])


def test_golden_metrics_reproduce(tmp_path: Path, frame: Path, runs: dict[str, Path]) -> None:
    started = time.monotonic()
    out = tmp_path / "m"
    result = _compute(out, frame, runs)
    assert result.exit_code == 0, result.output
    assert time.monotonic() - started < 60.0
    payload = _payload(out)
    assert payload["schema"] == "omopmap-metrics/1"
    assert payload["interval"] == {
        "confidence": 0.95,
        "continuity_correction": False,
        "method": "wilson-score",
        "z": 1.959963984540054,
    }
    assert payload["frame"]["n"] == 60
    assert payload["frame"]["verdicts"] == {"accept": 48, "defer": 1, "no-target": 11}
    assert payload["recall_denominator"] == 48
    for run, hits in GOLDEN.items():
        assert _hits(payload, run) == hits, run
        for cut in payload["recall"][run]["at"]:
            assert cut["recall"] == wilson(cut["recall"]["count"], 48).model_dump()
    assert payload["recall"]["rrf"]["run_type"] == "fuse"
    assert payload["frame"]["accept"] == wilson(48, 60).model_dump()
    # Every file the run wrote is recorded in its manifest, written last.
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_type"] == "metrics"
    assert sorted(o["name"] for o in manifest["outputs"]) == ["frame_of_record.tsv", "metrics.json"]
    assert {i["name"] for i in manifest["inputs"]} == {
        "concepts",
        "gold",
        "run:embed",
        "run:lexical",
        "run:rrf",
    }


def test_golden_matches_an_independent_recount(
    tmp_path: Path, frame: Path, runs: dict[str, Path]
) -> None:
    out = tmp_path / "m"
    assert _compute(out, frame, runs).exit_code == 0
    payload = _payload(out)
    rows = read_frame_rows()
    accepted = [(i, int(r["targets"])) for i, r in enumerate(rows) if r["verdict"] == "accept"]
    for name, key in (("lexical", "lex"), ("embed", "emb"), ("rrf", "fused")):
        ranked = _ranked(runs[key], len(rows))
        expected = tuple(
            sum(1 for i, gold in accepted if gold in ranked[i][:k]) for k in (1, 5, 10)
        )
        assert _hits(payload, name) == expected, name
    # The per-query frame lists the first gold rank per run.
    with (out / "frame_of_record.tsv").open(encoding="utf-8") as handle:
        table = list(csv.DictReader(handle, delimiter="\t"))
    assert len(table) == 60
    lex = _ranked(runs["lex"], len(rows))
    for i, gold in accepted:
        rank = table[i]["rank:lexical"]
        expected_rank = lex[i].index(gold) + 1 if gold in lex[i] else None
        assert (int(rank) if rank else None) == expected_rank


def test_reruns_are_byte_identical(tmp_path: Path, frame: Path, runs: dict[str, Path]) -> None:
    assert _compute(tmp_path / "a", frame, runs).exit_code == 0
    assert _compute(tmp_path / "b", frame, runs).exit_code == 0
    for name in ("metrics.json", "frame_of_record.tsv", "run_manifest.json"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()


def test_split_moves_the_denominator(tmp_path: Path, frame: Path, runs: dict[str, Path]) -> None:
    out = tmp_path / "heldout"
    assert _compute(out, frame, runs, "--split", "heldout").exit_code == 0
    payload = _payload(out)
    assert payload["split"] == "heldout"
    assert payload["frame"]["n"] == 8 and payload["recall_denominator"] == 8
    # No held-out term has a Swedish surface, so lexical containment finds none.
    assert _hits(payload, "lexical") == (0, 0, 0)


def test_overrides_are_applied_counted_and_listed(
    tmp_path: Path, frame: Path, runs: dict[str, Path]
) -> None:
    rows = read_frame_rows()
    first_accept = next(r for r in rows if r["verdict"] == "accept" and r["split"] == "dev")
    overrides = tmp_path / "overrides.toml"
    overrides.write_text(
        f"""
[[override]]
query_id = "q5"
from = "no-target"
verdict = "accept"
targets = [{first_accept["targets"]}]
reason = "arbitration: a dictionary concept fits after all"

[[override]]
query_id = "q2"
from = "defer"
exclude = true
reason = "duplicate of another sampled query"

[[override]]
query_id = "{first_accept["query_id"]}"
from = "accept"
verdict = "no-target"
reason = "re-review: the listed target is a broader parent only"

[expect.all]
accept = 48
"no-target" = 11
defer = 0
""",
        encoding="utf-8",
    )
    out = tmp_path / "corrected"
    result = _compute(out, frame, runs, "--overrides", str(overrides))
    assert result.exit_code == 0, result.output
    payload = _payload(out)
    assert payload["overrides"]["applied"] == 3
    assert payload["overrides"]["by_kind"] == {"exclude": 1, "targets": 0, "verdict": 2}
    reasons = {r["query_id"]: r["reason"] for r in payload["overrides"]["records"]}
    assert reasons["q2"].startswith("duplicate")
    assert payload["frame"]["verdicts_before"] == {"accept": 48, "defer": 1, "no-target": 11}
    assert payload["frame"]["verdicts"] == {"accept": 48, "defer": 0, "no-target": 11}
    assert payload["frame"]["excluded"] == 1 and payload["frame"]["n"] == 59
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert "overrides" in {i["name"] for i in manifest["inputs"]}
    assert manifest["parameters"]["overrides_applied"] == 3
    with (out / "frame_of_record.tsv").open(encoding="utf-8") as handle:
        table = {r["query_id"]: r for r in csv.DictReader(handle, delimiter="\t")}
    assert table["q2"]["excluded"] == "1" and table["q2"]["override"] == "exclude"
    assert table["q5"]["verdict_before"] == "no-target" and table["q5"]["verdict"] == "accept"


# ---------------------------------------------------------------------------
# Corrupted inputs: one per gate, each with its own error and exit code.
# ---------------------------------------------------------------------------


def _mutated_frame(tmp_path: Path, edit: Any) -> Path:
    rows = read_frame_rows()
    edit(rows)
    return write_frame(tmp_path / "frame-bad.tsv", rows)


def _expect_gate(result: Result, out: Path, error: str, code: int = GATE_EXIT) -> None:
    assert result.exit_code == code, result.output
    assert error in result.output
    assert not out.exists(), "a failed gate must write nothing"


def _rerun_channels(tmp_path: Path, frame: Path, runs: dict[str, Path]) -> dict[str, Path]:
    """Runs over a different frame file (so the frame hash binds them to it)."""
    root = tmp_path / "rerun"
    runner = CliRunner()

    def step(key: str, *args: str) -> Path:
        out = root / key
        res = runner.invoke(app, ["retrieve", *args, "--out", str(out), "--created-at", CREATED_AT])
        assert res.exit_code == 0, res.output
        return out

    lex = step("lex", "lexical", "--concepts", str(CONCEPTS), "--queries", str(frame))
    emb = step("emb", "embed", "--index", str(runs["idx"]), "--queries", str(frame))
    fused = step("fused", "fuse", "--run", str(lex), "--run", str(emb))
    return {"lex": lex, "emb": emb, "fused": fused, "idx": runs["idx"]}


def test_gate_duplicate_query_id(tmp_path: Path, runs: dict[str, Path]) -> None:
    def edit(rows: list[dict[str, str]]) -> None:
        rows[3]["query_id"] = rows[4]["query_id"]

    bad = _mutated_frame(tmp_path, edit)
    out = tmp_path / "m"
    _expect_gate(
        _compute(out, bad, _rerun_channels(tmp_path, bad, runs)), out, "DuplicateQueryIdError"
    )


def test_gate_gold_not_in_dictionary(tmp_path: Path, runs: dict[str, Path]) -> None:
    def edit(rows: list[dict[str, str]]) -> None:
        index = next(i for i, r in enumerate(rows) if r["verdict"] == "accept")
        rows[index]["targets"] = "424242"  # invented id, absent from the fixture dictionary

    bad = _mutated_frame(tmp_path, edit)
    out = tmp_path / "m"
    _expect_gate(
        _compute(out, bad, _rerun_channels(tmp_path, bad, runs)), out, "GoldNotInDictionaryError"
    )


def test_gate_accept_without_target(tmp_path: Path, runs: dict[str, Path]) -> None:
    def edit(rows: list[dict[str, str]]) -> None:
        index = next(i for i, r in enumerate(rows) if r["verdict"] == "accept")
        rows[index]["targets"] = ""

    bad = _mutated_frame(tmp_path, edit)
    out = tmp_path / "m"
    _expect_gate(_compute(out, bad, _rerun_channels(tmp_path, bad, runs)), out, "VerdictError")


def test_gate_run_ranked_another_frame(tmp_path: Path, frame: Path, runs: dict[str, Path]) -> None:
    """Run/gold id-set mismatch: the runs ranked a frame with one more query."""
    rows = read_frame_rows()
    rows.append({**rows[-1], "query_id": "q-extra", "source_term": "demo extra term"})
    other = write_frame(tmp_path / "other.tsv", rows)
    other_runs = _rerun_channels(tmp_path, other, runs)
    out = tmp_path / "m"
    _expect_gate(_compute(out, frame, other_runs), out, "RunCoverageError")


def test_gate_fused_run_without_its_channels(
    tmp_path: Path, frame: Path, runs: dict[str, Path]
) -> None:
    out = tmp_path / "m"
    result = CliRunner().invoke(
        app,
        [
            "metrics",
            "compute",
            "--run",
            f"rrf={runs['fused']}",
            "--gold",
            str(frame),
            "--concepts",
            str(CONCEPTS),
            "--out",
            str(out),
        ],
    )
    _expect_gate(result, out, "RunCoverageError")


def test_gate_k_larger_than_run_depth(tmp_path: Path, frame: Path, runs: dict[str, Path]) -> None:
    out = tmp_path / "m"
    # The fused run kept its default top 50; k = 60 needs more.
    _expect_gate(_compute(out, frame, runs, "--k", "1,60"), out, "DepthExceededError")


def _tampered_run(tmp_path: Path, runs: dict[str, Path], edit: Any) -> dict[str, Path]:
    """A copy of the embed run whose rankings are edited and whose manifest is re-signed."""
    copy = tmp_path / "emb-edit"
    shutil.copytree(runs["emb"], copy)
    path = copy / "rankings.tsv"
    lines = path.read_text(encoding="utf-8").splitlines()
    edit(lines)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    from omopmap.artifacts.manifest import sha256_file

    manifest = json.loads((copy / "run_manifest.json").read_text(encoding="utf-8"))
    for record in manifest["outputs"]:
        if record["name"] == "rankings.tsv":
            record["sha256"] = sha256_file(path)
    (copy / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return {**runs, "emb": copy}


def test_gate_nan_score(tmp_path: Path, frame: Path, runs: dict[str, Path]) -> None:
    def edit(lines: list[str]) -> None:
        cells = lines[1].split("\t")
        cells[3] = "nan"
        lines[1] = "\t".join(cells)

    out = tmp_path / "m"
    _expect_gate(
        _compute(out, frame, _tampered_run(tmp_path, runs, edit)), out, "NonFiniteScoreError"
    )


def test_gate_duplicate_candidate(tmp_path: Path, frame: Path, runs: dict[str, Path]) -> None:
    def edit(lines: list[str]) -> None:
        first, second = lines[1].split("\t"), lines[2].split("\t")
        assert first[0] == second[0]
        second[2] = first[2]
        lines[2] = "\t".join(second)

    out = tmp_path / "m"
    _expect_gate(
        _compute(out, frame, _tampered_run(tmp_path, runs, edit)), out, "DuplicateCandidateError"
    )


def test_gate_ranks_out_of_sequence(tmp_path: Path, frame: Path, runs: dict[str, Path]) -> None:
    def edit(lines: list[str]) -> None:
        cells = lines[2].split("\t")
        cells[1] = "7"
        lines[2] = "\t".join(cells)

    out = tmp_path / "m"
    _expect_gate(
        _compute(out, frame, _tampered_run(tmp_path, runs, edit)), out, "RankingOrderError"
    )


def test_gate_override_contradicts_the_frame(
    tmp_path: Path, frame: Path, runs: dict[str, Path]
) -> None:
    overrides = tmp_path / "o.toml"
    overrides.write_text(
        '[[override]]\nquery_id = "q2"\nfrom = "accept"\nexclude = true\nreason = "r"\n',
        encoding="utf-8",
    )
    out = tmp_path / "m"
    _expect_gate(_compute(out, frame, runs, "--overrides", str(overrides)), out, "OverrideError")


def test_gate_override_without_reason(tmp_path: Path, frame: Path, runs: dict[str, Path]) -> None:
    overrides = tmp_path / "o.toml"
    overrides.write_text(
        '[[override]]\nquery_id = "q2"\nfrom = "defer"\nexclude = true\n', encoding="utf-8"
    )
    out = tmp_path / "m"
    _expect_gate(_compute(out, frame, runs, "--overrides", str(overrides)), out, "OverrideError")


def test_gate_expected_totals(tmp_path: Path, frame: Path, runs: dict[str, Path]) -> None:
    overrides = tmp_path / "o.toml"
    overrides.write_text(
        '[expect.all]\naccept = 47\n"no-target" = 12\ndefer = 1\n', encoding="utf-8"
    )
    out = tmp_path / "m"
    _expect_gate(
        _compute(out, frame, runs, "--overrides", str(overrides)), out, "ExpectedTotalsError"
    )


def test_gate_empty_denominator(tmp_path: Path, runs: dict[str, Path]) -> None:
    def edit(rows: list[dict[str, str]]) -> None:
        for row in rows:
            if row["split"] == "heldout":
                row["verdict"], row["targets"] = "no-target", ""

    bad = _mutated_frame(tmp_path, edit)
    out = tmp_path / "m"
    result = _compute(out, bad, _rerun_channels(tmp_path, bad, runs), "--split", "heldout")
    _expect_gate(result, out, "EmptyDenominatorError")


def test_tampered_rankings_exit_4(tmp_path: Path, frame: Path, runs: dict[str, Path]) -> None:
    copy = tmp_path / "emb-tampered"
    shutil.copytree(runs["emb"], copy)
    with (copy / "rankings.tsv").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    out = tmp_path / "m"
    _expect_gate(_compute(out, frame, {**runs, "emb": copy}), out, "RunHashError", RUN_HASH_EXIT)


@pytest.mark.parametrize("k", ["0,5", "5,5", "one"])
def test_bad_cut_offs_exit_1(tmp_path: Path, frame: Path, runs: dict[str, Path], k: str) -> None:
    out = tmp_path / "m"
    _expect_gate(_compute(out, frame, runs, "--k", k), out, "MetricsError", 1)


def test_occupied_output_exits_1(tmp_path: Path, frame: Path, runs: dict[str, Path]) -> None:
    out = tmp_path / "m"
    out.mkdir()
    (out / "keep.txt").write_text("x", encoding="utf-8")
    result = _compute(out, frame, runs)
    assert result.exit_code == 1 and "OutputExistsError" in result.output
    assert sorted(p.name for p in out.iterdir()) == ["keep.txt"]
