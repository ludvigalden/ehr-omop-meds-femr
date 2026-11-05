"""End-to-end ``omopmap validate runs`` over embed runs built from the committed fixtures.

The identical case exits 0 and writes ``validation.json`` plus a ``validate``
run manifest recording ``passed``; a rank swap beyond ``tie_eps`` exits 5 with
``DIVERGED``; tampered rankings, mixed channels and a taken output directory
all exit 1. The diverged copy is made honestly: the manifest's output digest
is refreshed over the swapped file, so the run still verifies against its own
manifest and only the rankings themselves differ.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner, Result

import omopmap.retrieval.cli as retrieval_cli
from omopmap.artifacts.manifest import sha256_file
from omopmap.artifacts.models import InputRecord
from omopmap.cli import app
from omopmap.retrieval.embed import Encoder
from omopmap.retrieval.manifest import RUN_MANIFEST_NAME, parse_run_manifest
from omopmap.retrieval.rankings import RANKINGS_NAME
from omopmap.validate_gpu_cpu import VALIDATION_NAME, RunComparison
from tests.retrieval.conftest import CONCEPTS, CREATED_AT, SOURCE_TERMS, FakeEncoder

TIE_EPS = 2e-3


def _run(*args: str) -> Result:
    return CliRunner().invoke(app, list(args))


@pytest.fixture(scope="module")
def fake_encoder() -> Iterator[FakeEncoder]:
    """Route every CLI ``--model`` to the deterministic 20-dim hashing encoder."""
    encoder = FakeEncoder()

    def factory(model: str) -> Encoder:
        del model
        return encoder

    original = retrieval_cli.ENCODER_FACTORY
    retrieval_cli.ENCODER_FACTORY = factory
    yield encoder
    retrieval_cli.ENCODER_FACTORY = original


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory, fake_encoder: FakeEncoder) -> dict[str, Path]:
    """Two embed runs over the same fixtures (identical), and one lexical run."""
    root = tmp_path_factory.mktemp("validate-runs")

    def step(key: str, *args: str) -> Path:
        out = root / key
        result = _run("retrieve", *args, "--out", str(out), "--created-at", CREATED_AT)
        assert result.exit_code == 0, result.output
        return out

    step("idx", "index", "--concepts", str(CONCEPTS))
    step("a", "embed", "--index", str(root / "idx"), "--queries", str(SOURCE_TERMS))
    step("b", "embed", "--index", str(root / "idx"), "--queries", str(SOURCE_TERMS))
    step("lex", "lexical", "--concepts", str(CONCEPTS), "--queries", str(SOURCE_TERMS))
    return {key: root / key for key in ("idx", "a", "b", "lex")}


def _copy_with_swapped_top_two(src: Path, dst: Path) -> None:
    """Copy a run, swap the top-2 concept ids of query 0, refresh the manifest digest.

    Only concept ids move; every score stays on its row, so ranks remain
    sequential and the score column stays descending. The two swapped scores
    must differ by at least ``tie_eps`` in the source run, or the swap would
    be tie-explained and the run would pass.
    """
    dst.mkdir()
    for name in (RANKINGS_NAME, RUN_MANIFEST_NAME):
        (dst / name).write_bytes((src / name).read_bytes())
    path = dst / RANKINGS_NAME
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    body = [line.rstrip("\n").split("\t") for line in lines[1:]]
    assert body[0][0] == "0" and body[0][1] == "1", "expected query 0 rank 1 first"
    assert float(body[0][3]) - float(body[1][3]) >= TIE_EPS, "top-2 gap is inside tie_eps"
    body[0][2], body[1][2] = body[1][2], body[0][2]
    path.write_text(lines[0] + "".join("\t".join(row) + "\n" for row in body), encoding="utf-8")
    manifest = parse_run_manifest((dst / RUN_MANIFEST_NAME).read_text(encoding="utf-8"))
    digest = sha256_file(path)
    outputs = tuple(
        InputRecord(
            name=record.name, sha256=digest if record.name == RANKINGS_NAME else record.sha256
        )
        for record in manifest.outputs
    )
    (dst / RUN_MANIFEST_NAME).write_text(
        manifest.model_copy(update={"outputs": outputs}).to_json(), encoding="utf-8"
    )


def test_identical_runs_pass_and_write_records(tmp_path: Path, runs: dict[str, Path]) -> None:
    out = tmp_path / "identical"
    result = _run(
        "validate",
        "runs",
        "--first",
        str(runs["a"]),
        "--second",
        str(runs["b"]),
        "--out",
        str(out),
        "--created-at",
        CREATED_AT,
    )
    assert result.exit_code == 0, result.output
    comparison = RunComparison.model_validate_json(
        (out / VALIDATION_NAME).read_text(encoding="utf-8")
    )
    assert comparison.passed and comparison.queries == 60
    assert comparison.identical_top_k == 60 and comparison.diverged_queries == ()
    assert comparison.max_score_delta == 0.0
    manifest = parse_run_manifest((out / RUN_MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest.run_type == "validate"
    assert manifest.parameters["passed"] is True
    assert manifest.parameters["queries"] == 60


def test_diverged_copy_exits_5(tmp_path: Path, runs: dict[str, Path]) -> None:
    diverged = tmp_path / "diverged"
    _copy_with_swapped_top_two(runs["a"], diverged)
    out = tmp_path / "verdict"
    result = _run(
        "validate",
        "runs",
        "--first",
        str(runs["a"]),
        "--second",
        str(diverged),
        "--out",
        str(out),
        "--created-at",
        CREATED_AT,
    )
    assert result.exit_code == 5
    assert "DIVERGED" in result.output
    comparison = RunComparison.model_validate_json(
        (out / VALIDATION_NAME).read_text(encoding="utf-8")
    )
    assert not comparison.passed
    assert comparison.diverged_queries == (0,)
    manifest = parse_run_manifest((out / RUN_MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest.parameters["passed"] is False


def test_tampered_rankings_without_manifest_update_exit_1(
    tmp_path: Path, runs: dict[str, Path]
) -> None:
    tampered = tmp_path / "tampered"
    tampered.mkdir()
    for name in (RANKINGS_NAME, RUN_MANIFEST_NAME):
        (tampered / name).write_bytes((runs["a"] / name).read_bytes())
    text = (tampered / RANKINGS_NAME).read_text(encoding="utf-8")
    (tampered / RANKINGS_NAME).write_text(text.replace("\t155\t", "\t154\t", 1), encoding="utf-8")
    out = tmp_path / "out"
    result = _run(
        "validate",
        "runs",
        "--first",
        str(runs["a"]),
        "--second",
        str(tampered),
        "--out",
        str(out),
    )
    assert result.exit_code == 1
    assert "differs from its manifest" in result.output
    assert not out.exists()


def test_lexical_vs_embed_exits_1_on_channels(tmp_path: Path, runs: dict[str, Path]) -> None:
    result = _run(
        "validate",
        "runs",
        "--first",
        str(runs["lex"]),
        "--second",
        str(runs["a"]),
        "--out",
        str(tmp_path / "out"),
    )
    assert result.exit_code == 1
    assert "different channels" in result.output


def test_existing_output_directory_is_refused(tmp_path: Path, runs: dict[str, Path]) -> None:
    out = tmp_path / "taken"
    out.mkdir()
    (out / "keep.txt").write_text("x", encoding="utf-8")
    result = _run(
        "validate",
        "runs",
        "--first",
        str(runs["a"]),
        "--second",
        str(runs["b"]),
        "--out",
        str(out),
    )
    assert result.exit_code == 1
    assert (out / "keep.txt").read_text(encoding="utf-8") == "x"


def test_malformed_scores_exit_1_not_traceback(tmp_path: Path) -> None:
    """Ascending scores within a query are a malformed run: clean exit 1."""
    from omopmap.artifacts.manifest import sha256_file
    from omopmap.retrieval.manifest import RUN_MANIFEST_NAME, parse_run_manifest
    from omopmap.retrieval.rankings import RANKINGS_NAME

    run = tmp_path / "run"
    run.mkdir()
    (run / RANKINGS_NAME).write_text(
        "query_index\trank\tconcept_id\tscore\n0\t1\t1\t0.1\n0\t2\t2\t0.9\n", encoding="utf-8"
    )
    manifest = {
        "run_type": "lexical",
        "producer": {"name": "omopmap", "version": "0.1.0"},
        "created_at": "2026-01-01T00:00:00Z",
        "seed": None,
        "encoder": None,
        "parameters": {"queries": 1},
        "inputs": [{"name": "queries", "sha256": "0" * 64}],
        "outputs": [{"name": RANKINGS_NAME, "sha256": sha256_file(run / RANKINGS_NAME)}],
        "truncated": None,
    }
    import json

    text = json.dumps(manifest)
    parse_run_manifest(text)  # the manifest itself is valid
    (run / RUN_MANIFEST_NAME).write_text(text, encoding="utf-8")
    from typer.testing import CliRunner

    from omopmap.cli import app

    result = CliRunner().invoke(
        app,
        [
            "validate",
            "runs",
            "--first",
            str(run),
            "--second",
            str(run),
            "--out",
            str(tmp_path / "o"),
        ],
    )
    assert result.exit_code == 1, result.output
    assert "RunError" in result.output and "malformed" in result.output
