"""End-to-end ``omopmap stats mcnemar``: lexical baseline vs embed candidate.

For every cut-off the test recomputes the 2x2 table, the discordant id
lists and the exact p-value independently — plain ``csv`` reads of both
``rankings.tsv`` files and the gold fixture, hits computed by hand,
p from a local two-sided binomial formula — and compares them cell by
cell with what the CLI wrote. Split restriction, query-table binding,
``--k`` validation, output-directory refusal and byte-identical reruns
are exercised against the same fixture runs.
"""

from __future__ import annotations

import csv
import json
import time
from collections.abc import Iterator
from math import comb
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner, Result

import omopmap.retrieval.cli as retrieval_cli
from omopmap.cli import app
from omopmap.retrieval.embed import Encoder
from tests.retrieval.conftest import CONCEPTS, CREATED_AT, SOURCE_TERMS, FakeEncoder

KS = "1,5,10"


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
def sv_labels(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Train-split terms as extra surfaces of their gold concept."""
    path = tmp_path_factory.mktemp("stats-labels") / "sv_labels.tsv"
    with SOURCE_TERMS.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    with path.open("w", encoding="utf-8") as handle:
        handle.write("concept_id\tlabel\n")
        for row in rows:
            if row["split"] == "train":
                handle.write(f"{row['concept_id']}\t{row['source_term']}\n")
    return path


@pytest.fixture(scope="module")
def runs(
    tmp_path_factory: pytest.TempPathFactory, sv_labels: Path, fake_encoder: FakeEncoder
) -> dict[str, Path]:
    """A lexical baseline run and an embed candidate run, both over the fixture table."""
    root = tmp_path_factory.mktemp("stats-runs")

    def step(key: str, *args: str) -> Path:
        out = root / key
        result = _run("retrieve", *args, "--out", str(out), "--created-at", CREATED_AT)
        assert result.exit_code == 0, result.output
        return out

    step(
        "lex",
        "lexical",
        "--concepts",
        str(CONCEPTS),
        "--queries",
        str(SOURCE_TERMS),
        "--labels",
        str(sv_labels),
    )
    step("idx", "index", "--concepts", str(CONCEPTS))
    step("emb", "embed", "--index", str(root / "idx"), "--queries", str(SOURCE_TERMS))
    return {"lex": root / "lex", "emb": root / "emb"}


def _two_sided_p(baseline_only: int, candidate_only: int) -> float:
    """Exact two-sided binomial p on the discordant pairs, from ``math.comb``."""
    n = baseline_only + candidate_only
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(baseline_only, candidate_only) + 1))
    return min(1.0, 2 * tail / 2**n)


def _ranked_by_row(run: Path, n_rows: int) -> list[list[int]]:
    """Ranked concept ids per query row, rank 1 first, read with plain csv."""
    with (run / "rankings.tsv").open(encoding="utf-8") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))[1:]
    ranked: list[list[int]] = [[] for _ in range(n_rows)]
    for row in rows:
        ranked[int(row[0])].append(int(row[2]))
    return ranked


def _gold_concepts() -> list[int]:
    with SOURCE_TERMS.open(encoding="utf-8") as handle:
        return [int(r["concept_id"]) for r in csv.DictReader(handle, delimiter="\t")]


def _invoke_mcnemar(out: Path, runs: dict[str, Path], *extra: str) -> Result:
    return _run(
        "stats",
        "mcnemar",
        "--baseline",
        str(runs["lex"]),
        "--candidate",
        str(runs["emb"]),
        "--gold",
        str(SOURCE_TERMS),
        "--out",
        str(out),
        "--k",
        KS,
        "--created-at",
        CREATED_AT,
        *extra,
    )


def test_mcnemar_matches_independent_per_cell_computation(
    tmp_path: Path, runs: dict[str, Path]
) -> None:
    started = time.monotonic()
    out = tmp_path / "mc"
    result = _invoke_mcnemar(out, runs)
    elapsed = time.monotonic() - started
    assert result.exit_code == 0, result.output
    assert elapsed < 60.0, f"mcnemar took {elapsed:.1f}s"
    payload: dict[str, Any] = json.loads((out / "mcnemar.json").read_text(encoding="utf-8"))
    assert payload["ks"] == [1, 5, 10]
    gold = _gold_concepts()
    base = _ranked_by_row(runs["lex"], len(gold))
    cand = _ranked_by_row(runs["emb"], len(gold))
    assert len(payload["results"]) == 3
    for entry, k in zip(payload["results"], (1, 5, 10), strict=True):
        hits_base = [gold[i] in base[i][:k] for i in range(len(gold))]
        hits_cand = [gold[i] in cand[i][:k] for i in range(len(gold))]
        pairs = list(zip(hits_base, hits_cand, strict=True))
        a = sum(1 for hb, hc in pairs if hb and hc)
        b_ids = sorted(str(i) for i, (hb, hc) in enumerate(pairs) if hb and not hc)
        c_ids = sorted(str(i) for i, (hb, hc) in enumerate(pairs) if hc and not hb)
        d = sum(1 for hb, hc in pairs if not hb and not hc)
        assert entry["k"] == k and entry["n"] == len(gold)
        assert entry["both_hit"] == a and entry["neither_hit"] == d
        assert entry["baseline_only"] == len(b_ids) and entry["candidate_only"] == len(c_ids)
        assert entry["baseline_hits"] == a + len(b_ids)
        assert entry["candidate_hits"] == a + len(c_ids)
        assert sorted(entry["baseline_only_ids"]) == b_ids
        assert sorted(entry["candidate_only_ids"]) == c_ids
        assert entry["p_value"] == pytest.approx(
            _two_sided_p(len(b_ids), len(c_ids)), rel=1e-12, abs=1e-15
        )
    # Per-query outcomes are present for every cut-off and agree with the cells.
    for k in ("1", "5", "10"):
        rows = payload["outcomes"][k]
        assert len(rows) == len(gold)
        assert len({row["query_id"] for row in rows}) == len(gold)


def test_split_restricts_the_queries(tmp_path: Path, runs: dict[str, Path]) -> None:
    heldout = tmp_path / "heldout"
    assert _invoke_mcnemar(heldout, runs, "--split", "heldout").exit_code == 0
    payload: dict[str, Any] = json.loads((heldout / "mcnemar.json").read_text(encoding="utf-8"))
    assert all(entry["n"] == 8 for entry in payload["results"])
    train = tmp_path / "train"
    assert _invoke_mcnemar(train, runs, "--split", "train").exit_code == 0
    payload = json.loads((train / "mcnemar.json").read_text(encoding="utf-8"))
    assert all(entry["n"] == 41 for entry in payload["results"])


def test_run_on_a_different_query_table_fails_closed(
    tmp_path: Path, runs: dict[str, Path], sv_labels: Path
) -> None:
    other = tmp_path / "other-terms.tsv"
    with SOURCE_TERMS.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    rows.append({"source_term": "demo term ej i tabellen", "concept_id": "5", "split": "train"})
    with other.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    other_run = tmp_path / "other-run"
    result = _run(
        "retrieve",
        "lexical",
        "--concepts",
        str(CONCEPTS),
        "--queries",
        str(other),
        "--labels",
        str(sv_labels),
        "--out",
        str(other_run),
        "--created-at",
        CREATED_AT,
    )
    assert result.exit_code == 0, result.output
    result = _run(
        "stats",
        "mcnemar",
        "--baseline",
        str(runs["lex"]),
        "--candidate",
        str(other_run),
        "--gold",
        str(SOURCE_TERMS),
        "--out",
        str(tmp_path / "out"),
    )
    assert result.exit_code == 1
    assert "query table" in result.output


def test_bad_k_and_taken_output_fail(tmp_path: Path, runs: dict[str, Path]) -> None:
    result = _invoke_mcnemar(tmp_path / "k0", runs, "--k", "0")
    assert result.exit_code == 1
    taken = tmp_path / "taken"
    taken.mkdir()
    (taken / "keep.txt").write_text("x", encoding="utf-8")
    result = _invoke_mcnemar(taken, runs)
    assert result.exit_code == 1
    assert (taken / "keep.txt").read_text(encoding="utf-8") == "x"


def test_rerun_is_byte_identical(tmp_path: Path, runs: dict[str, Path]) -> None:
    first, second = tmp_path / "one", tmp_path / "two"
    assert _invoke_mcnemar(first, runs).exit_code == 0
    assert _invoke_mcnemar(second, runs).exit_code == 0
    for name in ("mcnemar.json", "run_manifest.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
