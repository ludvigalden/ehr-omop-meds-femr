"""End-to-end retrieval on the committed fixtures with the fake encoder.

Exact expected numbers: lexical recovers exactly the 33 train-split terms whose
gold concept is eligible (standard and valid); the fake encoder's hash vectors
hit @1..@50 as pinned; RRF fusion combines both channels. Every number is
cross-checked with an independent in-test computation (brute-force containment;
direct numpy dot-product ranking), so a change in any step moves both.
"""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner, Result

import omopmap.retrieval.cli as retrieval_cli
from omopmap.cli import app
from omopmap.retrieval.lexical import Diacritics, Label, build_index, search
from omopmap.retrieval.manifest import parse_run_manifest
from tests.retrieval.conftest import CONCEPTS, CREATED_AT, SOURCE_TERMS, FakeEncoder

FIXTURES = CONCEPTS.parent


def _run(*args: str) -> Result:
    return CliRunner().invoke(app, list(args))


@pytest.fixture(scope="module")
def sv_labels(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Train-split terms as Swedish surfaces of their gold concept."""
    path = tmp_path_factory.mktemp("labels") / "sv_labels.tsv"
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
    root = tmp_path_factory.mktemp("runs")

    def step(key: str, *args: str) -> Path:
        out = root / key
        result = _run(*args, "--out", str(out), "--created-at", CREATED_AT)
        assert result.exit_code == 0, result.output
        return out

    step(
        "lex",
        "retrieve",
        "lexical",
        "--concepts",
        str(CONCEPTS),
        "--queries",
        str(SOURCE_TERMS),
        "--labels",
        str(sv_labels),
    )
    step("idx", "retrieve", "index", "--concepts", str(CONCEPTS))
    step(
        "emb",
        "retrieve",
        "embed",
        "--index",
        str(root / "idx"),
        "--queries",
        str(SOURCE_TERMS),
    )
    step("fused", "retrieve", "fuse", "--run", str(root / "lex"), "--run", str(root / "emb"))
    return {key: root / key for key in ("lex", "idx", "emb", "fused")}


def _gold() -> list[int]:
    with SOURCE_TERMS.open(encoding="utf-8") as handle:
        return [int(r["concept_id"]) for r in csv.DictReader(handle, delimiter="\t")]


def _ranked(run: Path) -> list[list[int]]:
    with (run / "rankings.tsv").open(encoding="utf-8") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))[1:]
    ranked: list[list[int]] = [[] for _ in _gold()]
    for row in rows:
        ranked[int(row[0])].append(int(row[2]))
    return ranked


def _hits_at(ranked: list[list[int]], k: int) -> int:
    gold = _gold()
    return sum(1 for query, target in enumerate(gold) if target in ranked[query][:k])


def test_lexical_recall_is_exactly_the_eligible_train_surfaces(
    runs: dict[str, Path], sv_labels: Path
) -> None:
    ranked = _ranked(runs["lex"])
    # 41 train terms, 8 of whose gold concepts are ineligible, and none of the
    # dev/heldout terms match anything (the concept names are English).
    assert _hits_at(ranked, 50) == 33
    # Independent brute force: same number from a direct scan over the labels.

    with CONCEPTS.open(encoding="utf-8") as handle:
        concepts = list(csv.DictReader(handle, delimiter="\t"))
    with sv_labels.open(encoding="utf-8") as handle:
        sv = [int(r["concept_id"]) for r in csv.DictReader(handle, delimiter="\t")]
    eligible = {
        int(r["concept_id"])
        for r in concepts
        if r["standard_concept"] == "S" and r["invalid_reason"] == ""
    }
    labels = [Label(int(r["concept_id"]), r["concept_name"]) for r in concepts]
    labels += [Label(cid, term) for cid, term in zip(sv, _train_terms(), strict=True)]
    index = build_index([lb for lb in labels if lb.concept_id in eligible], Diacritics.KEEP)
    hits = sum(
        1
        for term, target in zip(_train_terms(), _train_gold(), strict=True)
        if target in search(index, term, depth=50).ids
    )
    assert hits == 33 == _hits_at(ranked, 1)


def _train_terms() -> list[str]:
    with SOURCE_TERMS.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return [r["source_term"] for r in rows if r["split"] == "train"]


def _train_gold() -> list[int]:
    with SOURCE_TERMS.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return [int(r["concept_id"]) for r in rows if r["split"] == "train"]


def test_embed_and_fused_hits_are_pinned_and_cross_checked(runs: dict[str, Path]) -> None:
    emb, fused = _ranked(runs["emb"]), _ranked(runs["fused"])
    assert [_hits_at(emb, k) for k in (1, 5, 10, 50)] == [0, 3, 5, 14]
    assert [_hits_at(fused, k) for k in (1, 5, 10, 50)] == [21, 30, 36, 38]
    # Cross-check the embed channel independently: fake vectors, numpy ranking.
    spec = importlib.util.spec_from_file_location("fe", FIXTURES / "fake_encoder.py")
    assert spec is not None and spec.loader is not None
    fake = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fake)
    with CONCEPTS.open(encoding="utf-8") as handle:
        concepts = [
            r
            for r in csv.DictReader(handle, delimiter="\t")
            if r["standard_concept"] == "S" and r["invalid_reason"] == ""
        ]
    matrix = fake.encode([r["concept_name"] for r in concepts])
    ids = np.array([int(r["concept_id"]) for r in concepts], dtype=np.int64)
    with SOURCE_TERMS.open(encoding="utf-8") as handle:
        terms = [r["source_term"] for r in csv.DictReader(handle, delimiter="\t")]
    scores = fake.encode(terms).astype(np.float32) @ matrix.T.astype(np.float32)
    for query in range(len(terms)):
        top = np.lexsort((ids, -scores[query]))[:200]
        assert emb[query] == [int(i) for i in ids[top]]


def test_run_manifests_record_model_input_output_hashes(runs: dict[str, Path]) -> None:
    index = parse_run_manifest((runs["idx"] / "run_manifest.json").read_text(encoding="utf-8"))
    embed = parse_run_manifest((runs["emb"] / "run_manifest.json").read_text(encoding="utf-8"))
    assert index.run_type == "index" and embed.run_type == "embed"
    assert index.encoder is not None
    assert index.encoder.model_id == "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR"
    assert index.encoder.revision == "47b6bd041ba61311584bb2494edfda5c7d9b719f"
    assert embed.inputs_by_name()["index:matrix.npy"] == index.output_sha256("matrix.npy")


def test_runs_are_byte_identical_across_directories(
    tmp_path: Path, runs: dict[str, Path], fake_encoder: FakeEncoder
) -> None:
    first = runs["emb"]
    second = tmp_path / "again"
    result = _run(
        "retrieve",
        "embed",
        "--index",
        str(runs["idx"]),
        "--queries",
        str(SOURCE_TERMS),
        "--out",
        str(second),
        "--created-at",
        CREATED_AT,
    )
    assert result.exit_code == 0, result.output
    for name in ("rankings.tsv", "run_manifest.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_tampered_matrix_exits_4(tmp_path: Path, runs: dict[str, Path]) -> None:
    broken = tmp_path / "broken-index"
    broken.mkdir()
    for name in ("matrix.npy", "ids.txt", "run_manifest.json"):
        (broken / name).write_bytes((runs["idx"] / name).read_bytes())
    matrix = bytearray((broken / "matrix.npy").read_bytes())
    matrix[-1] ^= 0xFF
    (broken / "matrix.npy").write_bytes(matrix)
    result = _run(
        "retrieve",
        "embed",
        "--index",
        str(broken),
        "--queries",
        str(SOURCE_TERMS),
        "--out",
        str(tmp_path / "out"),
    )
    assert result.exit_code == 4
    assert not (tmp_path / "out").exists()


def test_encoder_mismatch_against_the_index_fails_closed(
    tmp_path: Path, runs: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    def other_encoder(model: str) -> FakeEncoder:
        del model
        return FakeEncoder(revision="f" * 40)

    monkeypatch.setattr(retrieval_cli, "ENCODER_FACTORY", other_encoder)
    result = _run(
        "retrieve",
        "embed",
        "--index",
        str(runs["idx"]),
        "--queries",
        str(SOURCE_TERMS),
        "--out",
        str(tmp_path / "out"),
    )
    assert result.exit_code == 1
    assert "another encoder" in result.output
    assert not (tmp_path / "out").exists()


def test_fuse_refuses_mismatched_query_tables(
    tmp_path: Path, runs: dict[str, Path], sv_labels: Path, fake_encoder: FakeEncoder
) -> None:
    other = tmp_path / "other-queries.tsv"
    other.write_text("source_term\nfeber\n", encoding="utf-8")
    other_run = tmp_path / "other"
    result = _run(
        "retrieve",
        "embed",
        "--index",
        str(runs["idx"]),
        "--queries",
        str(other),
        "--out",
        str(other_run),
        "--created-at",
        CREATED_AT,
    )
    assert result.exit_code == 0, result.output
    fused = _run(
        "retrieve",
        "fuse",
        "--run",
        str(runs["lex"]),
        "--run",
        str(other_run),
        "--out",
        str(tmp_path / "fused"),
    )
    assert fused.exit_code == 1
    assert "same query table" in fused.output


def test_fuse_checks_channel_rankings_against_their_manifests(
    tmp_path: Path, runs: dict[str, Path]
) -> None:
    tampered = tmp_path / "tampered"
    tampered.mkdir()
    for name in ("rankings.tsv", "run_manifest.json"):
        (tampered / name).write_bytes((runs["lex"] / name).read_bytes())
    (tampered / "rankings.tsv").write_text(
        (tampered / "rankings.tsv")
        .read_text(encoding="utf-8")
        .replace("1\t1\t", "1\t1\t9999\t", 1),
        encoding="utf-8",
    )
    result = _run(
        "retrieve",
        "fuse",
        "--run",
        str(tampered),
        "--run",
        str(runs["emb"]),
        "--out",
        str(tmp_path / "fused"),
    )
    assert result.exit_code == 1
    assert "differs from its manifest" in result.output


def test_existing_output_directory_is_refused(tmp_path: Path, runs: dict[str, Path]) -> None:
    out = tmp_path / "taken"
    out.mkdir()
    (out / "keep.txt").write_text("x", encoding="utf-8")
    result = _run("retrieve", "index", "--concepts", str(CONCEPTS), "--out", str(out))
    assert result.exit_code == 1
    assert (out / "keep.txt").read_text(encoding="utf-8") == "x"


def test_gate_passes_under_budget(
    tmp_path: Path, runs: dict[str, Path], fake_encoder: FakeEncoder
) -> None:
    out = tmp_path / "gate-ok"
    result = _run(
        "gate",
        "--concepts",
        str(CONCEPTS),
        "--queries",
        str(SOURCE_TERMS),
        "--out",
        str(out),
        "--budget-hours",
        "1000",
    )
    assert result.exit_code == 0, result.output
    assert (out / "gate.json").is_file() and not (out / "GATE_STOP.json").exists()


def test_gate_over_2x_budget_writes_stop_and_exits_3(
    tmp_path: Path, fake_encoder: FakeEncoder
) -> None:
    out = tmp_path / "gate-stop"
    result = _run(
        "gate",
        "--concepts",
        str(CONCEPTS),
        "--queries",
        str(SOURCE_TERMS),
        "--out",
        str(out),
        "--budget-hours",
        "0.000001",
        "--fixed-seconds",
        "3600",
    )
    assert result.exit_code == 3
    assert "GATE STOP" in result.output
    stop = out / "GATE_STOP.json"
    assert stop.is_file()
    assert '"stop": true' in stop.read_text(encoding="utf-8")
