"""Shared fixtures: an adjudicated frame of record over the committed term fixture.

The frame reuses the B2 ``source_terms`` table (so lexical/embed runs over it
already exist as patterns in the retrieval tests) and adds the adjudication
columns: a query id, a verdict and, for accepted queries, the acceptable
target set. The verdicts are a pure function of the fixture: a term whose
gold concept is an eligible target is ``accept`` with that concept as its
target, except two hand-picked rows that are turned into ``defer`` /
``no-target`` so all three verdicts exist; a term whose gold concept is not
eligible is ``no-target`` — exactly the situation the frame-of-record model
exists for (nothing in the candidate dictionary fits).
"""

from __future__ import annotations

import csv
from collections.abc import Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

import omopmap.retrieval.cli as retrieval_cli
from omopmap.cli import app
from omopmap.retrieval.cli import eligible_labels
from omopmap.retrieval.embed import Encoder
from tests.retrieval.conftest import CONCEPTS, CREATED_AT, SOURCE_TERMS, FakeEncoder

FRAME_COLUMNS = ["query_id", "source_term", "split", "verdict", "targets"]


def read_source_terms() -> list[dict[str, str]]:
    """The committed B2 term fixture rows."""
    with SOURCE_TERMS.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_frame_rows() -> list[dict[str, str]]:
    """The frame rows, deterministically derived from the committed fixtures."""
    terms = read_source_terms()
    eligible = {label.concept_id for label in eligible_labels(CONCEPTS, False)}
    rows: list[dict[str, str]] = []
    for index, term in enumerate(terms):
        query_id = f"q{index}"
        gold = int(term["concept_id"])
        if index == 2:  # a reviewed query nobody could decide on
            verdict, targets = "defer", ""
        elif index == 5:  # reviewed, and nothing in the dictionary fits
            verdict, targets = "no-target", ""
        elif gold in eligible:
            verdict, targets = "accept", str(gold)
        else:
            verdict, targets = "no-target", ""
        rows.append(
            {
                "query_id": query_id,
                "source_term": term["source_term"],
                "split": term["split"],
                "verdict": verdict,
                "targets": targets,
            }
        )
    return rows


def write_frame(path: Path, rows: list[dict[str, str]] | None = None) -> Path:
    """Write the frame (or the given rows) as a TSV with the expected header."""
    rows = read_frame_rows() if rows is None else rows
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=FRAME_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.fixture(scope="module")
def fake_encoder() -> Iterator[FakeEncoder]:
    """Route every CLI ``--model`` to the deterministic fake encoder."""
    encoder = FakeEncoder()

    def factory(model: str) -> Encoder:
        del model
        return encoder

    original = retrieval_cli.ENCODER_FACTORY
    retrieval_cli.ENCODER_FACTORY = factory
    yield encoder
    retrieval_cli.ENCODER_FACTORY = original


@pytest.fixture(scope="module")
def frame(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The frame-of-record TSV."""
    return write_frame(tmp_path_factory.mktemp("metrics-frame") / "frame.tsv")


@pytest.fixture(scope="module")
def runs(
    tmp_path_factory: pytest.TempPathFactory, frame: Path, fake_encoder: FakeEncoder
) -> dict[str, Path]:
    """A lexical run, an embed run and their fusion, all over the frame table."""
    root = tmp_path_factory.mktemp("metrics-runs")
    runner = CliRunner()

    def step(key: str, *args: str) -> Path:
        out = root / key
        result = runner.invoke(
            app, ["retrieve", *args, "--out", str(out), "--created-at", CREATED_AT]
        )
        assert result.exit_code == 0, result.output
        return out

    labels = root / "sv_labels.tsv"
    with labels.open("w", encoding="utf-8") as handle:
        handle.write("concept_id\tlabel\n")
        for row in read_source_terms():
            if row["split"] == "train":
                handle.write(f"{row['concept_id']}\t{row['source_term']}\n")
    lex = step(
        "lex",
        "lexical",
        "--concepts",
        str(CONCEPTS),
        "--queries",
        str(frame),
        "--labels",
        str(labels),
    )
    idx = step("idx", "index", "--concepts", str(CONCEPTS))
    emb = step("emb", "embed", "--index", str(idx), "--queries", str(frame))
    fused = step(
        "fused",
        "fuse",
        "--run",
        str(lex),
        "--run",
        str(emb),
    )
    return {"lex": lex, "idx": idx, "emb": emb, "fused": fused}
