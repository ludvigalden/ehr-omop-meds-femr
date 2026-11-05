"""Shared fixtures for the adjudication tests.

A nine-case workspace over the committed fixtures: each case's source term is
an exact concept name, so the fake-encoder embed run puts the case's gold
concept at rank 1 and it always reaches the sheet. Two runs are built with
distinctive names (SYSALPHA = lexical, SYSBETA = embed) so the blinding scan
can look for them by name, and the sheet/key pair is built once per module
with a fixed seed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

import omopmap.retrieval.cli as retrieval_cli
from omopmap.adjudication import KEY_NAME, SHEET_NAME, AdjudicationKey
from omopmap.cli import app
from omopmap.retrieval.embed import Encoder
from omopmap.retrieval.fusion import ScoredRanking
from omopmap.retrieval.rankings import RANKINGS_NAME, decode_scored
from tests.metrics.conftest import read_source_terms
from tests.retrieval.conftest import CONCEPTS, CREATED_AT, FakeEncoder

_FAKE = FakeEncoder()


def _fake_factory(model: str) -> Encoder:
    """Route every CLI ``--model`` to the deterministic fake encoder."""
    del model
    return _FAKE


SEED = 12345
N_CASES = 9
CASE_GOLD: tuple[tuple[int, str], ...] = (
    (1, "demo disorder alpha"),
    (2, "demo medication alpha"),
    (3, "demo procedure alpha"),
    (5, "demo finding alpha"),
    (6, "demo disorder beta"),
    (7, "demo medication beta"),
    (8, "demo procedure beta"),
    (10, "demo finding beta"),
    (12, "demo medication gamma"),
)
LEX_NAME, EMB_NAME = "SYSALPHA", "SYSBETA"
CASE_SPLITS: tuple[str, ...] = (
    "train",
    "train",
    "dev",
    "dev",
    "heldout",
    "train",
    "dev",
    "heldout",
    "heldout",
)


def write_cases_table(path: Path) -> Path:
    """The query table the runs rank: ids, system/code, terms, splits."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("query_id\tsystem\tcode\tsource_term\tsplit\n")
        for index, (_, name) in enumerate(CASE_GOLD):
            handle.write(f"q{index}\tDEMOSYS\tS{index:03d}\t{name}\t{CASE_SPLITS[index]}\n")
    return path


def write_labels(path: Path) -> Path:
    """Train-split surfaces as the lexical run's extra labels."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("concept_id\tlabel\n")
        for term in read_source_terms():
            if term["split"] == "train":
                handle.write(f"{term['concept_id']}\t{term['source_term']}\n")
    return path


def run_cli(*args: str) -> int:
    """Invoke one ``omopmap`` command and return its exit code."""
    return CliRunner().invoke(app, list(args)).exit_code


def run_step(out: Path, *args: str) -> None:
    """Invoke one ``omopmap`` command that takes ``--out`` and require exit 0."""
    result = CliRunner().invoke(app, [*args, "--out", str(out), "--created-at", CREATED_AT])
    assert result.exit_code == 0, result.output


class AdjudicationWorkspace:
    """The built sheet and key, the runs behind them, and the gold concepts."""

    def __init__(
        self, root: Path, cases: Path, lex: Path, emb: Path, sheet: Path, key: Path
    ) -> None:
        self.root = root
        self.cases = cases
        self.lex = lex
        self.emb = emb
        self.sheet = sheet
        self.key = key

    @property
    def sheet_bytes(self) -> bytes:
        return (self.sheet / SHEET_NAME).read_bytes()

    @property
    def key_model(self) -> AdjudicationKey:
        return AdjudicationKey.model_validate_json(
            (self.key / KEY_NAME).read_text(encoding="utf-8")
        )

    def gold_of(self, index: int) -> int:
        """The gold concept of case ``index`` (the embed run's rank 1)."""
        return CASE_GOLD[index][0]

    def embed_rankings(self) -> tuple[ScoredRanking, ...]:
        """The embed run's decoded rankings."""
        data = (self.emb / RANKINGS_NAME).read_bytes()
        return tuple(decode_scored(data, channel="embed", queries=N_CASES))

    def lexical_rankings(self) -> tuple[ScoredRanking, ...]:
        """The lexical run's decoded rankings."""
        data = (self.lex / RANKINGS_NAME).read_bytes()
        return tuple(decode_scored(data, channel="lexical", queries=N_CASES))

    def build(self, sheet_out: Path, key_out: Path, *extra: str) -> int:
        """Invoke ``adjudicate build`` over this workspace's runs and cases."""
        return run_cli(
            "adjudicate",
            "build",
            "--run",
            f"{LEX_NAME}={self.lex}",
            "--run",
            f"{EMB_NAME}={self.emb}",
            "--cases",
            str(self.cases),
            "--concepts",
            str(CONCEPTS),
            "--sheet-out",
            str(sheet_out),
            "--key-out",
            str(key_out),
            "--created-at",
            CREATED_AT,
            *extra,
        )


@pytest.fixture(scope="module")
def workspace(tmp_path_factory: pytest.TempPathFactory) -> AdjudicationWorkspace:
    """The adjudication workspace: cases, both runs, and one built sheet/key."""
    root = tmp_path_factory.mktemp("adjudication-ws")
    original = retrieval_cli.ENCODER_FACTORY
    retrieval_cli.ENCODER_FACTORY = _fake_factory
    try:
        cases = write_cases_table(root / "cases.tsv")
        write_labels(root / "labels.tsv")
        idx = root / "idx"
        run_step(idx, "retrieve", "index", "--concepts", str(CONCEPTS))
        emb = root / "emb"
        run_step(emb, "retrieve", "embed", "--index", str(idx), "--queries", str(cases))
        lex = root / "lex"
        run_step(
            lex,
            "retrieve",
            "lexical",
            "--concepts",
            str(CONCEPTS),
            "--queries",
            str(cases),
            "--labels",
            str(root / "labels.tsv"),
        )
    finally:
        retrieval_cli.ENCODER_FACTORY = original
    built = AdjudicationWorkspace(root, cases, lex, emb, root / "sheet", root / "key")
    exit_code = built.build(root / "sheet", root / "key", "--seed", str(SEED))
    assert exit_code == 0
    return built
