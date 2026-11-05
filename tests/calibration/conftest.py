"""Shared fixtures for the calibration tests.

Two kinds of fixture live here:

- ``make_evidence``: a keyword helper over the :class:`Evidence` dataclass, so
  the pure-function tests can spell out one feature per argument.
- The CLI workspace fixtures: a 24-query frame of record (12 train, 6 dev,
  6 heldout) with a lexical run, an embed run (depth 50) and proposals read
  from the embed run's own rankings. The first ten train terms are exact
  concept names, so the fake encoder puts the gold concept at rank 1 with a
  top-1 score near 1.0; six of them carry rank-1, uncapped proposals, which
  makes ``calibrate fit --learn-s1`` pass: the pool of decided rank-1
  proposals is six rows, all exact.
"""

from __future__ import annotations

import csv
from collections.abc import Iterator
from pathlib import Path
from types import TracebackType
from typing import Self

import pytest
from typer.testing import CliRunner

import omopmap.retrieval.cli as retrieval_cli
from omopmap.calibration import Evidence, Label
from omopmap.cli import app
from omopmap.retrieval.embed import Encoder
from omopmap.retrieval.fusion import ScoredRanking
from omopmap.retrieval.rankings import RANKINGS_NAME, decode_scored
from tests.metrics.conftest import read_source_terms
from tests.retrieval.conftest import CONCEPTS, CREATED_AT, FakeEncoder

N_QUERIES = 24
TRAIN_NAMES: tuple[tuple[int, str], ...] = (
    (1, "demo disorder alpha"),
    (2, "demo medication alpha"),
    (3, "demo procedure alpha"),
    (5, "demo finding alpha"),
    (6, "demo disorder beta"),
    (7, "demo medication beta"),
    (8, "demo procedure beta"),
    (10, "demo finding beta"),
    (12, "demo medication gamma"),
    (15, "demo finding gamma"),
)
UNRELATED_TARGET = 20

# One entry per query row: the proposal kind, the mapping path, the verdict
# and the targets. ``rank1``/``rank2``/``rank3`` propose the concept at that
# rank of the embed run; ``absent`` proposes an eligible concept the embed run
# did not return; ``invalid`` proposes a concept outside the dictionary.
PROPOSAL_KINDS: dict[int, str] = {
    0: "rank1",
    1: "rank1",
    2: "rank1",
    3: "absent",
    4: "rank1",
    5: "rank2",
    6: "rank1",
    7: "rank2",
    8: "rank1",
    9: "rank1",
    10: "rank3",
    11: "absent",
    12: "invalid",
    13: "rank1",
    14: "rank2",
    15: "absent",
    16: "rank1",
    17: "rank2",
    18: "rank1",
    19: "absent",
    20: "rank1",
    21: "rank2",
    22: "rank3",
    23: "absent",
}
PROPOSAL_PATHS: dict[int, str] = {
    row: ("truncation" if row % 7 == 0 else "committed") for row in range(N_QUERIES)
}
INVALID_CONCEPT = 4  # a committed fixture concept with invalid_reason = D


def source_term(row: int) -> str:
    """The source term of query row ``row``: an exact concept name or an invented text."""
    if row < len(TRAIN_NAMES):
        return TRAIN_NAMES[row][1]
    if row == 13:
        return ""  # a query with no source text
    return f"demo symptom text {row}"


def split_of(row: int) -> str:
    """``train`` for rows 0-11, ``dev`` for rows 12-17, ``heldout`` for rows 18-23."""
    if row < 12:
        return "train"
    if row < 18:
        return "dev"
    return "heldout"


def verdict_of(row: int) -> tuple[str, str]:
    """The (verdict, targets) of query row ``row``."""
    if row < len(TRAIN_NAMES):
        return "accept", str(TRAIN_NAMES[row][0])
    if row == 13 or row == 21:
        return "no-target", ""
    return "accept", str(UNRELATED_TARGET)


def write_cases(path: Path, rows: list[dict[str, str]]) -> Path:
    """Write the frame-of-record table with its system/code columns."""
    columns = ["query_id", "system", "code", "source_term", "split", "verdict", "targets"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_proposals(path: Path, rows: list[dict[str, str]]) -> Path:
    """Write the proposals table."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, delimiter="\t", fieldnames=["query_id", "concept_id", "path"]
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


class EncoderSwitch:
    """Routes CLI ``--model`` to a fake encoder whose revision can change."""

    def __init__(self) -> None:
        self._encoder = FakeEncoder()
        self._original: object = None

    def switch(self, revision: str) -> None:
        self._encoder = FakeEncoder(revision=revision)

    def factory(self, model: str) -> Encoder:
        del model
        return self._encoder

    def __enter__(self) -> Self:
        self._original = retrieval_cli.ENCODER_FACTORY
        retrieval_cli.ENCODER_FACTORY = self.factory
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        del exc_type, exc, tb
        retrieval_cli.ENCODER_FACTORY = self._original


def run_step(out: Path, *args: str) -> None:
    """Invoke one ``omopmap`` command and require exit 0."""
    result = CliRunner().invoke(app, [*args, "--out", str(out), "--created-at", CREATED_AT])
    assert result.exit_code == 0, result.output


def embed_rankings(runs: Path, n_queries: int) -> tuple[ScoredRanking, ...]:
    """Decode a channel run's rankings."""
    data = (runs / RANKINGS_NAME).read_bytes()
    return tuple(decode_scored(data, channel="embed", queries=n_queries))


def proposal_concept(
    kind: str,
    ranking: ScoredRanking,
    eligible: frozenset[int],
) -> int:
    """The concept id a proposal of ``kind`` picks from one embed ranking."""
    if kind == "invalid":
        return INVALID_CONCEPT
    if kind == "absent":
        return next(c for c in sorted(eligible) if c not in set(ranking.ids))
    rank = int(kind[len("rank") :])
    return ranking.ids[rank - 1]


def make_evidence(
    *,
    query_id: str = "demo query",
    split: str = "train",
    concept_id: int = 1,
    eligible: bool = True,
    has_text: bool = True,
    retrieved: bool = True,
    rank: int | None = 1,
    s1: float | None = 0.9,
    s2: float | None = 0.4,
    margin: float | None = 0.5,
    path: str = "committed",
    lexical_rank: int | None = 5,
    lexical_agrees: bool | None = False,
    external_agrees: bool | None = None,
    label: Label = "exact",
) -> Evidence:
    """One :class:`Evidence` row with every feature spelled out."""
    return Evidence(
        query_id=query_id,
        split=split,
        concept_id=concept_id,
        eligible=eligible,
        has_text=has_text,
        path=path,
        retrieved=retrieved,
        rank=rank,
        s1=s1,
        s2=s2,
        margin=margin,
        lexical_rank=lexical_rank,
        lexical_agrees=lexical_agrees,
        external_agrees=external_agrees,
        label=label,
    )


def build_workspace(root: Path, switch: EncoderSwitch) -> Workspace:
    """The calibration CLI workspace: cases, proposals, runs, and their picks."""
    from omopmap.retrieval.cli import eligible_labels

    root.mkdir(parents=True, exist_ok=True)
    eligible = frozenset(label.concept_id for label in eligible_labels(CONCEPTS, False))

    draft = root / "draft.tsv"
    with draft.open("w", encoding="utf-8", newline="") as handle:
        handle.write("query_id\tsource_term\n")
        for row in range(N_QUERIES):
            handle.write(f"q{row:02d}\t{source_term(row)}\n")

    idx = root / "idx"
    run_step(idx, "retrieve", "index", "--concepts", str(CONCEPTS))
    emb_draft = root / "emb-draft"
    run_step(
        emb_draft,
        "retrieve",
        "embed",
        "--index",
        str(idx),
        "--queries",
        str(draft),
        "--depth",
        "50",
    )
    rankings = embed_rankings(emb_draft, N_QUERIES)
    picks = {
        row: proposal_concept(PROPOSAL_KINDS[row], rankings[row], eligible)
        for row in range(N_QUERIES)
    }

    cases = write_cases(
        root / "cases.tsv",
        [
            {
                "query_id": f"q{row:02d}",
                "system": "DEMOSYS",
                "code": f"S{row:03d}",
                "source_term": source_term(row),
                "split": split_of(row),
                "verdict": verdict,
                "targets": targets,
            }
            for row in range(N_QUERIES)
            for verdict, targets in (verdict_of(row),)
        ],
    )
    proposals = write_proposals(
        root / "proposals.tsv",
        [
            {
                "query_id": f"q{row:02d}",
                "concept_id": str(picks[row]),
                "path": PROPOSAL_PATHS[row],
            }
            for row in range(N_QUERIES)
        ],
    )

    emb = root / "emb"
    run_step(
        emb, "retrieve", "embed", "--index", str(idx), "--queries", str(cases), "--depth", "50"
    )
    labels = root / "labels.tsv"
    with labels.open("w", encoding="utf-8", newline="") as handle:
        handle.write("concept_id\tlabel\n")
        for term in read_source_terms():
            if term["split"] == "train":
                handle.write(f"{term['concept_id']}\t{term['source_term']}\n")
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
        str(labels),
    )

    switch.switch("0" * 40)
    idx_other = root / "idx-other"
    run_step(idx_other, "retrieve", "index", "--concepts", str(CONCEPTS))
    emb_other = root / "emb-other"
    run_step(
        emb_other,
        "retrieve",
        "embed",
        "--index",
        str(idx_other),
        "--queries",
        str(cases),
        "--depth",
        "50",
    )
    switch.switch(FakeEncoder().revision)
    return Workspace(
        cases=cases,
        proposals=proposals,
        idx=idx,
        emb=emb,
        lex=lex,
        idx_other=idx_other,
        emb_other=emb_other,
        picks=picks,
    )


class Workspace:
    """Everything the calibration CLI tests need, built once per module."""

    def __init__(
        self,
        *,
        cases: Path,
        proposals: Path,
        idx: Path,
        emb: Path,
        lex: Path,
        idx_other: Path,
        emb_other: Path,
        picks: dict[int, int],
    ) -> None:
        self.cases = cases
        self.proposals = proposals
        self.idx = idx
        self.emb = emb
        self.lex = lex
        self.idx_other = idx_other
        self.emb_other = emb_other
        self.picks = picks

    def run_embed_lexical(self, cases: Path, emb: Path, lex: Path) -> None:
        """Embed and lexical runs over ``cases`` into new directories."""
        run_step(
            emb,
            "retrieve",
            "embed",
            "--index",
            str(self.idx),
            "--queries",
            str(cases),
            "--depth",
            "50",
        )
        labels = self.lex.parent / "labels.tsv"
        run_step(
            lex,
            "retrieve",
            "lexical",
            "--concepts",
            str(CONCEPTS),
            "--queries",
            str(cases),
            "--labels",
            str(labels),
        )


@pytest.fixture(scope="module")
def workspace(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Workspace]:
    """The calibration CLI workspace, built once per module.

    The fake encoder stays patched for the whole module, so the tests can run
    further embed steps (e.g. over a perturbed table) without touching the
    real model.
    """
    root = tmp_path_factory.mktemp("calibration-ws")
    switch = EncoderSwitch()
    with switch:
        yield build_workspace(root, switch)
