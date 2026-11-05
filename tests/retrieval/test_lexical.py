"""Lexical containment: Swedish-aware normalisation and goldens on the committed fixtures."""

from __future__ import annotations

import csv

import pytest

from omopmap.retrieval.lexical import (
    LEXICAL_CHANNEL,
    Diacritics,
    Label,
    build_index,
    normalise,
    search,
    tokens,
)
from tests.retrieval.conftest import CONCEPTS


def test_normalise_casefolds_collapses_and_strips_edges() -> None:
    assert normalise("  Smärta  I\tBRÖSTET. ") == "smärta i bröstet"
    assert normalise("«Feber»!") == "feber"
    assert normalise("ﬁber") == "fiber"  # NFKC ligature


def test_swedish_letters_are_kept_by_default() -> None:
    assert tokens("Får") != tokens("far")
    assert tokens("Får", Diacritics.STRIP) == tokens("far", Diacritics.STRIP)
    assert normalise("ÅÄÖ", Diacritics.STRIP) == "aao"


def test_containment_score_counts_unknown_query_tokens() -> None:
    index = build_index([Label(1, "ont i magen"), Label(2, "magen")])
    hit = search(index, "ont i magen idag", depth=5)
    assert hit.channel == LEXICAL_CHANNEL
    assert hit.ids == (1, 2)
    assert hit.scores == (0.75, 0.25)


def test_best_label_wins_and_ties_break_on_id() -> None:
    index = build_index(
        [Label(9, "hosta"), Label(4, "torr hosta"), Label(4, "hosta", "extra"), Label(2, "x")]
    )
    assert search(index, "hosta", depth=5).ids == (4, 9)
    assert search(index, "hosta", depth=1).ids == (4,)


def test_empty_query_and_empty_labels() -> None:
    index = build_index([Label(1, " . "), Label(2, "puls")])
    assert index.size == 1
    assert search(index, "  !  ", depth=3).ids == ()
    with pytest.raises(ValueError):
        search(index, "puls", depth=0)


def _fixture_labels() -> list[Label]:
    with CONCEPTS.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return [Label(int(r["concept_id"]), r["concept_name"]) for r in rows]


def test_goldens_on_fixture_concepts() -> None:
    index = build_index(_fixture_labels())
    assert index.size == 200
    # Hand-derived: ids 1 and 101-103 hold all three tokens (tie at 1.0, by id);
    # ids 2-5 share "demo" and "alpha" (2 of 3).
    alpha = search(index, "demo disorder alpha", depth=8)
    assert alpha.ids == (1, 101, 102, 103, 2, 3, 4, 5)
    assert alpha.scores == (1.0, 1.0, 1.0, 1.0, 2 / 3, 2 / 3, 2 / 3, 2 / 3)
    swedish = search(index, "smärta i bröstet", depth=10)
    assert swedish.ids == ()
