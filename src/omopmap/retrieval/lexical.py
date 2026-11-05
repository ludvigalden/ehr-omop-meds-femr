"""Lexical baseline: normalised-token containment, with no corpus statistics.

A query matches a label by the share of the query's distinct tokens that occur
in the label:

    containment(query, label) = |tokens(query) & tokens(label)| / |tokens(query)|

The denominator counts every distinct query token, including tokens that occur
in no label, so a query that is half unknown words can score at most 0.5. A
concept can carry several labels (its OMOP name and any extra surfaces, such
as trusted Swedish synonyms), and its score is the maximum over them. Ranking
is score descending, then concept id ascending, so the order is a pure function
of the inputs.

No IDF, BM25 or document frequency is used, by design: the score does not move
when the candidate set changes, which makes it a fixed reference point for the
embedding and fusion channels.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from omopmap.retrieval.fusion import ScoredRanking

LEXICAL_CHANNEL = "lexical"

# Characters stripped from both ends of a normalised string before it is split.
_EDGE_PUNCTUATION = " \t.,;:!?()[]{}\"'«»"


class Diacritics(StrEnum):
    """What normalisation does with diacritics.

    ``KEEP`` preserves them, so Swedish å, ä and ö stay distinct letters ("får"
    and "far" are different words). ``STRIP`` removes combining marks after
    canonical decomposition, which helps only when one side of the comparison
    has lost its diacritics, for example ASCII-folded source data.
    """

    KEEP = "keep"
    STRIP = "strip"


def normalise(text: str, diacritics: Diacritics = Diacritics.KEEP) -> str:
    """NFKC, casefold, optional diacritic removal, whitespace collapsed, edge punctuation off."""
    value = unicodedata.normalize("NFKC", text).casefold()
    if diacritics is Diacritics.STRIP:
        decomposed = unicodedata.normalize("NFD", value)
        value = unicodedata.normalize(
            "NFC", "".join(c for c in decomposed if not unicodedata.combining(c))
        )
    return " ".join(value.split()).strip(_EDGE_PUNCTUATION)


def tokens(text: str, diacritics: Diacritics = Diacritics.KEEP) -> frozenset[str]:
    """Distinct whitespace tokens of the normalised text."""
    return frozenset(normalise(text, diacritics).split())


@dataclass(frozen=True, slots=True)
class Label:
    """One searchable surface string attached to one concept."""

    concept_id: int
    text: str
    surface: str = "name"


@dataclass(frozen=True, slots=True)
class LexicalIndex:
    """Inverted index from token to the labels that contain it."""

    diacritics: Diacritics
    label_concepts: tuple[int, ...]
    label_tokens: tuple[frozenset[str], ...]
    postings: dict[str, tuple[int, ...]] = field(repr=False)

    @property
    def size(self) -> int:
        """Number of indexed (non-empty) labels."""
        return len(self.label_concepts)


def build_index(labels: Iterable[Label], diacritics: Diacritics = Diacritics.KEEP) -> LexicalIndex:
    """Index ``labels``; labels that normalise to no tokens are skipped."""
    concepts: list[int] = []
    label_tokens: list[frozenset[str]] = []
    postings: dict[str, list[int]] = {}
    for label in labels:
        toks = tokens(label.text, diacritics)
        if not toks:
            continue
        position = len(concepts)
        concepts.append(label.concept_id)
        label_tokens.append(toks)
        for token in toks:
            postings.setdefault(token, []).append(position)
    return LexicalIndex(
        diacritics=diacritics,
        label_concepts=tuple(concepts),
        label_tokens=tuple(label_tokens),
        postings={t: tuple(p) for t, p in postings.items()},
    )


def search(index: LexicalIndex, query: str, depth: int) -> ScoredRanking:
    """Top ``depth`` concepts for ``query`` by containment; empty if the query has no tokens."""
    if depth < 1:
        raise ValueError(f"depth must be >= 1, got {depth}")
    query_tokens = tokens(query, index.diacritics)
    if not query_tokens:
        return ScoredRanking(channel=LEXICAL_CHANNEL, ids=(), scores=())
    shared: dict[int, int] = {}
    for token in query_tokens:
        for position in index.postings.get(token, ()):
            shared[position] = shared.get(position, 0) + 1
    best: dict[int, int] = {}
    for position, count in shared.items():
        concept_id = index.label_concepts[position]
        if count > best.get(concept_id, 0):
            best[concept_id] = count
    denominator = len(query_tokens)
    ranked = sorted(best.items(), key=lambda item: (-item[1], item[0]))[:depth]
    return ScoredRanking(
        channel=LEXICAL_CHANNEL,
        ids=tuple(concept_id for concept_id, _ in ranked),
        scores=tuple(count / denominator for _, count in ranked),
    )


def search_all(index: LexicalIndex, queries: Sequence[str], depth: int) -> list[ScoredRanking]:
    """``search`` for each query, in order."""
    return [search(index, query, depth) for query in queries]
