"""Dictionary to concept set: exclude ``unused``, separate ``CARE_SITE``, dedupe, enrich.

The concept set is the set of distinct vocabulary codes the dictionary knows,
each with the entry types it appears under. Entries of type ``unused`` carry no
learned meaning and are excluded. ``CARE_SITE/*`` codes are site-local ids of the
release's training data, not standard concepts, so they are counted and kept
out. With an Athena ``CONCEPT`` table each concept can be enriched with its OMOP
``concept_id``, ``domain_id`` and ``standard_concept``; a concept the table does
not contain stays in the set and is reported as unresolved.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from omopmap.artifacts.contract import CARE_SITE_VOCABULARY
from omopmap.artifacts.errors import ArtifactError
from omopmap.artifacts.manifest import concept_list_sha256
from omopmap.clmbr.model import ENTRY_TYPES, ClmbrDictionary, ConceptRef, EntryType


class EnrichmentError(ArtifactError):
    """The Athena CONCEPT table is unusable for enrichment."""


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class OmopConcept(_Frozen):
    """OMOP attributes of a concept, from an Athena CONCEPT row (no names)."""

    concept_id: int
    domain_id: str
    standard_concept: str | None


class ClmbrConcept(_Frozen):
    """One distinct concept of the dictionary."""

    ref: ConceptRef
    entry_types: tuple[EntryType, ...] = Field(min_length=1)
    omop: OmopConcept | None = None

    @property
    def code(self) -> str:
        return self.ref.code


class ClmbrConceptSet(_Frozen):
    """The distinct concepts of a dictionary, ``CARE_SITE`` excluded, sorted by code."""

    concepts: tuple[ClmbrConcept, ...]
    entry_type_counts: dict[EntryType, int]
    care_site_excluded: int = Field(ge=0)
    enriched: bool = False

    def codes(self) -> list[str]:
        return [c.code for c in self.concepts]

    def by_vocabulary(self) -> dict[str, int]:
        counts = Counter(c.ref.vocabulary_id for c in self.concepts)
        return dict(sorted(counts.items()))

    def concept_list_sha256(self) -> str:
        return concept_list_sha256(self.codes())

    def unresolved(self) -> list[str]:
        """Codes an enrichment did not resolve (empty when not enriched)."""
        if not self.enriched:
            return []
        return [c.code for c in self.concepts if c.omop is None]


def build_concept_set(dictionary: ClmbrDictionary) -> ClmbrConceptSet:
    """Distinct non-``unused`` codes with their entry types; ``CARE_SITE`` separated."""
    entry_counts: Counter[EntryType] = Counter(e.type for e in dictionary.entries)
    types: defaultdict[str, set[EntryType]] = defaultdict(set)
    refs: dict[str, ConceptRef] = {}
    care_site: set[str] = set()
    for entry in dictionary.entries:
        if entry.type == "unused":
            continue
        ref = entry.ref()
        if ref.vocabulary_id == CARE_SITE_VOCABULARY:
            care_site.add(ref.code)
            continue
        refs[ref.code] = ref
        types[ref.code].add(entry.type)
    concepts = tuple(
        ClmbrConcept(ref=refs[code], entry_types=tuple(t for t in ENTRY_TYPES if t in types[code]))
        for code in sorted(refs)
    )
    return ClmbrConceptSet(
        concepts=concepts,
        entry_type_counts={t: entry_counts.get(t, 0) for t in ENTRY_TYPES},
        care_site_excluded=len(care_site),
    )


def _concept_index(concept: pl.DataFrame) -> Mapping[str, OmopConcept]:
    required = ["concept_id", "vocabulary_id", "concept_code", "domain_id", "standard_concept"]
    missing = [c for c in required if c not in concept.columns]
    if missing:
        raise EnrichmentError(f"CONCEPT lacks columns {missing}")
    try:
        frame = concept.select(
            pl.col("concept_id").cast(pl.Int64, strict=True),
            (
                pl.col("vocabulary_id").cast(pl.String).str.strip_chars()
                + pl.lit("/")
                + pl.col("concept_code").cast(pl.String).str.strip_chars()
            ).alias("code"),
            pl.col("domain_id").cast(pl.String),
            pl.col("standard_concept").cast(pl.String).str.strip_chars().replace("", None),
        )
    except pl.exceptions.PolarsError as exc:
        raise EnrichmentError(f"CONCEPT has unreadable columns: {exc}") from exc
    duplicated = frame.filter(pl.col("code").is_duplicated()).get_column("code").unique()
    if duplicated.len():
        raise EnrichmentError(
            f"CONCEPT has {duplicated.len()} vocabulary_id/concept_code pairs on several rows"
        )
    index: dict[str, OmopConcept] = {}
    for concept_id, code, domain_id, standard in frame.iter_rows():
        index[code] = OmopConcept(
            concept_id=concept_id, domain_id=domain_id or "", standard_concept=standard
        )
    return index


def enrich(concept_set: ClmbrConceptSet, concept: pl.DataFrame) -> ClmbrConceptSet:
    """Attach OMOP attributes from an Athena CONCEPT table; nothing is dropped."""
    index = _concept_index(concept)
    concepts = tuple(c.model_copy(update={"omop": index.get(c.code)}) for c in concept_set.concepts)
    return concept_set.model_copy(update={"concepts": concepts, "enriched": True})


def concept_set_frame(concept_set: ClmbrConceptSet) -> pl.DataFrame:
    """The ``concept_set.parquet`` rows (sorted by code)."""
    return pl.DataFrame(
        {
            "vocabulary_id": [c.ref.vocabulary_id for c in concept_set.concepts],
            "concept_code": [c.ref.concept_code for c in concept_set.concepts],
            "code": concept_set.codes(),
            "entry_types": [list(c.entry_types) for c in concept_set.concepts],
        },
        schema={
            "vocabulary_id": pl.String,
            "concept_code": pl.String,
            "code": pl.String,
            "entry_types": pl.List(pl.String),
        },
    )
