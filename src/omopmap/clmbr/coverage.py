"""Dictionary coverage of a MEDS extract: which distinct codes the concept set knows.

Coverage counts distinct vocabulary codes (reserved ``MEDS_*`` codes excluded)
and how many of them are in the concept set, overall and per vocabulary id. It
also lists the most frequent unrecognised codes, code strings only, and never a
code seen in fewer than ``min_events`` events, so rare codes of a small cohort
are not exposed. It measures; it does not adjust or map anything.
"""

from __future__ import annotations

from collections.abc import Collection

import polars as pl

from omopmap.artifacts.contract import CONTRACT_VERSION, RESERVED_CODES
from omopmap.artifacts.models import (
    CoverageReport,
    UnrecognisedCode,
    VocabularyCoverage,
    producer,
)

#: Codes seen in fewer events are never listed (the contract default).
DEFAULT_MIN_EVENTS = 10
#: Default maximum length of the unrecognised-code list.
DEFAULT_TOP_N = 20


def _ratio(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def coverage_report(
    events: pl.DataFrame,
    concept_codes: Collection[str],
    *,
    extract_manifest_sha256: str,
    concept_list_sha256: str,
    created_at: str,
    top_n: int = DEFAULT_TOP_N,
    min_events: int = DEFAULT_MIN_EVENTS,
) -> CoverageReport:
    """Build the coverage report of ``events`` (a MEDS event frame) against a concept set."""
    known = pl.Series("code", sorted(set(concept_codes)), dtype=pl.String)
    per_code = (
        events.filter(
            ~pl.col("code").is_in(pl.Series(sorted(RESERVED_CODES), dtype=pl.String).implode())
        )
        .group_by("code")
        .len("events")
        .with_columns(
            pl.col("code").str.split_exact("/", 1).struct.field("field_0").alias("vocabulary_id"),
            pl.col("code").is_in(known.implode()).alias("recognised"),
        )
    )
    by_vocabulary = tuple(
        VocabularyCoverage(
            vocabulary_id=vocabulary_id,
            distinct_codes=distinct,
            recognised_codes=recognised,
            share=_ratio(recognised, distinct),
        )
        for vocabulary_id, distinct, recognised in per_code.group_by("vocabulary_id")
        .agg(pl.len().alias("distinct"), pl.col("recognised").sum().alias("recognised"))
        .sort("vocabulary_id")
        .select("vocabulary_id", "distinct", "recognised")
        .iter_rows()
    )
    distinct_codes = per_code.height
    recognised_codes = int(per_code.get_column("recognised").sum())
    unrecognised = tuple(
        UnrecognisedCode(code=code, events=count)
        for code, count in per_code.filter(~pl.col("recognised") & (pl.col("events") >= min_events))
        .sort(["events", "code"], descending=[True, False])
        .head(top_n)
        .select("code", "events")
        .iter_rows()
    )
    return CoverageReport(
        contract_version=CONTRACT_VERSION,
        producer=producer(),
        created_at=created_at,
        extract_manifest_sha256=extract_manifest_sha256,
        concept_list_sha256=concept_list_sha256,
        distinct_codes=distinct_codes,
        recognised_codes=recognised_codes,
        share=_ratio(recognised_codes, distinct_codes),
        by_vocabulary=by_vocabulary,
        unrecognised_top_n=top_n,
        unrecognised_min_events=min_events,
        unrecognised=unrecognised,
    )
