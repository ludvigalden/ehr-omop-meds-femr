"""The in-memory event table every MEDS producer builds before writing.

Producers (the synthetic generator, the OMOP converter) emit a Polars frame with
exactly :data:`EVENT_SCHEMA`; :func:`normalise_events` then sorts it into the
contract order. Keeping one frame shape means the writer, the split assignment
and the label builder all work on the same columns.
"""

from __future__ import annotations

from typing import Final

import polars as pl

#: Columns and Polars dtypes of an event frame (the MEDS data columns).
EVENT_SCHEMA: Final = pl.Schema(
    {
        "subject_id": pl.Int64(),
        "time": pl.Datetime("us"),
        "code": pl.String(),
        "numeric_value": pl.Float32(),
        "text_value": pl.String(),
    }
)

#: Columns and Polars dtypes of a code-metadata frame (``metadata/codes.parquet``).
CODES_SCHEMA: Final = pl.Schema(
    {
        "code": pl.String(),
        "description": pl.String(),
        "parent_codes": pl.List(pl.String()),
    }
)


def empty_events() -> pl.DataFrame:
    """An event frame with no rows."""
    return pl.DataFrame(schema=EVENT_SCHEMA)


def as_event_frame(frame: pl.DataFrame) -> pl.DataFrame:
    """Select and cast ``frame`` to :data:`EVENT_SCHEMA` (strict: no silent coercion)."""
    return frame.select(
        [pl.col(name).cast(dtype, strict=True) for name, dtype in EVENT_SCHEMA.items()]
    )


def normalise_events(frame: pl.DataFrame) -> pl.DataFrame:
    """Sort events into contract order and drop exact duplicate rows.

    Order is ``(subject_id, time)`` with null times first, then ``code``,
    ``numeric_value`` and ``text_value`` so that the result does not depend on
    the input order: two producers given the same events write the same bytes.
    """
    return (
        as_event_frame(frame)
        .unique(maintain_order=False)
        .sort(
            ["subject_id", "time", "code", "numeric_value", "text_value"],
            nulls_last=False,
        )
    )


def codes_frame(
    events: pl.DataFrame,
    descriptions: dict[str, str] | None = None,
    parents: dict[str, list[str]] | None = None,
) -> pl.DataFrame:
    """One ``codes.parquet`` row per distinct code in ``events``, sorted by code.

    ``descriptions`` and ``parents`` are optional lookups; codes without an entry
    get a null description and null parents.
    """
    descriptions = descriptions or {}
    parents = parents or {}
    codes: list[str] = sorted(events.get_column("code").unique().to_list())
    return pl.DataFrame(
        {
            "code": codes,
            "description": [descriptions.get(c) for c in codes],
            "parent_codes": [parents.get(c) for c in codes],
        },
        schema=CODES_SCHEMA,
    )
