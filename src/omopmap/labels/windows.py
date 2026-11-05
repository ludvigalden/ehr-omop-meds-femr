"""Prediction times and binary labels: "outcome within N days after an index event".

The task is generic: the caller names the index codes, the outcome codes and
the window; nothing is specific to a care setting or site. A code pattern is
an exact code string, or a prefix ending in ``*`` (``Visit/*`` matches every
``Visit/`` code).

Semantics, per subject:

- Every timed event whose code matches an index pattern yields a prediction
  time equal to that event's time (``first_index_only`` keeps only the
  earliest). The model may use events with ``time <= prediction_time``.
- The label is true when an outcome event falls in the window
  ``(prediction_time, prediction_time + window_days]``: strictly after the
  prediction time, inclusive at the end.
- Censoring: with no outcome in the window and the subject's last event
  before the window end, the outcome is unknown, so the prediction time is
  dropped (``MEDS_DEATH`` ends follow-up the same way as the last event).
- ``exclude_prior_outcome`` drops prediction times at or after a subject's
  first outcome, so a label never asks whether something already happened.
- ``min_age_years`` drops prediction times before the subject reached that
  age (age from ``MEDS_BIRTH``).

Multiple index events at the same time collapse into one prediction time.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Final, Self

import polars as pl
from pydantic import BaseModel, ConfigDict, Field, model_validator

from omopmap.artifacts.contract import BIRTH_CODE, is_valid_data_code
from omopmap.artifacts.models import TaskDefinition, TaskName

_PREFIX: Final = "*"
_DAYS_PER_YEAR: Final = 365.25


def _valid_pattern(pattern: str) -> bool:
    if pattern.endswith(_PREFIX):
        stem = pattern[: -len(_PREFIX)]
        return bool(stem) and _PREFIX not in stem
    return is_valid_data_code(pattern)


class WindowTask(BaseModel):
    """A binary "outcome within ``window_days`` after an index event" task."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: TaskName
    index_codes: tuple[str, ...] = Field(min_length=1)
    outcome_codes: tuple[str, ...] = Field(min_length=1)
    window_days: int = Field(ge=1, le=36_500)
    first_index_only: bool = False
    exclude_prior_outcome: bool = True
    min_age_years: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _patterns(self) -> Self:
        for pattern in (*self.index_codes, *self.outcome_codes):
            if not _valid_pattern(pattern):
                raise ValueError(f"{pattern!r} is neither a code string nor a `prefix*` pattern")
        return self

    def definition(self) -> TaskDefinition:
        """The ``task.json`` record for this task."""
        prior = "; prediction times at or after a first outcome are excluded" * int(
            self.exclude_prior_outcome
        )
        first = " (first index event per subject only)" * int(self.first_index_only)
        age = f"; subjects younger than {self.min_age_years} years excluded" * int(
            self.min_age_years > 0
        )
        description = (
            f"True when an event matching {list(self.outcome_codes)} occurs within "
            f"{self.window_days} days after an index event matching "
            f"{list(self.index_codes)}{first}. Prediction time is the index event time; "
            "the window excludes the prediction time and includes its end; prediction "
            f"times without full follow-up and no outcome are dropped{prior}{age}."
        )
        return TaskDefinition(
            name=self.name, label_type="boolean", description=description, categories=None
        )


def matches(codes: tuple[str, ...]) -> pl.Expr:
    """A boolean expression over ``code`` for a set of patterns."""
    exact = [c for c in codes if not c.endswith(_PREFIX)]
    prefixes = [c[: -len(_PREFIX)] for c in codes if c.endswith(_PREFIX)]
    expr = pl.col("code").is_in(exact)
    for prefix in prefixes:
        expr = expr | pl.col("code").str.starts_with(prefix)
    return expr


def build_labels(events: pl.DataFrame, task: WindowTask) -> pl.DataFrame:
    """Labels in the MEDS label schema, sorted by ``(subject_id, prediction_time)``.

    ``events`` is an event frame (all subjects of an extract).
    """
    timed = events.filter(pl.col("time").is_not_null())
    window = timedelta(days=task.window_days)

    index = (
        timed.filter(matches(task.index_codes))
        .select("subject_id", prediction_time=pl.col("time"))
        .unique()
    )
    if task.first_index_only:
        index = index.group_by("subject_id").agg(pl.col("prediction_time").min())

    last = timed.group_by("subject_id").agg(pl.col("time").max().alias("last_time"))
    births = timed.filter(pl.col("code") == BIRTH_CODE).select("subject_id", birth=pl.col("time"))
    outcomes = timed.filter(matches(task.outcome_codes)).select(
        "subject_id", outcome_time=pl.col("time")
    )
    first_outcome = outcomes.group_by("subject_id").agg(
        pl.col("outcome_time").min().alias("first_outcome")
    )

    candidates = (
        index.join(last, on="subject_id", how="left")
        .join(births, on="subject_id", how="left")
        .join(first_outcome, on="subject_id", how="left")
        .with_columns(window_end=pl.col("prediction_time") + window)
    )
    if task.exclude_prior_outcome:
        candidates = candidates.filter(
            pl.col("first_outcome").is_null()
            | (pl.col("first_outcome") > pl.col("prediction_time"))
        )
    if task.min_age_years > 0:
        min_age = timedelta(days=task.min_age_years * _DAYS_PER_YEAR)
        candidates = candidates.filter(
            pl.col("birth").is_not_null() & (pl.col("prediction_time") - pl.col("birth") >= min_age)
        )

    hits = (
        candidates.select("subject_id", "prediction_time", "window_end")
        .join(outcomes, on="subject_id", how="inner")
        .filter(
            (pl.col("outcome_time") > pl.col("prediction_time"))
            & (pl.col("outcome_time") <= pl.col("window_end"))
        )
        .select("subject_id", "prediction_time")
        .unique()
        .with_columns(pl.lit(True).alias("hit"))
    )
    labelled = (
        candidates.join(hits, on=["subject_id", "prediction_time"], how="left")
        .with_columns(pl.col("hit").fill_null(False))
        .filter(pl.col("hit") | (pl.col("last_time") >= pl.col("window_end")))
    )
    return (
        labelled.select(
            pl.col("subject_id").cast(pl.Int64),
            pl.col("prediction_time").cast(pl.Datetime("us")),
            pl.col("hit").alias("boolean_value"),
            pl.lit(None, dtype=pl.Int64).alias("integer_value"),
            pl.lit(None, dtype=pl.Float64).alias("float_value"),
            pl.lit(None, dtype=pl.String).alias("categorical_value"),
        )
        .unique(["subject_id", "prediction_time"])
        .sort(["subject_id", "prediction_time"])
    )
