"""``build_labels`` window semantics, censoring, filters, patterns and schema."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Final

import polars as pl
import pytest
from pydantic import ValidationError

from omopmap.artifacts import LABELS
from omopmap.labels.windows import WindowTask, build_labels
from omopmap.meds.events import EVENT_SCHEMA

T0: Final = datetime(2024, 1, 1, 8, 0)
WINDOW_END: Final = datetime(2024, 1, 31, 8, 0)

type EventRow = tuple[int, datetime | None, str, float | None, str | None]


def _frame(rows: list[EventRow]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=EVENT_SCHEMA, orient="row")


def _task(
    *,
    window_days: int = 30,
    first_index_only: bool = False,
    exclude_prior_outcome: bool = False,
    min_age_years: int = 0,
) -> WindowTask:
    return WindowTask(
        name="er_to_ip_30d",
        index_codes=("Visit/ER",),
        outcome_codes=("Visit/IP",),
        window_days=window_days,
        first_index_only=first_index_only,
        exclude_prior_outcome=exclude_prior_outcome,
        min_age_years=min_age_years,
    )


def _labelled(events: pl.DataFrame, task: WindowTask) -> pl.DataFrame:
    return build_labels(events, task).sort(["subject_id", "prediction_time"])


def _rows(subject: int, events: list[tuple[datetime, str]]) -> list[EventRow]:
    return [(subject, time, code, None, None) for time, code in events]


def test_outcome_within_window_inclusive_at_end_is_true() -> None:
    events = _frame(
        _rows(1, [(T0, "Visit/ER"), (WINDOW_END, "Visit/IP")])
        + _rows(2, [(T0, "Visit/ER"), (datetime(2024, 1, 16), "Visit/IP")])
    )
    labelled = _labelled(events, _task())
    assert labelled.get_column("boolean_value").to_list() == [True, True]


def test_outcome_at_prediction_time_is_not_counted() -> None:
    events = _frame(
        _rows(
            1,
            [
                (T0, "Visit/ER"),
                (T0, "Visit/IP"),
                (WINDOW_END, "DEMOLAB/L1"),
            ],
        )
    )
    labelled = _labelled(events, _task())
    assert labelled.get_column("boolean_value").to_list() == [False]


def test_outcome_one_microsecond_after_window_end_is_false() -> None:
    late = WINDOW_END + timedelta(microseconds=1)
    events = _frame(_rows(1, [(T0, "Visit/ER"), (late, "Visit/IP")]))
    labelled = _labelled(events, _task())
    assert labelled.get_column("boolean_value").to_list() == [False]


def test_no_outcome_and_insufficient_follow_up_is_dropped() -> None:
    events = _frame(_rows(1, [(T0, "Visit/ER")]))
    assert _labelled(events, _task()).height == 0


def test_no_outcome_and_sufficient_follow_up_is_false() -> None:
    events = _frame(_rows(1, [(T0, "Visit/ER"), (datetime(2024, 2, 1), "DEMOLAB/L1")]))
    labelled = _labelled(events, _task())
    assert labelled.get_column("boolean_value").to_list() == [False]


def test_exclude_prior_outcome_drops_index_after_first_outcome() -> None:
    events = _frame(
        _rows(
            1,
            [
                (datetime(2024, 1, 1), "Visit/IP"),
                (datetime(2024, 2, 1), "Visit/ER"),
                (datetime(2024, 3, 5), "DEMOLAB/L1"),
            ],
        )
    )
    assert _labelled(events, _task(exclude_prior_outcome=True)).height == 0
    labelled = _labelled(events, _task(exclude_prior_outcome=False))
    assert labelled.get_column("boolean_value").to_list() == [False]


def test_first_index_only_keeps_the_earliest_index() -> None:
    events = _frame(
        _rows(
            1,
            [
                (T0, "Visit/ER"),
                (datetime(2024, 2, 1), "Visit/ER"),
                (datetime(2024, 2, 15), "Visit/IP"),
                (datetime(2024, 3, 10), "DEMOLAB/L1"),
            ],
        )
    )
    all_rows = _labelled(events, _task(first_index_only=False))
    assert all_rows.get_column("prediction_time").to_list() == [T0, datetime(2024, 2, 1)]
    assert all_rows.get_column("boolean_value").to_list() == [False, True]
    first_only = _labelled(events, _task(first_index_only=True))
    assert first_only.get_column("prediction_time").to_list() == [T0]
    assert first_only.get_column("boolean_value").to_list() == [False]


def test_min_age_years_filters_by_birth() -> None:
    events = _frame(
        [
            (1, datetime(2000, 1, 1), "MEDS_BIRTH", None, None),
            (1, datetime(2010, 1, 1), "Visit/ER", None, None),
            (1, datetime(2010, 3, 1), "DEMOLAB/L1", None, None),
            (2, datetime(1980, 1, 1), "MEDS_BIRTH", None, None),
            (2, datetime(2024, 1, 1), "Visit/ER", None, None),
            (2, datetime(2024, 1, 6), "Visit/IP", None, None),
        ]
    )
    labelled = _labelled(events, _task(min_age_years=18, exclude_prior_outcome=True))
    assert labelled.get_column("subject_id").to_list() == [2]
    assert labelled.get_column("boolean_value").to_list() == [True]


def test_prefix_pattern_matches_both_visit_codes() -> None:
    events = _frame(
        _rows(
            1,
            [
                (T0, "Visit/ER"),
                (datetime(2024, 1, 3), "DEMOOBS/O001"),
                (datetime(2024, 2, 1), "Visit/IP"),
                (datetime(2024, 3, 5), "DEMOLAB/L1"),
            ],
        )
    )
    task = WindowTask(
        name="visit_to_obs_30d",
        index_codes=("Visit/*",),
        outcome_codes=("DEMOOBS/*",),
        window_days=30,
        exclude_prior_outcome=False,
    )
    labelled = _labelled(events, task)
    assert labelled.get_column("prediction_time").to_list() == [
        T0,
        datetime(2024, 2, 1),
    ]
    assert labelled.get_column("boolean_value").to_list() == [True, False]


def test_output_schema_and_order() -> None:
    events = _frame(
        _rows(1, [(T0, "Visit/ER"), (datetime(2024, 1, 6), "Visit/IP")])
        + _rows(2, [(T0, "Visit/ER"), (datetime(2024, 2, 1), "DEMOLAB/L1")])
    )
    labelled = build_labels(events, _task())
    assert labelled.columns == list(LABELS.names)
    assert labelled.get_column("boolean_value").null_count() == 0
    for column in ("integer_value", "float_value", "categorical_value"):
        assert labelled.get_column(column).null_count() == labelled.height
    assert labelled.equals(labelled.sort(["subject_id", "prediction_time"]))
    assert not labelled.select("subject_id", "prediction_time").is_duplicated().any()


@pytest.mark.parametrize(
    ("pattern_field", "pattern"),
    [
        ("index_codes", "bad"),
        ("index_codes", "*"),
        ("index_codes", "X/"),
        ("outcome_codes", "bad"),
        ("outcome_codes", "*"),
        ("outcome_codes", "X/"),
    ],
)
def test_invalid_patterns_are_rejected(pattern_field: str, pattern: str) -> None:
    values: dict[str, object] = {
        "name": "er_to_ip_30d",
        "index_codes": ("Visit/ER",),
        "outcome_codes": ("Visit/IP",),
        "window_days": 30,
        pattern_field: (pattern,),
    }
    with pytest.raises(ValidationError):
        WindowTask(**values)  # type: ignore[arg-type]


def test_invalid_task_name_is_rejected() -> None:
    with pytest.raises(ValidationError):
        WindowTask(
            name="Bad-Name",
            index_codes=("Visit/ER",),
            outcome_codes=("Visit/IP",),
            window_days=30,
        )


def test_definition_is_a_boolean_task() -> None:
    task = _task()
    definition = task.definition()
    assert definition.name == task.name
    assert definition.label_type == "boolean"
    assert definition.categories is None
    assert definition.description
