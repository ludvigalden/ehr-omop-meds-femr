"""Split and shard assignment: purity, seed sensitivity, fractions, shard sizes."""

from __future__ import annotations

import polars as pl
import pytest
from pydantic import ValidationError

from omopmap.meds.splits import SplitConfig, assign_splits, shard_plan, split_of

IDS_100 = list(range(1, 101))
IDS_200 = list(range(1, 201))


def test_split_is_pure_in_other_subjects() -> None:
    """A subject's split never depends on which other subjects exist."""
    small = assign_splits(IDS_100, SplitConfig(seed=0))
    large = assign_splits(IDS_200, SplitConfig(seed=0))
    first = small.filter(pl.col("subject_id") <= 100)
    assert first.equals(large.filter(pl.col("subject_id") <= 100))


def test_different_seeds_move_some_subject() -> None:
    config_a = SplitConfig(seed=0)
    config_b = SplitConfig(seed=1)
    assert any(
        split_of(config_a, subject) != split_of(config_b, subject) for subject in range(1, 1001)
    )


def test_fractions_are_respected() -> None:
    config = SplitConfig(seed=7)
    assigned = assign_splits(list(range(1, 10_001)), config)
    counts = {split: count for split, count in assigned.group_by("split").len().iter_rows()}
    for split, fraction in config.fractions():
        observed = counts.get(split, 0) / 10_000
        assert abs(observed - fraction) <= 0.02


def test_zero_fraction_split_stays_empty() -> None:
    config = SplitConfig(seed=0, train=0.5, tuning=0.0, held_out=0.5)
    assigned = assign_splits(list(range(1, 1001)), config)
    assert "tuning" not in set(assigned.get_column("split").to_list())


def test_fractions_must_sum_to_one() -> None:
    with pytest.raises(ValidationError):
        SplitConfig(train=0.6, tuning=0.2, held_out=0.1)


def test_empty_split_yields_single_empty_shard() -> None:
    splits = pl.DataFrame(
        {
            "subject_id": list(range(1, 26)),
            "split": ["train"] * 25,
        },
        schema={"subject_id": pl.Int64, "split": pl.String},
    )
    plan = shard_plan(splits, subjects_per_shard=10)
    assert [len(shard) for shard in plan["train"]] == [10, 10, 5]
    flattened = [subject for shard in plan["train"] for subject in shard]
    assert flattened == list(range(1, 26))
    assert plan["tuning"] == [[]]
    assert plan["held_out"] == [[]]
