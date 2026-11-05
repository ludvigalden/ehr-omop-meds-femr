"""Deterministic, seeded, subject-level split and shard assignment.

A subject's split is a pure function of ``(seed, subject_id)``: the first eight
bytes of ``SHA-256("<seed>:<subject_id>")`` read as an unsigned integer and
scaled to ``[0, 1)``, bucketed by the cumulative split fractions. The result
does not depend on which other subjects are present, so adding subjects to an
extract never moves an existing subject to another split, and the assignment
can be reproduced from the seed alone.

Shards are contiguous runs of a split's subjects in ascending ``subject_id``
order, at most ``subjects_per_shard`` subjects each.
"""

from __future__ import annotations

import hashlib
import math
from typing import Final, Self

import polars as pl
from pydantic import BaseModel, ConfigDict, Field, model_validator

from omopmap.artifacts.contract import SPLITS, Split

_SCALE: Final = float(1 << 64)


class SplitConfig(BaseModel):
    """Split fractions, seed and shard size."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    train: float = Field(default=0.8, ge=0.0, le=1.0)
    tuning: float = Field(default=0.1, ge=0.0, le=1.0)
    held_out: float = Field(default=0.1, ge=0.0, le=1.0)
    seed: int = Field(default=0, ge=0)
    subjects_per_shard: int = Field(default=10_000, ge=1)

    @model_validator(mode="after")
    def _fractions_sum_to_one(self) -> Self:
        total = self.train + self.tuning + self.held_out
        if not math.isclose(total, 1.0, abs_tol=1e-9):
            raise ValueError(f"split fractions must sum to 1, got {total}")
        return self

    def fractions(self) -> tuple[tuple[Split, float], ...]:
        return (("train", self.train), ("tuning", self.tuning), ("held_out", self.held_out))


def subject_unit(seed: int, subject_id: int) -> float:
    """The subject's position in ``[0, 1)``: a pure function of seed and id."""
    digest = hashlib.sha256(f"{seed}:{subject_id}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big") / _SCALE


def split_of(config: SplitConfig, subject_id: int) -> Split:
    """The split ``subject_id`` is assigned to under ``config``."""
    unit = subject_unit(config.seed, subject_id)
    cumulative = 0.0
    last: Split = "train"
    for split, fraction in config.fractions():
        if fraction <= 0.0:
            continue
        cumulative += fraction
        last = split
        if unit < cumulative:
            return split
    # Floating-point slack at the top of the interval goes to the last non-empty split.
    return last


def assign_splits(subject_ids: list[int], config: SplitConfig) -> pl.DataFrame:
    """``subject_splits`` rows for ``subject_ids``: unique, sorted by ``subject_id``."""
    ids = sorted(set(subject_ids))
    return pl.DataFrame(
        {"subject_id": ids, "split": [split_of(config, s) for s in ids]},
        schema={"subject_id": pl.Int64, "split": pl.String},
    )


def shard_plan(splits: pl.DataFrame, subjects_per_shard: int) -> dict[Split, list[list[int]]]:
    """Subject ids per shard for every split, in shard order.

    A split with no subjects gets a single empty shard, because the contract
    requires ``0.parquet`` in every split directory.
    """
    plan: dict[Split, list[list[int]]] = {}
    for split in SPLITS:
        ids: list[int] = (
            splits.filter(pl.col("split") == split).get_column("subject_id").sort().to_list()
        )
        if not ids:
            plan[split] = [[]]
            continue
        count = math.ceil(len(ids) / subjects_per_shard)
        plan[split] = [
            ids[i * subjects_per_shard : (i + 1) * subjects_per_shard] for i in range(count)
        ]
    return plan
