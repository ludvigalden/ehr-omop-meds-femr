"""Compute gate: project a job's wall time from a measured sample, stop if over budget.

Rented GPU time is paid by the hour, and a bulk embedding pass over a large
dictionary is the one step that can run for hours. So a job first measures
real throughput on a small sample (for example 1,000 labels), extrapolates each
stage linearly to its full size, adds a fixed allowance for the steps that do
not scale with the sample (model download, retrieval, I/O), and compares the
total with the budget:

    projected = fixed + sum over stages of full_items * measured_seconds / measured_items
    stop      = projected > factor * budget        (factor 2 by default)

A projection more than twice the estimate means the hardware, the input size or
the estimate itself is wrong, and continuing would spend money on a run nobody
budgeted for. The job writes ``GATE_STOP.json`` and exits 3 before any bulk
work starts. ``gate.json`` is written either way, so a passing run also records
its measured rates.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field, model_validator

GATE_NAME = "gate.json"
GATE_STOP_NAME = "GATE_STOP.json"
GATE_STOP_EXIT = 3
DEFAULT_FACTOR = 2.0


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class StageMeasurement(_Model):
    """Throughput measured on a sample, and the stage's full size."""

    name: str = Field(min_length=1)
    measured_items: int = Field(gt=0)
    measured_seconds: float = Field(ge=0.0)
    full_items: int = Field(ge=0)

    @property
    def items_per_second(self) -> float:
        if self.measured_seconds == 0.0:
            return float("inf")
        return self.measured_items / self.measured_seconds

    @property
    def projected_seconds(self) -> float:
        return self.full_items * self.measured_seconds / self.measured_items


class GateDecision(_Model):
    """The projection, the threshold it was held against, and the verdict."""

    stages: tuple[StageMeasurement, ...]
    fixed_seconds: float = Field(ge=0.0)
    budget_seconds: float = Field(gt=0.0)
    factor: float = Field(gt=0.0)
    projected_seconds: float = Field(ge=0.0)
    threshold_seconds: float = Field(gt=0.0)
    stop: bool

    @model_validator(mode="after")
    def _consistent(self) -> GateDecision:
        if self.stop != (self.projected_seconds > self.threshold_seconds):
            raise ValueError("stop must equal projected_seconds > threshold_seconds")
        return self

    def to_json(self) -> str:
        """Deterministic JSON with per-stage projections included."""
        payload = self.model_dump(mode="json")
        for stage, record in zip(self.stages, payload["stages"], strict=True):
            record["projected_seconds"] = stage.projected_seconds
        return json.dumps(payload, sort_keys=True, indent=2) + "\n"


def decide(
    stages: Sequence[StageMeasurement],
    *,
    budget_seconds: float,
    fixed_seconds: float = 0.0,
    factor: float = DEFAULT_FACTOR,
) -> GateDecision:
    """Project the full job and decide whether it may proceed (pure)."""
    names = [s.name for s in stages]
    if len(set(names)) != len(names):
        raise ValueError(f"stage names repeat: {names}")
    projected = fixed_seconds + sum(s.projected_seconds for s in stages)
    threshold = factor * budget_seconds
    return GateDecision(
        stages=tuple(stages),
        fixed_seconds=fixed_seconds,
        budget_seconds=budget_seconds,
        factor=factor,
        projected_seconds=projected,
        threshold_seconds=threshold,
        stop=projected > threshold,
    )


@dataclass(frozen=True, slots=True)
class Timed[T]:
    """A value and the wall seconds it took to produce."""

    value: T
    seconds: float


def timed[T](work: Callable[[], T], clock: Callable[[], float] = time.perf_counter) -> Timed[T]:
    """Run ``work`` and measure it with ``clock`` (injectable for tests)."""
    start = clock()
    value = work()
    return Timed(value=value, seconds=clock() - start)
