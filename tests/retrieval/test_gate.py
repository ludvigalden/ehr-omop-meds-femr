"""Compute gate: projection arithmetic and the 2x-budget stop rule."""

from __future__ import annotations

import json
from collections.abc import Iterator

import pytest

from omopmap.retrieval.gate import StageMeasurement, decide, timed


def _stage(name: str, measured: int, seconds: float, full: int) -> StageMeasurement:
    return StageMeasurement(
        name=name, measured_items=measured, measured_seconds=seconds, full_items=full
    )


def test_projection_is_linear_plus_fixed() -> None:
    decision = decide(
        [_stage("labels", 1000, 2.0, 600_000), _stage("queries", 1000, 0.5, 70_000)],
        budget_seconds=3600.0,
        fixed_seconds=900.0,
    )
    assert decision.projected_seconds == pytest.approx(900 + 1200 + 35)
    assert decision.threshold_seconds == 7200.0
    assert not decision.stop


def test_stops_strictly_above_twice_budget() -> None:
    at = decide([_stage("labels", 10, 1.0, 200)], budget_seconds=10.0)
    assert at.projected_seconds == 20.0 and not at.stop
    over = decide([_stage("labels", 10, 1.0, 201)], budget_seconds=10.0)
    assert over.stop


def test_factor_and_duplicate_stages() -> None:
    assert decide([_stage("a", 1, 1.0, 3)], budget_seconds=1.0, factor=3.0).stop is False
    with pytest.raises(ValueError, match="repeat"):
        decide([_stage("a", 1, 1.0, 1), _stage("a", 1, 1.0, 1)], budget_seconds=1.0)


def test_gate_json_is_deterministic_and_lists_stage_projections() -> None:
    decision = decide([_stage("labels", 4, 2.0, 8)], budget_seconds=1.0)
    text = decision.to_json()
    assert text == decision.to_json() and text.endswith("\n")
    payload = json.loads(text)
    assert payload["stages"][0]["projected_seconds"] == 4.0
    assert payload["stop"] is True


def test_timed_uses_the_injected_clock() -> None:
    ticks: Iterator[float] = iter([10.0, 12.5])
    result = timed(lambda: "done", clock=lambda: next(ticks))
    assert (result.value, result.seconds) == ("done", 2.5)
