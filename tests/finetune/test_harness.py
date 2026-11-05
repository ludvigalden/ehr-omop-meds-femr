"""Fine-tune harness: dataset construction, leakage guards, MNRL gradients, early stopping."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl
import pytest

from omopmap.finetune import (
    Dataset,
    FinetuneConfig,
    FinetuneError,
    FitResult,
    LeakageError,
    NumpyTrainer,
    PairRow,
    PairTable,
    Trainer,
    TrainExample,
    build_dataset,
    candidate_ids,
    false_negative_mask,
    fit,
    mine_negatives,
    mnrl_loss_and_grad,
    pack_batches,
    read_pairs,
)
from omopmap.retrieval.embed import Encoded, EncoderInfo


def _frame(rows: Sequence[tuple[str, str, str]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "source_term": [r[0] for r in rows],
            "concept_id": [r[1] for r in rows],
            "split": [r[2] for r in rows],
        }
    )


def test_heldout_rows_are_dropped_at_read_time_and_only_counted() -> None:
    table = read_pairs(
        _frame(
            [
                ("demo alfa", "1", "train"),
                ("demo beta", "2", "heldout"),
                ("demo gamma", "3", "dev"),
                ("demo delta", "4", "heldout"),
            ]
        )
    )
    assert table.heldout_rows == 2
    assert [(r.row, r.split) for r in table.rows] == [(0, "train"), (2, "dev")]
    assert all("beta" not in r.text and "delta" not in r.text for r in table.rows)


@pytest.mark.parametrize(
    ("rows", "match"),
    [
        ([("demo alfa", "1", "test")], "unknown split"),
        ([("demo alfa", "x", "train")], "not an integer"),
        ([(" ", "1", "train")], "empty source term"),
    ],
)
def test_read_pairs_fails_closed(rows: list[tuple[str, str, str]], match: str) -> None:
    with pytest.raises(FinetuneError, match=match):
        read_pairs(_frame(rows))


def test_missing_columns_fail_closed() -> None:
    with pytest.raises(FinetuneError, match="lacks columns"):
        read_pairs(pl.DataFrame({"source_term": ["a"], "concept_id": ["1"]}))


NAMES = {i: f"demo concept {i}" for i in range(1, 21)}


def _table(rows: Sequence[tuple[str, int, str]], heldout: int = 0) -> PairTable:
    return PairTable(
        rows=tuple(
            PairRow(row=i, text=t, concept_id=c, split="train" if s == "train" else "dev")
            for i, (t, c, s) in enumerate(rows)
        ),
        heldout_rows=heldout,
    )


def test_train_dev_term_overlap_is_a_leakage_error() -> None:
    table = _table([("Demo Alfa", 1, "train"), ("demo alfa", 2, "dev")])
    with pytest.raises(LeakageError, match="both the train and dev"):
        build_dataset(table, NAMES, {0: (3,), 1: (4,)}, FinetuneConfig())


def test_mined_negatives_never_include_any_gold_of_the_term() -> None:
    """A term with two gold rows: neither gold concept may be a negative for either row."""
    table = _table([("demo alfa", 1, "train"), ("demo alfa", 2, "train"), ("demo beta", 3, "dev")])
    rankings = {0: (2, 1, 5, 6, 7), 1: (1, 2, 5, 6, 7), 2: (3, 8)}
    dataset = build_dataset(table, NAMES, rankings, FinetuneConfig(hard_negatives=2))
    for example in dataset.examples:
        assert example.gold == frozenset({1, 2})
        assert not set(example.negatives) & example.gold
        assert example.negatives == (5, 6)


def test_mine_negatives_respects_cap_rank_order_and_eligibility() -> None:
    assert mine_negatives((9, 1, 99, 4, 5, 6), frozenset({1}), frozenset({4, 5, 6, 9}), 2) == (9, 4)


def test_effective_negatives_are_uniform_and_counts_recorded() -> None:
    table = _table(
        [
            ("demo alfa", 1, "train"),
            ("demo beta", 2, "train"),
            ("demo gamma", 99, "train"),  # gold is not an eligible target
            ("demo delta", 4, "dev"),
        ],
        heldout=3,
    )
    rankings = {0: (5, 6, 7), 1: (5,), 2: (5,), 3: (4, 9)}
    dataset = build_dataset(table, NAMES, rankings, FinetuneConfig())
    assert dataset.hard_negatives == 1
    assert all(len(e.negatives) == 1 for e in dataset.examples)
    assert dataset.counts == {
        "heldout_rows_excluded": 3,
        "train_rows": 3,
        "dev_rows": 1,
        "gold_not_eligible": 1,
        "train_examples": 2,
        "dev_queries": 1,
        "dev_pool": 2,
    }
    assert dataset.dev_pool == (4, 9)


def test_dataset_needs_train_and_dev() -> None:
    with pytest.raises(FinetuneError, match="no dev pair"):
        build_dataset(_table([("demo alfa", 1, "train")]), NAMES, {0: (2,)}, FinetuneConfig())
    with pytest.raises(FinetuneError, match="no train pair"):
        build_dataset(_table([("demo alfa", 1, "dev")]), NAMES, {0: (2,)}, FinetuneConfig())


def test_config_caps_hard_negatives_at_eight() -> None:
    with pytest.raises(ValueError):
        FinetuneConfig(hard_negatives=9)
    config = FinetuneConfig()
    assert (config.scale, config.hard_negatives, config.batch_size) == (20.0, 8, 16)
    assert (config.learning_rate, config.patience, config.monitor_k) == (2e-5, 2, 10)
    assert config.seed == 20261106


def _examples(positives: Sequence[int]) -> list[TrainExample]:
    return [
        TrainExample(row=i, anchor=f"a{i}", positive=p, negatives=(), gold=frozenset({p}))
        for i, p in enumerate(positives)
    ]


def test_batches_never_hold_two_anchors_with_the_same_positive() -> None:
    examples = _examples([1, 1, 1, 2, 3, 2, 4, 5, 6, 1])
    batches = pack_batches(examples, 4, seed=7, epoch=1)
    assert sorted(i for b in batches for i in b) == list(range(len(examples)))
    for batch in batches:
        positives = [examples[i].positive for i in batch]
        assert len(positives) == len(set(positives)) and len(batch) <= 4
    assert batches == pack_batches(examples, 4, seed=7, epoch=1)
    assert batches != pack_batches(examples, 4, seed=7, epoch=2)


def test_false_negative_mask_hides_other_gold_but_never_the_target() -> None:
    batch = [
        TrainExample(row=0, anchor="a", positive=1, negatives=(3,), gold=frozenset({1, 2})),
        TrainExample(row=1, anchor="b", positive=2, negatives=(4,), gold=frozenset({2})),
    ]
    assert candidate_ids(batch) == [1, 2, 3, 4]
    mask = false_negative_mask(batch)
    assert mask.tolist() == [[False, True, False, False], [False, False, False, False]]


def test_mnrl_gradient_matches_finite_differences() -> None:
    rng = np.random.default_rng(3)
    scores = rng.normal(size=(3, 7))
    mask = np.zeros((3, 7), dtype=np.bool_)
    mask[0, 4] = True
    loss, grad = mnrl_loss_and_grad(scores, mask)
    eps = 1e-6
    for i in range(3):
        for j in range(7):
            bumped = scores.copy()
            bumped[i, j] += eps
            numeric = (mnrl_loss_and_grad(bumped, mask)[0] - loss) / eps
            assert grad[i, j] == pytest.approx(numeric, abs=1e-4)
    assert grad[0, 4] == 0.0


def test_numpy_trainer_gradient_matches_finite_differences() -> None:
    trainer = NumpyTrainer(seed=1)
    batch = [
        TrainExample(row=0, anchor="demo alfa", positive=1, negatives=(3,), gold=frozenset({1})),
        TrainExample(row=1, anchor="demo beta", positive=2, negatives=(4,), gold=frozenset({2})),
    ]
    names = {1: "concept one", 2: "concept two", 3: "concept three", 4: "concept four"}
    loss, grad = trainer.batch_loss_and_grad(batch, names, 20.0)
    rng = np.random.default_rng(0)
    for _ in range(5):
        i, j = int(rng.integers(grad.shape[0])), int(rng.integers(grad.shape[1]))
        eps = 1e-6
        trainer.weights[i, j] += eps
        bumped, _ = trainer.batch_loss_and_grad(batch, names, 20.0)
        trainer.weights[i, j] -= eps
        assert grad[i, j] == pytest.approx((bumped - loss) / eps, abs=1e-3)


class ScriptedTrainer:
    """A trainer whose dev recall follows a script, to test the early-stopping loop."""

    def __init__(self, scores: Sequence[float]) -> None:
        self.scores = list(scores)
        self.epoch = 0
        self.restored: object = None

    def info(self) -> EncoderInfo:
        return EncoderInfo(
            model_id="scripted", revision="0" * 40, pooling="none", max_length=1, dim=1
        )

    def encode(self, texts: Sequence[str]) -> Encoded:
        return Encoded(vectors=np.ones((len(texts), 1), dtype=np.float32), truncated=0)

    def begin(self, total_steps: int, config: FinetuneConfig) -> None:
        del total_steps, config

    def train_epoch(
        self,
        batches: Sequence[Sequence[TrainExample]],
        names: dict[int, str],
        config: FinetuneConfig,
    ) -> float:
        del batches, names, config
        self.epoch += 1
        return 1.0 / self.epoch

    def snapshot(self) -> object:
        return self.epoch

    def restore(self, state: object) -> None:
        self.restored = state


def _scripted_fit(
    scores: Sequence[float], max_epochs: int = 6
) -> tuple[ScriptedTrainer, FitResult]:
    trainer = ScriptedTrainer(scores)
    dataset = Dataset(
        examples=tuple(_examples([1, 2])),
        dev=(),
        dev_pool=(),
        names={1: "x", 2: "y"},
        hard_negatives=0,
        counts={},
    )

    def monitor(t: Trainer, d: Dataset, k: int) -> float:
        del t, d, k
        return trainer.scores[trainer.epoch]

    result = fit(trainer, dataset, FinetuneConfig(max_epochs=max_epochs), monitor=monitor)
    return trainer, result


def test_early_stopping_with_patience_two_restores_the_best_epoch() -> None:
    trainer, result = _scripted_fit([0.2, 0.5, 0.4, 0.5, 0.9, 0.9, 0.9])
    assert result.best_epoch == 1
    assert result.early_stopped and result.epochs_run == 3
    assert trainer.restored == 1


def test_no_improvement_restores_the_untuned_weights() -> None:
    trainer, result = _scripted_fit([0.5, 0.5, 0.4])
    assert result.best_epoch == 0 and result.early_stopped
    assert trainer.restored == 0


def test_runs_to_max_epochs_while_improving() -> None:
    trainer, result = _scripted_fit([0.1, 0.2, 0.3, 0.4], max_epochs=3)
    assert result.best_epoch == 3 and not result.early_stopped
    assert result.epochs_run == 3 and trainer.restored == 3
