"""``TorchTrainer`` against a tiny offline BERT — no download, CPU only, seconds.

The trainer is model-agnostic: give it any ``BertModel`` and tokenizer and
it trains, monitors and snapshots. The test builds a randomly initialised
mini-BERT (one layer, hidden size 16) and a ``PreTrainedTokenizerFast``
over an invented WordPiece vocabulary, sets ``HF_HUB_OFFLINE=1`` so no
network path is even reachable, and checks the encoder contract
(unit-norm float32 rows), one optimisation pass, and the snapshot/restore
round-trip.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import pytest

from omopmap.finetune import BIOLORD, FinetuneConfig, TorchTrainer, TrainExample

pytest.importorskip("torch")
pytest.importorskip("tokenizers")
pytest.importorskip("transformers")

import torch  # noqa: E402
from tokenizers import (  # noqa: E402
    Tokenizer,
    models,
    normalizers,
    pre_tokenizers,
    processors,
)
from transformers import BertConfig, BertModel, PreTrainedTokenizerFast  # noqa: E402

SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
VOCAB_WORDS = [
    "demo",
    "alfa",
    "beta",
    "gamma",
    "delta",
    "disorder",
    "finding",
    "one",
    "two",
    "three",
    "four",
    "five",
]
NAMES = {
    101: "demo alfa",
    102: "demo beta",
    103: "demo gamma",
    104: "demo delta",
    201: "demo disorder one",
    202: "demo disorder two",
    203: "demo finding one",
    204: "demo finding two",
}


def offline_tokenizer() -> PreTrainedTokenizerFast:
    """A small WordPiece tokenizer built entirely from an invented vocabulary."""
    vocab = {token: i for i, token in enumerate([*SPECIALS, *VOCAB_WORDS])}
    wordpiece = Tokenizer(models.WordPiece(vocab, unk_token="[UNK]"))
    wordpiece.normalizer = normalizers.Sequence([normalizers.Lowercase()])
    wordpiece.pre_tokenizer = pre_tokenizers.Whitespace()
    wordpiece.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])],
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=wordpiece,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )


@pytest.fixture()
def trainer(monkeypatch: pytest.MonkeyPatch) -> TorchTrainer:
    """A tiny CPU BERT behind the trainer, seeded and offline."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    # torch's manual_seed is untyped in the installed stubs
    torch.manual_seed(0)  # pyright: ignore[reportUnknownMemberType]
    config = BertConfig(
        vocab_size=len(SPECIALS) + len(VOCAB_WORDS),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
    )
    model = BertModel(config)
    return TorchTrainer(spec=BIOLORD, device="cpu", model=model, tokenizer=offline_tokenizer())


def _batches() -> list[list[TrainExample]]:
    rows = [
        TrainExample(
            row=0, anchor="demo alfa", positive=101, negatives=(201,), gold=frozenset({101})
        ),
        TrainExample(
            row=1, anchor="demo beta", positive=102, negatives=(202,), gold=frozenset({102})
        ),
        TrainExample(
            row=2, anchor="demo gamma", positive=103, negatives=(203,), gold=frozenset({103})
        ),
        TrainExample(
            row=3, anchor="demo delta", positive=104, negatives=(204,), gold=frozenset({104})
        ),
    ]
    return [rows[:2], rows[2:]]


def test_encode_returns_unit_norm_float32_rows(trainer: TorchTrainer) -> None:
    encoded = trainer.encode(["demo alfa", "demo beta", "demo disorder one"])
    vectors = encoded.vectors
    assert vectors.dtype == np.float32
    assert vectors.shape == (3, 16)
    assert bool(np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5))
    assert trainer.info().dim == 16 and trainer.info().device == "cpu"


def test_train_epoch_runs_a_finite_loss_and_moves_weights(trainer: TorchTrainer) -> None:
    trainer.begin(total_steps=4, config=FinetuneConfig())
    # torch state dicts are untyped; compare them as str-keyed Any maps
    before = cast("dict[Any, Any]", trainer.snapshot())
    batches: Sequence[Sequence[TrainExample]] = _batches()
    loss = trainer.train_epoch(batches, NAMES, FinetuneConfig())
    assert np.isfinite(loss) and float(loss) > 0.0
    after = cast("dict[Any, Any]", trainer.snapshot())
    assert any(not torch.equal(after[key], value) for key, value in before.items())


def test_snapshot_restore_round_trips_the_encoding_exactly(trainer: TorchTrainer) -> None:
    probe = ["demo alfa", "demo finding two"]
    pre = trainer.encode(probe).vectors.copy()
    state = trainer.snapshot()
    trainer.begin(total_steps=4, config=FinetuneConfig())
    trainer.train_epoch(_batches(), NAMES, FinetuneConfig())
    trained = trainer.encode(probe).vectors
    assert not np.array_equal(pre, trained)  # training actually moved the encoding
    trainer.restore(state)
    assert np.array_equal(pre, trainer.encode(probe).vectors)
