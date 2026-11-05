"""``retrieve index|embed --model-dir``: a fine-tuned checkpoint, verified, loaded from disk.

The chain is the real one, kept hermetic and CPU-only:

1. ``omopmap finetune`` in full mode, with ``TRAINER_FACTORY`` returning a
   ``TorchTrainer`` around a tiny randomly initialised BERT and an invented
   WordPiece vocabulary (no download). It writes ``checkpoint/`` and a run
   manifest recording every checkpoint file's SHA-256.
2. ``retrieve index --model-dir`` and ``retrieve embed --model-dir`` load that
   checkpoint through the production ``TransformerEncoder`` local route
   (``local_files_only``), with ``HF_HUB_OFFLINE=1`` so no network path is
   even reachable.

Then the fail-closed cases: a tampered or extra checkpoint file (exit 4,
nothing written), a dry-run directory, a contradicting ``--model``, and an
index built by another encoder (exit 1).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from typer.testing import CliRunner, Result

import omopmap.finetune as finetune_module
from omopmap.cli import app
from omopmap.finetune import FinetuneConfig, TorchTrainer, Trainer
from omopmap.retrieval.checkpoint import CHECKPOINT_DIR, checkpoint_digest, open_checkpoint
from omopmap.retrieval.embed import BIOLORD, Encoder, ModelSpec
from tests.retrieval.conftest import CONCEPTS, CREATED_AT, SOURCE_TERMS

pytest.importorskip("torch")
pytest.importorskip("transformers")

import torch  # noqa: E402
from transformers import BertConfig, BertModel  # noqa: E402

from tests.finetune.test_torch_trainer import SPECIALS, VOCAB_WORDS, offline_tokenizer  # noqa: E402


def _invoke(*args: str) -> Result:
    return CliRunner().invoke(app, list(args))


@pytest.fixture(scope="module")
def offline() -> Iterator[None]:
    patch = pytest.MonkeyPatch()
    patch.setenv("HF_HUB_OFFLINE", "1")
    patch.setenv("TRANSFORMERS_OFFLINE", "1")
    yield
    patch.undo()


@pytest.fixture(scope="module")
def tuned(tmp_path_factory: pytest.TempPathFactory, offline: None) -> Path:
    """A full-mode ``finetune`` output directory around a tiny offline BERT."""
    del offline

    def factory(spec: ModelSpec, config: FinetuneConfig) -> Trainer:
        del config
        # torch's manual_seed is untyped in the installed stubs
        torch.manual_seed(0)  # pyright: ignore[reportUnknownMemberType]
        model = BertModel(
            BertConfig(
                vocab_size=len(SPECIALS) + len(VOCAB_WORDS),
                hidden_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                intermediate_size=32,
            )
        )
        return TorchTrainer(spec=spec, device="cpu", model=model, tokenizer=offline_tokenizer())

    original = finetune_module.TRAINER_FACTORY
    finetune_module.TRAINER_FACTORY = factory
    try:
        out = tmp_path_factory.mktemp("tuned") / "ft"
        result = _invoke(
            "finetune",
            "--pairs",
            str(SOURCE_TERMS),
            "--concepts",
            str(CONCEPTS),
            "--out",
            str(out),
            "--epochs",
            "1",
            "--created-at",
            CREATED_AT,
        )
    finally:
        finetune_module.TRAINER_FACTORY = original
    assert result.exit_code == 0, result.output
    return out


def _index(out: Path, *extra: str) -> Result:
    return _invoke(
        "retrieve",
        "index",
        "--concepts",
        str(CONCEPTS),
        "--out",
        str(out),
        "--created-at",
        CREATED_AT,
        *extra,
    )


def _embed(index: Path, out: Path, *extra: str) -> Result:
    return _invoke(
        "retrieve",
        "embed",
        "--index",
        str(index),
        "--queries",
        str(SOURCE_TERMS),
        "--out",
        str(out),
        "--created-at",
        CREATED_AT,
        *extra,
    )


def _manifest(run: Path) -> dict[str, Any]:
    return json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))


def test_checkpoint_identity_is_the_digest_of_its_recorded_files(tuned: Path) -> None:
    checkpoint = open_checkpoint(tuned)
    recorded = {
        o["name"]: o["sha256"]
        for o in _manifest(tuned)["outputs"]
        if o["name"].startswith(CHECKPOINT_DIR + "/")
    }
    assert recorded and dict(checkpoint.files) == recorded
    assert checkpoint.checkpoint_sha256 == checkpoint_digest(recorded)
    assert checkpoint.spec == BIOLORD
    assert checkpoint.model_label == f"biolord+ft:{checkpoint.checkpoint_sha256[:12]}"


def test_index_and_embed_load_the_checkpoint_offline(tuned: Path, tmp_path: Path) -> None:
    digest = open_checkpoint(tuned).checkpoint_sha256
    index = tmp_path / "idx"
    result = _index(index, "--model-dir", str(tuned))
    assert result.exit_code == 0, result.output
    built = _manifest(index)
    assert built["encoder"]["model_id"] == BIOLORD.model_id
    assert built["encoder"]["revision"] == BIOLORD.revision
    assert built["encoder"]["checkpoint_sha256"] == digest
    assert built["encoder"]["dim"] == 16 and built["encoder"]["pooling"] == "mean"
    assert built["parameters"]["model"] == f"biolord+ft:{digest[:12]}"
    assert {i["name"]: i["sha256"] for i in built["inputs"]}["checkpoint"] == digest
    matrix: npt.NDArray[np.float32] = np.load(index / "matrix.npy")
    assert matrix.dtype == np.float32 and matrix.shape[1] == 16
    assert bool(np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-5))

    ranked = tmp_path / "emb"
    result = _embed(index, ranked, "--model-dir", str(tuned))
    assert result.exit_code == 0, result.output
    assert _manifest(ranked)["encoder"]["checkpoint_sha256"] == digest
    # Same checkpoint, same inputs, fixed timestamp: byte-identical rerun.
    again = tmp_path / "idx2"
    assert _index(again, "--model-dir", str(tuned)).exit_code == 0
    assert (again / "matrix.npy").read_bytes() == (index / "matrix.npy").read_bytes()
    assert (again / "run_manifest.json").read_bytes() == (index / "run_manifest.json").read_bytes()


def test_model_flag_must_name_the_base_model(tuned: Path, tmp_path: Path) -> None:
    out = tmp_path / "idx"
    assert _index(out, "--model-dir", str(tuned), "--model", "biolord").exit_code == 0
    wrong = tmp_path / "wrong"
    result = _index(wrong, "--model-dir", str(tuned), "--model", "sapbert")
    assert result.exit_code == 1 and "contradicts" in result.output
    assert not wrong.exists()


def _copy(tuned: Path, tmp_path: Path) -> Path:
    import shutil

    copy = tmp_path / "ft-copy"
    shutil.copytree(tuned, copy)
    return copy


def test_tampered_checkpoint_file_exits_4(tuned: Path, tmp_path: Path) -> None:
    copy = _copy(tuned, tmp_path)
    weights = sorted((copy / CHECKPOINT_DIR).glob("*.safetensors")) or sorted(
        (copy / CHECKPOINT_DIR).glob("*.bin")
    )
    assert weights
    data = bytearray(weights[0].read_bytes())
    data[-1] ^= 0xFF
    weights[0].write_bytes(bytes(data))
    out = tmp_path / "idx"
    result = _index(out, "--model-dir", str(copy))
    assert result.exit_code == 4, result.output
    assert "CheckpointHashError" in result.output and "differs from the recorded" in result.output
    assert not out.exists()


def test_unrecorded_or_missing_checkpoint_file_exits_4(tuned: Path, tmp_path: Path) -> None:
    copy = _copy(tuned, tmp_path)
    (copy / CHECKPOINT_DIR / "extra.bin").write_bytes(b"x")
    out = tmp_path / "idx"
    result = _index(out, "--model-dir", str(copy))
    assert result.exit_code == 4 and "unrecorded" in result.output
    (copy / CHECKPOINT_DIR / "extra.bin").unlink()
    (copy / CHECKPOINT_DIR / "config.json").unlink()
    result = _index(out, "--model-dir", str(copy))
    assert result.exit_code == 4 and "missing" in result.output
    assert not out.exists()


def test_embed_refuses_a_tampered_checkpoint_with_exit_4(tuned: Path, tmp_path: Path) -> None:
    index = tmp_path / "idx"
    assert _index(index, "--model-dir", str(tuned)).exit_code == 0
    copy = _copy(tuned, tmp_path)
    (copy / CHECKPOINT_DIR / "config.json").write_text("{}", encoding="utf-8")
    out = tmp_path / "emb"
    result = _embed(index, out, "--model-dir", str(copy))
    assert result.exit_code == 4 and not out.exists()


def test_dry_run_and_non_finetune_dirs_exit_1(tmp_path: Path) -> None:
    dry = tmp_path / "dry"
    result = _invoke(
        "finetune",
        "--dry-run",
        "--pairs",
        str(SOURCE_TERMS),
        "--concepts",
        str(CONCEPTS),
        "--out",
        str(dry),
        "--epochs",
        "1",
        "--created-at",
        CREATED_AT,
    )
    assert result.exit_code == 0, result.output
    out = tmp_path / "idx"
    result = _index(out, "--model-dir", str(dry))
    assert result.exit_code == 1 and "dry run" in result.output
    result = _index(out, "--model-dir", str(tmp_path / "nowhere"))
    assert result.exit_code == 1 and "run_manifest.json" in result.output
    assert not out.exists()


def test_index_from_another_encoder_is_refused(
    tuned: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An index built from the checkpoint cannot be queried with the base model, and back."""
    import omopmap.retrieval.cli as retrieval_cli
    from tests.retrieval.conftest import FakeEncoder

    index = tmp_path / "idx"
    assert _index(index, "--model-dir", str(tuned)).exit_code == 0

    def factory(model: str) -> Encoder:
        del model
        return FakeEncoder()

    monkeypatch.setattr(retrieval_cli, "ENCODER_FACTORY", factory)
    base = tmp_path / "emb-base"
    result = _embed(index, base, "--model", "sapbert")
    assert result.exit_code == 1 and "another encoder" in result.output
    base_index = tmp_path / "idx-base"
    assert _index(base_index, "--model", "sapbert").exit_code == 0
    tuned_query = tmp_path / "emb-tuned"
    result = _embed(base_index, tuned_query, "--model-dir", str(tuned))
    assert result.exit_code == 1 and "another encoder" in result.output
