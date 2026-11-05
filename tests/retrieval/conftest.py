"""Shared fixtures for the retrieval tests: the B2 fake encoder behind the Encoder protocol."""

from __future__ import annotations

import importlib.util
from collections.abc import Iterator, Sequence
from pathlib import Path
from types import ModuleType

import pytest

import omopmap.retrieval.cli as retrieval_cli
from omopmap.retrieval.embed import SAPBERT, Encoded, Encoder, EncoderInfo

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
CONCEPTS = FIXTURES / "concepts.csv"
SOURCE_TERMS = FIXTURES / "source_terms.csv"
CREATED_AT = "2026-01-01T00:00:00Z"


def _load_fake() -> ModuleType:
    spec = importlib.util.spec_from_file_location("fake_encoder", FIXTURES / "fake_encoder.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_FAKE = _load_fake()


class FakeEncoder:
    """The deterministic hashing encoder, reporting the SapBERT pin as its identity."""

    def __init__(self, revision: str = SAPBERT.revision) -> None:
        self.revision = revision

    def info(self) -> EncoderInfo:
        return EncoderInfo(
            model_id=SAPBERT.model_id,
            revision=self.revision,
            pooling=SAPBERT.pooling,
            max_length=128,
            dim=20,
        )

    def encode(self, texts: Sequence[str]) -> Encoded:
        return Encoded(vectors=_FAKE.encode(list(texts)), truncated=0)


@pytest.fixture(scope="module")
def fake_encoder() -> Iterator[FakeEncoder]:
    """Route every CLI ``--model`` to the fake encoder, for the whole module."""
    encoder = FakeEncoder()

    def factory(model: str) -> Encoder:
        del model
        return encoder

    original = retrieval_cli.ENCODER_FACTORY
    retrieval_cli.ENCODER_FACTORY = factory
    yield encoder
    retrieval_cli.ENCODER_FACTORY = original
