"""Shared fixtures for the dictionary/translate tests: a fake translator."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence

import pytest

import omopmap.translate as translate_module
from omopmap.translate import TRANSLATE_MODEL, TranslatorInfo, TranslatorSpec


class FakeTranslator:
    """A deterministic stand-in: ``en:<text>`` per input, or empty on demand.

    ``empty_on`` names inputs the model "fails" on, to exercise the smoke
    gate. Every call is recorded, so tests can assert what the smoke run
    translated.
    """

    def __init__(self, empty_on: set[str] | None = None) -> None:
        self.empty_on: set[str] = empty_on if empty_on is not None else set()
        self.calls: list[list[str]] = []

    def info(self) -> TranslatorInfo:
        return TranslatorInfo(
            model_id=TRANSLATE_MODEL.model_id,
            revision=TRANSLATE_MODEL.revision,
        )

    def translate(self, texts: Sequence[str]) -> list[str]:
        self.calls.append(list(texts))
        return ["" if text in self.empty_on else f"en:{text}" for text in texts]


def fake_factory(translator: FakeTranslator) -> Callable[[TranslatorSpec], FakeTranslator]:
    def make(spec: TranslatorSpec) -> FakeTranslator:
        del spec
        return translator

    return make


@pytest.fixture()
def fake_translator(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeTranslator]:
    """Route the CLI translator factory to the fake, for one test."""
    translator = FakeTranslator()
    monkeypatch.setattr(translate_module, "TRANSLATOR_FACTORY", fake_factory(translator))
    yield translator


# Re-exported so the tests here can request the retrieval fake-encoder fixture.
from tests.retrieval.conftest import fake_encoder as fake_encoder  # noqa: E402  # pyright: ignore[reportUnusedImport]
