"""The translate pipeline: the fake translator, the smoke gate, the manifest.

Nothing here touches torch or the network: the CLI translator factory is
swapped for the deterministic fake in ``conftest``.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

import pytest
from typer.testing import CliRunner, Result

from omopmap.cli import app
from omopmap.meds.convert import read_table
from omopmap.retrieval.manifest import RUN_MANIFEST_NAME, parse_run_manifest
from omopmap.translate import (
    SMOKE_N,
    TRANSLATE_MODEL,
    TRANSLATIONS_NAME,
    SmokeGateError,
    TranslationError,
    TranslatorSpec,
    translate_all,
)
from tests.dictionary.conftest import FakeTranslator
from tests.dictionary.conftest import fake_factory as fake_factory
from tests.retrieval.conftest import CREATED_AT

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
SOURCE_TERMS = FIXTURES / "source_terms.csv"


def _terms(column: str = "source_term") -> list[str]:
    with SOURCE_TERMS.open(encoding="utf-8") as handle:
        return [
            "" if row[column] is None else str(row[column])
            for row in csv.DictReader(handle, delimiter="\t")
        ]


def test_translate_all_preserves_order_and_originals() -> None:
    translator = FakeTranslator()
    texts = ["huvudvärk", "feber", "", "hosta"]
    pairs = translate_all(texts, translator)
    assert pairs == [
        ("huvudvärk", "en:huvudvärk"),
        ("feber", "en:feber"),
        ("", ""),  # empty input: passed through, never translated
        ("hosta", "en:hosta"),
    ]
    # The smoke run covered the first SMOKE_N non-empty inputs only.
    assert translator.calls[0] == ["huvudvärk", "feber", "hosta"][:SMOKE_N]
    assert translator.calls[1] == ["huvudvärk", "feber", "hosta"]


def test_smoke_gate_aborts_on_empty_translation() -> None:
    translator = FakeTranslator(empty_on={"feber"})
    with pytest.raises(SmokeGateError, match="empty translation of 'feber'"):
        translate_all(["huvudvärk", "feber", "hosta"], translator)
    # The full run never started: one call, the smoke run.
    assert len(translator.calls) == 1


def test_smoke_gate_counts_mismatch_fails() -> None:
    class Truncating(FakeTranslator):
        def translate(self, texts: Sequence[str]) -> list[str]:  # noqa: ARG002
            return []

    with pytest.raises(SmokeGateError, match="outputs for"):
        translate_all(["feber"], Truncating())


def test_no_nonempty_text_is_an_error() -> None:
    with pytest.raises(TranslationError, match="no non-empty"):
        translate_all(["", "  "], FakeTranslator())


def test_smoke_n_must_be_positive() -> None:
    with pytest.raises(TranslationError, match="smoke_n"):
        translate_all(["feber"], FakeTranslator(), smoke_n=0)


def test_pinned_revision_is_a_full_commit() -> None:
    assert len(TRANSLATE_MODEL.revision) == 40
    with pytest.raises(ValueError, match="40-hex"):
        TranslatorSpec(model_id="google/translategemma-4b-it", revision="main")


def _run(*args: str) -> Result:
    return CliRunner().invoke(app, list(args))


def test_cli_translate_writes_table_and_manifest(
    tmp_path: Path, fake_translator: FakeTranslator
) -> None:
    out = tmp_path / "translated"
    result = _run(
        "translate",
        "run",
        "--queries",
        str(SOURCE_TERMS),
        "--out",
        str(out),
        "--created-at",
        CREATED_AT,
    )
    assert result.exit_code == 0, result.output
    table = read_table(out / TRANSLATIONS_NAME)
    terms = _terms()
    assert table.height == len(terms)
    assert table.get_column("source_term").to_list() == terms
    assert table.get_column("translated_term").to_list() == [
        "" if term.strip() == "" else f"en:{term}" for term in terms
    ]
    manifest = parse_run_manifest((out / RUN_MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest.run_type == "translate"
    assert manifest.parameters["smoke_n"] == SMOKE_N
    assert manifest.parameters["queries"] == len(terms)
    assert manifest.translator is not None
    assert manifest.translator.model_id == TRANSLATE_MODEL.model_id
    assert manifest.translator.revision == TRANSLATE_MODEL.revision
    assert manifest.encoder is None


def test_cli_smoke_gate_failure_exits_1_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import omopmap.translate as translate_module

    translator = FakeTranslator(empty_on={_terms()[0]})
    monkeypatch.setattr(translate_module, "TRANSLATOR_FACTORY", fake_factory(translator))
    out = tmp_path / "translated"
    result = _run(
        "translate",
        "run",
        "--queries",
        str(SOURCE_TERMS),
        "--out",
        str(out),
    )
    assert result.exit_code == 1
    assert "smoke gate failed" in result.output
    # The abort happened before any output directory was created.
    assert not out.exists()


def test_cli_missing_query_column_exits_1(tmp_path: Path, fake_translator: FakeTranslator) -> None:
    result = _run(
        "translate",
        "run",
        "--queries",
        str(SOURCE_TERMS),
        "--query-column",
        "no_such_column",
        "--out",
        str(tmp_path / "out"),
    )
    assert result.exit_code == 1
    assert "no column" in result.output


def test_cli_reruns_are_byte_identical(tmp_path: Path, fake_translator: FakeTranslator) -> None:
    manifest_bytes: list[bytes] = []
    for run in ("r1", "r2"):
        out = tmp_path / run / "translated"
        result = _run(
            "translate",
            "run",
            "--queries",
            str(SOURCE_TERMS),
            "--out",
            str(out),
            "--created-at",
            CREATED_AT,
        )
        assert result.exit_code == 0, result.output
        manifest_bytes.append((out / RUN_MANIFEST_NAME).read_bytes())
    assert manifest_bytes[0] == manifest_bytes[1]


def test_cli_unpinned_revision_exits_1_and_writes_nothing(
    tmp_path: Path, fake_translator: FakeTranslator
) -> None:
    out = tmp_path / "translated"
    result = CliRunner().invoke(
        app,
        [
            "translate",
            "run",
            "--queries",
            str(SOURCE_TERMS),
            "--out",
            str(out),
            "--revision",
            "main",
        ],
    )
    assert result.exit_code == 1
    assert "TranslationError" in result.output and "40-hex" in result.output
    assert not out.exists()
    assert fake_translator.calls == []
