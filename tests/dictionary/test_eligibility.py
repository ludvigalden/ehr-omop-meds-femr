"""Eligibility filter: exact expected rows on the committed synthetic fixture.

The expected sets are recomputed independently in this file with a plain
Python predicate over the fixture constants -- never through the library
under test -- and then compared for exact set equality. A change in the
eligibility contract moves both sides.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from typer.testing import CliRunner, Result

from omopmap.cli import app
from omopmap.dictionary import (
    TARGETS_NAME,
    DictionaryConfig,
    DictionaryError,
    build_targets,
    clinical_scope,
    config_from_options,
    counts_by_vocabulary,
    filter_eligible,
)
from omopmap.meds.convert import read_table
from omopmap.retrieval.manifest import RUN_MANIFEST_NAME, parse_run_manifest
from tests.fixtures.build_fixtures import CONCEPT_HEADER, concepts
from tests.retrieval.conftest import CREATED_AT

AS_OF = date(2026, 1, 1)
FIXTURE_HEADER = CONCEPT_HEADER.split("\t")


def fixture_frame() -> pl.DataFrame:
    """The committed fixture's 200 concept rows as a frame of strings."""
    return pl.DataFrame(concepts(), schema=FIXTURE_HEADER, orient="row")


def independently_eligible(
    rows: list[list[str]], scope: dict[str, frozenset[str] | None], as_of: str
) -> list[tuple[int, str, str, str, str]]:
    """A plain-Python reimplementation of the eligibility contract."""
    out: list[tuple[int, str, str, str, str]] = []
    for row in rows:
        standard, invalid_reason, valid_end = row[5], row[9], row[8]
        domains = scope.get(row[3], "out-of-scope")
        if standard != "S" or invalid_reason != "":
            continue
        if domains == "out-of-scope":
            continue
        if domains is not None and row[2] not in domains:
            continue
        if valid_end < as_of:
            continue
        out.append((int(row[0]), row[1], row[3], row[2], row[6]))
    return sorted(out)


def test_default_scope_yields_exactly_the_four_standard_valid_real_vocab_rows() -> None:
    """Default scope on the fixture: the four real-vocabulary rows only.

    The demo vocabularies are out of scope; every demo row therefore
    drops, and the pinned expectation is the full, hand-counted result.
    """
    targets = build_targets(fixture_frame(), clinical_scope(), as_of=AS_OF)
    expected = [
        (181, "demo snomed finding alpha", "SNOMED", "Condition", "900000001"),
        (182, "demo snomed procedure beta", "SNOMED", "Procedure", "900000002"),
        (183, "demo loinc measurement gamma", "LOINC", "Measurement", "90000-1"),
        (184, "demo rxnorm drug delta", "RxNorm", "Drug", "9000001"),
    ]
    got = [
        (int(cid), name, vocab, domain, code)
        for cid, name, vocab, domain, code in targets.iter_rows()
    ]
    assert got == expected
    assert targets.columns == [
        "concept_id",
        "concept_name",
        "vocabulary_id",
        "domain_id",
        "concept_code",
    ]
    assert targets.schema["concept_id"] == pl.Int64


def test_democond_scope_matches_the_independent_computation() -> None:
    scope: dict[str, frozenset[str] | None] = {"DEMOCOND": None}
    config = DictionaryConfig(scope=scope)
    rows = concepts()
    targets = build_targets(fixture_frame(), config, as_of=AS_OF)
    got = sorted(
        (int(cid), name, vocab, domain, code)
        for cid, name, vocab, domain, code in targets.iter_rows()
    )
    assert got == independently_eligible(rows, scope, AS_OF.isoformat())
    assert got, "the DEMOCOND scope must admit the fixture's main-pass rows"


def test_domain_restricted_scope_matches_the_independent_computation() -> None:
    scope: dict[str, frozenset[str] | None] = {
        "DEMOVOCAB": frozenset({"Metadata", "Unit"}),
        "DEMOCOND": frozenset({"Condition"}),
    }
    config = DictionaryConfig(scope=scope)
    rows = concepts()
    targets = build_targets(fixture_frame(), config, as_of=AS_OF)
    got = sorted(
        (int(cid), name, vocab, domain, code)
        for cid, name, vocab, domain, code in targets.iter_rows()
    )
    assert got == independently_eligible(rows, scope, AS_OF.isoformat())
    # Every surviving DEMOVOCAB row is in a scoped domain.
    assert {row[3] for row in got if row[2] == "DEMOVOCAB"} <= {"Metadata", "Unit"}


def mini_frame() -> pl.DataFrame:
    """A nine-row synthetic frame covering each contract clause separately.

    Invented vocabulary (MINIVOC, OTHERVOC) and invented codes; nothing
    real-vocabulary-shaped.
    """
    rows = [
        # concept_id, name, domain, vocab, standard, code, valid_end, invalid_reason
        ("1", "kept standard", "Demo", "MINIVOC", "S", "M-1", "2099-12-31", ""),
        ("2", "invalid standard", "Demo", "MINIVOC", "S", "M-2", "2023-12-31", "D"),
        ("3", "classification level", "Demo", "MINIVOC", "C", "M-3", "2099-12-31", ""),
        ("4", "non-standard", "Demo", "MINIVOC", "", "M-4", "2099-12-31", ""),
        ("5", "ends exactly today", "Demo", "MINIVOC", "S", "M-5", "2026-01-01", ""),
        ("6", "ended, not flagged", "Demo", "MINIVOC", "S", "M-6", "2025-06-30", ""),
        ("7", "null invalid reason", "Demo", "MINIVOC", "S", "M-7", "2099-12-31", None),
        ("8", "out-of-scope vocab", "Demo", "OTHERVOC", "S", "O-1", "2099-12-31", ""),
        ("9", "out-of-scope domain", "Other", "MINIVOC", "S", "M-9", "2099-12-31", ""),
    ]
    return pl.DataFrame(
        {
            "concept_id": [r[0] for r in rows],
            "concept_name": [r[1] for r in rows],
            "domain_id": [r[2] for r in rows],
            "vocabulary_id": [r[3] for r in rows],
            "standard_concept": [r[4] for r in rows],
            "concept_code": [r[5] for r in rows],
            "valid_end_date": [r[6] for r in rows],
            "invalid_reason": [r[7] for r in rows],
        }
    )


SCOPE: dict[str, frozenset[str] | None] = {"MINIVOC": frozenset({"Demo"})}


def test_each_contract_clause_independently() -> None:
    targets = build_targets(mini_frame(), DictionaryConfig(scope=SCOPE), as_of=AS_OF)
    got = [int(cid) for (cid,) in targets.select("concept_id").iter_rows()]
    # 1 kept standard; 5 end-date boundary is inclusive; 7 null invalid
    # reason is still current; 2/3/4/6/8/9 all drop.
    assert got == [1, 5, 7]


def test_null_end_date_is_never_current() -> None:
    frame = mini_frame().vstack(
        pl.DataFrame(
            {
                "concept_id": ["10"],
                "concept_name": ["null end date"],
                "domain_id": ["Demo"],
                "vocabulary_id": ["MINIVOC"],
                "concept_code": ["M-10"],
                "standard_concept": ["S"],
                "valid_end_date": [None],
                "invalid_reason": [""],
            },
            schema=mini_frame().schema,
        )
    )
    targets = build_targets(frame, DictionaryConfig(scope=SCOPE), as_of=AS_OF)
    assert [int(cid) for (cid,) in targets.select("concept_id").iter_rows()] == [1, 5, 7]


def _dated_frame(end_dates: list[str]) -> pl.DataFrame:
    """Standard, valid MINIVOC rows that differ only in their end date."""
    n = len(end_dates)
    return pl.DataFrame(
        {
            "concept_id": [str(100 + i) for i in range(n)],
            "concept_name": [f"dated {i}" for i in range(n)],
            "domain_id": ["Demo"] * n,
            "vocabulary_id": ["MINIVOC"] * n,
            "concept_code": [f"M-{100 + i}" for i in range(n)],
            "standard_concept": ["S"] * n,
            "valid_end_date": end_dates,
            "invalid_reason": [""] * n,
        },
        schema=mini_frame().schema,
    )


def test_athena_compact_end_dates_compare_chronologically() -> None:
    """Athena writes ``YYYYMMDD``; the comparison must be by date, not by raw string.

    As a raw string comparison against an ISO reference date,
    ``"20260301" >= "2026-06-15"`` holds (``0`` sorts after ``-``), so a
    concept that expired in March would stay eligible in June.
    """
    mid_year = date(2026, 6, 15)
    ends = ["20260301", "20260614", "20260615", "20260616", "2026-03-01", "2026-06-15"]
    targets = build_targets(_dated_frame(ends), DictionaryConfig(scope=SCOPE), as_of=mid_year)
    kept = [int(cid) for (cid,) in targets.select("concept_id").iter_rows()]
    # 100/101 expired before the reference date; 102 is the inclusive
    # boundary; 103 is current; 104/105 are 100/102 in ISO form.
    assert kept == [102, 103, 105]


def test_empty_end_date_is_never_current() -> None:
    targets = build_targets(_dated_frame(["", "-"]), DictionaryConfig(scope=SCOPE), as_of=AS_OF)
    assert targets.height == 0


def test_build_cli_streams_athena_formatted_table(tmp_path: Path) -> None:
    """The lazy CLI path applies the same contract to Athena-formatted dates."""
    header = (
        "concept_id\tconcept_name\tdomain_id\tvocabulary_id\tconcept_class_id\t"
        "standard_concept\tconcept_code\tvalid_start_date\tvalid_end_date\tinvalid_reason"
    )
    rows = [
        "1\tdemo current\tCondition\tSNOMED\tdemo\tS\tDEMO-1\t20200101\t20991231\t",
        "2\tdemo expired\tCondition\tSNOMED\tdemo\tS\tDEMO-2\t20200101\t20260301\t",
        "3\tdemo boundary\tCondition\tSNOMED\tdemo\tS\tDEMO-3\t20200101\t20260615\t",
        "4\tdemo retired\tCondition\tSNOMED\tdemo\tS\tDEMO-4\t20200101\t20991231\tD",
        "5\tdemo out of domain\tGender\tSNOMED\tdemo\tS\tDEMO-5\t20200101\t20991231\t",
    ]
    concepts_path = tmp_path / "CONCEPT.csv"
    concepts_path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    out = tmp_path / "targets"
    result = _run(
        "dictionary",
        "build",
        "--concepts",
        str(concepts_path),
        "--out",
        str(out),
        "--as-of",
        "2026-06-15",
        "--created-at",
        CREATED_AT,
    )
    assert result.exit_code == 0, result.output
    table = read_table(out / TARGETS_NAME)
    assert table.get_column("concept_id").to_list() == [1, 3]
    assert table.columns == [
        "concept_id",
        "concept_name",
        "vocabulary_id",
        "domain_id",
        "concept_code",
    ]


def test_build_cli_non_integer_concept_id_exits_1(tmp_path: Path) -> None:
    header = (
        "concept_id\tconcept_name\tdomain_id\tvocabulary_id\tconcept_class_id\t"
        "standard_concept\tconcept_code\tvalid_start_date\tvalid_end_date\tinvalid_reason"
    )
    row = "x1\tdemo bad id\tCondition\tSNOMED\tdemo\tS\tDEMO-1\t20200101\t20991231\t"
    concepts_path = tmp_path / "CONCEPT.csv"
    concepts_path.write_text(f"{header}\n{row}\n", encoding="utf-8")
    result = _run(
        "dictionary",
        "build",
        "--concepts",
        str(concepts_path),
        "--out",
        str(tmp_path / "o"),
        "--as-of",
        AS_OF.isoformat(),
    )
    assert result.exit_code == 1
    assert "DictionaryError" in result.output


def test_duplicate_concept_id_fails_closed() -> None:
    frame = pl.concat([mini_frame(), mini_frame().head(1)])
    with pytest.raises(DictionaryError, match="duplicate concept_id"):
        build_targets(frame, DictionaryConfig(scope=SCOPE), as_of=AS_OF)


def test_missing_column_fails_closed() -> None:
    frame = mini_frame().drop("invalid_reason")
    with pytest.raises(DictionaryError, match="lacks columns"):
        filter_eligible(frame, DictionaryConfig(scope=SCOPE), as_of=AS_OF)


def test_empty_scope_admits_nothing() -> None:
    with pytest.raises(ValueError, match="at least one vocabulary"):
        DictionaryConfig(scope={})


def test_scope_option_parsing() -> None:
    config = config_from_options(["SNOMED", "DEMOVOCAB=Metadata,Unit"])
    assert config.scope == {
        "SNOMED": None,
        "DEMOVOCAB": frozenset({"Metadata", "Unit"}),
    }
    with pytest.raises(DictionaryError, match="names no domain"):
        config_from_options(["DEMOVOCAB="])
    with pytest.raises(DictionaryError, match="empty vocabulary"):
        config_from_options(["=Unit"])


def test_counts_by_vocabulary() -> None:
    targets = build_targets(fixture_frame(), clinical_scope(), as_of=AS_OF)
    assert counts_by_vocabulary(targets) == {
        "LOINC": 1,
        "RxNorm": 1,
        "SNOMED": 2,
    }


def _run(*args: str) -> Result:
    return CliRunner().invoke(app, list(args))


def test_build_cli_writes_table_and_manifest(tmp_path: Path) -> None:
    out = tmp_path / "targets"
    result = _run(
        "dictionary",
        "build",
        "--concepts",
        str(Path(__file__).resolve().parent.parent / "fixtures" / "concepts.csv"),
        "--out",
        str(out),
        "--as-of",
        AS_OF.isoformat(),
        "--created-at",
        CREATED_AT,
    )
    assert result.exit_code == 0, result.output
    table = read_table(out / TARGETS_NAME)
    assert table.height == 4
    manifest = parse_run_manifest((out / RUN_MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest.run_type == "dictionary"
    assert manifest.parameters["eligible"] == 4
    assert manifest.parameters["as_of"] == AS_OF.isoformat()
    assert manifest.parameters["vocabulary:SNOMED"] == 2
    assert manifest.parameters["vocabulary:LOINC"] == 1
    assert manifest.parameters["vocabulary:RxNorm"] == 1
    assert manifest.encoder is None and manifest.translator is None


def test_build_cli_reruns_are_byte_identical(tmp_path: Path) -> None:
    concepts_path = Path(__file__).resolve().parent.parent / "fixtures" / "concepts.csv"
    manifest_bytes: list[bytes] = []
    table_bytes: list[bytes] = []
    for run in ("r1", "r2"):
        out = tmp_path / run / "targets"
        result = _run(
            "dictionary",
            "build",
            "--concepts",
            str(concepts_path),
            "--out",
            str(out),
            "--as-of",
            AS_OF.isoformat(),
            "--created-at",
            CREATED_AT,
        )
        assert result.exit_code == 0, result.output
        manifest_bytes.append((out / RUN_MANIFEST_NAME).read_bytes())
        table_bytes.append((out / TARGETS_NAME).read_bytes())
    assert manifest_bytes[0] == manifest_bytes[1]
    assert table_bytes[0] == table_bytes[1]


def test_build_cli_missing_column_exits_1(tmp_path: Path) -> None:
    broken = tmp_path / "broken.tsv"
    broken.write_text("concept_id\tconcept_name\n1\tname\n", encoding="utf-8")
    result = _run(
        "dictionary",
        "build",
        "--concepts",
        str(broken),
        "--out",
        str(tmp_path / "out"),
    )
    assert result.exit_code == 1
    assert "lacks columns" in result.output


def test_build_cli_existing_output_refused(tmp_path: Path) -> None:
    concepts_path = Path(__file__).resolve().parent.parent / "fixtures" / "concepts.csv"
    out = tmp_path / "targets"
    out.mkdir()
    (out / "something.txt").write_text("occupied", encoding="utf-8")
    result = _run(
        "dictionary",
        "build",
        "--concepts",
        str(concepts_path),
        "--out",
        str(out),
    )
    assert result.exit_code == 1
    assert "not empty" in result.output


def test_retrieve_index_consumes_the_target_table(tmp_path: Path, fake_encoder: Any) -> None:
    concepts_path = Path(__file__).resolve().parent.parent / "fixtures" / "concepts.csv"
    out = tmp_path / "targets"
    built = _run(
        "dictionary",
        "build",
        "--concepts",
        str(concepts_path),
        "--out",
        str(out),
        "--as-of",
        AS_OF.isoformat(),
    )
    assert built.exit_code == 0, built.output
    index_dir = tmp_path / "idx"
    result = _run(
        "retrieve",
        "index",
        "--concepts",
        str(out / TARGETS_NAME),
        "--out",
        str(index_dir),
        "--created-at",
        CREATED_AT,
    )
    assert result.exit_code == 0, result.output
    ids = (index_dir / "ids.txt").read_text(encoding="utf-8").splitlines()
    # Exactly the four eligible concepts, in concept_id order.
    assert ids == ["181", "182", "183", "184"]
    # The lexical channel consumes the same table without eligibility columns.
    lexical = _run(
        "retrieve",
        "lexical",
        "--concepts",
        str(out / TARGETS_NAME),
        "--queries",
        str(Path(__file__).resolve().parent.parent / "fixtures" / "source_terms.csv"),
        "--out",
        str(tmp_path / "lex"),
    )
    assert lexical.exit_code == 0, lexical.output
