"""End-to-end ``omopmap finetune --dry-run`` on the committed fixtures.

The dry run exercises the full loop — held-out exclusion, leakage guards,
negative mining, batching, dev-recall monitoring, early stopping and the
manifest — with the hermetic numpy encoder, so it runs in well under a
second. Reruns with a fixed ``--created-at`` are byte-identical; the
held-out guard is proven by recording every string the trainer encodes;
the ``--negatives`` path uses a dedicated invented English pairs table so
lexical mining finds candidates for every train anchor.
"""

from __future__ import annotations

import csv
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner, Result

from omopmap.cli import app
from omopmap.finetune import NumpyTrainer
from omopmap.retrieval.embed import Encoded
from tests.retrieval.conftest import CONCEPTS, CREATED_AT, SOURCE_TERMS

PAIR_COLUMNS = ["source_term", "concept_id", "split"]


def _run(*args: str) -> Result:
    return CliRunner().invoke(app, list(args))


def _finetune(out: Path, pairs: Path = SOURCE_TERMS, *extra: str) -> Result:
    return _run(
        "finetune",
        "--dry-run",
        "--pairs",
        str(pairs),
        "--concepts",
        str(CONCEPTS),
        "--out",
        str(out),
        "--created-at",
        CREATED_AT,
        *extra,
    )


@pytest.fixture(scope="module")
def first_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One completed dry run, timed; later tests read its records."""
    out = tmp_path_factory.mktemp("finetune") / "ft-first"
    started = time.monotonic()
    result = _finetune(out)
    elapsed = time.monotonic() - started
    assert result.exit_code == 0, result.output
    assert elapsed < 30.0, f"dry run took {elapsed:.1f}s"
    return out


def _report(out: Path) -> dict[str, Any]:
    return json.loads((out / "finetune.json").read_text(encoding="utf-8"))


def test_report_records_the_contract_and_counts(first_run: Path) -> None:
    report = _report(first_run)
    assert report["mode"] == "dry-run" and report["trainer"] == "numpy-trigram-linear"
    config = report["config"]
    assert config["scale"] == 20.0 and config["hard_negatives"] == 8
    assert config["batch_size"] == 16 and config["learning_rate"] == 2e-05
    assert config["seed"] == 20261106 and config["patience"] == 2
    assert config["monitor_k"] == 10
    counts = report["counts"]
    assert counts["heldout_rows_excluded"] == 8
    assert counts["train_rows"] == 41 and counts["dev_rows"] == 11
    assert report["fit"]["history"][0]["epoch"] == 0
    best = report["fit"]["best_epoch"]
    recalls = {record["epoch"]: record["dev_recall"] for record in report["fit"]["history"]}
    assert recalls[best] == max(recalls.values())


def test_manifest_and_absent_checkpoint(first_run: Path) -> None:
    from omopmap.retrieval.manifest import RUN_MANIFEST_NAME, parse_run_manifest

    manifest = parse_run_manifest((first_run / RUN_MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest.run_type == "finetune"
    assert manifest.seed == 20261106
    assert manifest.encoder is None
    assert not (first_run / "checkpoint").exists()


def test_reruns_are_byte_identical(first_run: Path, tmp_path: Path) -> None:
    second = tmp_path / "ft-second"
    assert _finetune(second).exit_code == 0
    for name in ("finetune.json", "run_manifest.json"):
        assert (first_run / name).read_bytes() == (second / name).read_bytes()


def test_heldout_terms_are_never_encoded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace every held-out term with a sentinel and record everything encoded."""
    pairs = tmp_path / "sentinel-terms.tsv"
    with SOURCE_TERMS.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    sentinels = 0
    for row in rows:
        if row["split"] == "heldout":
            sentinels += 1
            row["source_term"] = f"zz heldout sentinel {sentinels}"
    assert sentinels == 8
    with pairs.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=PAIR_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    seen: list[str] = []
    original = NumpyTrainer.encode

    def recording(trainer: NumpyTrainer, texts: Sequence[str]) -> Encoded:
        seen.extend(texts)
        return original(trainer, texts)

    monkeypatch.setattr(NumpyTrainer, "encode", recording)
    out = tmp_path / "ft-guarded"
    result = _finetune(out, pairs)
    assert result.exit_code == 0, result.output
    assert seen, "the recording wrapper must see the run's encoded texts"
    assert not any("zz heldout sentinel" in text for text in seen)


def _eligible_concepts() -> list[tuple[int, str]]:
    """Standard, valid fixture concepts as (id, name), sorted by id."""
    with CONCEPTS.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return sorted(
        (int(r["concept_id"]), r["concept_name"])
        for r in rows
        if r["standard_concept"] == "S" and r["invalid_reason"] == ""
    )


def _english_pairs(path: Path) -> Path:
    """An invented pairs table over eligible concepts whose terms share tokens.

    Lexical mining needs every train anchor to rank some non-gold concept;
    these invented English terms overlap the fixture concept names, which
    the committed Swedish fixture terms do not.
    """
    eligible = _eligible_concepts()
    train_terms = [
        "demo disorder",
        "demo medication",
        "demo procedure",
        "demo lab test",
        "demo finding",
        "acute demo",
        "demo phase",
        "acute demo phase",
    ]
    dev_terms = [
        "demo disorder phase",
        "demo medication phase",
        "demo finding phase",
        "demo lab phase",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=PAIR_COLUMNS)
        writer.writeheader()
        for i, term in enumerate([*train_terms, *dev_terms]):
            gold = eligible[i % len(eligible)][0]
            split = "train" if i < len(train_terms) else "dev"
            writer.writerow({"source_term": term, "concept_id": str(gold), "split": split})
    return path


@pytest.fixture()
def english_pairs(tmp_path: Path) -> Path:
    return _english_pairs(tmp_path / "english-pairs.tsv")


def test_negatives_from_a_lexical_run(tmp_path: Path, english_pairs: Path) -> None:
    run = tmp_path / "neg-run"
    result = _run(
        "retrieve",
        "lexical",
        "--concepts",
        str(CONCEPTS),
        "--queries",
        str(english_pairs),
        "--out",
        str(run),
        "--created-at",
        CREATED_AT,
    )
    assert result.exit_code == 0, result.output
    out = tmp_path / "ft-negatives"
    result = _finetune(out, english_pairs, "--negatives", str(run))
    assert result.exit_code == 0, result.output
    assert _report(out)["negatives_source"] == "run:lexical"


def test_negatives_from_a_different_query_table_fail(tmp_path: Path, english_pairs: Path) -> None:
    other = tmp_path / "other-pairs.tsv"
    with english_pairs.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    rows.append(
        {"source_term": "demo extra", "concept_id": rows[0]["concept_id"], "split": "train"}
    )
    with other.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=PAIR_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    run = tmp_path / "other-run"
    result = _run(
        "retrieve",
        "lexical",
        "--concepts",
        str(CONCEPTS),
        "--queries",
        str(other),
        "--out",
        str(run),
        "--created-at",
        CREATED_AT,
    )
    assert result.exit_code == 0, result.output
    result = _finetune(tmp_path / "ft-mismatch", english_pairs, "--negatives", str(run))
    assert result.exit_code == 1
    assert "query table" in result.output


def test_train_dev_term_overlap_is_a_leakage_error(tmp_path: Path) -> None:
    leak = tmp_path / "leak.tsv"
    with SOURCE_TERMS.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    first_dev = next(i for i, row in enumerate(rows) if row["split"] == "dev")
    train_term = next(row["source_term"] for row in rows if row["split"] == "train")
    rows[first_dev]["source_term"] = train_term.upper()
    with leak.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=PAIR_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    result = _finetune(tmp_path / "ft-leak", leak)
    assert result.exit_code == 1
    assert "LeakageError" in result.output


def test_hard_negatives_nine_is_rejected(tmp_path: Path) -> None:
    result = _finetune(tmp_path / "ft-nine", SOURCE_TERMS, "--hard-negatives", "9")
    assert result.exit_code == 1


def test_existing_output_directory_is_refused(tmp_path: Path) -> None:
    out = tmp_path / "taken"
    out.mkdir()
    (out / "keep.txt").write_text("x", encoding="utf-8")
    result = _finetune(out)
    assert result.exit_code == 1
    assert (out / "keep.txt").read_text(encoding="utf-8") == "x"
