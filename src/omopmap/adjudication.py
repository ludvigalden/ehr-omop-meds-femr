"""Blinded adjudication sheets: built from retrieval runs, ingested as a frame of record.

A reviewer who sees which system proposed a candidate, at what rank and with
what score, anchors on that: the top-ranked candidate gets accepted more
readily, and a favoured model's proposals get the benefit of the doubt. The
evaluation then measures the reviewer's trust in the system, not the mapping.
So the sheet a reviewer works from carries none of it.

**Build** (``omopmap adjudicate build``):

- Every run given (lexical, embedding, fused, under user-chosen names)
  contributes its top ``depth`` candidates per case. Candidates are pooled
  per case and deduplicated, so a concept proposed by two systems is shown
  once and agreement between systems is not visible either.
- The pooled candidates are put in concept-id order, then shuffled with a
  ``random.Random`` seeded by the string ``"<seed>|<system>|<code>"``, where
  ``system`` and ``code`` identify the case's source code. The order is a pure
  function of the seed and the case: rebuilding gives identical bytes, and it
  does not depend on run order, ranks or scores.
- The sheet holds only the case id, the source term, each candidate's
  concept id, name, vocabulary and domain, and blank verdict columns. No
  rank, score, run name, model id or source query id.
- A separate key file, written to a different directory, maps each sheet row
  back to every (run, rank, score) that proposed it, and to the query it came
  from. It also records the SHA-256 of the blank sheet, of its read-only
  columns, and the row count. Reviewers get the sheet directory only.

**Ingest** (``omopmap adjudicate ingest``) re-joins a completed sheet with
the key. It fails closed when the row count or the read-only columns differ
from what was built (a re-sorted, filtered or edited sheet), and when a case
is incomplete or contradictory. The output is a frame of record in the format
``omopmap metrics compute`` reads (``query_id``, ``split``, ``verdict``,
``targets``), with the verdicts of ``omopmap.metrics``.

Verdict columns per row: ``acceptable`` (``y``/``n``/blank), ``other_targets``
(acceptable concept ids the reviewer found by searching outside the list,
``;``-separated), ``case_verdict`` (``accept``, ``no-target`` or ``defer``;
needed when no candidate is acceptable) and free-text ``notes``.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import polars as pl
import typer
from pydantic import BaseModel, ConfigDict, Field

from omopmap.metrics import VERDICTS, EvaluatedRun, Verdict, check_verdict

SHEET_NAME = "sheet.csv"
KEY_NAME = "adjudication_key.json"
FRAME_NAME = "adjudicated.tsv"
REPORT_NAME = "ingest_report.json"
KEY_SCHEMA = "omopmap-adjudication-key/1"
DEFAULT_SEED = 20261104
DEFAULT_DEPTH = 10
GATE_EXIT = 6
HASH_EXIT = 4

READ_ONLY_COLUMNS = (
    "case_id",
    "row",
    "source_term",
    "concept_id",
    "concept_name",
    "vocabulary_id",
    "domain_id",
)
VERDICT_COLUMNS = ("acceptable", "other_targets", "case_verdict", "notes")
SHEET_COLUMNS = READ_ONLY_COLUMNS + VERDICT_COLUMNS
FRAME_COLUMNS = ("query_id", "system", "code", "source_term", "split", "verdict", "targets")


class AdjudicationError(ValueError):
    """Input that cannot be read or interpreted (exit 1)."""


class SheetGateError(AdjudicationError):
    """A completed sheet is incomplete or contradicts itself (exit 6)."""


class TamperedSheetError(AdjudicationError):
    """A sheet or key no longer matches what was built (exit 4)."""


# --------------------------------------------------------------------------
# Building.
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Case:
    """One query to adjudicate; ``row`` is its position in the table the runs ranked."""

    row: int
    query_id: str
    system: str
    code: str
    source_term: str
    split: str


@dataclass(frozen=True, slots=True)
class ConceptInfo:
    name: str
    vocabulary_id: str
    domain_id: str


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class KeyProposal(_Frozen):
    run: str
    rank: int = Field(ge=1)
    score: float


class KeyRow(_Frozen):
    row: int = Field(ge=0, description="Row number within the case; 0 is a no-candidate row.")
    concept_id: int | None
    proposals: tuple[KeyProposal, ...]


class KeyCase(_Frozen):
    case_id: str
    query_row: int = Field(ge=0)
    query_id: str
    system: str
    code: str
    split: str
    rows: tuple[KeyRow, ...]


class KeyRun(_Frozen):
    name: str
    run_type: str
    channel: str
    rankings_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class AdjudicationKey(_Frozen):
    """The sealed key: everything the sheet hides, plus the sheet's fingerprints."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True, populate_by_name=True)

    schema_: str = Field(alias="schema", default=KEY_SCHEMA)
    seed: int
    depth: int = Field(ge=1)
    queries_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sheet_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    content_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$", description="Digest of the sheet's read-only columns."
    )
    n_rows: int = Field(ge=0)
    n_cases: int = Field(ge=0)
    runs: tuple[KeyRun, ...]
    cases: tuple[KeyCase, ...]

    def to_json(self) -> str:
        payload = self.model_dump(mode="json", by_alias=True)
        return json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def shuffle_key(seed: int, system: str, code: str) -> str:
    """The per-case shuffle seed string, ``"<seed>|<system>|<code>"``."""
    return f"{seed}|{system}|{code}"


def blind_order(concept_ids: Sequence[int], *, seed: int, system: str, code: str) -> list[int]:
    """Concept ids in the sheet's order: sorted, then shuffled by the case's seed string.

    Sorting first makes the result independent of run order, ranks and scores.
    ``random.Random`` seeded with a string hashes it with SHA-512, which is
    stable across platforms and Python versions.
    """
    ordered = sorted(set(concept_ids))
    random.Random(shuffle_key(seed, system, code)).shuffle(ordered)
    return ordered


def pool_candidates(
    case: Case, runs: Sequence[EvaluatedRun], depth: int
) -> dict[int, list[KeyProposal]]:
    """Each run's top ``depth`` candidates for the case, pooled by concept."""
    pooled: dict[int, list[KeyProposal]] = {}
    for run in runs:
        ranking = run.rankings[case.row]
        for rank, (concept_id, score) in enumerate(
            zip(ranking.ids[:depth], ranking.scores[:depth], strict=True), 1
        ):
            pooled.setdefault(concept_id, []).append(
                KeyProposal(run=run.name, rank=rank, score=score)
            )
    return pooled


def _csv_bytes(rows: Sequence[Sequence[str]]) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def content_digest(rows: Sequence[Sequence[str]]) -> str:
    """SHA-256 over the read-only cells of the sheet rows (header excluded).

    Computed from parsed cell values, so a spreadsheet that re-quotes the file
    on save still matches; any change to a read-only value does not.
    """
    digest = hashlib.sha256()
    for row in rows:
        cells = row[: len(READ_ONLY_COLUMNS)]
        digest.update(("\x1f".join(cells) + "\x1e").encode("utf-8"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class BuiltSheet:
    sheet: bytes
    key: AdjudicationKey


def build_sheet(
    cases: Sequence[Case],
    runs: Sequence[EvaluatedRun],
    concepts: dict[int, ConceptInfo],
    *,
    seed: int,
    depth: int,
    queries_sha256: str,
) -> BuiltSheet:
    """The blinded sheet bytes and the sealed key (pure)."""
    if depth < 1:
        raise AdjudicationError("--depth must be at least 1")
    if not runs:
        raise AdjudicationError("give at least one run")
    if not cases:
        raise AdjudicationError("there are no cases to adjudicate")
    keys = [(case.system, case.code) for case in cases]
    repeated = sorted({k for k in keys if keys.count(k) > 1})
    if repeated:
        raise AdjudicationError(
            f"system/code pairs repeat, so their shuffles would too: {repeated[:3]}"
        )
    width = max(4, len(str(len(cases))))
    rows: list[list[str]] = []
    key_cases: list[KeyCase] = []
    for number, case in enumerate(cases, 1):
        case_id = f"C{number:0{width}d}"
        pooled = pool_candidates(case, runs, depth)
        key_rows: list[KeyRow] = []
        order = blind_order(list(pooled), seed=seed, system=case.system, code=case.code)
        for position, concept_id in enumerate(order, 1):
            info = concepts.get(concept_id)
            if info is None:
                raise AdjudicationError(f"concept {concept_id} is not in --concepts")
            rows.append(
                [
                    case_id,
                    str(position),
                    case.source_term,
                    str(concept_id),
                    info.name,
                    info.vocabulary_id,
                    info.domain_id,
                    "",
                    "",
                    "",
                    "",
                ]
            )
            key_rows.append(
                KeyRow(
                    row=position,
                    concept_id=concept_id,
                    proposals=tuple(sorted(pooled[concept_id], key=lambda p: p.run)),
                )
            )
        if not order:
            rows.append([case_id, "0", case.source_term, "", "", "", "", "", "", "", ""])
            key_rows.append(KeyRow(row=0, concept_id=None, proposals=()))
        key_cases.append(
            KeyCase(
                case_id=case_id,
                query_row=case.row,
                query_id=case.query_id,
                system=case.system,
                code=case.code,
                split=case.split,
                rows=tuple(key_rows),
            )
        )
    sheet = _csv_bytes([list(SHEET_COLUMNS), *rows])
    key = AdjudicationKey(
        seed=seed,
        depth=depth,
        queries_sha256=queries_sha256,
        sheet_sha256=hashlib.sha256(sheet).hexdigest(),
        content_sha256=content_digest(rows),
        n_rows=len(rows),
        n_cases=len(cases),
        runs=tuple(
            KeyRun(
                name=run.name,
                run_type=run.run_type,
                channel=run.channel,
                rankings_sha256=run.rankings_sha256,
            )
            for run in sorted(runs, key=lambda r: r.name)
        ),
        cases=tuple(key_cases),
    )
    return BuiltSheet(sheet=sheet, key=key)


# --------------------------------------------------------------------------
# Ingesting.
# --------------------------------------------------------------------------


def parse_sheet(data: bytes) -> list[list[str]]:
    """Rows of a completed sheet (header checked, then dropped)."""
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise AdjudicationError(f"the sheet is not UTF-8: {exc}") from exc
    records = list(csv.reader(io.StringIO(text)))
    if not records or tuple(records[0]) != SHEET_COLUMNS:
        raise TamperedSheetError(f"the sheet header is not {','.join(SHEET_COLUMNS)}")
    body = [r for r in records[1:] if any(cell.strip() for cell in r)]
    for number, record in enumerate(body, 2):
        if len(record) != len(SHEET_COLUMNS):
            raise TamperedSheetError(f"sheet line {number} has {len(record)} cells")
    return body


@dataclass(frozen=True, slots=True)
class Decision:
    """One adjudicated case, ready for the frame of record."""

    case: KeyCase
    source_term: str
    verdict: Verdict
    targets: frozenset[int]
    presented_accepted: frozenset[int]
    other_targets: frozenset[int]


def _ids(raw: str, where: str) -> frozenset[int]:
    parts = [p.strip() for p in raw.replace(",", ";").split(";") if p.strip()]
    try:
        return frozenset(int(p) for p in parts)
    except ValueError as exc:
        raise SheetGateError(f"{where}: other_targets {raw!r} are not integer ids") from exc


def decide(rows: Sequence[list[str]], key: AdjudicationKey) -> tuple[Decision, ...]:
    """Re-join completed rows with the key; fail closed on tampering or an undecided case."""
    if len(rows) != key.n_rows:
        raise TamperedSheetError(
            f"the sheet has {len(rows)} rows, the key built {key.n_rows}; rows were added, "
            "removed or filtered"
        )
    if content_digest(rows) != key.content_sha256:
        raise TamperedSheetError(
            "the sheet's read-only columns differ from what was built (re-sorted or edited)"
        )
    position = 0
    decisions: list[Decision] = []
    for case in key.cases:
        block = rows[position : position + len(case.rows)]
        position += len(case.rows)
        accepted: set[int] = set()
        other: set[int] = set()
        stated: set[str] = set()
        for record, key_row in zip(block, case.rows, strict=True):
            where = f"case {case.case_id} row {key_row.row}"
            mark = record[SHEET_COLUMNS.index("acceptable")].strip().lower()
            if mark not in ("", "y", "n"):
                raise SheetGateError(f"{where}: acceptable must be y, n or blank, got {mark!r}")
            if mark == "y":
                if key_row.concept_id is None:
                    raise SheetGateError(f"{where}: the no-candidate row cannot be acceptable")
                accepted.add(key_row.concept_id)
            other |= _ids(record[SHEET_COLUMNS.index("other_targets")], where)
            verdict = record[SHEET_COLUMNS.index("case_verdict")].strip().lower()
            if verdict:
                if verdict not in VERDICTS:
                    raise SheetGateError(f"{where}: case_verdict must be one of {list(VERDICTS)}")
                stated.add(verdict)
        targets = frozenset(accepted | other)
        if len(stated) > 1:
            raise SheetGateError(f"case {case.case_id}: conflicting verdicts {sorted(stated)}")
        given = next(iter(stated), None)
        if targets:
            if given not in (None, "accept"):
                raise SheetGateError(f"case {case.case_id} is {given} but lists acceptable targets")
            verdict_value: Verdict = "accept"
        elif given in ("no-target", "defer"):
            verdict_value = "no-target" if given == "no-target" else "defer"
        elif given == "accept":
            raise SheetGateError(f"case {case.case_id} is accept but no target is marked")
        else:
            raise SheetGateError(
                f"case {case.case_id} is undecided: mark a row y, list other_targets, or set "
                "case_verdict to no-target or defer"
            )
        check_verdict(case.query_id, verdict_value, targets)
        decisions.append(
            Decision(
                case=case,
                source_term=block[0][SHEET_COLUMNS.index("source_term")],
                verdict=verdict_value,
                targets=targets,
                presented_accepted=frozenset(accepted),
                other_targets=frozenset(other - accepted),
            )
        )
    return tuple(decisions)


def encode_frame(decisions: Sequence[Decision]) -> bytes:
    """The frame of record, one row per case in query-table order."""
    lines = ["\t".join(FRAME_COLUMNS)]
    for d in sorted(decisions, key=lambda x: x.case.query_row):
        cells = [
            d.case.query_id,
            d.case.system,
            d.case.code,
            d.source_term,
            d.case.split,
            d.verdict,
            ";".join(str(t) for t in sorted(d.targets)),
        ]
        if any("\t" in c or "\n" in c for c in cells):
            raise AdjudicationError(f"case {d.case.case_id}: a value holds a tab or newline")
        lines.append("\t".join(cells))
    return ("\n".join(lines) + "\n").encode("utf-8")


class IngestReport(_Frozen):
    """``ingest_report.json``: verdict counts and where accepted targets came from."""

    cases: int = Field(ge=0)
    verdicts: dict[str, int]
    accepted_from_list: int = Field(
        ge=0, description="Accepted cases with at least one acceptable presented candidate."
    )
    accepted_by_search_only: int = Field(
        ge=0, description="Accepted cases whose targets were all found outside the list."
    )
    presented_targets: int = Field(ge=0)
    other_targets: int = Field(ge=0)

    def to_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, indent=2) + "\n"


def ingest_report(decisions: Sequence[Decision]) -> IngestReport:
    counts = Counter(d.verdict for d in decisions)
    accepted = [d for d in decisions if d.verdict == "accept"]
    return IngestReport(
        cases=len(decisions),
        verdicts={v: counts[v] for v in VERDICTS},
        accepted_from_list=sum(1 for d in accepted if d.presented_accepted),
        accepted_by_search_only=sum(1 for d in accepted if not d.presented_accepted),
        presented_targets=sum(len(d.presented_accepted) for d in decisions),
        other_targets=sum(len(d.other_targets) for d in decisions),
    )


# --------------------------------------------------------------------------
# Input parsing and CLI.
# --------------------------------------------------------------------------


def _column(frame: pl.DataFrame, name: str) -> list[str]:
    return ["" if v is None else str(v) for v in frame.get_column(name).to_list()]


def parse_cases(frame: pl.DataFrame, *, id_column: str, text_column: str) -> tuple[Case, ...]:
    """Cases from the query table the runs ranked (``system`` and ``code`` required)."""
    needed = {id_column, text_column, "system", "code"}
    missing = sorted(needed - set(frame.columns))
    if missing:
        raise AdjudicationError(f"the case table lacks columns {missing}")
    splits = _column(frame, "split") if "split" in frame.columns else [""] * frame.height
    ids, texts = _column(frame, id_column), _column(frame, text_column)
    systems, codes = _column(frame, "system"), _column(frame, "code")
    cases: list[Case] = []
    seen: set[str] = set()
    for row in range(frame.height):
        query_id = ids[row].strip()
        if not query_id or query_id in seen:
            raise AdjudicationError(f"case table row {row}: empty or repeated query id")
        seen.add(query_id)
        if not systems[row].strip() or not codes[row].strip():
            raise AdjudicationError(f"query {query_id!r} needs a system and a code")
        cases.append(
            Case(row, query_id, systems[row].strip(), codes[row].strip(), texts[row], splits[row])
        )
    return tuple(cases)


def parse_concepts(frame: pl.DataFrame) -> dict[int, ConceptInfo]:
    needed = {"concept_id", "concept_name", "vocabulary_id", "domain_id"}
    missing = sorted(needed - set(frame.columns))
    if missing:
        raise AdjudicationError(f"--concepts lacks columns {missing}")
    out: dict[int, ConceptInfo] = {}
    for cid, name, vocabulary, domain in frame.select(
        "concept_id", "concept_name", "vocabulary_id", "domain_id"
    ).iter_rows():
        out[int(str(cid))] = ConceptInfo(str(name or ""), str(vocabulary or ""), str(domain or ""))
    return out


def _nested(a: Path, b: Path) -> bool:
    a, b = a.resolve(), b.resolve()
    return a == b or a in b.parents or b in a.parents


def _fail(exc: Exception) -> typer.Exit:
    from omopmap.meds.writer import OutputExistsError
    from omopmap.metrics import MetricsError, RunHashError

    typer.echo(f"{type(exc).__name__}: {exc}", err=True)
    if isinstance(exc, SheetGateError):
        return typer.Exit(GATE_EXIT)
    if isinstance(exc, TamperedSheetError | RunHashError):
        return typer.Exit(HASH_EXIT)
    if isinstance(exc, AdjudicationError | MetricsError | OutputExistsError | OSError | ValueError):
        return typer.Exit(1)
    raise exc


app = typer.Typer(
    name="adjudicate",
    help="Blinded adjudication sheets and their re-join into a frame of record.",
    no_args_is_help=True,
)


@app.command("build")
def build(
    run: Annotated[
        list[str], typer.Option("--run", help="NAME=DIR of a lexical, embed or fused run; repeat.")
    ],
    cases: Annotated[
        Path,
        typer.Option("--cases", help="The query table the runs ranked: query_id, system, code."),
    ],
    concepts: Annotated[Path, typer.Option("--concepts", help="Concept table for names.")],
    sheet_out: Annotated[Path, typer.Option("--sheet-out", help="New directory for reviewers.")],
    key_out: Annotated[Path, typer.Option("--key-out", help="New directory for the sealed key.")],
    seed: Annotated[int, typer.Option("--seed", help="Shuffle seed.")] = DEFAULT_SEED,
    depth: Annotated[int, typer.Option("--depth", help="Candidates taken from each run.")] = (
        DEFAULT_DEPTH
    ),
    id_column: Annotated[str, typer.Option("--id-column")] = "query_id",
    text_column: Annotated[str, typer.Option("--text-column")] = "source_term",
    created_at: Annotated[str | None, typer.Option("--created-at")] = None,
) -> None:
    """Write the blinded ``sheet.csv`` and, separately, the sealed key.

    Exit 0 on success; 1 unreadable input or occupied output; 4 a run's
    rankings differ from its manifest.
    """
    from omopmap.artifacts.manifest import sha256_file
    from omopmap.artifacts.models import utc_timestamp
    from omopmap.meds.convert import read_table
    from omopmap.meds.writer import prepare_output
    from omopmap.metrics import load_runs, named_runs
    from omopmap.retrieval.manifest import RUN_MANIFEST_NAME, build_run_manifest

    try:
        if _nested(sheet_out, key_out):
            raise AdjudicationError("--sheet-out and --key-out must be separate directories")
        specs = named_runs(run)
        if any(name is None for name, _ in specs):
            raise AdjudicationError("name every run: --run NAME=DIR")
        case_rows = parse_cases(read_table(cases), id_column=id_column, text_column=text_column)
        queries_sha = sha256_file(cases)
        runs = load_runs(specs, frame_sha256=queries_sha, n_queries=len(case_rows))
        built = build_sheet(
            case_rows,
            runs,
            parse_concepts(read_table(concepts)),
            seed=seed,
            depth=depth,
            queries_sha256=queries_sha,
        )
        prepare_output(sheet_out)
        prepare_output(key_out)
        (sheet_out / SHEET_NAME).write_bytes(built.sheet)
        (key_out / KEY_NAME).write_text(built.key.to_json(), encoding="utf-8")
        inputs = {"cases": queries_sha, "concepts": sha256_file(concepts)}
        inputs.update({f"run:{r.name}": r.rankings_sha256 for r in runs})
        manifest = build_run_manifest(
            "adjudicate",
            created_at=created_at or utc_timestamp(),
            seed=seed,
            parameters={"step": "build", "depth": depth, "cases": len(case_rows)},
            inputs=inputs,
            outputs={KEY_NAME: sha256_file(key_out / KEY_NAME), SHEET_NAME: built.key.sheet_sha256},
        )
        (key_out / RUN_MANIFEST_NAME).write_text(manifest.to_json(), encoding="utf-8")
    except Exception as exc:
        raise _fail(exc) from exc
    typer.echo(f"{built.key.n_cases} cases, {built.key.n_rows} sheet rows")


def load_key(directory: Path) -> tuple[AdjudicationKey, str]:
    """Read the sealed key; fail closed unless it hashes to its build manifest."""
    from omopmap.artifacts.manifest import sha256_file
    from omopmap.retrieval.manifest import RUN_MANIFEST_NAME, RunManifestError, parse_run_manifest

    path, manifest_path = directory / KEY_NAME, directory / RUN_MANIFEST_NAME
    if not path.is_file() or not manifest_path.is_file():
        raise AdjudicationError(f"{directory} lacks {KEY_NAME} or {RUN_MANIFEST_NAME}")
    try:
        recorded = parse_run_manifest(manifest_path.read_text(encoding="utf-8")).output_sha256(
            KEY_NAME
        )
    except RunManifestError as exc:
        raise AdjudicationError(str(exc)) from exc
    digest = sha256_file(path)
    if digest != recorded:
        raise TamperedSheetError(f"{path}: SHA-256 {digest} differs from its manifest")
    try:
        return AdjudicationKey.model_validate_json(path.read_text(encoding="utf-8")), digest
    except ValueError as exc:
        raise AdjudicationError(f"{KEY_NAME} is invalid: {exc}") from exc


@app.command("ingest")
def ingest(
    sheet: Annotated[Path, typer.Option("--sheet", help="The completed sheet.")],
    key: Annotated[Path, typer.Option("--key", help="The key directory from `build`.")],
    out: Annotated[Path, typer.Option("--out", help="New output directory.")],
    created_at: Annotated[str | None, typer.Option("--created-at")] = None,
) -> None:
    """Re-join a completed sheet with its key into a frame of record (``adjudicated.tsv``).

    Exit 0 on success; 1 unreadable input or occupied output; 4 the sheet or
    key differs from what was built; 6 a case is undecided or contradictory.
    """
    from omopmap.artifacts.manifest import sha256_file
    from omopmap.artifacts.models import utc_timestamp
    from omopmap.meds.writer import prepare_output
    from omopmap.retrieval.manifest import RUN_MANIFEST_NAME, build_run_manifest

    try:
        sealed, key_sha = load_key(key)
        decisions = decide(parse_sheet(sheet.read_bytes()), sealed)
        report = ingest_report(decisions)
        files = {
            FRAME_NAME: encode_frame(decisions),
            REPORT_NAME: report.to_json().encode("utf-8"),
        }
        prepare_output(out)
        for name, data in files.items():
            (out / name).write_bytes(data)
        manifest = build_run_manifest(
            "adjudicate",
            created_at=created_at or utc_timestamp(),
            parameters={"step": "ingest", "cases": report.cases},
            inputs={"key": key_sha, "sheet": sha256_file(sheet)},
            outputs={name: sha256_file(out / name) for name in files},
        )
        (out / RUN_MANIFEST_NAME).write_text(manifest.to_json(), encoding="utf-8")
    except Exception as exc:
        raise _fail(exc) from exc
    typer.echo(
        " ".join(f"{v}={n}" for v, n in report.verdicts.items())
        + f"; accepted by search only: {report.accepted_by_search_only}"
    )
