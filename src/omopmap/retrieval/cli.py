"""``omopmap retrieve`` and ``omopmap gate``: the retrieval core's file I/O.

Commands:
- ``retrieve lexical``: containment ranking of query terms over concept labels.
- ``retrieve index``: embed the concept labels into a frozen matrix and id map.
- ``retrieve embed``: open a frozen index (hash-asserted), embed queries, rank by cosine.

``index`` and ``embed`` load either a pinned base model (``--model``) or a
fine-tuned checkpoint (``--model-dir``: a ``finetune`` output directory whose
checkpoint files must hash to its run manifest; loaded from disk only).
- ``retrieve fuse``: RRF over the rank orders of two or more channel runs.
- ``gate``: measure encoder throughput on a sample, project the full job, stop if over 2x budget.

Every command writes into a new directory and finishes with ``run_manifest.json``.

Exit codes: 0 success; 1 bad input or a failed check; 3 the compute gate
stopped the job (``GATE_STOP.json`` written); 4 a frozen index's matrix or id
map, or a ``--model-dir`` checkpoint's files, do not match the digests their
manifest recorded.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import polars as pl
import typer

from omopmap.artifacts.manifest import sha256_file
from omopmap.artifacts.models import utc_timestamp
from omopmap.meds.convert import read_table
from omopmap.meds.writer import OutputExistsError, prepare_output
from omopmap.retrieval.checkpoint import (
    Checkpoint,
    CheckpointError,
    CheckpointHashError,
    open_checkpoint,
)
from omopmap.retrieval.embed import MODELS, Encoder, EncoderError, TransformerEncoder
from omopmap.retrieval.fusion import RRF_DEPTH, RRF_K, FusionError, fuse
from omopmap.retrieval.gate import (
    DEFAULT_FACTOR,
    GATE_NAME,
    GATE_STOP_EXIT,
    GATE_STOP_NAME,
    StageMeasurement,
    decide,
    timed,
)
from omopmap.retrieval.lexical import Diacritics, Label, build_index, search_all
from omopmap.retrieval.manifest import (
    RUN_MANIFEST_NAME,
    ParamValue,
    RunManifest,
    RunManifestError,
    RunType,
    build_run_manifest,
    parse_run_manifest,
)
from omopmap.retrieval.rankings import (
    RANKINGS_NAME,
    RankingsFormatError,
    decode_rankings,
    encode_fused,
    encode_scored,
)
from omopmap.retrieval.retrieve import (
    IDS_NAME,
    MATRIX_NAME,
    FrozenIndexError,
    IndexHashError,
    encode_ids,
    encode_matrix,
    open_index,
    top_k,
)

app = typer.Typer(
    name="retrieve", help="Rank OMOP concepts for source terms.", no_args_is_help=True
)

INDEX_HASH_EXIT = 4
DEFAULT_TOP_K = 50

EncoderFactory = Callable[[str], Encoder]


def _transformer_encoder(model: str) -> Encoder:
    if model not in MODELS:
        raise typer.BadParameter(f"--model must be one of {sorted(MODELS)}")
    return TransformerEncoder(MODELS[model])


ENCODER_FACTORY: EncoderFactory = _transformer_encoder
"""How ``--model`` becomes an encoder. Tests replace it with a deterministic stand-in."""

CheckpointEncoderFactory = Callable[[Checkpoint], Encoder]


def _checkpoint_encoder(checkpoint: Checkpoint) -> Encoder:
    return TransformerEncoder(
        checkpoint.spec, source=checkpoint.path, checkpoint_sha256=checkpoint.checkpoint_sha256
    )


CHECKPOINT_ENCODER_FACTORY: CheckpointEncoderFactory = _checkpoint_encoder
"""How a verified ``--model-dir`` checkpoint becomes an encoder (local files only)."""


def resolve_encoder(model: str | None, model_dir: Path | None) -> tuple[Encoder, str, str | None]:
    """The encoder, the ``model`` label runs record, and the checkpoint digest if any.

    ``--model-dir`` is verified (``open_checkpoint``) before anything is loaded.
    Giving both ``--model`` and ``--model-dir`` is allowed only when ``--model``
    names the checkpoint's base model.
    """
    if model_dir is None:
        key = model or "sapbert"
        return ENCODER_FACTORY(key), key, None
    checkpoint = open_checkpoint(model_dir)
    if model is not None and model != checkpoint.spec.key:
        raise CheckpointError(
            f"--model {model} contradicts the checkpoint's base model {checkpoint.spec.key}"
        )
    return (
        CHECKPOINT_ENCODER_FACTORY(checkpoint),
        checkpoint.model_label,
        checkpoint.checkpoint_sha256,
    )


_INPUT_ERRORS = (
    OutputExistsError,
    FusionError,
    FrozenIndexError,
    RankingsFormatError,
    RunManifestError,
    EncoderError,
    ValueError,
    OSError,
)


def _fail(exc: Exception, code: int = 1) -> typer.Exit:
    typer.echo(f"{type(exc).__name__}: {exc}", err=True)
    return typer.Exit(code)


def eligible_labels(concepts: Path, all_concepts: bool) -> list[Label]:
    """Concept names as labels; standard, valid concepts only unless ``all_concepts``.

    A built target table (``dictionary build``) has already had its
    eligibility decision applied and carries no eligibility columns, so it
    is consumed as-is.
    """
    frame = read_table(concepts)
    missing = {"concept_id", "concept_name"} - set(frame.columns)
    if missing:
        raise ValueError(f"{concepts.name} lacks columns {sorted(missing)}")
    if not all_concepts and {"standard_concept", "invalid_reason"} <= set(frame.columns):
        frame = frame.filter(
            (pl.col("standard_concept").fill_null("") == "S")
            & (pl.col("invalid_reason").fill_null("") == "")
        )
    return [
        Label(concept_id=int(cid), text=str(name))
        for cid, name in frame.select("concept_id", "concept_name").iter_rows()
    ]


def _extra_labels(path: Path | None) -> list[Label]:
    """Extra surfaces (e.g. Swedish synonyms) as ``concept_id``/``label`` rows."""
    if path is None:
        return []
    frame = read_table(path)
    if {"concept_id", "label"} - set(frame.columns):
        raise ValueError(f"{path.name} needs columns concept_id and label")
    return [
        Label(concept_id=int(cid), text=str(text), surface="extra")
        for cid, text in frame.select("concept_id", "label").iter_rows()
    ]


def _queries(path: Path, column: str) -> list[str]:
    frame = read_table(path)
    if column not in frame.columns:
        raise ValueError(f"{path.name} has no column {column!r}")
    return [str(v) if v is not None else "" for v in frame.get_column(column).to_list()]


def _write_run(
    out: Path,
    run_type: RunType,
    files: dict[str, bytes],
    *,
    parameters: dict[str, ParamValue],
    inputs: dict[str, str],
    created_at: str | None,
    encoder: Encoder | None = None,
    truncated: int | None = None,
) -> RunManifest:
    """Write ``files``, then the run manifest last (a run without one is incomplete)."""
    for name, data in files.items():
        (out / name).write_bytes(data)
    manifest = build_run_manifest(
        run_type,
        created_at=created_at or utc_timestamp(),
        parameters=parameters,
        inputs=inputs,
        outputs={name: sha256_file(out / name) for name in files},
        encoder=None if encoder is None else encoder.info(),
        truncated=truncated,
    )
    (out / RUN_MANIFEST_NAME).write_text(manifest.to_json(), encoding="utf-8")
    typer.echo(f"wrote {out}")
    return manifest


def _read_run_manifest(run: Path) -> RunManifest:
    path = run / RUN_MANIFEST_NAME
    if not path.is_file():
        raise RunManifestError(f"{run} has no {RUN_MANIFEST_NAME}")
    return parse_run_manifest(path.read_text(encoding="utf-8"))


ConceptsOpt = Annotated[
    Path, typer.Option("--concepts", help="OMOP CONCEPT table (.csv/.tsv/.parquet).")
]
EncoderModelOpt = Annotated[
    str | None,
    typer.Option("--model", help=f"One of {sorted(MODELS)} (default sapbert)."),
]
ModelDirOpt = Annotated[
    Path | None,
    typer.Option(
        "--model-dir",
        help="A `finetune` output directory: load its hash-verified checkpoint from disk.",
    ),
]
QueriesOpt = Annotated[Path, typer.Option("--queries", help="Table of source terms.")]
QueryColumnOpt = Annotated[str, typer.Option("--query-column", help="Column holding the terms.")]
OutOpt = Annotated[Path, typer.Option("--out", help="New output directory.")]
ModelOpt = Annotated[str, typer.Option("--model", help=f"One of {sorted(MODELS)}.")]
AllConceptsOpt = Annotated[
    bool,
    typer.Option("--all-concepts", help="Keep non-standard and invalid concepts as candidates."),
]
CreatedAtOpt = Annotated[
    str | None, typer.Option("--created-at", help="Fixed manifest timestamp, for byte identity.")
]


@app.command("lexical")
def lexical(
    concepts: ConceptsOpt,
    queries: QueriesOpt,
    out: OutOpt,
    query_column: QueryColumnOpt = "source_term",
    labels: Annotated[
        Path | None, typer.Option("--labels", help="Extra concept_id/label surfaces.")
    ] = None,
    depth: Annotated[int, typer.Option("--depth", help="Candidates kept per query.")] = RRF_DEPTH,
    diacritics: Annotated[Diacritics, typer.Option("--diacritics")] = Diacritics.KEEP,
    all_concepts: AllConceptsOpt = False,
    created_at: CreatedAtOpt = None,
) -> None:
    """Rank concepts by normalised-token containment (no corpus statistics)."""
    try:
        prepare_output(out)
        names = eligible_labels(concepts, all_concepts)
        eligible = {label.concept_id for label in names}
        # Extra surfaces only add strings to concepts already in the candidate set;
        # they never widen it.
        extra = [label for label in _extra_labels(labels) if label.concept_id in eligible]
        index = build_index([*names, *extra], diacritics)
        texts = _queries(queries, query_column)
        rankings = search_all(index, texts, depth)
        inputs = {"concepts": sha256_file(concepts), "queries": sha256_file(queries)}
        if labels is not None:
            inputs["labels"] = sha256_file(labels)
        _write_run(
            out,
            "lexical",
            {RANKINGS_NAME: encode_scored(rankings)},
            parameters={
                "all_concepts": all_concepts,
                "depth": depth,
                "diacritics": diacritics.value,
                "queries": len(texts),
                "query_column": query_column,
            },
            inputs=inputs,
            created_at=created_at,
        )
    except _INPUT_ERRORS as exc:
        raise _fail(exc) from exc


@app.command("index")
def index(
    concepts: ConceptsOpt,
    out: OutOpt,
    model: EncoderModelOpt = None,
    model_dir: ModelDirOpt = None,
    all_concepts: AllConceptsOpt = False,
    created_at: CreatedAtOpt = None,
) -> None:
    """Embed concept names into a frozen matrix plus id map."""
    try:
        encoder, label, checkpoint_sha = resolve_encoder(model, model_dir)
    except CheckpointHashError as exc:
        raise _fail(exc, INDEX_HASH_EXIT) from exc
    except (CheckpointError, *_INPUT_ERRORS) as exc:
        raise _fail(exc) from exc
    try:
        prepare_output(out)
        labels = eligible_labels(concepts, all_concepts)
        encoded = encoder.encode([label.text for label in labels])
        inputs = {"concepts": sha256_file(concepts)}
        if checkpoint_sha is not None:
            inputs["checkpoint"] = checkpoint_sha
        _write_run(
            out,
            "index",
            {
                MATRIX_NAME: encode_matrix(encoded.vectors),
                IDS_NAME: encode_ids([label.concept_id for label in labels]),
            },
            parameters={"all_concepts": all_concepts, "model": label},
            inputs=inputs,
            created_at=created_at,
            encoder=encoder,
            truncated=encoded.truncated,
        )
    except _INPUT_ERRORS as exc:
        raise _fail(exc) from exc


@app.command("embed")
def embed(
    index_dir: Annotated[Path, typer.Option("--index", help="Directory from `retrieve index`.")],
    queries: QueriesOpt,
    out: OutOpt,
    query_column: QueryColumnOpt = "source_term",
    model: EncoderModelOpt = None,
    model_dir: ModelDirOpt = None,
    depth: Annotated[int, typer.Option("--depth", help="Candidates kept per query.")] = RRF_DEPTH,
    created_at: CreatedAtOpt = None,
) -> None:
    """Rank concepts by cosine similarity against a hash-verified frozen index."""
    try:
        encoder, label, checkpoint_sha = resolve_encoder(model, model_dir)
        built = _read_run_manifest(index_dir)
        frozen = open_index(
            (index_dir / MATRIX_NAME).read_bytes(),
            (index_dir / IDS_NAME).read_bytes(),
            expected_matrix_sha256=built.output_sha256(MATRIX_NAME),
            expected_ids_sha256=built.output_sha256(IDS_NAME),
        )
    except (IndexHashError, CheckpointHashError) as exc:
        raise _fail(exc, INDEX_HASH_EXIT) from exc
    except (CheckpointError, *_INPUT_ERRORS) as exc:
        raise _fail(exc) from exc
    try:
        info = encoder.info()
        identity = (info.model_id, info.revision, info.checkpoint_sha256)
        if (
            built.encoder is None
            or (
                built.encoder.model_id,
                built.encoder.revision,
                built.encoder.checkpoint_sha256,
            )
            != identity
        ):
            what = f"{info.model_id}@{info.revision}" + (
                "" if checkpoint_sha is None else f" + checkpoint {checkpoint_sha[:12]}"
            )
            raise FrozenIndexError(f"the index was built with another encoder than {what}")
        prepare_output(out)
        texts = _queries(queries, query_column)
        encoded = encoder.encode(texts)
        rankings = top_k(frozen, encoded.vectors, depth)
        _write_run(
            out,
            "embed",
            {RANKINGS_NAME: encode_scored(rankings)},
            parameters={
                "depth": depth,
                "model": label,
                "queries": len(texts),
                "query_column": query_column,
            },
            inputs={
                "index:" + IDS_NAME: frozen.ids_sha256,
                "index:" + MATRIX_NAME: frozen.matrix_sha256,
                "queries": sha256_file(queries),
                **({} if checkpoint_sha is None else {"checkpoint": checkpoint_sha}),
            },
            created_at=created_at,
            encoder=encoder,
            truncated=encoded.truncated,
        )
    except _INPUT_ERRORS as exc:
        raise _fail(exc) from exc


@app.command("fuse")
def fuse_runs(
    runs: Annotated[
        list[Path], typer.Option("--run", help="A channel run directory; give two or more.")
    ],
    out: OutOpt,
    k: Annotated[int, typer.Option("--k", help="RRF damping constant.")] = RRF_K,
    depth: Annotated[int, typer.Option("--depth", help="Entries taken per channel.")] = RRF_DEPTH,
    top: Annotated[int, typer.Option("--top-k", help="Fused candidates kept.")] = DEFAULT_TOP_K,
    created_at: CreatedAtOpt = None,
) -> None:
    """Fuse channel runs by reciprocal rank fusion; only rank orders are read."""
    try:
        if len(runs) < 2:
            raise ValueError("give at least two --run directories")
        prepare_output(out)
        channels: list[tuple[str, bytes, RunManifest]] = []
        for run in runs:
            manifest = _read_run_manifest(run)
            if manifest.run_type not in ("lexical", "embed"):
                raise ValueError(f"{run} is a {manifest.run_type} run, not a channel run")
            data = (run / RANKINGS_NAME).read_bytes()
            if sha256_file(run / RANKINGS_NAME) != manifest.output_sha256(RANKINGS_NAME):
                raise RunManifestError(f"{run}/{RANKINGS_NAME} differs from its manifest")
            channels.append((manifest.channel(), data, manifest))
        names = [name for name, _, _ in channels]
        if len(set(names)) != len(names):
            raise ValueError(f"channels repeat: {sorted(names)}")
        shapes = {
            (m.inputs_by_name().get("queries"), m.parameters.get("queries")) for _, _, m in channels
        }
        if len(shapes) != 1:
            raise ValueError("the runs were not computed on the same query table")
        n_queries = channels[0][2].parameters.get("queries")
        if not isinstance(n_queries, int) or isinstance(n_queries, bool):
            raise RunManifestError("channel runs must record an integer `queries` parameter")
        per_channel = [
            decode_rankings(data, channel=name, queries=n_queries) for name, data, _ in channels
        ]
        fused = [
            fuse(list(query_rankings), k=k, depth=depth, top_k=top)
            for query_rankings in zip(*per_channel, strict=True)
        ]
        _write_run(
            out,
            "fuse",
            {RANKINGS_NAME: encode_fused(fused)},
            parameters={"depth": depth, "k": k, "top_k": top},
            inputs={f"run:{name}": m.output_sha256(RANKINGS_NAME) for name, _, m in channels},
            created_at=created_at,
        )
    except _INPUT_ERRORS as exc:
        raise _fail(exc) from exc


def gate(
    concepts: ConceptsOpt,
    queries: QueriesOpt,
    out: OutOpt,
    budget_hours: Annotated[
        float, typer.Option("--budget-hours", help="Estimated wall time of the full job.")
    ],
    query_column: QueryColumnOpt = "source_term",
    model: ModelOpt = "sapbert",
    sample: Annotated[int, typer.Option("--sample", help="Texts timed per stage.")] = 1000,
    fixed_seconds: Annotated[
        float, typer.Option("--fixed-seconds", help="Allowance for unscaled steps.")
    ] = 0.0,
    factor: Annotated[float, typer.Option("--factor", help="Stop above factor x budget.")] = (
        DEFAULT_FACTOR
    ),
    all_concepts: AllConceptsOpt = False,
    created_at: CreatedAtOpt = None,
) -> None:
    """Project the full embedding job from a timed sample; exit 3 if over factor x budget."""
    try:
        prepare_output(out)
        labels = [label.text for label in eligible_labels(concepts, all_concepts)]
        texts = [q for q in _queries(queries, query_column) if q]
        if not labels or not texts or sample < 1:
            raise ValueError("the gate needs at least one label, one query and --sample >= 1")
        encoder = ENCODER_FACTORY(model)
        encoder.encode(labels[:1])  # load the model outside the timed region
        stages: list[StageMeasurement] = []
        for name, items in (("labels", labels), ("queries", texts)):
            batch = items[:sample]
            run = timed(lambda batch=batch: encoder.encode(batch))
            stages.append(
                StageMeasurement(
                    name=name,
                    measured_items=len(batch),
                    measured_seconds=run.seconds,
                    full_items=len(items),
                )
            )
        decision = decide(
            stages,
            budget_seconds=budget_hours * 3600.0,
            fixed_seconds=fixed_seconds,
            factor=factor,
        )
        files = {GATE_NAME: decision.to_json().encode("utf-8")}
        if decision.stop:
            files[GATE_STOP_NAME] = files[GATE_NAME]
        _write_run(
            out,
            "gate",
            files,
            parameters={
                "all_concepts": all_concepts,
                "budget_seconds": budget_hours * 3600.0,
                "factor": factor,
                "fixed_seconds": fixed_seconds,
                "model": model,
                "query_column": query_column,
                "sample": sample,
            },
            inputs={"concepts": sha256_file(concepts), "queries": sha256_file(queries)},
            created_at=created_at,
            encoder=encoder,
        )
    except _INPUT_ERRORS as exc:
        raise _fail(exc) from exc
    if decision.stop:
        typer.echo(
            f"GATE STOP: projected {decision.projected_seconds:.1f}s > "
            f"threshold {decision.threshold_seconds:.1f}s",
            err=True,
        )
        raise typer.Exit(GATE_STOP_EXIT)
