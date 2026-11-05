"""Contrastive fine-tuning of the retrieval encoder (MNRL with mined hard negatives).

The task: pull each Swedish source term (the anchor) towards the name of its
gold OMOP concept (the positive) and away from the concepts the retriever
confuses it with. The objective is the multiple-negatives ranking loss
(MNRL): for a batch of ``B`` anchors, every anchor is scored against all
``B`` positives and all ``B x h`` hard negatives in the batch by cosine
similarity times a scale of 20, and cross-entropy pushes the anchor's own
positive to the top. In-batch positives of other anchors act as free extra
negatives; the mined hard negatives are the informative ones, because they
are exactly the near misses the untuned model ranks above the gold concept.

Contract (from the source experiments; see ``context/reference/fine-tuning.md``):
cosine scale 20, at most 8 hard negatives per anchor, batch 16, learning rate
2e-5, a fixed seed, at most 4 epochs, dev-split recall@10 evaluated after
every epoch with early stopping at patience 2, and the best-dev weights
restored at the end.

Leakage guards, all fail closed:

- the held-out split is dropped when the pair table is read; only its row
  count is kept, so no held-out text reaches training, mining or monitoring;
- anchors come from the train split only; the dev split only drives early
  stopping;
- a source term that appears (after normalisation) in both train and dev is
  an error, since dev would then monitor memorised anchors;
- hard negatives exclude every gold concept of the anchor's term, from any
  of its rows;
- in a batch, no two anchors share a positive concept, and a candidate that
  is a gold concept of the anchor (other than its own positive column) is
  masked out of the softmax, so no true match is ever trained as a negative.

Two trainers implement one protocol:

- ``TorchTrainer`` (``embed`` extra): the pinned Hugging Face encoder with
  its own pooling rule, AdamW, a linear schedule with 10% warmup, and fp16
  autocast on CUDA. ``--dry-run`` never imports it.
- ``NumpyTrainer``: the hermetic stand-in used by ``--dry-run`` and the
  tests. It is a real, if small, encoder: hashed character-trigram features
  times a learned linear map, L2-normalised. It optimises the identical MNRL
  objective with analytic gradients and plain SGD at its own step size, so
  the dataset, guards, batching, monitor, early stopping and manifest are
  all exercised end to end without torch or a model download.

The functions below are pure; ``omopmap finetune`` does the file I/O.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol

import numpy as np
import numpy.typing as npt
import polars as pl
import typer
from pydantic import BaseModel, ConfigDict, Field

from omopmap.retrieval.checkpoint import CHECKPOINT_DIR
from omopmap.retrieval.embed import (
    BIOLORD,
    MAX_LENGTH,
    MODELS,
    Encoded,
    EncoderError,
    EncoderInfo,
    ModelSpec,
    check_encoded,
    l2_normalise,
)
from omopmap.retrieval.lexical import normalise

FINETUNE_NAME = "finetune.json"

SEED = 20261106
SCALE = 20.0
MAX_HARD_NEGATIVES = 8
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
MAX_EPOCHS = 4
PATIENCE = 2
MONITOR_K = 10
WARMUP_RATIO = 0.1

Split = Literal["train", "dev"]
Mode = Literal["dry-run", "full"]
F64 = npt.NDArray[np.float64]


class FinetuneError(ValueError):
    """The fine-tune cannot run on these inputs."""


class LeakageError(FinetuneError):
    """A split boundary would leak evaluation data into training or monitoring."""


class FinetuneConfig(BaseModel):
    """The training contract. Defaults are the source experiments' values."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scale: float = Field(default=SCALE, gt=0)
    hard_negatives: int = Field(default=MAX_HARD_NEGATIVES, ge=0, le=MAX_HARD_NEGATIVES)
    batch_size: int = Field(default=BATCH_SIZE, ge=2)
    learning_rate: float = Field(default=LEARNING_RATE, gt=0)
    seed: int = SEED
    max_epochs: int = Field(default=MAX_EPOCHS, ge=1)
    patience: int = Field(default=PATIENCE, ge=1)
    monitor_k: int = Field(default=MONITOR_K, ge=1)
    warmup_ratio: float = Field(default=WARMUP_RATIO, ge=0, lt=1)


# --------------------------------------------------------------------------
# Dataset construction
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PairRow:
    """One (source term, gold concept) row of the user's pair table."""

    row: int
    text: str
    concept_id: int
    split: Split


@dataclass(frozen=True, slots=True)
class PairTable:
    """The train and dev rows of a pair table; held-out rows are counted, never kept."""

    rows: tuple[PairRow, ...]
    heldout_rows: int


def read_pairs(
    frame: pl.DataFrame,
    *,
    text_column: str = "source_term",
    gold_column: str = "concept_id",
    split_column: str = "split",
) -> PairTable:
    """Parse a pair table, dropping held-out rows before anything else looks at them."""
    missing = sorted({text_column, gold_column, split_column} - set(frame.columns))
    if missing:
        raise FinetuneError(f"the pair table lacks columns {missing}")
    splits = frame.get_column(split_column).cast(pl.String).fill_null("").to_list()
    unknown = sorted({s for s in splits if s not in ("train", "dev", "heldout")})
    if unknown:
        raise FinetuneError(f"unknown split values {unknown}; expected train, dev, heldout")
    heldout = sum(1 for s in splits if s == "heldout")
    kept = frame.with_row_index("row_index").filter(pl.col(split_column) != "heldout")
    rows: list[PairRow] = []
    for record in kept.iter_rows(named=True):
        text = "" if record[text_column] is None else str(record[text_column])
        if not text.strip():
            raise FinetuneError(f"row {record['row_index']}: empty source term")
        try:
            concept_id = int(str(record[gold_column]))
        except ValueError as exc:
            raise FinetuneError(
                f"row {record['row_index']}: gold id {record[gold_column]!r} is not an integer"
            ) from exc
        split: Split = "train" if record[split_column] == "train" else "dev"
        rows.append(
            PairRow(row=int(record["row_index"]), text=text, concept_id=concept_id, split=split)
        )
    return PairTable(rows=tuple(rows), heldout_rows=heldout)


def gold_by_term(rows: Sequence[PairRow]) -> dict[str, frozenset[int]]:
    """Every gold concept of each normalised source term, over train and dev rows."""
    gold: dict[str, set[int]] = {}
    for row in rows:
        gold.setdefault(normalise(row.text), set()).add(row.concept_id)
    return {term: frozenset(ids) for term, ids in gold.items()}


def check_split_leakage(rows: Sequence[PairRow]) -> None:
    """Fail closed when a normalised source term occurs in both train and dev."""
    by_split: dict[Split, set[str]] = {"train": set(), "dev": set()}
    for row in rows:
        by_split[row.split].add(normalise(row.text))
    shared = sorted(by_split["train"] & by_split["dev"])
    if shared:
        raise LeakageError(
            f"{len(shared)} source term(s) occur in both the train and dev splits; "
            "dev would monitor memorised anchors"
        )


@dataclass(frozen=True, slots=True)
class TrainExample:
    """One anchor with its positive and its mined hard negatives (concept ids)."""

    row: int
    anchor: str
    positive: int
    negatives: tuple[int, ...]
    gold: frozenset[int]


@dataclass(frozen=True, slots=True)
class DevQuery:
    """One dev query: its text and its gold concepts."""

    text: str
    gold: frozenset[int]


@dataclass(frozen=True, slots=True)
class Dataset:
    """Everything the training loop needs, plus the counts the manifest records."""

    examples: tuple[TrainExample, ...]
    dev: tuple[DevQuery, ...]
    dev_pool: tuple[int, ...]
    names: dict[int, str]
    hard_negatives: int
    counts: dict[str, int]


def mine_negatives(
    ranking: Sequence[int], gold: frozenset[int], eligible: frozenset[int], cap: int
) -> tuple[int, ...]:
    """The first ``cap`` ranked concepts that are eligible and not gold, in rank order."""
    out: list[int] = []
    for concept_id in ranking:
        if len(out) >= cap:
            break
        if concept_id in gold or concept_id not in eligible:
            continue
        out.append(concept_id)
    return tuple(out)


def build_dataset(
    table: PairTable,
    names: dict[int, str],
    rankings: Mapping[int, Sequence[int]],
    config: FinetuneConfig,
) -> Dataset:
    """Training examples and the dev monitor from pairs, concept names and mined rankings.

    ``rankings`` maps a pair-table row index to the retriever's ranked
    concept ids for that row's source term. Pairs whose gold concept is not
    an eligible target are dropped and counted (there is no positive text to
    train on). Every anchor gets the same number of hard negatives: the
    smaller of the cap and the fewest any anchor has.
    """
    check_split_leakage(table.rows)
    gold = gold_by_term(table.rows)
    eligible = frozenset(names)
    counts = {
        "heldout_rows_excluded": table.heldout_rows,
        "train_rows": 0,
        "dev_rows": 0,
        "gold_not_eligible": 0,
    }
    raw: list[tuple[PairRow, tuple[int, ...]]] = []
    dev: dict[str, DevQuery] = {}
    pool: set[int] = set()
    for row in table.rows:
        counts[f"{row.split}_rows"] += 1
        term_gold = gold[normalise(row.text)]
        if row.concept_id not in eligible:
            counts["gold_not_eligible"] += 1
            continue
        if row.row not in rankings:
            raise FinetuneError(f"no retriever ranking for pair row {row.row}")
        mined = mine_negatives(rankings[row.row], term_gold, eligible, config.hard_negatives)
        if row.split == "train":
            raw.append((row, mined))
        else:
            key = normalise(row.text)
            reachable = term_gold & eligible
            dev.setdefault(key, DevQuery(text=row.text, gold=reachable))
            pool.update(reachable)
            pool.update(mined)
    if not raw:
        raise FinetuneError("no train pair has an eligible gold concept")
    if not dev:
        raise FinetuneError("no dev pair has an eligible gold concept; early stopping needs dev")
    effective = min(config.hard_negatives, *(len(mined) for _, mined in raw))
    if config.hard_negatives > 0 and effective == 0:
        raise FinetuneError("no hard negatives could be mined for at least one anchor")
    examples = tuple(
        TrainExample(
            row=row.row,
            anchor=row.text,
            positive=row.concept_id,
            negatives=mined[:effective],
            gold=gold[normalise(row.text)],
        )
        for row, mined in raw
    )
    counts["train_examples"] = len(examples)
    counts["dev_queries"] = len(dev)
    counts["dev_pool"] = len(pool)
    return Dataset(
        examples=examples,
        dev=tuple(dev[key] for key in sorted(dev)),
        dev_pool=tuple(sorted(pool)),
        names=names,
        hard_negatives=effective,
        counts=counts,
    )


# --------------------------------------------------------------------------
# Batching and the MNRL objective
# --------------------------------------------------------------------------


def pack_batches(
    examples: Sequence[TrainExample], batch_size: int, *, seed: int, epoch: int
) -> list[list[int]]:
    """Shuffle deterministically per epoch, then pack batches with distinct positives.

    An example whose positive is already in the open batch waits for the
    next one, so an in-batch positive is never another anchor's true match.
    """
    order = [int(i) for i in np.random.default_rng([seed, epoch]).permutation(len(examples))]
    batches: list[list[int]] = []
    pending = order
    while pending:
        batch: list[int] = []
        seen: set[int] = set()
        rest: list[int] = []
        for index in pending:
            positive = examples[index].positive
            if len(batch) < batch_size and positive not in seen:
                batch.append(index)
                seen.add(positive)
            else:
                rest.append(index)
        batches.append(batch)
        pending = rest
    return batches


def candidate_ids(batch: Sequence[TrainExample]) -> list[int]:
    """Candidate concept ids of a batch: all positives first, then all hard negatives."""
    return [e.positive for e in batch] + [n for e in batch for n in e.negatives]


def false_negative_mask(batch: Sequence[TrainExample]) -> npt.NDArray[np.bool_]:
    """``True`` where a candidate is a gold concept of the anchor, except its own positive."""
    candidates = candidate_ids(batch)
    mask = np.zeros((len(batch), len(candidates)), dtype=np.bool_)
    for i, example in enumerate(batch):
        for j, concept_id in enumerate(candidates):
            if j != i and concept_id in example.gold:
                mask[i, j] = True
    return mask


def mnrl_loss_and_grad(scores: F64, mask: npt.NDArray[np.bool_]) -> tuple[float, F64]:
    """Mean cross-entropy of each row against its diagonal target, and d loss / d scores.

    ``scores`` are already scaled; masked entries are excluded from the softmax.
    """
    logits = np.where(mask, -np.inf, scores)
    peak = logits.max(axis=1, keepdims=True)
    exp = np.exp(logits - peak)
    probs = exp / exp.sum(axis=1, keepdims=True)
    n = scores.shape[0]
    rows: npt.NDArray[np.intp] = np.arange(n, dtype=np.intp)
    loss = float(-np.mean(np.log(probs[rows, rows])))
    grad = probs.copy()
    grad[rows, rows] -= 1.0
    return loss, grad / n


# --------------------------------------------------------------------------
# Trainers
# --------------------------------------------------------------------------


class Trainer(Protocol):
    """A trainable encoder: the training loop needs only these operations."""

    def info(self) -> EncoderInfo:
        """Identity of the encoder being trained."""
        ...

    def encode(self, texts: Sequence[str]) -> Encoded:
        """L2-normalised float32 vectors, one row per text."""
        ...

    def begin(self, total_steps: int, config: FinetuneConfig) -> None:
        """Set up the optimiser for ``total_steps`` updates."""
        ...

    def train_epoch(
        self,
        batches: Sequence[Sequence[TrainExample]],
        names: dict[int, str],
        config: FinetuneConfig,
    ) -> float:
        """One pass over ``batches``; return the mean batch loss."""
        ...

    def snapshot(self) -> object:
        """A copy of the current weights."""
        ...

    def restore(self, state: object) -> None:
        """Load weights from a ``snapshot``."""
        ...


NGRAM_DIM = 512
NUMPY_DIM = 32
NUMPY_STEP = 0.5


def trigram_features(texts: Sequence[str]) -> F64:
    """Hashed character-trigram counts of the normalised text, L2-normalised rows."""
    out = np.zeros((len(texts), NGRAM_DIM), dtype=np.float64)
    for row, text in enumerate(texts):
        padded = f"  {normalise(text)} "
        for start in range(len(padded) - 2):
            gram = padded[start : start + 3].encode("utf-8")
            digest = hashlib.blake2b(gram, digest_size=4).digest()
            out[row, int.from_bytes(digest, "little") % NGRAM_DIM] += 1.0
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return out / norms


@dataclass
class NumpyTrainer:
    """Hashed trigrams times a learned linear map: a small real encoder in pure numpy."""

    seed: int = SEED
    step: float = NUMPY_STEP
    weights: F64 = field(init=False, repr=False)

    def __post_init__(self) -> None:
        rng = np.random.default_rng([self.seed, 0])
        self.weights = rng.normal(0.0, 1.0 / math.sqrt(NGRAM_DIM), (NGRAM_DIM, NUMPY_DIM))

    def info(self) -> EncoderInfo:
        return EncoderInfo(
            model_id="numpy-trigram-linear",
            revision="0" * 40,
            pooling="none",
            max_length=MAX_LENGTH,
            dim=NUMPY_DIM,
        )

    def _forward(self, features: F64) -> tuple[F64, F64]:
        projected = features @ self.weights
        norms = np.linalg.norm(projected, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return projected / norms, norms

    def encode(self, texts: Sequence[str]) -> Encoded:
        vectors, _ = self._forward(trigram_features(texts))
        return check_encoded(
            Encoded(vectors=l2_normalise(vectors), truncated=0), len(texts), NUMPY_DIM
        )

    def begin(self, total_steps: int, config: FinetuneConfig) -> None:
        del total_steps, config

    def batch_loss_and_grad(
        self, batch: Sequence[TrainExample], names: dict[int, str], scale: float
    ) -> tuple[float, F64]:
        """MNRL loss of one batch and its gradient with respect to the weights."""
        anchor_x = trigram_features([e.anchor for e in batch])
        cand_x = trigram_features([names[c] for c in candidate_ids(batch)])
        anchors, anchor_norms = self._forward(anchor_x)
        cands, cand_norms = self._forward(cand_x)
        scores = scale * (anchors @ cands.T)
        loss, d_scores = mnrl_loss_and_grad(scores, false_negative_mask(batch))
        d_anchors = scale * d_scores @ cands
        d_cands = scale * d_scores.T @ anchors

        def through_norm(d_v: F64, v: F64, norms: F64) -> F64:
            return (d_v - v * np.sum(v * d_v, axis=1, keepdims=True)) / norms

        grad = anchor_x.T @ through_norm(d_anchors, anchors, anchor_norms)
        grad += cand_x.T @ through_norm(d_cands, cands, cand_norms)
        return loss, grad

    def train_epoch(
        self,
        batches: Sequence[Sequence[TrainExample]],
        names: dict[int, str],
        config: FinetuneConfig,
    ) -> float:
        losses: list[float] = []
        for batch in batches:
            loss, grad = self.batch_loss_and_grad(batch, names, config.scale)
            self.weights = self.weights - self.step * grad
            losses.append(loss)
        return float(np.mean(losses)) if losses else 0.0

    def snapshot(self) -> object:
        return self.weights.copy()

    def restore(self, state: object) -> None:
        if not isinstance(state, np.ndarray):
            raise FinetuneError("not a numpy trainer snapshot")
        restored: F64 = np.array(state, dtype=np.float64, copy=True)
        self.weights = restored


@dataclass
class TorchTrainer:
    """The pinned Hugging Face encoder, fine-tuned in plain torch.

    Pooling follows the model's ``ModelSpec`` (CLS for SapBERT, mean for
    BioLORD), inputs are truncated to 128 tokens, AdamW runs a linear
    schedule with warmup, and on CUDA the forward pass uses fp16 autocast
    with gradient scaling. Pass ``model`` and ``tokenizer`` to train an
    already-loaded encoder (the tests use a tiny randomly initialised one).
    """

    spec: ModelSpec
    device: str | None = None
    model: Any = None
    tokenizer: Any = None
    max_length: int = MAX_LENGTH
    _state: dict[str, Any] = field(default_factory=dict[str, Any], repr=False)

    def _load(self) -> dict[str, Any]:
        if self._state:
            return self._state
        try:
            torch: Any = importlib.import_module("torch")
            transformers: Any = importlib.import_module("transformers")
        except ImportError as exc:
            raise EncoderError("fine-tuning needs the `embed` extra (torch, transformers)") from exc
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = self.tokenizer or transformers.AutoTokenizer.from_pretrained(
            self.spec.model_id, revision=self.spec.revision
        )
        model = self.model or transformers.AutoModel.from_pretrained(
            self.spec.model_id, revision=self.spec.revision
        )
        self._state.update(
            torch=torch,
            transformers=transformers,
            device=device,
            tokenizer=tokenizer,
            model=model.to(device),
        )
        return self._state

    def info(self) -> EncoderInfo:
        state = self._load()
        torch = state["torch"]
        device: str = state["device"]
        on_gpu = device.startswith("cuda")
        return EncoderInfo(
            model_id=self.spec.model_id,
            revision=self.spec.revision,
            pooling=self.spec.pooling,
            max_length=self.max_length,
            dim=int(state["model"].config.hidden_size),
            device=device,
            gpu=str(torch.cuda.get_device_name(0)) if on_gpu else None,
            precision="fp16-autocast" if on_gpu else "fp32",
            library_versions=(
                ("torch", str(torch.__version__)),
                ("transformers", str(state["transformers"].__version__)),
            ),
        )

    def _embed(self, texts: Sequence[str]) -> Any:
        state = self._load()
        torch = state["torch"]
        inputs = state["tokenizer"](
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(state["device"])
        hidden = state["model"](**inputs).last_hidden_state
        if self.spec.pooling == "cls":
            pooled = hidden[:, 0]
        else:
            mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return torch.nn.functional.normalize(pooled.float(), p=2, dim=1)

    def encode(self, texts: Sequence[str]) -> Encoded:
        state = self._load()
        torch, model = state["torch"], state["model"]
        dim = int(model.config.hidden_size)
        out = np.zeros((len(texts), dim), dtype=np.float32)
        model.eval()
        with torch.no_grad():
            for start in range(0, len(texts), 256):
                batch = texts[start : start + 256]
                out[start : start + len(batch)] = self._embed(batch).cpu().numpy()
        return check_encoded(Encoded(vectors=out, truncated=0), len(texts), dim)

    def begin(self, total_steps: int, config: FinetuneConfig) -> None:
        state = self._load()
        torch, transformers = state["torch"], state["transformers"]
        torch.manual_seed(config.seed)
        optimiser = torch.optim.AdamW(state["model"].parameters(), lr=config.learning_rate)
        schedule = transformers.get_linear_schedule_with_warmup(
            optimiser,
            num_warmup_steps=max(1, int(config.warmup_ratio * total_steps)),
            num_training_steps=max(1, total_steps),
        )
        on_gpu = str(state["device"]).startswith("cuda")
        state.update(
            optimiser=optimiser,
            schedule=schedule,
            scaler=torch.amp.GradScaler("cuda") if on_gpu else None,
        )

    def train_epoch(
        self,
        batches: Sequence[Sequence[TrainExample]],
        names: dict[int, str],
        config: FinetuneConfig,
    ) -> float:
        state = self._load()
        if "optimiser" not in state:
            raise FinetuneError("call begin() before train_epoch()")
        torch, model = state["torch"], state["model"]
        optimiser, schedule, scaler = state["optimiser"], state["schedule"], state["scaler"]
        device: str = state["device"]
        model.train()
        losses: list[float] = []
        for batch in batches:
            mask = torch.as_tensor(false_negative_mask(batch), device=device)
            target = torch.arange(len(batch), device=device)
            optimiser.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda" if scaler is not None else "cpu",
                dtype=torch.float16,
                enabled=scaler is not None,
            ):
                anchors = self._embed([e.anchor for e in batch])
                cands = self._embed([names[c] for c in candidate_ids(batch)])
            scores = config.scale * anchors.float() @ cands.float().T
            loss = torch.nn.functional.cross_entropy(
                scores.masked_fill(mask, float("-inf")), target
            )
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimiser)
                scaler.update()
            else:
                loss.backward()
                optimiser.step()
            schedule.step()
            losses.append(float(loss.detach().cpu()))
        return float(np.mean(losses)) if losses else 0.0

    def snapshot(self) -> object:
        model = self._load()["model"]
        return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    def restore(self, state: object) -> None:
        if not isinstance(state, dict):
            raise FinetuneError("not a torch trainer snapshot")
        self._load()["model"].load_state_dict(state)

    def save(self, path: Path) -> None:
        """Write the model and tokenizer in Hugging Face format."""
        state = self._load()
        state["model"].save_pretrained(path)
        state["tokenizer"].save_pretrained(path)


# --------------------------------------------------------------------------
# Monitor and loop
# --------------------------------------------------------------------------


def pool_recall(trainer: Trainer, dataset: Dataset, k: int) -> float:
    """Share of dev queries with a gold concept in the top ``k`` of the dev pool."""
    pool = list(dataset.dev_pool)
    pool_vectors = trainer.encode([dataset.names[c] for c in pool]).vectors
    query_vectors = trainer.encode([q.text for q in dataset.dev]).vectors
    scores = query_vectors @ pool_vectors.T
    ids = np.asarray(pool, dtype=np.int64)
    hits = 0
    for row, query in zip(scores, dataset.dev, strict=True):
        order = np.lexsort((ids, -row))[:k]
        hits += any(int(ids[i]) in query.gold for i in order)
    return hits / len(dataset.dev)


class EpochRecord(BaseModel):
    """One evaluation point: epoch 0 is the untuned encoder."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    epoch: int = Field(ge=0)
    train_loss: float | None
    dev_recall: float = Field(ge=0.0, le=1.0)
    batches: int = Field(ge=0)


class FitResult(BaseModel):
    """What the loop did: every evaluation, the best epoch, and why it stopped."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    history: tuple[EpochRecord, ...]
    best_epoch: int
    best_dev_recall: float
    early_stopped: bool
    epochs_run: int


Monitor = Callable[[Trainer, Dataset, int], float]


def fit(
    trainer: Trainer,
    dataset: Dataset,
    config: FinetuneConfig,
    *,
    monitor: Monitor = pool_recall,
) -> FitResult:
    """Train with dev recall@k early stopping; leave the best-dev weights loaded.

    The untuned encoder is evaluated first (epoch 0) and competes for best,
    so a fine-tune that never improves dev recall restores the base weights.
    Improvement is strict; ``patience`` epochs without improvement stop the
    run.
    """
    examples = dataset.examples
    plans = [
        pack_batches(examples, config.batch_size, seed=config.seed, epoch=epoch)
        for epoch in range(1, config.max_epochs + 1)
    ]
    trainer.begin(sum(len(plan) for plan in plans), config)
    score = monitor(trainer, dataset, config.monitor_k)
    history = [EpochRecord(epoch=0, train_loss=None, dev_recall=score, batches=0)]
    best_epoch, best_score, best_state = 0, score, trainer.snapshot()
    stale = 0
    early_stopped = False
    for epoch, plan in enumerate(plans, start=1):
        batches = [[examples[i] for i in indices] for indices in plan]
        loss = trainer.train_epoch(batches, dataset.names, config)
        score = monitor(trainer, dataset, config.monitor_k)
        history.append(
            EpochRecord(epoch=epoch, train_loss=loss, dev_recall=score, batches=len(plan))
        )
        if score > best_score:
            best_epoch, best_score, best_state = epoch, score, trainer.snapshot()
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                early_stopped = True
                break
    trainer.restore(best_state)
    return FitResult(
        history=tuple(history),
        best_epoch=best_epoch,
        best_dev_recall=best_score,
        early_stopped=early_stopped,
        epochs_run=len(history) - 1,
    )


def mine_with_encoder(
    trainer: Trainer, names: dict[int, str], rows: Sequence[PairRow], depth: int
) -> dict[int, tuple[int, ...]]:
    """Rank the targets for every pair row with the untuned encoder (the B3 retriever)."""
    from omopmap.retrieval.retrieve import (
        encode_ids,
        encode_matrix,
        open_index,
        sha256_bytes,
        top_k,
    )

    ids = sorted(names)
    matrix = trainer.encode([names[i] for i in ids]).vectors
    matrix_bytes, ids_bytes = encode_matrix(matrix), encode_ids(ids)
    index = open_index(
        matrix_bytes,
        ids_bytes,
        expected_matrix_sha256=sha256_bytes(matrix_bytes),
        expected_ids_sha256=sha256_bytes(ids_bytes),
    )
    queries = trainer.encode([row.text for row in rows]).vectors
    rankings = top_k(index, queries, min(depth, len(ids)))
    return {row.row: ranking.ids for row, ranking in zip(rows, rankings, strict=True)}


class FinetuneReport(BaseModel):
    """``finetune.json``: the contract that ran, the dataset counts and the loop history."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: Mode
    trainer: str
    config: FinetuneConfig
    hard_negatives_effective: int
    negatives_source: str
    counts: dict[str, int]
    fit: FitResult

    def to_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, indent=2) + "\n"


# --------------------------------------------------------------------------
# CLI: all file I/O lives here.
# --------------------------------------------------------------------------

TrainerFactory = Callable[[ModelSpec, FinetuneConfig], Trainer]


def _torch_trainer(spec: ModelSpec, config: FinetuneConfig) -> Trainer:
    del config
    return TorchTrainer(spec)


TRAINER_FACTORY: TrainerFactory = _torch_trainer
"""How full mode builds its trainer. Tests may replace it."""


def finetune(
    pairs: Annotated[Path, typer.Option("--pairs", help="source_term / concept_id / split table.")],
    concepts: Annotated[
        Path, typer.Option("--concepts", help="Target table or OMOP CONCEPT table.")
    ],
    out: Annotated[Path, typer.Option("--out", help="New output directory.")],
    negatives: Annotated[
        Path | None,
        typer.Option(
            "--negatives",
            help="A lexical/embed run over --pairs to mine hard negatives from. "
            "Default: mine with the untuned encoder.",
        ),
    ] = None,
    model: Annotated[str, typer.Option("--model", help=f"One of {sorted(MODELS)}.")] = BIOLORD.key,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Train the numpy stand-in encoder; no torch, no download."),
    ] = False,
    epochs: Annotated[int, typer.Option("--epochs", help="Maximum epochs.")] = MAX_EPOCHS,
    seed: Annotated[int, typer.Option("--seed")] = SEED,
    hard_negatives: Annotated[
        int, typer.Option("--hard-negatives", help="Hard negatives per anchor, at most 8.")
    ] = MAX_HARD_NEGATIVES,
    created_at: Annotated[
        str | None, typer.Option("--created-at", help="Fixed manifest timestamp.")
    ] = None,
) -> None:
    """Fine-tune the encoder with MNRL on the train split; early-stop on dev recall@10.

    The held-out split is never read beyond its row count. Writes
    ``finetune.json``, ``run_manifest.json`` (last) and, in full mode, the
    best-dev ``checkpoint/``. Exit 1 on any input, leakage or training failure.
    """
    from omopmap.artifacts.manifest import sha256_file
    from omopmap.artifacts.models import utc_timestamp
    from omopmap.meds.convert import read_table
    from omopmap.meds.writer import OutputExistsError, prepare_output
    from omopmap.retrieval.cli import eligible_labels
    from omopmap.retrieval.manifest import RUN_MANIFEST_NAME, RunManifestError, build_run_manifest
    from omopmap.retrieval.rankings import RankingsFormatError
    from omopmap.runs import RunError, load_run

    try:
        if model not in MODELS:
            raise FinetuneError(f"--model must be one of {sorted(MODELS)}")
        config = FinetuneConfig(seed=seed, max_epochs=epochs, hard_negatives=hard_negatives)
        table = read_pairs(read_table(pairs))
        names = {label.concept_id: label.text for label in eligible_labels(concepts, False)}
        if not names:
            raise FinetuneError("the concept table has no eligible targets")
        trainer: Trainer = (
            NumpyTrainer(seed=seed) if dry_run else TRAINER_FACTORY(MODELS[model], config)
        )
        inputs = {"concepts": sha256_file(concepts), "pairs": sha256_file(pairs)}
        if negatives is None:
            depth = config.hard_negatives + 64
            rankings = mine_with_encoder(trainer, names, table.rows, depth)
            source = "untuned-encoder"
        else:
            run = load_run(negatives, queries_sha256=inputs["pairs"])
            rankings = {row.row: run.rankings[row.row].ids for row in table.rows}
            source = f"run:{run.manifest.channel()}"
            inputs["negatives"] = run.rankings_sha256
        dataset = build_dataset(table, names, rankings, config)
        prepare_output(out)
        result = fit(trainer, dataset, config)
        report = FinetuneReport(
            mode="dry-run" if dry_run else "full",
            trainer="numpy-trigram-linear" if dry_run else "torch",
            config=config,
            hard_negatives_effective=dataset.hard_negatives,
            negatives_source=source,
            counts=dataset.counts,
            fit=result,
        )
        files: dict[str, str] = {}
        (out / FINETUNE_NAME).write_text(report.to_json(), encoding="utf-8")
        files[FINETUNE_NAME] = sha256_file(out / FINETUNE_NAME)
        if not dry_run and isinstance(trainer, TorchTrainer):
            trainer.save(out / CHECKPOINT_DIR)
            for path in sorted((out / CHECKPOINT_DIR).rglob("*")):
                if path.is_file():
                    files[path.relative_to(out).as_posix()] = sha256_file(path)
        manifest = build_run_manifest(
            "finetune",
            created_at=created_at or utc_timestamp(),
            parameters={
                "batch_size": config.batch_size,
                "dry_run": dry_run,
                "early_stopped": result.early_stopped,
                "best_epoch": result.best_epoch,
                "hard_negatives": config.hard_negatives,
                "hard_negatives_effective": dataset.hard_negatives,
                "learning_rate": config.learning_rate,
                "max_epochs": config.max_epochs,
                "model": model,
                "monitor": f"dev_recall@{config.monitor_k}",
                "patience": config.patience,
                "scale": config.scale,
                "trainer": report.trainer,
                "warmup_ratio": config.warmup_ratio,
                **{f"count:{name}": value for name, value in dataset.counts.items()},
            },
            inputs=inputs,
            outputs=files,
            encoder=None if dry_run else trainer.info(),
            seed=config.seed,
        )
        (out / RUN_MANIFEST_NAME).write_text(manifest.to_json(), encoding="utf-8")
        typer.echo(
            f"best epoch {result.best_epoch} (dev recall@{config.monitor_k} "
            f"{result.best_dev_recall:.3f}); {result.epochs_run} epoch(s) run"
            + (", early stopped" if result.early_stopped else "")
        )
    except (
        FinetuneError,
        RunError,
        RunManifestError,
        RankingsFormatError,
        EncoderError,
        OutputExistsError,
        OSError,
        ValueError,
    ) as exc:
        typer.echo(f"{type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(1) from exc
