"""Text encoders behind one protocol, with pinned model revisions.

Every retrieval step that needs vectors depends only on ``Encoder``: give it
texts, get back an ``(n, dim)`` float32 matrix of L2-normalised rows plus the
number of inputs that were truncated. The tests use a deterministic hashing
encoder. Real runs use ``TransformerEncoder``, which loads a pinned Hugging
Face checkpoint on first use through the ``embed`` extra (torch, transformers),
so importing this module never imports torch.

Each model is pinned to a full commit revision and its own pooling rule:
SapBERT (multilingual, XLM-R based) uses the [CLS] vector, and BioLORD-2023-M
uses attention-mask mean pooling, as its model card prescribes. Inputs are
truncated to 128 tokens and the truncations are counted, never silent. On a
GPU the model runs in fp16. Outputs are returned as float32.

A fine-tuned encoder is the same architecture loaded from a local checkpoint
directory instead of the hub (``TransformerEncoder(spec, source=...)``, with
``local_files_only``, so nothing is downloaded). It keeps its base model's
pooling and length settings, and reports the base model id and revision plus
the checkpoint's own digest, so manifests tell base and fine-tuned vectors
apart. ``omopmap.retrieval.checkpoint`` verifies the directory before this
class ever sees it.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import numpy.typing as npt

Pooling = Literal["cls", "mean"]
Matrix = npt.NDArray[np.float32]

MAX_LENGTH = 128
BATCH_SIZE = 256


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """A Hugging Face model pinned to one commit, with its pooling rule."""

    key: str
    model_id: str
    revision: str
    pooling: Pooling

    def __post_init__(self) -> None:
        if len(self.revision) != 40 or any(c not in "0123456789abcdef" for c in self.revision):
            raise ValueError(f"{self.key}: revision must be a full 40-hex commit")


SAPBERT = ModelSpec(
    key="sapbert",
    model_id="cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR",
    revision="47b6bd041ba61311584bb2494edfda5c7d9b719f",
    pooling="cls",
)
BIOLORD = ModelSpec(
    key="biolord",
    model_id="FremyCompany/BioLORD-2023-M",
    revision="4ea2ea2c89ef63365f7fcd91406a0cd8ac36d2e2",
    pooling="mean",
)
MODELS: dict[str, ModelSpec] = {spec.key: spec for spec in (SAPBERT, BIOLORD)}


@dataclass(frozen=True, slots=True)
class EncoderInfo:
    """What a run manifest records about the encoder that produced its vectors."""

    model_id: str
    revision: str
    pooling: str
    max_length: int
    dim: int
    device: str = "cpu"
    gpu: str | None = None
    precision: str = "fp32"
    library_versions: tuple[tuple[str, str], ...] = ()
    checkpoint_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class Encoded:
    """Encoder output: L2-normalised float32 rows and the truncation count."""

    vectors: Matrix
    truncated: int


class Encoder(Protocol):
    """Anything that turns texts into L2-normalised vectors."""

    def info(self) -> EncoderInfo:
        """Identity of the model, for manifests."""
        ...

    def encode(self, texts: Sequence[str]) -> Encoded:
        """Encode ``texts`` in order; one row per text."""
        ...


class EncoderError(RuntimeError):
    """The encoder could not be loaded or produced malformed output."""


def l2_normalise(vectors: npt.NDArray[np.floating[Any]]) -> Matrix:
    """Row-wise L2 normalisation to float32; zero rows stay zero."""
    rows = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (rows / norms).astype(np.float32, copy=False)


def check_encoded(encoded: Encoded, n: int, dim: int) -> Encoded:
    """Fail closed unless the output has ``n`` finite rows of width ``dim``."""
    shape = encoded.vectors.shape
    if shape != (n, dim):
        raise EncoderError(f"encoder returned shape {shape}, expected {(n, dim)}")
    if not bool(np.isfinite(encoded.vectors).all()):
        raise EncoderError("encoder returned non-finite values")
    return encoded


@dataclass
class TransformerEncoder:
    """A pinned Hugging Face encoder, loaded on the first ``encode`` call.

    With ``source`` set, the weights and tokenizer come from that local
    directory (a verified fine-tuned checkpoint of ``spec``) with
    ``local_files_only=True``; ``checkpoint_sha256`` is then its identity.
    """

    spec: ModelSpec
    source: Path | None = None
    checkpoint_sha256: str | None = None
    device: str | None = None
    batch_size: int = BATCH_SIZE
    max_length: int = MAX_LENGTH
    _state: dict[str, Any] = field(default_factory=dict[str, Any], repr=False)

    def _load(self) -> dict[str, Any]:
        if self._state:
            return self._state
        try:
            torch: Any = importlib.import_module("torch")
            transformers: Any = importlib.import_module("transformers")
        except ImportError as exc:
            raise EncoderError(
                "the embedding encoder needs the `embed` extra (torch, transformers)"
            ) from exc
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.source is None:
            location: dict[str, Any] = {"revision": self.spec.revision}
            name = self.spec.model_id
        else:
            location = {"local_files_only": True}
            name = str(self.source)
        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(name, **location)
            model = transformers.AutoModel.from_pretrained(name, **location).to(device)
        except OSError as exc:
            raise EncoderError(f"could not load the encoder from {name}: {exc}") from exc
        if device.startswith("cuda"):
            model = model.half()
        model.eval()
        self._state.update(
            torch=torch,
            transformers=transformers,
            device=device,
            tokenizer=tokenizer,
            model=model,
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
            precision="fp16" if on_gpu else "fp32",
            library_versions=(
                ("torch", str(torch.__version__)),
                ("transformers", str(state["transformers"].__version__)),
            ),
            checkpoint_sha256=self.checkpoint_sha256,
        )

    def encode(self, texts: Sequence[str]) -> Encoded:
        state = self._load()
        torch, tokenizer, model = state["torch"], state["tokenizer"], state["model"]
        dim = int(model.config.hidden_size)
        out = np.zeros((len(texts), dim), dtype=np.float32)
        truncated = 0
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = list(texts[start : start + self.batch_size])
                full = tokenizer(batch, add_special_tokens=True, truncation=False)["input_ids"]
                truncated += sum(1 for ids in full if len(ids) > self.max_length)
                inputs = tokenizer(
                    batch,
                    add_special_tokens=True,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(state["device"])
                hidden = model(**inputs).last_hidden_state
                if self.spec.pooling == "cls":
                    pooled = hidden[:, 0]
                else:
                    mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
                out[start : start + len(batch)] = pooled.cpu().numpy()
        return check_encoded(Encoded(vectors=out, truncated=truncated), len(texts), dim)
