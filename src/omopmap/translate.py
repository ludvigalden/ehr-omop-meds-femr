"""Optional Swedish-to-English machine translation for retrieval queries.

The retrieval channels operate on concept names in English; Swedish source
terms can therefore only reach the embedding channel after translation.
This step is optional and sits behind the ``embed`` extra: the real
translator loads a pinned Hugging Face checkpoint on first use, and
importing this module never imports torch.

The translation contract. Carried over from the source experiment:

- the model is pinned to a full commit revision and loaded the way its
  model card prescribes: ``AutoProcessor`` plus
  ``AutoModelForImageTextToText``, one user chat message per input whose
  content is a structured text part with ``source_lang_code`` ``sv`` and
  ``target_lang_code`` ``en`` (no free-text instruction), left padding;
- decoding is greedy (``do_sample=False``, ``num_beams=1``,
  ``max_new_tokens=64``), so one input always yields the same translation;
- on a GPU the model runs in bf16;
- the original string is preserved per row -- the translation is
  additional, never a replacement.

Added here: a smoke check runs first -- the first N non-empty inputs are
translated and every one must come back non-empty, otherwise the step
aborts before any output exists. (The source's smoke step projected the
full run's wall time instead; that cost gate is ``omopmap gate``'s job.)

The hermetic tests inject a deterministic fake translator; the real
transformer path is exercised only in opt-in, model-present runs.

These experiments ran on rented GPUs (see
``context/reference/translation.md``); no real-data result is claimed by
this module.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Protocol

import polars as pl
import typer

from omopmap.meds.convert import read_table

TRANSLATIONS_NAME = "translations.parquet"

SMOKE_N = 8
MAX_NEW_TOKENS = 64


class TranslationError(ValueError):
    """The translation step cannot run."""


class SmokeGateError(TranslationError):
    """The smoke gate failed: the model produced empty output on real input."""


@dataclass(frozen=True, slots=True)
class TranslatorSpec:
    """A Hugging Face translation model pinned to one commit."""

    model_id: str
    revision: str

    def __post_init__(self) -> None:
        if len(self.revision) != 40 or any(c not in "0123456789abcdef" for c in self.revision):
            raise ValueError(
                f"{self.model_id}: revision must be a full 40-hex commit, got {self.revision!r}"
            )


#: The source experiment's pinned model; the revision is carried in full.
TRANSLATE_MODEL = TranslatorSpec(
    model_id="google/translategemma-4b-it",
    revision="10042cb0e6e7fdce748996a71dc3dc432a4e0c89",
)


@dataclass(frozen=True, slots=True)
class TranslatorInfo:
    """What a run manifest records about the translator that ran."""

    model_id: str
    revision: str
    device: str = "cpu"
    gpu: str | None = None
    precision: str = "fp32"
    max_new_tokens: int = MAX_NEW_TOKENS
    library_versions: tuple[tuple[str, str], ...] = ()


class Translator(Protocol):
    """Anything that translates Swedish texts to English, in order."""

    def info(self) -> TranslatorInfo:
        """Identity of the model, for manifests."""
        ...

    def translate(self, texts: Sequence[str]) -> list[str]:
        """Translate ``texts`` in order; exactly one string per input."""
        ...


def check_smoke_gate(translations: Sequence[str], sources: Sequence[str]) -> None:
    """Fail closed unless every smoke translation is non-empty."""
    if len(translations) != len(sources):
        raise SmokeGateError(
            f"the smoke run returned {len(translations)} outputs for {len(sources)} inputs"
        )
    for source, translation in zip(sources, translations, strict=True):
        if not translation.strip():
            raise SmokeGateError(
                f"the smoke gate failed: empty translation of {source!r}; "
                "aborting before the full run"
            )


def translate_all(
    texts: Sequence[str],
    translator: Translator,
    *,
    smoke_n: int = SMOKE_N,
) -> list[tuple[str, str]]:
    """Translate every text under the smoke gate, keeping the original per row.

    Empty inputs are passed through untranslated (and never feed the smoke
    gate). The returned pairs preserve the input order: ``(original,
    translation)``.
    """
    if smoke_n < 1:
        raise TranslationError("smoke_n must be at least 1")
    pending = [(index, text) for index, text in enumerate(texts) if text.strip()]
    if not pending:
        raise TranslationError("there are no non-empty texts to translate")
    smoke = [text for _, text in pending[:smoke_n]]
    check_smoke_gate(translator.translate(smoke), smoke)
    translated = translator.translate([text for _, text in pending])
    if len(translated) != len(pending):
        raise TranslationError(
            f"the translator returned {len(translated)} outputs for {len(pending)} inputs"
        )
    out: list[tuple[str, str]] = [(text, "") for text in texts]
    for (index, original), translation in zip(pending, translated, strict=True):
        out[index] = (original, translation)
    return out


SOURCE_LANG = "sv"
TARGET_LANG = "en"


def chat_message(text: str) -> list[dict[str, Any]]:
    """The model's translation request for one input: a structured text part.

    The language pair travels in the structured fields; no instruction text
    is composed around the source term.
    """
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "source_lang_code": SOURCE_LANG,
                    "target_lang_code": TARGET_LANG,
                    "text": text,
                }
            ],
        }
    ]


@dataclass
class GreedyTranslator:
    """The pinned Hugging Face translator, loaded on the first ``translate`` call.

    The processor applies the model's chat template to one structured
    message per input, with left padding so every generated continuation
    starts at the same column. Decoding is greedy and bounded:
    ``do_sample=False``, ``num_beams=1``, ``max_new_tokens`` as configured.
    On CUDA the model runs in bf16; on CPU, fp32.
    """

    spec: TranslatorSpec
    device: str | None = None
    batch_size: int = 16
    max_new_tokens: int = MAX_NEW_TOKENS
    _state: dict[str, Any] = field(default_factory=dict[str, Any], repr=False)

    def _load(self) -> dict[str, Any]:
        if self._state:
            return self._state
        try:
            torch: Any = importlib.import_module("torch")
            transformers: Any = importlib.import_module("transformers")
        except ImportError as exc:
            raise TranslationError(
                "the translator needs the `embed` extra (torch, transformers)"
            ) from exc
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        processor = transformers.AutoProcessor.from_pretrained(
            self.spec.model_id, revision=self.spec.revision
        )
        processor.tokenizer.padding_side = "left"
        model = transformers.AutoModelForImageTextToText.from_pretrained(
            self.spec.model_id,
            revision=self.spec.revision,
            dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        ).to(device)
        model.eval()
        self._state.update(
            torch=torch,
            transformers=transformers,
            device=device,
            processor=processor,
            model=model,
        )
        return self._state

    def info(self) -> TranslatorInfo:
        state = self._load()
        torch: Any = state["torch"]
        device: str = state["device"]
        on_gpu = device.startswith("cuda")
        return TranslatorInfo(
            model_id=self.spec.model_id,
            revision=self.spec.revision,
            device=device,
            gpu=str(torch.cuda.get_device_name(0)) if on_gpu else None,
            precision="bf16" if on_gpu else "fp32",
            max_new_tokens=self.max_new_tokens,
            library_versions=(
                ("torch", str(torch.__version__)),
                ("transformers", str(state["transformers"].__version__)),
            ),
        )

    def translate(self, texts: Sequence[str]) -> list[str]:
        state = self._load()
        torch, processor, model = state["torch"], state["processor"], state["model"]
        device: str = state["device"]
        out: list[str] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            inputs = processor.apply_chat_template(
                [chat_message(text) for text in batch],
                tokenize=True,
                add_generation_prompt=True,
                padding=True,
                return_dict=True,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=self.max_new_tokens,
                )
            new_tokens = generated[:, inputs["input_ids"].shape[1] :]
            decoded: list[str] = processor.batch_decode(new_tokens, skip_special_tokens=True)
            out.extend(text.strip() for text in decoded)
        return out


TranslatorFactory = Callable[[TranslatorSpec], Translator]

TRANSLATOR_FACTORY: TranslatorFactory = GreedyTranslator
"""How the CLI builds a translator. Tests replace it with a deterministic stand-in."""


def _make_translator(revision: str | None) -> Translator:
    try:
        spec = (
            TRANSLATE_MODEL
            if revision is None
            else TranslatorSpec(model_id=TRANSLATE_MODEL.model_id, revision=revision)
        )
    except ValueError as exc:
        raise TranslationError(str(exc)) from exc
    return TRANSLATOR_FACTORY(spec)


# --------------------------------------------------------------------------
# CLI: all file I/O lives here; the functions above are pure except the
# transformer-backed translator.
# --------------------------------------------------------------------------

app = typer.Typer(
    name="translate",
    help="Translate Swedish source terms to English (optional; needs the embed extra).",
    no_args_is_help=True,
)


@app.command("run")
def run(
    queries: Annotated[Path, typer.Option("--queries", help="Table of source terms.")],
    out: Annotated[Path, typer.Option("--out", help="New output directory.")],
    query_column: Annotated[
        str, typer.Option("--query-column", help="Column holding the Swedish terms.")
    ] = "source_term",
    smoke_n: Annotated[
        int, typer.Option("--smoke-n", help="Inputs translated before the full run.")
    ] = SMOKE_N,
    revision: Annotated[
        str | None,
        typer.Option(
            "--revision",
            help="Override the pinned model revision (full 40-hex commit).",
        ),
    ] = None,
    created_at: Annotated[
        str | None, typer.Option("--created-at", help="Fixed manifest timestamp.")
    ] = None,
) -> None:
    """Translate the query column sv->en and write ``translations.parquet``.

    Every input row is preserved in order with its original string; the
    English translation is additional. The smoke gate aborts the run --
    before any output is written -- if the model returns empty text.
    """
    from omopmap.artifacts.manifest import sha256_file
    from omopmap.artifacts.models import utc_timestamp
    from omopmap.meds.writer import OutputExistsError, prepare_output
    from omopmap.retrieval.manifest import (
        RUN_MANIFEST_NAME,
        TranslatorRecord,
        build_run_manifest,
    )

    try:
        frame = read_table(queries)
        if query_column not in frame.columns:
            raise TranslationError(f"{queries.name} has no column {query_column!r}")
        texts = [
            "" if value is None else str(value)
            for value in frame.get_column(query_column).to_list()
        ]
        translator = _make_translator(revision)
        pairs = translate_all(texts, translator, smoke_n=smoke_n)
        prepare_output(out)
        result = pl.DataFrame(
            {
                "source_term": [original for original, _ in pairs],
                "translated_term": [translation for _, translation in pairs],
            }
        )
        result.write_parquet(out / TRANSLATIONS_NAME)
        info = translator.info()
        manifest = build_run_manifest(
            "translate",
            created_at=created_at or utc_timestamp(),
            parameters={
                "max_new_tokens": info.max_new_tokens,
                "query_column": query_column,
                "queries": len(texts),
                "smoke_n": smoke_n,
            },
            inputs={"queries": sha256_file(queries)},
            outputs={TRANSLATIONS_NAME: sha256_file(out / TRANSLATIONS_NAME)},
            translator=TranslatorRecord(
                model_id=info.model_id,
                revision=info.revision,
                device=info.device,
                gpu=info.gpu,
                precision=info.precision,
                max_new_tokens=info.max_new_tokens,
                libraries=dict(info.library_versions),
            ),
        )
        (out / RUN_MANIFEST_NAME).write_text(manifest.to_json(), encoding="utf-8")
        typer.echo(f"translated {len(pairs)} terms into {out}")
    except (TranslationError, OutputExistsError, OSError) as exc:
        typer.echo(f"{type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(1) from exc
