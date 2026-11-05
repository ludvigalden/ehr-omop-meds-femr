"""Run manifests for retrieval steps: what ran, on what, producing what.

Every retrieval command writes ``run_manifest.json`` next to its outputs. It
records the encoder (model id, full revision, pooling, max length, dim, device,
GPU name and precision when present), the step's parameters, the seed, the
package version, and the SHA-256 of every input and output file. With that
record a result can be traced to the exact model commit and the exact bytes it
was computed from, and a frozen index can be re-opened only if its matrix and id
map still hash to what the build recorded.

The JSON is deterministic: sorted keys, two-space indent, a trailing newline,
and input/output records sorted by name. Records reuse the artifact-contract
types (``InputRecord``, ``Producer``) rather than defining parallel ones.
"""

from __future__ import annotations

import json
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from omopmap.artifacts.models import InputRecord, Producer, Timestamp, producer
from omopmap.retrieval.embed import EncoderInfo

RUN_MANIFEST_NAME = "run_manifest.json"
RunType = Literal[
    "lexical",
    "index",
    "embed",
    "fuse",
    "gate",
    "dictionary",
    "translate",
    "finetune",
    "mcnemar",
    "validate",
    "metrics",
    "calibrate",
    "adjudicate",
]
ParamValue = bool | int | float | str


class RunManifestError(ValueError):
    """A run manifest is missing, malformed, or lacks a record it must have."""


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class EncoderRecord(_Model):
    """The encoder that produced the vectors of a run."""

    model_id: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    pooling: str
    max_length: int = Field(ge=1)
    dim: int = Field(ge=1)
    device: str
    gpu: str | None = Field(description="GPU name when the run used one; null on CPU.")
    precision: str
    libraries: dict[str, str] = Field(description="Library name to version.")
    checkpoint_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
        description="Digest of the fine-tuned checkpoint loaded on top of the base model; "
        "null for the pinned base model itself.",
    )

    @classmethod
    def from_info(cls, info: EncoderInfo) -> EncoderRecord:
        return cls(
            model_id=info.model_id,
            revision=info.revision,
            pooling=info.pooling,
            max_length=info.max_length,
            dim=info.dim,
            device=info.device,
            gpu=info.gpu,
            precision=info.precision,
            libraries=dict(info.library_versions),
            checkpoint_sha256=info.checkpoint_sha256,
        )


class TranslatorRecord(_Model):
    """The translation model that produced a translate step's strings."""

    model_id: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    device: str
    gpu: str | None = Field(description="GPU name when the run used one; null on CPU.")
    precision: str
    max_new_tokens: int = Field(ge=1)
    libraries: dict[str, str] = Field(description="Library name to version.")


class RunManifest(_Model):
    """One retrieval step's record."""

    run_type: RunType
    producer: Producer
    created_at: Timestamp
    seed: int | None
    encoder: EncoderRecord | None
    translator: TranslatorRecord | None = None
    parameters: dict[str, ParamValue]
    inputs: tuple[InputRecord, ...]
    outputs: tuple[InputRecord, ...]
    truncated: int | None = Field(ge=0, description="Inputs truncated by the encoder, if any.")

    @model_validator(mode="after")
    def _sorted_unique(self) -> Self:
        for what, records in (("inputs", self.inputs), ("outputs", self.outputs)):
            names = [r.name for r in records]
            if names != sorted(set(names)):
                raise ValueError(f"{what} must be sorted by name and unique")
        return self

    def output_sha256(self, name: str) -> str:
        """The recorded digest of output ``name``; fails closed if it is absent."""
        for record in self.outputs:
            if record.name == name:
                return record.sha256
        raise RunManifestError(f"the run manifest records no output named {name!r}")

    def inputs_by_name(self) -> dict[str, str]:
        """Input name to recorded sha256."""
        return {record.name: record.sha256 for record in self.inputs}

    def channel(self) -> str:
        """Channel name for fusion: the run type, plus the model for embedding runs."""
        model = self.parameters.get("model")
        return self.run_type if model is None else f"{self.run_type}:{model}"

    def to_json(self) -> str:
        """Deterministic JSON: sorted keys, two-space indent, trailing newline."""
        payload = self.model_dump(mode="json")
        return json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def build_run_manifest(
    run_type: RunType,
    *,
    created_at: str,
    parameters: dict[str, ParamValue],
    inputs: dict[str, str],
    outputs: dict[str, str],
    encoder: EncoderInfo | None = None,
    translator: TranslatorRecord | None = None,
    seed: int | None = None,
    truncated: int | None = None,
) -> RunManifest:
    """Assemble a manifest from name-to-sha256 maps (pure)."""
    return RunManifest(
        run_type=run_type,
        producer=producer(),
        created_at=created_at,
        seed=seed,
        encoder=None if encoder is None else EncoderRecord.from_info(encoder),
        translator=translator,
        parameters=dict(sorted(parameters.items())),
        inputs=tuple(InputRecord(name=n, sha256=s) for n, s in sorted(inputs.items())),
        outputs=tuple(InputRecord(name=n, sha256=s) for n, s in sorted(outputs.items())),
        truncated=truncated,
    )


def parse_run_manifest(text: str) -> RunManifest:
    """Parse manifest JSON; raise ``RunManifestError`` on any defect."""
    try:
        return RunManifest.model_validate_json(text)
    except ValidationError as exc:
        raise RunManifestError(f"{RUN_MANIFEST_NAME} is invalid: {exc}") from exc
