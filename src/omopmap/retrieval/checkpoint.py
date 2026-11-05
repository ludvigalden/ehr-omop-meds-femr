"""Opening a fine-tuned encoder checkpoint only if it still matches its run manifest.

``omopmap finetune`` (full mode) writes ``checkpoint/`` (model and tokenizer in
Hugging Face format) and, last, ``run_manifest.json`` recording the SHA-256 of
every checkpoint file and the base model it was tuned from. A fine-tuned
encoder is identified by those bytes, not by a hub revision: two checkpoints
tuned from the same base with different data share the base's model id and
revision but are different encoders. So before any weight is loaded,
``open_checkpoint`` checks that

- the directory holds a ``finetune`` run manifest from a full (not dry-run)
  run, with an encoder record and a known model key;
- the files under ``checkpoint/`` are exactly the recorded ones (none missing,
  none added);
- every file hashes to its recorded SHA-256.

Any difference raises ``CheckpointHashError`` (exit 4 in the CLI, like a
frozen index whose matrix no longer matches); a missing or unusable manifest
raises ``CheckpointError`` (exit 1). The checkpoint's identity,
``checkpoint_sha256``, is the SHA-256 of its sorted ``<path>\\t<sha256>\\n``
listing; run manifests record it next to the base model id and revision, so an
index built with one checkpoint cannot be queried with another.

Everything here reads local files only; nothing touches the network.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from omopmap.artifacts.manifest import sha256_file
from omopmap.retrieval.embed import MODELS, ModelSpec
from omopmap.retrieval.manifest import (
    RUN_MANIFEST_NAME,
    EncoderRecord,
    RunManifestError,
    parse_run_manifest,
)

CHECKPOINT_DIR = "checkpoint"
_PREFIX = CHECKPOINT_DIR + "/"


class CheckpointError(ValueError):
    """A model directory is not a usable fine-tuned checkpoint."""


class CheckpointHashError(CheckpointError):
    """The checkpoint's files differ from what its run manifest recorded."""


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """A verified fine-tuned checkpoint and the base model it was tuned from."""

    path: Path
    spec: ModelSpec
    base: EncoderRecord
    files: tuple[tuple[str, str], ...]
    checkpoint_sha256: str

    @property
    def model_label(self) -> str:
        """The ``model`` parameter runs record: base key plus a checkpoint prefix."""
        return f"{self.spec.key}+ft:{self.checkpoint_sha256[:12]}"


def checkpoint_digest(files: Mapping[str, str]) -> str:
    """SHA-256 of the sorted ``<relative path>\\t<sha256>\\n`` listing of a checkpoint."""
    if not files:
        raise CheckpointError("a checkpoint needs at least one file")
    listing = "".join(f"{name}\t{files[name]}\n" for name in sorted(files))
    return hashlib.sha256(listing.encode("utf-8")).hexdigest()


def open_checkpoint(model_dir: Path) -> Checkpoint:
    """Verify a ``finetune`` output directory and describe its checkpoint (no weights loaded)."""
    manifest_path = model_dir / RUN_MANIFEST_NAME
    if not manifest_path.is_file():
        raise CheckpointError(f"{model_dir} has no {RUN_MANIFEST_NAME}")
    try:
        manifest = parse_run_manifest(manifest_path.read_text(encoding="utf-8"))
    except RunManifestError as exc:
        raise CheckpointError(str(exc)) from exc
    if manifest.run_type != "finetune":
        raise CheckpointError(f"{model_dir} is a {manifest.run_type} run, not a finetune run")
    if manifest.parameters.get("dry_run") is not False or manifest.encoder is None:
        raise CheckpointError(f"{model_dir} is a dry run; it has no checkpoint to load")
    key = manifest.parameters.get("model")
    if not isinstance(key, str) or key not in MODELS:
        raise CheckpointError(f"{model_dir}: the manifest records no known `model` key")
    spec = MODELS[key]
    if (manifest.encoder.model_id, manifest.encoder.revision) != (spec.model_id, spec.revision):
        raise CheckpointError(
            f"{model_dir}: the recorded base encoder is not the pinned {spec.model_id}"
        )
    recorded = {
        record.name: record.sha256 for record in manifest.outputs if record.name.startswith(_PREFIX)
    }
    if not recorded:
        raise CheckpointError(f"{model_dir}: the manifest records no {CHECKPOINT_DIR}/ files")
    root = model_dir / CHECKPOINT_DIR
    on_disk = (
        {p.relative_to(model_dir).as_posix() for p in root.rglob("*") if not p.is_dir()}
        if root.is_dir()
        else set[str]()
    )
    missing = sorted(set(recorded) - on_disk)
    extra = sorted(on_disk - set(recorded))
    if missing or extra:
        raise CheckpointHashError(
            f"{model_dir}: checkpoint files differ from the manifest "
            f"(missing {missing[:5]}, unrecorded {extra[:5]})"
        )
    for name in sorted(recorded):
        path = model_dir / name
        if path.is_symlink():
            raise CheckpointHashError(f"{path} is a symlink; checkpoint files must be regular")
        digest = sha256_file(path)
        if digest != recorded[name]:
            raise CheckpointHashError(
                f"{path}: SHA-256 {digest} differs from the recorded {recorded[name]}"
            )
    return Checkpoint(
        path=root,
        spec=spec,
        base=manifest.encoder,
        files=tuple(sorted(recorded.items())),
        checkpoint_sha256=checkpoint_digest(recorded),
    )
