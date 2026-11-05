"""Resolve the pinned CLMBR dictionary file and check its SHA-256.

Both acquisition routes yield the same pinned bytes:

* ``local``: ``<model_dir>/<filename>``, a copy of the release's dictionary on
  disk. No network access. This is the route for an offline or sensitive-data
  environment: copy the file in, and the hash check proves it is the pinned
  one.
* ``hf``: ``hf_hub_download`` of the pinned revision of the release. The model
  is gated, so the download uses ``HF_TOKEN`` from the environment; the token is
  passed through, redacted from every error message, and never written
  anywhere. Needs the ``hf`` extra.

Every failure to obtain exactly the pinned bytes raises ``SourceError``: a hash
mismatch, a repo or revision that differs from the pin, a missing file, an
unreachable or refused download. There is no fallback from one route to the
other and no unchecked path.
"""

from __future__ import annotations

import importlib
import os
import tomllib
from collections.abc import Callable
from importlib import resources
from pathlib import Path
from typing import Literal, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from omopmap.artifacts.manifest import sha256_file

SourceKind = Literal["local", "hf"]
_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class SourceError(Exception):
    """The pinned dictionary could not be obtained or did not match its pin."""


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class Pins(_Frozen):
    """``pins.toml``: the release, the commit, the file and its hashes."""

    schema_version: Literal[1]
    repo: str = Field(min_length=1)
    licence: str = Field(min_length=1)
    filename: str = Field(min_length=1, pattern=r"^[^/\\]+$")
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    sha256: str = Field(pattern=_SHA256_PATTERN)
    concept_list_sha256: str = Field(pattern=_SHA256_PATTERN)


class ResolvedDictionary(_Frozen):
    """A dictionary file whose bytes equal the pin."""

    path: Path
    kind: SourceKind
    repo: str
    revision: str
    filename: str
    sha256: str


def load_pins(path: Path | None = None) -> Pins:
    """Parse ``pins.toml`` (the packaged one unless ``path`` is given); fail closed."""
    try:
        if path is None:
            text = resources.files("omopmap.clmbr").joinpath("pins.toml").read_text("utf-8")
        else:
            text = path.read_text(encoding="utf-8")
        return Pins.model_validate(tomllib.loads(text))
    except (OSError, tomllib.TOMLDecodeError, ValidationError) as exc:
        raise SourceError(f"cannot read the dictionary pins: {exc}") from exc


def check_hash(path: Path, pins: Pins, kind: SourceKind) -> str:
    """Return the file's SHA-256 if it equals the pin; raise otherwise."""
    if not path.is_file():
        raise SourceError(f"dictionary file {path} does not exist")
    actual = sha256_file(path)
    if actual != pins.sha256:
        raise SourceError(
            f"{path.name}: SHA-256 {actual} differs from the pinned {pins.sha256}; "
            f"refusing to import anything but the pinned file via {kind}"
        )
    return actual


def resolve_local(model_dir: Path, pins: Pins) -> ResolvedDictionary:
    """The pinned dictionary from a local directory (no network)."""
    if not model_dir.is_dir():
        raise SourceError(f"model directory {model_dir} does not exist")
    path = model_dir / pins.filename
    digest = check_hash(path, pins, "local")
    return ResolvedDictionary(
        path=path,
        kind="local",
        repo=pins.repo,
        revision=pins.revision,
        filename=pins.filename,
        sha256=digest,
    )


class Downloader(Protocol):
    """``hf_hub_download``'s shape, as used here (injectable for tests)."""

    def __call__(self, *, repo_id: str, filename: str, revision: str, token: str | bool) -> str: ...


def _hf_downloader() -> Downloader:
    # Imported by name so that the module, and type checking, need no
    # optional extra; the call shape is pinned by ``Downloader``.
    try:
        hub = importlib.import_module("huggingface_hub")
    except ImportError as exc:
        raise SourceError(
            "the hf source needs huggingface_hub; install omopmap with the 'hf' extra"
        ) from exc
    hf_hub_download = cast("Callable[..., object]", hub.hf_hub_download)

    def download(*, repo_id: str, filename: str, revision: str, token: str | bool) -> str:
        path = hf_hub_download(repo_id=repo_id, filename=filename, revision=revision, token=token)
        return str(path)

    return download


def resolve_hf(
    repo: str,
    revision: str | None,
    pins: Pins,
    downloader: Downloader | None = None,
) -> ResolvedDictionary:
    """The pinned dictionary from the Hugging Face release.

    ``revision`` defaults to the pin and must equal it when given.
    """
    if repo != pins.repo:
        raise SourceError(f"repo {repo!r} differs from the pinned {pins.repo!r}")
    if revision is not None and revision != pins.revision:
        raise SourceError(f"revision {revision} differs from the pinned {pins.revision}")
    download = downloader or _hf_downloader()
    token: str | bool = os.environ.get("HF_TOKEN") or False
    try:
        local = Path(
            download(repo_id=repo, filename=pins.filename, revision=pins.revision, token=token)
        )
    except Exception as exc:  # every download failure is the same verdict: unreachable
        detail = str(exc)
        if isinstance(token, str) and token:
            detail = detail.replace(token, "<redacted>")
        raise SourceError(
            f"cannot download {pins.filename} from {repo}@{pins.revision}: "
            f"{type(exc).__name__}: {detail}"
        ) from exc
    digest = check_hash(local, pins, "hf")
    return ResolvedDictionary(
        path=local,
        kind="hf",
        repo=repo,
        revision=pins.revision,
        filename=pins.filename,
        sha256=digest,
    )
