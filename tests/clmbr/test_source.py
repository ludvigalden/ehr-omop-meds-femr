"""Pinned acquisition routes: one pin, hash checks, unreachable download (exit-2 verdicts)."""

from __future__ import annotations

from pathlib import Path

import pytest

from omopmap.clmbr.source import (
    Downloader,
    SourceError,
    load_pins,
    resolve_hf,
    resolve_local,
)
from tests.clmbr.dictionaries import REVISION, pins_for, write_layouts


def _returning(path: str) -> Downloader:
    def download(*, repo_id: str, filename: str, revision: str, token: str | bool) -> str:
        return path

    return download


def test_packaged_pins_are_complete() -> None:
    pins = load_pins()
    assert pins.repo == "StanfordShahLab/clmbr-t-base"
    assert pins.filename == "dictionary.msgpack"
    assert len(pins.revision) == 40 and len(pins.sha256) == 64
    assert len(pins.concept_list_sha256) == 64


def test_local_and_hf_routes_check_the_identical_pin(tmp_path: Path) -> None:
    path = write_layouts(tmp_path / "model")["flat"]
    pins = pins_for(path)
    local = resolve_local(tmp_path / "model", pins)
    hf = resolve_hf(pins.repo, None, pins, _returning(str(path)))
    assert (local.sha256, local.revision, local.filename, local.repo) == (
        hf.sha256,
        hf.revision,
        hf.filename,
        hf.repo,
    )
    assert local.sha256 == pins.sha256 and local.revision == pins.revision
    changed = tmp_path / "changed" / "dictionary.msgpack"
    changed.parent.mkdir()
    changed.write_bytes(path.read_bytes() + b"\x00")
    with pytest.raises(SourceError, match="differs from the pinned"):
        resolve_local(changed.parent, pins)
    with pytest.raises(SourceError, match="differs from the pinned"):
        resolve_hf(pins.repo, None, pins, _returning(str(changed)))


def test_local_source_missing_file_or_dir(tmp_path: Path) -> None:
    path = write_layouts(tmp_path / "m")["flat"]
    pins = pins_for(path)
    with pytest.raises(SourceError, match="does not exist"):
        resolve_local(tmp_path / "absent", pins)
    path.unlink()
    with pytest.raises(SourceError, match="does not exist"):
        resolve_local(tmp_path / "m", pins)


def test_hf_source_downloads_pinned_revision_with_env_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write_layouts(tmp_path)["flat"]
    seen: dict[str, object] = {}

    def downloader(*, repo_id: str, filename: str, revision: str, token: str | bool) -> str:
        seen.update(repo_id=repo_id, filename=filename, revision=revision, token=token)
        return str(path)

    monkeypatch.setenv("HF_TOKEN", "invented-test-token")
    pins = pins_for(path)
    resolved = resolve_hf(pins.repo, REVISION, pins, downloader)
    assert resolved.kind == "hf" and resolved.revision == REVISION
    assert seen == {
        "repo_id": pins.repo,
        "filename": "dictionary.msgpack",
        "revision": REVISION,
        "token": "invented-test-token",
    }


def test_hf_without_token_passes_false(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = write_layouts(tmp_path)["flat"]
    monkeypatch.delenv("HF_TOKEN", raising=False)
    tokens: list[str | bool] = []

    def downloader(*, repo_id: str, filename: str, revision: str, token: str | bool) -> str:
        tokens.append(token)
        return str(path)

    resolve_hf("demo-org/demo-model", None, pins_for(path), downloader)
    assert tokens == [False]


def test_hf_source_unreachable_never_echoes_the_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "invented-secret-token-value"
    monkeypatch.setenv("HF_TOKEN", secret)

    def offline(*, repo_id: str, filename: str, revision: str, token: str | bool) -> str:
        raise ConnectionError(f"network is unreachable (Authorization: Bearer {token})")

    pins = pins_for(write_layouts(tmp_path)["flat"])
    with pytest.raises(SourceError, match="cannot download .*ConnectionError") as info:
        resolve_hf(pins.repo, None, pins, offline)
    assert secret not in str(info.value) and "<redacted>" in str(info.value)


def test_hf_source_rejects_other_repo_or_revision(tmp_path: Path) -> None:
    pins = pins_for(write_layouts(tmp_path)["flat"])
    with pytest.raises(SourceError, match="differs from the pinned"):
        resolve_hf("someone/else", None, pins, _returning(""))
    with pytest.raises(SourceError, match="revision .* differs"):
        resolve_hf(pins.repo, "f" * 40, pins, _returning(""))


def test_malformed_or_placeholder_pins_file_is_refused(tmp_path: Path) -> None:
    bad = tmp_path / "pins.toml"
    bad.write_text('schema_version = 1\nrepo = "x"\n')
    with pytest.raises(SourceError, match="cannot read the dictionary pins"):
        load_pins(bad)
    placeholder = tmp_path / "placeholder.toml"
    placeholder.write_text(
        'schema_version = 1\nrepo = "x"\nlicence = "y"\nfilename = "dictionary.msgpack"\n'
        f'revision = "{"0" * 40}"\nsha256 = "TBD"\nconcept_list_sha256 = "{"0" * 64}"\n'
    )
    with pytest.raises(SourceError, match="cannot read the dictionary pins"):
        load_pins(placeholder)
