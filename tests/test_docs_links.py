"""Hermetic link and index checks for the documentation set.

Verifies, without network access:

- every relative markdown link in ``README.md`` and
  ``context/reference/*.md`` resolves to a file that exists, and every
  link that carries a ``#fragment`` resolves to a heading in the target
  file;
- every ``context/reference`` file the README names exists;
- every file under ``context/reference/`` is named in the README, so the
  reference index cannot silently drop a document.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

import pytest

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
README: Final[Path] = REPO_ROOT / "README.md"
REFERENCE_DIR: Final[Path] = REPO_ROOT / "context" / "reference"

_MD_LINK: Final[re.Pattern[str]] = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
_FENCED_BLOCK: Final[re.Pattern[str]] = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE: Final[re.Pattern[str]] = re.compile(r"`[^`\n]*`")


def _markdown_links(path: Path) -> list[tuple[str, str]]:
    """Return (link target, source description) pairs for one file."""
    text = path.read_text(encoding="utf-8")
    text = _FENCED_BLOCK.sub("", text)
    text = _INLINE_CODE.sub("", text)
    return [(target, f"{path.name}") for target in _MD_LINK.findall(text)]


def _docs() -> list[Path]:
    return [README, *sorted(REFERENCE_DIR.glob("*.md"))]


def _github_slug(heading: str) -> str:
    slug = heading.strip().lower()
    slug = re.sub(r"[^\w\s\-]", "", slug, flags=re.UNICODE)
    return re.sub(r"\s", "-", slug)


def _headings(path: Path) -> set[str]:
    slugs: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^#{1,6}\s+(.*)$", line)
        if match:
            slugs.add(_github_slug(match.group(1)))
    return slugs


def test_docs_exist() -> None:
    assert README.is_file()
    assert REFERENCE_DIR.is_dir()
    assert REFERENCE_DIR.glob("*.md")


@pytest.mark.parametrize(
    ("doc", "target", "source"),
    [(doc, target, source) for doc in _docs() for target, source in _markdown_links(doc)],
)
def test_relative_links_resolve(doc: Path, target: str, source: str) -> None:
    if "://" in target or target.startswith("mailto:"):
        pytest.skip("absolute URL, not checked hermetically")
    stripped = target.split("#", 1)[0]
    assert ".." not in stripped or (doc.parent / stripped).resolve().is_file(), (
        f"{source}: link target {target!r} does not resolve"
    )
    resolved = (doc.parent / stripped).resolve()
    assert resolved.is_file(), f"{source}: link target {target!r} does not exist"
    fragment = target.split("#", 1)
    if len(fragment) == 2:
        assert fragment[1] in _headings(resolved), (
            f"{source}: link {target!r} names no heading in {resolved.name}"
        )


def test_readme_reference_links_all_exist() -> None:
    readme_targets = {
        target.split("#", 1)[0]
        for target, _ in _markdown_links(README)
        if "context/reference/" in target and "://" not in target
    }
    assert readme_targets, "the README must link into context/reference/"
    for rel in readme_targets:
        assert (REPO_ROOT / rel).is_file(), f"README links missing file {rel!r}"


def test_every_reference_file_is_listed_in_the_readme() -> None:
    readme_targets = {
        target.split("#", 1)[0]
        for target, _ in _markdown_links(README)
        if "context/reference/" in target and "://" not in target
    }
    on_disk = {f"context/reference/{p.name}" for p in REFERENCE_DIR.glob("*.md")}
    missing = on_disk - readme_targets
    assert not missing, f"context/reference files absent from the README: {sorted(missing)}"
