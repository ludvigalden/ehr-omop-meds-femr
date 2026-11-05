"""Synthetic CLMBR dictionaries in the three accepted layouts, invented codes only.

Nothing here comes from a real dictionary: vocabulary ids are ``DEMO*`` (one
with a space, as real ids can have), codes are made up, and ``CARE_SITE`` ids
are invented numbers.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import cast

from omopmap.clmbr.source import Pins

# (code_string, type, property-or-None, text_string)
Entry = tuple[str, str, str | None, str]

ENTRIES: tuple[Entry, ...] = (
    ("DEMOCOND/C001", "code", None, ""),
    ("DEMOCOND/C002", "code", None, ""),
    ("DEMOCOND/C001", "code", None, ""),  # duplicate entry, same code
    ("DEMOLAB/L01", "numeric", "DEMOLAB/L01", ""),
    ("DEMOLAB/L01", "numeric", "DEMOLAB/L01", ""),  # second value bin
    ("DEMOLAB/L01", "text", "DEMOLAB/L01", "high"),
    ("DEMOLAB/L02", "text", "DEMOLAB/L02", "low"),
    ("DEMO Extension/X/9", "code", None, ""),  # space in the id, slash in the code
    ("CARE_SITE/900001", "code", None, ""),
    ("CARE_SITE/900002", "code", None, ""),
    ("DEMOCOND/C999", "unused", None, ""),
    ("DEMODRUG/D999", "unused", None, ""),
)

#: Expected concept set (CARE_SITE and unused excluded), with entry types.
EXPECTED: dict[str, tuple[str, ...]] = {
    "DEMO Extension/X/9": ("code",),
    "DEMOCOND/C001": ("code",),
    "DEMOCOND/C002": ("code",),
    "DEMOLAB/L01": ("numeric", "text"),
    "DEMOLAB/L02": ("text",),
}
EXPECTED_ENTRY_TYPES = {"code": 6, "numeric": 2, "text": 2, "unused": 2}


def _entry(
    code: str, kind: str, prop: str | None, text: str, hierarchical: bool, i: int
) -> dict[str, object]:
    entry: dict[str, object] = {
        "code_string": code,
        "text_string": text,
        "type": kind,
        "val_start": float(i),
        "val_end": float(i + 1),
        "weight": -0.5 + i / 100,
    }
    if hierarchical and prop is not None:
        entry["property"] = prop
    return entry


def dictionary_object(
    entries: Iterable[Entry] = ENTRIES, *, hierarchical: bool = True, key: str = "vocab"
) -> dict[str, object]:
    return {
        "age_stats": {"mean": 40.0, "std": 12.5},
        "is_hierarchical": hierarchical,
        key: [_entry(*e, hierarchical=hierarchical, i=i) for i, e in enumerate(entries)],
    }


def pack(obj: object) -> bytes:
    packb = cast("Callable[..., bytes]", importlib.import_module("msgpack").packb)
    return packb(obj, use_bin_type=True)


def write_msgpack(path: Path, obj: object) -> Path:
    path.write_bytes(pack(obj))
    return path


def write_layouts(directory: Path, entries: Iterable[Entry] = ENTRIES) -> dict[str, Path]:
    """Write the three accepted shapes of the same dictionary.

    ``with_property``: msgpack, ``is_hierarchical`` true; ``flat``: msgpack,
    ``is_hierarchical`` false (written as ``dictionary.msgpack``, the shape the
    import reads); ``json``: entries under ``regular``.
    """
    entries = tuple(entries)
    directory.mkdir(parents=True, exist_ok=True)
    return {
        "flat": write_msgpack(
            directory / "dictionary.msgpack", dictionary_object(entries, hierarchical=False)
        ),
        "with_property": write_msgpack(
            directory / "dictionary_with_property.msgpack",
            dictionary_object(entries, hierarchical=True),
        ),
        "json": _write_json(
            directory / "dictionary.json",
            dictionary_object(entries, hierarchical=False, key="regular"),
        ),
    }


def _write_json(path: Path, obj: object) -> Path:
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


REVISION = "0123456789abcdef0123456789abcdef01234567"


def pins_for(
    path: Path,
    *,
    concept_list: str | None = None,
    revision: str = REVISION,
    sha: str | None = None,
) -> Pins:
    """Pins that accept the given synthetic dictionary file (one pin for both routes)."""
    from omopmap.clmbr.importer import build_concept_set
    from omopmap.clmbr.model import load_dictionary

    return Pins(
        schema_version=1,
        repo="demo-org/demo-model",
        licence="invented test licence",
        filename="dictionary.msgpack",
        revision=revision,
        sha256=sha or sha256(path),
        concept_list_sha256=concept_list
        or build_concept_set(load_dictionary(path)).concept_list_sha256(),
    )
