"""Typed CLMBR dictionary structures and the loader for its three on-disk layouts.

FEMR stores the CLMBR vocabulary as ``dictionary.msgpack``: a map with
``age_stats``, ``is_hierarchical`` and ``vocab``, a list of entries. Each entry
has a ``code_string`` (``<vocabulary_id>/<concept_code>``), a ``type``
(``code``, ``numeric``, ``text`` or ``unused``), a ``weight``, a value bin
(``val_start``, ``val_end``), a ``text_string`` and, in the hierarchical layout,
a ``property`` on ``numeric`` and ``text`` entries. Three layouts are accepted, described by shape:

* msgpack with ``is_hierarchical`` true, whose entries may carry ``property``;
* msgpack with ``is_hierarchical`` false, without ``property`` (the release
  publishes its ``dictionary.msgpack`` in this shape);
* JSON, whose entry list sits under ``regular``.

Unknown keys are rejected at every level, so a format change fails loudly.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from omopmap.artifacts.contract import is_vocabulary_code, split_code
from omopmap.artifacts.errors import ArtifactError

# msgpack ships no type information; its decoder returns untyped data, recorded
# here as ``object`` and checked field by field below.
_unpackb = cast("Callable[..., object]", importlib.import_module("msgpack").unpackb)

EntryType = Literal["code", "numeric", "text", "unused"]
ENTRY_TYPES: tuple[EntryType, ...] = ("code", "numeric", "text", "unused")

_TOP_KEYS_MSGPACK = frozenset({"age_stats", "is_hierarchical", "vocab"})
_TOP_KEYS_JSON = frozenset({"age_stats", "is_hierarchical", "regular"})
_ENTRY_KEYS = frozenset(
    {"code_string", "text_string", "type", "val_end", "val_start", "weight", "property"}
)
_REQUIRED_ENTRY_KEYS = _ENTRY_KEYS - {"property"}


class DictionaryFormatError(ArtifactError):
    """The dictionary file does not have one of the accepted layouts."""


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class ConceptRef(_Frozen):
    """A vocabulary code ``<vocabulary_id>/<concept_code>``, split on the first ``/``."""

    vocabulary_id: str = Field(min_length=1)
    concept_code: str = Field(min_length=1)

    @property
    def code(self) -> str:
        """The MEDS code string."""
        return f"{self.vocabulary_id}/{self.concept_code}"

    @classmethod
    def parse(cls, code: str) -> ConceptRef:
        """Parse a code string; fail closed on anything that is not a vocabulary code."""
        if not is_vocabulary_code(code):
            raise DictionaryFormatError(f"not a <vocabulary_id>/<concept_code> string: {code!r}")
        vocabulary_id, concept_code = split_code(code)
        return cls(vocabulary_id=vocabulary_id, concept_code=concept_code)


class AgeStats(_Frozen):
    """Age normalisation statistics stored with the dictionary (never exported)."""

    mean: float
    std: float


class ClmbrEntry(_Frozen):
    """One vocabulary entry, in dictionary order."""

    index: int = Field(ge=0)
    code_string: str
    type: EntryType
    weight: float
    val_start: float
    val_end: float
    text: str
    property: str | None = None

    def ref(self) -> ConceptRef:
        """The entry's concept reference (fails closed on a malformed code string)."""
        return ConceptRef.parse(self.code_string)


class ClmbrDictionary(_Frozen):
    """A decoded CLMBR dictionary."""

    is_hierarchical: bool
    age_stats: AgeStats
    entries: tuple[ClmbrEntry, ...]


def _require_keys(obj: Mapping[str, object], allowed: frozenset[str], what: str) -> None:
    unknown = sorted(set(obj) - allowed)
    if unknown:
        raise DictionaryFormatError(f"{what}: unknown keys {unknown}")


def _as_mapping(obj: object, what: str) -> Mapping[str, object]:
    if not isinstance(obj, dict):
        raise DictionaryFormatError(f"{what} is not a map")
    mapping = cast("dict[object, object]", obj)
    if not all(isinstance(k, str) for k in mapping):
        raise DictionaryFormatError(f"{what} has non-string keys")
    return cast("Mapping[str, object]", mapping)


def _number(value: object, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise DictionaryFormatError(f"{what} is not a number")
    return float(value)


def _entry(index: int, raw: object) -> ClmbrEntry:
    what = f"entry {index}"
    obj = _as_mapping(raw, what)
    _require_keys(obj, _ENTRY_KEYS, what)
    missing = sorted(_REQUIRED_ENTRY_KEYS - set(obj))
    if missing:
        raise DictionaryFormatError(f"{what}: missing keys {missing}")
    entry_type = obj["type"]
    if entry_type not in ENTRY_TYPES:
        raise DictionaryFormatError(f"{what}: unknown entry type {entry_type!r}")
    code_string, text, prop = obj["code_string"], obj["text_string"], obj.get("property")
    if not isinstance(code_string, str) or not isinstance(text, str):
        raise DictionaryFormatError(f"{what}: code_string and text_string must be strings")
    if prop is not None and not isinstance(prop, str):
        raise DictionaryFormatError(f"{what}: property must be a string")
    return ClmbrEntry(
        index=index,
        code_string=code_string,
        type=entry_type,
        weight=_number(obj["weight"], f"{what} weight"),
        val_start=_number(obj["val_start"], f"{what} val_start"),
        val_end=_number(obj["val_end"], f"{what} val_end"),
        text=text,
        property=prop,
    )


def dictionary_from_object(
    obj: object, *, entries_key: Literal["vocab", "regular"]
) -> ClmbrDictionary:
    """Build a ``ClmbrDictionary`` from a decoded msgpack or JSON object."""
    top = _as_mapping(obj, "dictionary")
    allowed = _TOP_KEYS_MSGPACK if entries_key == "vocab" else _TOP_KEYS_JSON
    _require_keys(top, allowed, "dictionary")
    missing = sorted(allowed - set(top))
    if missing:
        raise DictionaryFormatError(f"dictionary: missing keys {missing}")
    hierarchical = top["is_hierarchical"]
    if not isinstance(hierarchical, bool):
        raise DictionaryFormatError("dictionary: is_hierarchical must be a boolean")
    age = _as_mapping(top["age_stats"], "age_stats")
    _require_keys(age, frozenset({"mean", "std"}), "age_stats")
    raw_entries = top[entries_key]
    if not isinstance(raw_entries, list):
        raise DictionaryFormatError(f"dictionary: {entries_key} is not a list")
    entries = cast("Sequence[object]", raw_entries)
    try:
        return ClmbrDictionary(
            is_hierarchical=hierarchical,
            age_stats=AgeStats(
                mean=_number(age.get("mean"), "age_stats.mean"),
                std=_number(age.get("std"), "age_stats.std"),
            ),
            entries=tuple(_entry(i, raw) for i, raw in enumerate(entries)),
        )
    except ValidationError as exc:
        raise DictionaryFormatError(f"dictionary does not match the model: {exc}") from exc


def parse_dictionary(data: bytes, *, layout: Literal["msgpack", "json"]) -> ClmbrDictionary:
    """Decode dictionary bytes in the given layout."""
    try:
        if layout == "msgpack":
            return dictionary_from_object(_unpackb(data, raw=False), entries_key="vocab")
        obj: object = json.loads(data.decode("utf-8"))
    except (ValueError, TypeError) as exc:
        raise DictionaryFormatError(f"cannot decode the {layout} dictionary: {exc}") from exc
    return dictionary_from_object(obj, entries_key="regular")


def layout_of(path: Path) -> Literal["msgpack", "json"]:
    """The layout implied by a dictionary file's suffix."""
    match path.suffix:
        case ".msgpack":
            return "msgpack"
        case ".json":
            return "json"
        case _:
            raise DictionaryFormatError(f"{path.name}: expected a .msgpack or .json dictionary")


def load_dictionary(path: Path) -> ClmbrDictionary:
    """Read and decode a dictionary file (no hash check; see ``omopmap.clmbr.source``)."""
    return parse_dictionary(path.read_bytes(), layout=layout_of(path))
