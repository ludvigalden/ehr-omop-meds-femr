"""Loader: three layouts, strict keys, typed entries."""

from __future__ import annotations

from pathlib import Path

import pytest

from omopmap.clmbr.importer import build_concept_set
from omopmap.clmbr.model import (
    ConceptRef,
    DictionaryFormatError,
    load_dictionary,
    parse_dictionary,
)
from tests.clmbr.dictionaries import (
    ENTRIES,
    EXPECTED,
    EXPECTED_ENTRY_TYPES,
    dictionary_object,
    pack,
    write_layouts,
    write_msgpack,
)


def test_three_layouts_give_the_same_concept_set(tmp_path: Path) -> None:
    paths = write_layouts(tmp_path)
    sets = {name: build_concept_set(load_dictionary(p)) for name, p in paths.items()}
    for name, concept_set in sets.items():
        assert {c.code: c.entry_types for c in concept_set.concepts} == EXPECTED, name
        assert concept_set.entry_type_counts == EXPECTED_ENTRY_TYPES, name
        assert concept_set.care_site_excluded == 2, name
    hashes = {s.concept_list_sha256() for s in sets.values()}
    assert len(hashes) == 1


def test_layout_flags_and_properties(tmp_path: Path) -> None:
    paths = write_layouts(tmp_path)
    with_property = load_dictionary(paths["with_property"])
    flat = load_dictionary(paths["flat"])
    assert with_property.is_hierarchical and not flat.is_hierarchical
    assert len(with_property.entries) == len(ENTRIES)
    assert [e.index for e in with_property.entries] == list(range(len(ENTRIES)))
    assert with_property.entries[3].property == "DEMOLAB/L01"
    assert all(e.property is None for e in flat.entries)
    assert with_property.age_stats.mean == 40.0


def test_unknown_top_level_key_is_rejected() -> None:
    obj = dictionary_object()
    obj["extra"] = 1
    with pytest.raises(DictionaryFormatError, match="unknown keys"):
        parse_dictionary(pack(obj), layout="msgpack")


def test_unknown_entry_key_and_type_are_rejected(tmp_path: Path) -> None:
    obj = dictionary_object()
    entries = obj["vocab"]
    assert isinstance(entries, list)
    bad_key = dictionary_object()
    bad_entries = bad_key["vocab"]
    assert isinstance(bad_entries, list)
    bad_entries[0]["surprise"] = True  # pyright: ignore[reportIndexIssue]
    with pytest.raises(DictionaryFormatError, match="unknown keys"):
        load_dictionary(write_msgpack(tmp_path / "a.msgpack", bad_key))
    bad_type = dictionary_object([("DEMOCOND/C1", "mystery", None, "")])
    with pytest.raises(DictionaryFormatError, match="unknown entry type"):
        load_dictionary(write_msgpack(tmp_path / "b.msgpack", bad_type))


def test_json_layout_needs_regular_key(tmp_path: Path) -> None:
    path = tmp_path / "d.json"
    path.write_text('{"age_stats": {"mean": 1, "std": 1}, "is_hierarchical": false, "vocab": []}')
    with pytest.raises(DictionaryFormatError):
        load_dictionary(path)


def test_garbage_and_wrong_suffix_fail_closed(tmp_path: Path) -> None:
    junk = tmp_path / "dictionary.msgpack"
    junk.write_bytes(b"\xc1\xc1\xc1")
    with pytest.raises(DictionaryFormatError):
        load_dictionary(junk)
    other = tmp_path / "dictionary.bin"
    other.write_bytes(b"")
    with pytest.raises(DictionaryFormatError, match="expected a .msgpack or .json"):
        load_dictionary(other)


def test_malformed_code_string_fails_on_import(tmp_path: Path) -> None:
    path = write_msgpack(tmp_path / "d.msgpack", dictionary_object([("NOSLASH", "code", None, "")]))
    with pytest.raises(DictionaryFormatError, match="not a <vocabulary_id>"):
        build_concept_set(load_dictionary(path))


def test_concept_ref_splits_on_first_slash() -> None:
    ref = ConceptRef.parse("DEMO Extension/X/9")
    assert (ref.vocabulary_id, ref.concept_code, ref.code) == (
        "DEMO Extension",
        "X/9",
        "DEMO Extension/X/9",
    )
