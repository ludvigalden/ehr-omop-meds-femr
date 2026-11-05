"""``blind_order``: sorted, then shuffled by a per-case seed string."""

from __future__ import annotations

from omopmap.adjudication import blind_order, shuffle_key

IDS = list(range(1, 11))


def test_order_is_deterministic() -> None:
    """Same inputs give the same shuffled order."""
    first = blind_order(IDS, seed=1, system="DEMOSYS", code="S001")
    second = blind_order(IDS, seed=1, system="DEMOSYS", code="S001")
    assert first == second
    assert sorted(first) == sorted(IDS)


def test_input_order_and_duplicates_are_irrelevant() -> None:
    """Shuffled or repeated inputs produce the identical order."""
    baseline = blind_order(IDS, seed=4, system="DEMOSYS", code="S001")
    scrambled = list(reversed(IDS)) + IDS[:3]
    assert blind_order(scrambled, seed=4, system="DEMOSYS", code="S001") == baseline


def test_a_different_seed_shuffles_differently() -> None:
    """Two seeds give different orders over ten ids."""
    first = blind_order(IDS, seed=1, system="DEMOSYS", code="S001")
    second = blind_order(IDS, seed=2, system="DEMOSYS", code="S001")
    assert first != second


def test_a_different_system_or_code_shuffles_differently() -> None:
    """The shuffle key includes the case's system and code."""
    baseline = blind_order(IDS, seed=1, system="DEMOSYS", code="S001")
    assert blind_order(IDS, seed=1, system="DEMOSYS", code="S002") != baseline
    assert blind_order(IDS, seed=1, system="DEMOSYSX", code="S001") != baseline


def test_shuffle_key_format() -> None:
    """The key is exactly ``seed|system|code``."""
    assert shuffle_key(7, "DEMOSYS", "S001") == "7|DEMOSYS|S001"
