"""Frozen-matrix retrieval: hash-asserted open, id-map pairing, top-k order and ties."""

from __future__ import annotations

import numpy as np
import pytest

from omopmap.retrieval.embed import l2_normalise
from omopmap.retrieval.retrieve import (
    FrozenIndexError,
    IndexHashError,
    encode_ids,
    encode_matrix,
    open_index,
    sha256_bytes,
    top_k,
)


def _blobs() -> tuple[bytes, bytes]:
    matrix = l2_normalise(np.array([[1, 0], [0, 1], [1, 1], [1, 0]], dtype=np.float32))
    return encode_matrix(matrix), encode_ids([40, 10, 30, 20])


def _open(matrix: bytes, ids: bytes, **override: str):  # type: ignore[no-untyped-def]
    digests = {
        "expected_matrix_sha256": sha256_bytes(matrix),
        "expected_ids_sha256": sha256_bytes(ids),
    }
    digests.update(override)
    return open_index(matrix, ids, **digests)


def test_top_k_orders_by_score_then_concept_id() -> None:
    index = _open(*_blobs())
    [hit] = top_k(index, l2_normalise(np.array([[1.0, 0.0]], dtype=np.float32)), k=3)
    # Rows 0 and 3 are identical (ids 40 and 20): the tie breaks on the id.
    assert hit.ids == (20, 40, 30)
    assert hit.scores[0] == pytest.approx(1.0) and hit.scores[2] == pytest.approx(2**-0.5)


def test_tie_at_the_cut_is_resolved_by_id_not_partition_order() -> None:
    index = _open(*_blobs())
    [hit] = top_k(index, l2_normalise(np.array([[1.0, 0.0]], dtype=np.float32)), k=1)
    assert hit.ids == (20,)


def test_chunking_does_not_change_results() -> None:
    index = _open(*_blobs())
    queries = l2_normalise(np.random.default_rng(3).normal(size=(9, 2)).astype(np.float32))
    assert top_k(index, queries, k=4, chunk=2) == top_k(index, queries, k=4, chunk=1024)


def test_matrix_hash_mismatch_raises_before_decode() -> None:
    matrix, ids = _blobs()
    with pytest.raises(IndexHashError, match="matrix.npy"):
        _open(matrix, ids, expected_matrix_sha256="0" * 64)


def test_id_map_hash_mismatch_raises() -> None:
    matrix, ids = _blobs()
    reordered = encode_ids([10, 40, 30, 20])
    with pytest.raises(IndexHashError, match="ids.txt"):
        _open(matrix, reordered, expected_ids_sha256=sha256_bytes(ids))


@pytest.mark.parametrize(
    ("ids", "match"),
    [
        (b"40\n10\n30\n", "matrix rows"),
        (b"40\n10\n30\n20", "newline"),
        (b"40\nten\n30\n20\n", "non-integer"),
        (b"40\n40\n30\n20\n", "repeats"),
    ],
)
def test_malformed_id_map_fails_closed(ids: bytes, match: str) -> None:
    matrix, _ = _blobs()
    with pytest.raises(FrozenIndexError, match=match):
        _open(matrix, ids)


def test_unnormalised_matrix_and_wrong_query_dim_fail() -> None:
    raw = encode_matrix(np.array([[2.0, 0.0]], dtype=np.float32))
    with pytest.raises(FrozenIndexError, match="L2"):
        _open(raw, encode_ids([1]))
    index = _open(*_blobs())
    with pytest.raises(FrozenIndexError, match="dim"):
        top_k(index, np.zeros((1, 3), dtype=np.float32), k=1)
    with pytest.raises(FrozenIndexError):
        encode_matrix(np.zeros((2, 2), dtype=np.float64))  # type: ignore[arg-type]
