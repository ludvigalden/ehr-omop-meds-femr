"""Top-k retrieval over a frozen embedding matrix with a hash-asserted id map.

A frozen index is two byte blobs:
- ``matrix.npy``: an ``(n, dim)`` float32 array of L2-normalised label vectors;
- ``ids.txt``: the concept id of each matrix row, one decimal integer per line.

Row ``i`` of the matrix belongs to line ``i`` of the id map. That pairing is the
index's weak point: re-sorting, filtering or re-embedding one side without the
other gives plausible-looking but wrong results, and nothing in the numbers
shows it. So the index is opened only against expected SHA-256 digests of both
blobs (normally from the run manifest written when the index was built), and
any difference raises ``IndexHashError`` before a single score is computed.
Embedding a large dictionary is the expensive step, so a verified frozen matrix
also lets every later query batch reuse it byte for byte.

All functions here are pure: callers pass bytes in and get bytes or values out.
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from omopmap.retrieval.embed import Matrix
from omopmap.retrieval.fusion import ScoredRanking

EMBED_CHANNEL = "embed"
MATRIX_NAME = "matrix.npy"
IDS_NAME = "ids.txt"
_NORM_TOLERANCE = 1e-3


class FrozenIndexError(ValueError):
    """A frozen index is malformed or inconsistent."""


class IndexHashError(FrozenIndexError):
    """A frozen-index blob does not match its expected SHA-256."""


def sha256_bytes(data: bytes) -> str:
    """Hex SHA-256 of ``data``."""
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True, slots=True)
class FrozenIndex:
    """A verified matrix and its id map."""

    matrix: Matrix
    ids: npt.NDArray[np.int64]
    matrix_sha256: str
    ids_sha256: str

    @property
    def rows(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def dim(self) -> int:
        return int(self.matrix.shape[1])


def encode_matrix(matrix: Matrix) -> bytes:
    """``.npy`` bytes of a float32 matrix (deterministic for equal arrays)."""
    if matrix.dtype != np.float32 or matrix.ndim != 2:
        raise FrozenIndexError(f"matrix must be 2-D float32, got {matrix.ndim}-D {matrix.dtype}")
    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(matrix), allow_pickle=False)
    return buffer.getvalue()


def encode_ids(ids: Sequence[int]) -> bytes:
    """Id-map bytes: one decimal id per line, each line ending in ``\\n``."""
    if len(set(ids)) != len(ids):
        raise FrozenIndexError("the id map repeats a concept id")
    return "".join(f"{int(i)}\n" for i in ids).encode("ascii")


def _decode_ids(data: bytes) -> npt.NDArray[np.int64]:
    text = data.decode("ascii")
    if text and not text.endswith("\n"):
        raise FrozenIndexError("the id map does not end with a newline")
    try:
        values = [int(line) for line in text.splitlines()]
    except ValueError as exc:
        raise FrozenIndexError(f"the id map holds a non-integer line: {exc}") from exc
    if len(set(values)) != len(values):
        raise FrozenIndexError("the id map repeats a concept id")
    return np.asarray(values, dtype=np.int64)


def open_index(
    matrix_bytes: bytes,
    ids_bytes: bytes,
    *,
    expected_matrix_sha256: str,
    expected_ids_sha256: str,
) -> FrozenIndex:
    """Verify both blobs against their expected digests, then decode them.

    Raises ``IndexHashError`` on a digest mismatch, and ``FrozenIndexError`` if
    the decoded matrix and id map disagree in length or the rows are not unit
    vectors.
    """
    matrix_sha, ids_sha = sha256_bytes(matrix_bytes), sha256_bytes(ids_bytes)
    if matrix_sha != expected_matrix_sha256:
        raise IndexHashError(
            f"{MATRIX_NAME}: sha256 {matrix_sha} differs from the expected {expected_matrix_sha256}"
        )
    if ids_sha != expected_ids_sha256:
        raise IndexHashError(
            f"{IDS_NAME}: sha256 {ids_sha} differs from the expected {expected_ids_sha256}"
        )
    try:
        loaded = np.load(io.BytesIO(matrix_bytes), allow_pickle=False)
    except ValueError as exc:
        raise FrozenIndexError(f"{MATRIX_NAME} is not a valid .npy array: {exc}") from exc
    matrix = np.asarray(loaded)
    if matrix.dtype != np.float32 or matrix.ndim != 2:
        raise FrozenIndexError(f"{MATRIX_NAME} must be 2-D float32, got {matrix.dtype}")
    ids = _decode_ids(ids_bytes)
    if ids.shape[0] != matrix.shape[0]:
        raise FrozenIndexError(f"{matrix.shape[0]} matrix rows but {ids.shape[0]} ids")
    norms = np.linalg.norm(matrix, axis=1)
    if matrix.shape[0] and not bool(np.all(np.abs(norms - 1.0) <= _NORM_TOLERANCE)):
        raise FrozenIndexError(f"{MATRIX_NAME} rows are not L2-normalised")
    return FrozenIndex(
        matrix=matrix.astype(np.float32, copy=False),
        ids=ids,
        matrix_sha256=matrix_sha,
        ids_sha256=ids_sha,
    )


def _top_row(scores: npt.NDArray[np.float32], ids: npt.NDArray[np.int64], k: int) -> ScoredRanking:
    """Top ``k`` of one score row, score descending, then concept id ascending."""
    n = scores.shape[0]
    candidates: npt.NDArray[np.intp]
    if k < n:
        kth = np.partition(scores, n - k)[n - k]
        # Keep every candidate tied with the k-th score, so the id tie-break
        # decides among them rather than partition order.
        candidates = np.flatnonzero(scores >= kth).astype(np.intp)
    else:
        candidates = np.arange(n, dtype=np.intp)
    order: npt.NDArray[np.intp] = candidates[np.lexsort((ids[candidates], -scores[candidates]))][:k]
    return ScoredRanking(
        channel=EMBED_CHANNEL,
        ids=tuple(int(i) for i in ids[order]),
        scores=tuple(float(s) for s in scores[order]),
    )


def top_k(index: FrozenIndex, queries: Matrix, k: int, *, chunk: int = 1024) -> list[ScoredRanking]:
    """Cosine top-``k`` concepts for each (L2-normalised) query row.

    Queries are scored in chunks of ``chunk`` rows, so the full
    queries x labels score matrix is never built at once.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if queries.ndim != 2 or queries.shape[1] != index.dim:
        raise FrozenIndexError(
            f"query vectors have shape {queries.shape}; the index has dim {index.dim}"
        )
    results: list[ScoredRanking] = []
    for start in range(0, queries.shape[0], chunk):
        block = (queries[start : start + chunk].astype(np.float32) @ index.matrix.T).astype(
            np.float32
        )
        results.extend(_top_row(row, index.ids, k) for row in block)
    return results
