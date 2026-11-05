"""Deterministic hashing encoder used in place of a real text encoder.

Implements the minimal ``encode(texts) -> np.ndarray`` contract the
retrieval modules expect (L2-normalised float32, one row per input),
with no ML dependencies: each dimension is derived from a seeded
BLAKE2b hash of the text, so identical inputs always produce identical
vectors on any machine — the property hermetic tests need from a
stand-in encoder.
"""

from __future__ import annotations

import hashlib

import numpy as np

_DIMENSIONS = 20
_SEED = b"omopmap-fake-encoder-v1"


def _hash_dimension(text: str, dimension: int) -> float:
    """Map (text, dimension) deterministically into [-1, 1)."""
    digest = hashlib.blake2b(
        f"{dimension}:{text}".encode(),
        digest_size=8,
        key=_SEED,
    ).digest()
    value = int.from_bytes(digest, "little")
    return (value / 2**64) * 2.0 - 1.0


def encode(texts: list[str]) -> np.ndarray:
    """Encode ``texts`` into an (n, 20) L2-normalised float32 matrix."""
    vectors = np.empty((len(texts), _DIMENSIONS), dtype=np.float32)
    for row, text in enumerate(texts):
        for dimension in range(_DIMENSIONS):
            vectors[row, dimension] = _hash_dimension(text, dimension)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # A zero vector is practically impossible (all 20 hashes zero) but
    # the fallback stays deterministic rather than dividing by zero.
    norms[norms == 0.0] = 1.0
    vectors /= norms
    return vectors.astype(np.float32, copy=False)
