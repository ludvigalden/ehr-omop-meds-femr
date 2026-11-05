"""Encoder protocol: pins, normalisation, output checks, lazy optional dependency."""

from __future__ import annotations

import numpy as np
import pytest

from omopmap.retrieval.embed import (
    BIOLORD,
    MODELS,
    SAPBERT,
    Encoded,
    Encoder,
    EncoderError,
    ModelSpec,
    TransformerEncoder,
    check_encoded,
    l2_normalise,
)
from tests.retrieval.conftest import FakeEncoder


def test_pins_are_full_revisions_with_their_pooling() -> None:
    assert SAPBERT.model_id == "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR"
    assert SAPBERT.revision == "47b6bd041ba61311584bb2494edfda5c7d9b719f"
    assert SAPBERT.pooling == "cls"
    assert BIOLORD.model_id == "FremyCompany/BioLORD-2023-M"
    assert BIOLORD.revision == "4ea2ea2c89ef63365f7fcd91406a0cd8ac36d2e2"
    assert BIOLORD.pooling == "mean"
    assert set(MODELS) == {"sapbert", "biolord"}
    with pytest.raises(ValueError, match="40-hex"):
        ModelSpec("x", "org/x", "47b6bd04", "cls")


def test_fake_encoder_satisfies_the_protocol() -> None:
    encoder: Encoder = FakeEncoder()
    out = encoder.encode(["feber", "hosta", "feber"])
    assert out.vectors.shape == (3, 20) and out.vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(out.vectors, axis=1), 1.0)
    assert np.array_equal(out.vectors[0], out.vectors[2])


def test_l2_normalise_keeps_zero_rows() -> None:
    out = l2_normalise(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert out.dtype == np.float32
    assert np.allclose(out, [[0.6, 0.8], [0.0, 0.0]])


def test_check_encoded_fails_closed() -> None:
    good = Encoded(vectors=np.zeros((2, 4), dtype=np.float32), truncated=0)
    assert check_encoded(good, 2, 4) is good
    with pytest.raises(EncoderError, match="shape"):
        check_encoded(good, 3, 4)
    bad = Encoded(vectors=np.full((1, 4), np.nan, dtype=np.float32), truncated=0)
    with pytest.raises(EncoderError, match="non-finite"):
        check_encoded(bad, 1, 4)


def test_constructing_a_transformer_encoder_loads_nothing() -> None:
    encoder = TransformerEncoder(SAPBERT)
    assert encoder.max_length == 128
    assert repr(encoder).startswith("TransformerEncoder(")
