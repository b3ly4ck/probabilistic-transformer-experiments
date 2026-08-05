"""Check 1 -- every intermediate tensor has the dimensionality stated in Part III."""

import torch

from src import mfvi
from tests.conftest import BATCH, SEQ, TOY


def test_content_stream_shape(model, tokens):
    qbar = model.content_stream(tokens)
    assert qbar.shape == (BATCH, SEQ, TOY.d)


def test_contracted_prefix_shape(model, tokens):
    qbar = model.content_stream(tokens)
    Bkey = mfvi.contract_prefix(qbar, model.T, model.r)
    # key axis is |D_t| at its widest: ROOT plus every prefix position
    assert Bkey.shape == (BATCH, TOY.n_channels, SEQ + 1, TOY.d)


def test_root_row_is_r(model, tokens):
    qbar = model.content_stream(tokens)
    Bkey = mfvi.contract_prefix(qbar, model.T, model.r)
    for batch in range(BATCH):
        assert torch.equal(Bkey[batch, :, 0, :], model.r)


def test_logits_shape(model, tokens):
    for readout in ("exact", "mfvi"):
        logits = model(tokens, readout=readout)
        assert logits.shape == (BATCH, SEQ, TOY.vocab_size)
        assert torch.isfinite(logits).all()


def test_parameter_list_is_the_factor_list(model):
    """§18 Check 1: no matrix in the computation that is not a factor."""
    names = {n for n, _ in model.named_parameters()}
    assert names == {"S", "T", "r", "b"}
