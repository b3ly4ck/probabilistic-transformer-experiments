"""Check 1 -- every intermediate tensor has the dimensionality stated in Part III."""

import torch

from src import mfvi
from tests.conftest import BATCH, SEQ


def test_content_stream_shape(model, tokens):
    qbar = model.content_stream(tokens)
    assert qbar.shape == (BATCH, SEQ, model.cfg.d)


def test_contracted_prefix_shape(model, tokens):
    qbar = model.content_stream(tokens)
    Bkey = mfvi.contract_prefix(qbar, model.T, model.r)
    # key axis is |D_t| at its widest: ROOT plus every prefix position
    assert Bkey.shape == (BATCH, model.cfg.n_channels, SEQ + 1, model.cfg.d)


def test_root_row_is_r(model, tokens):
    qbar = model.content_stream(tokens)
    Bkey = mfvi.contract_prefix(qbar, model.T, model.r)
    for batch in range(BATCH):
        assert torch.equal(Bkey[batch, :, 0, :], model.r)


def test_logits_shape(model, tokens):
    for readout in ("exact", "mfvi"):
        logits = model(tokens, readout=readout)
        assert logits.shape == (BATCH, SEQ, model.cfg.vocab_size)
        assert torch.isfinite(logits).all()


def test_parameter_list_is_the_factor_list(model):
    """§18 Check 1: no matrix in the computation that is not a factor.

    With the global head attached the set gains exactly ``B_global``, the factor
    of Wu & Tu Eq. (46) -- and nothing else.
    """
    names = {n for n, _ in model.named_parameters()}
    expected = {"S", "T", "r", "b"}
    if model.cfg.use_global_head:
        expected |= {"B_global"}
    assert names == expected


def test_global_factor_shape(model):
    if not model.cfg.use_global_head:
        assert model.B_global is None
        return
    assert model.B_global.shape == (model.cfg.n_global, model.cfg.d)
