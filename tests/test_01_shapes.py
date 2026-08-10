"""Check 1 — shapes.

Every intermediate tensor has the dimensionality stated in Part III: Q_W over the
vocabulary, Q_Z over the d labels, Q_c over D_t = {ROOT} u {1..t-1}, and the
contracted prefix scores B^(c)_{j,a} over (channel, prefix position, label).
"""

import torch

from conftest import toy_model


def test_content_stream_and_logits(idx, readout, schedule):
    m = toy_model(readout=readout, schedule=schedule)
    B, n = idx.shape
    d, h, V, K = m.cfg.d, m.cfg.h, m.cfg.vocab_size, m.cfg.n_dist

    qbar = m.content_stream(idx)
    assert qbar.shape == (B, n, d)

    Bk = m.contract(qbar)
    assert Bk.shape == (K, B, h, n, d)

    logits = m(idx)
    assert logits.shape == (B, n, V)
    assert torch.isfinite(logits).all()

    assert m.next_token_logits(idx).shape == (B, V)


def test_message_shapes(idx):
    m = toy_model()
    B, n = idx.shape
    d, h = m.cfg.d, m.cfg.h

    qbar = m.content_stream(idx)
    Bk = m.contract(qbar)

    G, alpha = m._arc_message(qbar, Bk)
    assert G.shape == (B, n, d)
    assert alpha.shape == (B, h, n, 1 + n)  # column 0 is ROOT

    assert m.exact_log_mu(Bk).shape == (B, n, d)

    for t in range(n + 1):
        B_full = m._slot_keys(Bk, t)
        assert B_full.shape == (B, h, 1 + t, d)
        ctx, a = m._slot_message(qbar[:, 0], B_full)
        assert ctx.shape == (B, d)
        assert a.shape == (B, h, 1 + t)


def test_global_head_shapes(idx):
    m = toy_model(readout="mfvi", n_global=5)
    qbar = m.content_stream(idx)
    msg, qg = m._global_message(qbar)
    assert msg.shape == qbar.shape
    assert qg.shape == idx.shape + (5,)
    assert m(idx).shape == idx.shape + (m.cfg.vocab_size,)


def test_lowrank_matches_full_arc_scores():
    """The Kruskal form is a reparameterisation of T^(c), not a different model."""
    m = toy_model(rank=3)
    T = m.arc_scores()
    assert T.shape == (m.cfg.n_dist, m.cfg.h, m.cfg.d, m.cfg.d)
    manual = torch.einsum("khar,khbr->khab", m.U, m.V)
    assert torch.equal(T, manual)


def test_parameter_split_is_reported():
    m = toy_model()
    p = m.num_parameters()
    assert p["embedding"] == m.S.numel() + m.b.numel()
    assert p["embedding"] + p["non_embedding"] == p["total"]
    assert p["total"] == sum(q.numel() for q in m.parameters())
