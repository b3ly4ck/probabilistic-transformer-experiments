"""Check 2 — normalisation.

All posteriors sum to 1 along their own variable axis, at every iteration. A drift
here means a misplaced softmax axis. Q_c must additionally put *zero* mass outside
D_t = {ROOT} u {1..t-1} — a posterior that normalises correctly over the wrong
support would still pass a naive sum-to-one assertion.
"""

import torch

from conftest import toy_model

TOL = 1e-12


def test_label_posteriors_normalised_every_iteration(idx):
    """Re-run the content stream by hand so every iterate, not just the last, is seen."""
    m = toy_model()
    Sw = m.S[idx]
    q = torch.softmax(Sw / m.cfg.lambda_Z, dim=-1)
    for _ in range(m.cfg.n_iters):
        assert torch.allclose(q.sum(-1), torch.ones_like(q.sum(-1)), atol=TOL)
        assert (q >= 0).all()
        G, alpha = m._arc_message(q, m.contract(q))
        assert torch.allclose(alpha.sum(-1), torch.ones_like(alpha.sum(-1)), atol=TOL)
        q = torch.softmax((Sw + G) / m.cfg.lambda_Z, dim=-1)
    assert torch.allclose(q.sum(-1), torch.ones_like(q.sum(-1)), atol=TOL)


def test_head_posterior_support_is_the_head_domain(idx):
    m = toy_model()
    n = idx.shape[1]
    qbar = m.content_stream(idx)
    _, alpha = m._arc_message(qbar, m.contract(qbar))

    a_root, a_pos = alpha[..., 0], alpha[..., 1:]
    ar = torch.arange(n)
    forbidden = ~(ar.view(-1, 1) > ar.view(1, -1))  # j >= i
    assert a_pos[:, :, forbidden].abs().max() == 0.0
    # slot 0 has D_0 = {ROOT}: all mass on the root column
    assert torch.allclose(a_root[:, :, 0], torch.ones_like(a_root[:, :, 0]), atol=TOL)


def test_slot_posteriors_normalised(idx):
    m = toy_model(n_global=4)
    qbar = m.content_stream(idx)
    Bk = m.contract(qbar)
    for t in range(idx.shape[1] + 1):
        B_full = m._slot_keys(Bk, t)
        trace: list = []
        m.slot_mfvi_readout(B_full, trace=trace)
        for qz, alpha, qg in trace:
            assert torch.allclose(qz.sum(-1), torch.ones_like(qz.sum(-1)), atol=TOL)
            assert torch.allclose(alpha.sum(-1), torch.ones_like(alpha.sum(-1)), atol=TOL)
            assert torch.allclose(qg.sum(-1), torch.ones_like(qg.sum(-1)), atol=TOL)


def test_readout_is_a_distribution_over_the_vocabulary(idx, readout):
    m = toy_model(readout=readout)
    p = torch.softmax(m(idx), dim=-1)
    assert torch.allclose(p.sum(-1), torch.ones_like(p.sum(-1)), atol=TOL)
    assert (p > 0).all()
