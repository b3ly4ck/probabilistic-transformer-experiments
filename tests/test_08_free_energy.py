"""Check 8 — the mean-field free energy is non-increasing.

Checks 1-7 are blind to an error *inside* the update equations: a model with a typo in
(2)-(4) still has correct shapes, normalised posteriors, intact causality, and will
happily overfit a single batch. This check is what tests the equations.

The MFVI updates are coordinate descent on

    F = E - lambda_W H(Q_W) - lambda_Z H(Q_Z) - lambda_H sum_c H(Q_c)

not on E alone — ``Q ∝ exp(-dE/dQ / lambda)`` is the stationarity condition of F, and E
is linear in each block, so every update is an exact block minimisation. If F rises, the
update rule is not the gradient of the energy that was written down.
"""

import torch

from src.energy import slot_free_energy
from conftest import toy_model

TOL = 1e-10


def _free(m, qw, qz, alpha, B_full, qg=None):
    return slot_free_energy(
        qw, qz, alpha, B_full, m.S, m.b,
        m.cfg.lambda_W, m.cfg.lambda_Z, m.cfg.lam_H,
        qg=qg, B_glob=m.B_glob, lambda_G=m.cfg.lambda_G,
    )


def _predictive_curve(m, idx, slot, tau):
    m.cfg.tau = tau
    Bk = m.contract(m.content_stream(idx))
    B_full = m._slot_keys(Bk, slot)
    qw0, _ = m._word_prior()
    trace: list = []
    m.slot_mfvi_readout(B_full, trace=trace)
    return [_free(m, qw0, qz, alpha, B_full, qg) for qz, alpha, qg in trace]


def test_predictive_slot_free_energy_is_non_increasing(idx):
    m = toy_model()
    for slot in range(1, idx.shape[1] + 1):
        curve = _predictive_curve(m, idx, slot, tau=8)
        diffs = torch.stack(curve[1:]) - torch.stack(curve[:-1])
        assert diffs.max() <= TOL, f"slot {slot}: free energy rose by {float(diffs.max()):.3e}"


def test_predictive_slot_free_energy_converges(idx):
    m = toy_model()
    curve = _predictive_curve(m, idx, idx.shape[1], tau=20)
    tail = torch.stack(curve[-4:])
    assert (tail.max(0).values - tail.min(0).values).max() < 1e-6


def test_free_energy_non_increasing_with_the_global_head(idx):
    """The B.3.3 global head is one more leaf off Z_t; it must not break coordinate descent."""
    m = toy_model(n_global=5)
    curve = _predictive_curve(m, idx, idx.shape[1], tau=8)
    diffs = torch.stack(curve[1:]) - torch.stack(curve[:-1])
    assert diffs.max() <= TOL


def test_observed_slot_free_energy_is_non_increasing(idx):
    """The content stream is the same energy with Q_W clamped to a one-hot (§17)."""
    m = toy_model()
    n = idx.shape[1]
    Bk = m.contract(m.content_stream(idx))
    V = m.cfg.vocab_size

    for slot in range(1, n):
        B_full = m._slot_keys(Bk, slot)
        qw = torch.zeros(idx.shape[0], V, dtype=m.S.dtype)
        qw.scatter_(1, idx[:, slot : slot + 1], 1.0)
        Sw = m.S[idx[:, slot]]

        q = torch.softmax(Sw / m.cfg.lambda_Z, dim=-1)
        curve = []
        for _ in range(8):
            ctx, alpha = m._slot_message(q, B_full)
            curve.append(_free(m, qw, q, alpha, B_full))
            q = torch.softmax((Sw + ctx) / m.cfg.lambda_Z, dim=-1)
            curve.append(_free(m, qw, q, alpha, B_full))
        diffs = torch.stack(curve[1:]) - torch.stack(curve[:-1])
        assert diffs.max() <= TOL, f"slot {slot}: free energy rose by {float(diffs.max()):.3e}"


def test_the_check_has_teeth(idx):
    """A check that never fails proves nothing, so break the update and watch it fail.

    The mutation is a sign error in the H-update — ``Q_c ∝ exp(-F/lambda_H)`` instead of
    ``exp(+F/lambda_H)``. That is the most realistic typo in these equations and the one
    checks 1-7 are completely blind to: shapes, normalisation, causality and the ability
    to overfit a single batch all survive it untouched. The free energy must not.
    """
    m = toy_model()
    Bk = m.contract(m.content_stream(idx))
    B_full = m._slot_keys(Bk, idx.shape[1])
    qw0, sbar = m._word_prior()

    qz = torch.softmax(sbar / m.cfg.lambda_Z, dim=-1).expand(idx.shape[0], m.cfg.d)
    curve = []
    for _ in range(6):
        logit = torch.einsum("ba,bcja->bcj", qz, B_full)
        alpha = torch.softmax(-logit / m.cfg.lam_H, dim=-1)  # the mutation
        curve.append(_free(m, qw0, qz, alpha, B_full))
        ctx = torch.einsum("bcj,bcja->ba", alpha, B_full)
        qz = torch.softmax((sbar + ctx) / m.cfg.lambda_Z, dim=-1)
        curve.append(_free(m, qw0, qz, alpha, B_full))

    rise = (torch.stack(curve[1:]) - torch.stack(curve[:-1])).max()
    assert rise > 1e-6, f"a sign-flipped update stayed monotone (max rise {float(rise):.3e})"
