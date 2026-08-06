"""Check 8 -- the mean-field free energy is non-increasing along the inner loop.

Checks 1-7 all pass on a model with a typo in the update equations: shapes,
normalisation, causality and overfitting are blind to whether the updates are
the coordinate-wise minimisers of the energy that was written down.  This check
is not.  If F rises, the update rule is not the gradient of E_t.
"""

import pytest
import torch

from src import mfvi
from src.config import PTConfig

BASE = dict(vocab_size=6, d=4, n_channels=2, n_rounds=8, lambda_Z=1.0, lambda_H=0.25)
CONFIGS = [
    pytest.param(PTConfig(**BASE), id="no_global"),
    pytest.param(PTConfig(**BASE, use_global_head=True, n_global=5), id="global"),
]


def _fixture(cfg, seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    S = torch.randn(cfg.vocab_size, cfg.d, generator=g)
    b = torch.randn(cfg.vocab_size, generator=g)
    Bkey = torch.randn(cfg.n_channels, 5, cfg.d, generator=g)  # h, K=5, d
    Bg = torch.randn(cfg.n_global, cfg.d, generator=g) if cfg.use_global_head else None
    w = 3
    Q_W = torch.zeros(cfg.vocab_size)
    Q_W[w] = 1.0  # observed mode: the word is clamped
    return S, b, Bkey, Bg, Q_W, w


def _trajectory(cfg, S, b, Bkey, Bg, Q_W, w):
    """Alternate the updates, recording F after every single one."""

    def F(QZ, Qc, QG):
        return mfvi.slot_free_energy(Q_W, QZ, Qc, Bkey, S, b, cfg, QG, Bg)

    Q_Z = mfvi.init_slot(S[w], cfg)
    Q_c = mfvi.update_Qc(Q_Z, Bkey, cfg)
    Q_G = mfvi.update_QG(Q_Z, Bg, cfg) if Bg is not None else None
    energies = [F(Q_Z, Q_c, Q_G)]
    for _ in range(cfg.n_rounds):
        Q_c = mfvi.update_Qc(Q_Z, Bkey, cfg)
        if Bg is not None:
            Q_G = mfvi.update_QG(Q_Z, Bg, cfg)
        energies.append(F(Q_Z, Q_c, Q_G))
        Q_Z = mfvi.update_QZ(S[w], Q_c, Bkey, cfg, Q_G, Bg)
        energies.append(F(Q_Z, Q_c, Q_G))
    return torch.stack(energies), Q_Z, Q_c, Q_G


@pytest.mark.parametrize("cfg", CONFIGS)
def test_free_energy_is_non_increasing(cfg):
    for seed in range(5):
        S, b, Bkey, Bg, Q_W, w = _fixture(cfg, seed)
        F, _, _, _ = _trajectory(cfg, S, b, Bkey, Bg, Q_W, w)
        diffs = F[1:] - F[:-1]
        assert (diffs <= 1e-6).all(), f"seed {seed}: free energy rose by {diffs.max()}"


@pytest.mark.parametrize("cfg", CONFIGS)
def test_free_energy_converges(cfg):
    S, b, Bkey, Bg, Q_W, w = _fixture(cfg)
    F, _, _, _ = _trajectory(cfg, S, b, Bkey, Bg, Q_W, w)
    assert abs(float(F[-1] - F[-2])) < 1e-6


@pytest.mark.parametrize("cfg", CONFIGS)
def test_the_G_update_is_an_exact_argmin(cfg):
    """The global head must be a coordinate minimiser too, not a descent step."""
    if not cfg.use_global_head:
        return
    g = torch.Generator()
    g.manual_seed(11)
    S, b, Bkey, Bg, Q_W, w = _fixture(cfg)
    _, Q_Z, Q_c, Q_G = _trajectory(cfg, S, b, Bkey, Bg, Q_W, w)
    F_star = mfvi.slot_free_energy(Q_W, Q_Z, Q_c, Bkey, S, b, cfg, Q_G, Bg)
    for _ in range(20):
        direction = torch.softmax(torch.randn(cfg.n_global, generator=g), dim=-1)
        for eps in (0.01, 0.05, 0.2):
            Q_pert = (1 - eps) * Q_G + eps * direction
            F_pert = mfvi.slot_free_energy(Q_W, Q_Z, Q_c, Bkey, S, b, cfg, Q_pert, Bg)
            assert F_pert >= F_star - 1e-9, f"perturbing Q_G lowered F by {F_star - F_pert}"


@pytest.mark.parametrize("cfg", CONFIGS)
def test_the_Z_update_is_the_minimiser_not_merely_a_descent_step(cfg):
    """Perturb Q_Z off the update's output; F must rise.

    Monotonicity alone would also hold for a rule that merely decreases F. This
    asserts the stronger property the derivation claims: given Q_c, the update
    lands on the argmin.
    """
    g = torch.Generator()
    g.manual_seed(7)
    S, b, Bkey, Bg, Q_W, w = _fixture(cfg)
    _, Q_Z, Q_c, Q_G = _trajectory(cfg, S, b, Bkey, Bg, Q_W, w)
    F_star = mfvi.slot_free_energy(Q_W, Q_Z, Q_c, Bkey, S, b, cfg, Q_G, Bg)

    for _ in range(20):
        direction = torch.softmax(torch.randn(cfg.d, generator=g), dim=-1)
        for eps in (0.01, 0.05, 0.2):
            Q_pert = (1 - eps) * Q_Z + eps * direction
            F_pert = mfvi.slot_free_energy(Q_W, Q_pert, Q_c, Bkey, S, b, cfg, Q_G, Bg)
            assert F_pert >= F_star - 1e-9, f"perturbation lowered F by {F_star - F_pert}"
