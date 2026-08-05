"""Check 8 -- the mean-field free energy is non-increasing along the inner loop.

Checks 1-7 all pass on a model with a typo in the update equations: shapes,
normalisation, causality and overfitting are blind to whether the updates are
the coordinate-wise minimisers of the energy that was written down.  This check
is not.  If F rises, the update rule is not the gradient of E_t.
"""

import torch

from src import mfvi
from src.config import PTConfig

CFG = PTConfig(vocab_size=6, d=4, n_channels=2, n_rounds=8, lambda_Z=1.0, lambda_H=0.25)


def _fixture(seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    S = torch.randn(CFG.vocab_size, CFG.d, generator=g)
    b = torch.randn(CFG.vocab_size, generator=g)
    Bkey = torch.randn(CFG.n_channels, 5, CFG.d, generator=g)  # h, K=5, d
    w = 3
    Q_W = torch.zeros(CFG.vocab_size)
    Q_W[w] = 1.0  # observed mode: the word is clamped
    return S, b, Bkey, Q_W, w


def _trajectory(S, b, Bkey, Q_W, w):
    """Alternate the updates, recording F after every single one."""
    Q_Z = mfvi.init_slot(S[w], CFG)
    Q_c = mfvi.update_Qc(Q_Z, Bkey, CFG)
    energies = [mfvi.slot_free_energy(Q_W, Q_Z, Q_c, Bkey, S, b, CFG)]
    for _ in range(CFG.n_rounds):
        Q_c = mfvi.update_Qc(Q_Z, Bkey, CFG)
        energies.append(mfvi.slot_free_energy(Q_W, Q_Z, Q_c, Bkey, S, b, CFG))
        Q_Z = mfvi.update_QZ(S[w], Q_c, Bkey, CFG)
        energies.append(mfvi.slot_free_energy(Q_W, Q_Z, Q_c, Bkey, S, b, CFG))
    return torch.stack(energies), Q_Z, Q_c


def test_free_energy_is_non_increasing():
    for seed in range(5):
        S, b, Bkey, Q_W, w = _fixture(seed)
        F, _, _ = _trajectory(S, b, Bkey, Q_W, w)
        diffs = F[1:] - F[:-1]
        assert (diffs <= 1e-6).all(), f"seed {seed}: free energy rose by {diffs.max()}"


def test_free_energy_converges():
    S, b, Bkey, Q_W, w = _fixture()
    F, _, _ = _trajectory(S, b, Bkey, Q_W, w)
    assert abs(float(F[-1] - F[-2])) < 1e-6


def test_the_Z_update_is_the_minimiser_not_merely_a_descent_step():
    """Perturb Q_Z off the update's output; F must rise.

    Monotonicity alone would also hold for a rule that merely decreases F. This
    asserts the stronger property the derivation claims: given Q_c, the update
    lands on the argmin.
    """
    g = torch.Generator()
    g.manual_seed(7)
    S, b, Bkey, Q_W, w = _fixture()
    _, Q_Z, Q_c = _trajectory(S, b, Bkey, Q_W, w)
    F_star = mfvi.slot_free_energy(Q_W, Q_Z, Q_c, Bkey, S, b, CFG)

    for _ in range(20):
        direction = torch.softmax(torch.randn(CFG.d, generator=g), dim=-1)
        for eps in (0.01, 0.05, 0.2):
            Q_pert = (1 - eps) * Q_Z + eps * direction
            F_pert = mfvi.slot_free_energy(Q_W, Q_pert, Q_c, Bkey, S, b, CFG)
            assert F_pert >= F_star - 1e-9, f"perturbation lowered F by {F_star - F_pert}"
