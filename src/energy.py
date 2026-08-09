"""The per-slot mean-field free energy.

Part III §17 gives the per-step energy of one slot,

    E_t(Q_W, Q_Z, {Q_c}) = - Σ_w Q_W(w) b_w
                           - Σ_{w,a} Q_W(w) Q_Z(a) S_{w,a}
                           - Σ_c Σ_{j∈D_t} Σ_a Q_c(j) Q_Z(a) B^(c)_{j,a}

and the three updates are its centred gradients under the entropic Frank-Wolfe
message weights (Wu & Tu §2.3.3). The functional those updates perform coordinate
descent on is therefore *not* ``E`` but

    F = E - λ_W H(Q_W) - λ_Z H(Q_Z) - λ_H Σ_c H(Q_c) [- λ_G H(Q_g)]

because ``Q ∝ exp(-∂E/∂Q / λ)`` is exactly the stationarity condition of ``F``.
``F`` is what must be non-increasing along the inner loop; ``E`` alone need not be.
This is validation check 8 of the research plan — the one check that tests the update
equations themselves rather than their shapes.

The observed (content) slot is the same functional with ``Q_W`` clamped to a one-hot,
so one function serves both modes.
"""

from typing import Optional

import torch


def _entropy(p: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """``-Σ p log p``, with the ``0 log 0 = 0`` convention."""
    return -(p * torch.where(p > 0, p.log(), torch.zeros_like(p))).sum(dim)


def slot_free_energy(
    qw: torch.Tensor,
    qz: torch.Tensor,
    alpha: torch.Tensor,
    B_full: torch.Tensor,
    S: torch.Tensor,
    b: Optional[torch.Tensor],
    lambda_W: float,
    lambda_Z: float,
    lambda_H: float,
    qg: Optional[torch.Tensor] = None,
    B_glob: Optional[torch.Tensor] = None,
    lambda_G: float = 1.0,
) -> torch.Tensor:
    """Mean-field free energy of one slot, ``(B,)``.

    ``qw``     ``(V,)`` or ``(B, V)`` — belief over the word variable.
    ``qz``     ``(B, d)``             — belief over the label variable.
    ``alpha``  ``(B, h, 1+t)``        — beliefs over the head variables, ROOT at 0.
    ``B_full`` ``(B, h, 1+t, d)``     — contracted arc scores over ``D_t``.
    """
    if qw.dim() == 1:
        qw = qw.unsqueeze(0).expand(qz.shape[0], -1)

    sbar = qw @ S  # (B, d)
    energy = -torch.einsum("ba,ba->b", qz, sbar)
    energy = energy - torch.einsum("bcj,bcja,ba->b", alpha, B_full, qz)
    if b is not None:
        energy = energy - qw @ b
    if qg is not None and B_glob is not None:
        energy = energy - torch.einsum("bk,ka,ba->b", qg, B_glob, qz)

    free = energy - lambda_W * _entropy(qw) - lambda_Z * _entropy(qz)
    free = free - lambda_H * _entropy(alpha).sum(-1)
    if qg is not None and B_glob is not None:
        free = free - lambda_G * _entropy(qg)
    return free
