"""Slot-level mean-field updates and the per-slot free energy (§17).

Everything here is a pure function over tensors: no module, no parameters, no
state.  Tests build the inputs by hand and call these directly, which is what
lets check 8 assert on the free energy without standing up a model.

Shape convention for a single slot, with ``...`` any batch dimensions:

    Bkey : (..., h, K, d)   contracted arc scores B^(c)[j, a] over the head
                            domain D_t, K = |D_t|.  Index 0 is ROOT, whose row
                            is r^(c); indices 1.. are prefix positions.
    m_W  : (..., d)         the word message.  Observed mode: S[w_t, :].
                            Predictive mode: sum_w Q_W(w) S[w, :].
    Q_Z  : (..., d)
    Q_c  : (..., h, K)
"""

from __future__ import annotations

import torch
from torch import Tensor

from .config import PTConfig

NEG_INF = float("-inf")


def contract_prefix(qbar: Tensor, T: Tensor, r: Tensor) -> Tensor:
    """B^(c)[j, a] = sum_b qbar_j(b) T^(c)[a, b], with the ROOT row set to r^(c).

    §17: "the prefix is frozen as qbar_1..qbar_{t-1} and contracted as before into
    B^(c)[j,a] -- exact at the log-potential level by linearity in the one-hot
    encoding of z_j".  The contraction is exact, not an approximation: the
    log-potential is linear in the head's one-hot, so its expectation under
    qbar_j is obtained by substituting qbar_j itself.

    Args:
        qbar: (batch, n, d) frozen filtering marginals, one per prefix position.
        T: (h, d, d) arc scores, ``T[c, a, b]`` with ``a`` dependent, ``b`` head.
        r: (h, d) root keys.

    Returns:
        (batch, h, n + 1, d) with index 0 along the key axis holding ROOT.
    """
    B = torch.einsum("bje,cae->bcja", qbar, T)
    root = r.unsqueeze(0).unsqueeze(2).expand(B.shape[0], -1, 1, -1)
    return torch.cat([root, B], dim=2)


def head_scores(Q_Z: Tensor, Bkey: Tensor) -> Tensor:
    """F_c(j) = sum_a Q_Z(a) B^(c)[j, a] -- the attention logits (§18, Check 1).

    Args:
        Q_Z: (..., d)
        Bkey: (..., h, K, d)

    Returns:
        (..., h, K)
    """
    return torch.einsum("...a,...hka->...hk", Q_Z, Bkey)


def update_Qc(Q_Z: Tensor, Bkey: Tensor, cfg: PTConfig, mask: Tensor | None = None) -> Tensor:
    """Q_c(j) proportional to exp( (1/lambda_H) sum_a Q_Z(a) B^(c)[j, a] ).

    Args:
        mask: (..., K) boolean, True where the key is inside D_t.  Keys outside
            the head domain are given -inf logits, so they receive exactly zero
            mass -- this is the strict lower-triangular mask of §18 Check 2.
    """
    logits = head_scores(Q_Z, Bkey) / cfg.lambda_H
    if mask is not None:
        logits = logits.masked_fill(~mask.unsqueeze(-2), NEG_INF)
    return torch.softmax(logits, dim=-1)


def head_message(Q_c: Tensor, Bkey: Tensor) -> Tensor:
    """sum_c sum_j Q_c(j) B^(c)[j, a] -- the attention output (§18, Check 1)."""
    return torch.einsum("...hk,...hka->...a", Q_c, Bkey)


def update_QZ(m_W: Tensor, Q_c: Tensor, Bkey: Tensor, cfg: PTConfig) -> Tensor:
    """Q_Z(a) proportional to exp( (1/lambda_Z) [ m_W(a) + sum_c sum_j Q_c(j) B^(c)[j,a] ] )."""
    return torch.softmax((m_W + head_message(Q_c, Bkey)) / cfg.lambda_Z, dim=-1)


def mfvi_readout_logits(Q_Z: Tensor, S: Tensor, b: Tensor, cfg: PTConfig) -> Tensor:
    """The MFVI readout: logits b_w + sum_a Q_Z(a) S[w, a], scaled by 1/lambda_W.

    This is the mean-field update of the word variable, applied once as the
    readout (§17.1: "finish with a single Q_W update: that softmax is the LM
    output layer").  It is *affine* in ``Q_Z``, hence the rank-(d+1) bottleneck.

    Under the §23.3 verdict this readout is the **ablation**, not the mainline.
    """
    return (b + torch.einsum("...a,va->...v", Q_Z, S)) / cfg.lambda_W


def word_message(Q_W: Tensor, S: Tensor) -> Tensor:
    """m_W(a) = sum_w Q_W(w) S[w, a] -- the predictive word message (§17)."""
    return torch.einsum("...v,va->...a", Q_W, S)


def init_slot(
    m_W: Tensor,
    cfg: PTConfig,
) -> Tensor:
    """Initialise Q_Z from its unary-type factors (§17.1, the paper's Eq. 7 rule).

    Observed mode passes ``m_W = S[w_t, :]``.  Predictive mode passes the
    prior-weighted mean embedding ``s_bar``, which §17.1 identifies with the
    [MASK] embedding of MLM and the query embedding of XLNet -- derived, not
    invented.
    """
    return torch.softmax(m_W / cfg.lambda_Z, dim=-1)


def run_slot_mfvi(
    m_W: Tensor,
    Bkey: Tensor,
    cfg: PTConfig,
    n_rounds: int | None = None,
    mask: Tensor | None = None,
    return_trace: bool = False,
):
    """Run the asynchronous inner loop: {Q_c} then Q_Z, for ``n_rounds`` (§17.1).

    Returns ``(Q_Z, Q_c)``, or ``(Q_Z, Q_c, trace)`` where ``trace`` is the list of
    ``(Q_Z, Q_c)`` pairs after each round, used by the free-energy check.
    """
    rounds = cfg.n_rounds if n_rounds is None else n_rounds
    Q_Z = init_slot(m_W, cfg)
    Q_c = None
    trace = []
    for _ in range(rounds):
        Q_c = update_Qc(Q_Z, Bkey, cfg, mask)
        Q_Z = update_QZ(m_W, Q_c, Bkey, cfg)
        if return_trace:
            trace.append((Q_Z, Q_c))
    if return_trace:
        return Q_Z, Q_c, trace
    return Q_Z, Q_c


def _entropy(Q: Tensor, dim: int = -1) -> Tensor:
    """Shannon entropy, with 0 log 0 = 0."""
    logQ = torch.where(Q > 0, torch.log(Q.clamp_min(torch.finfo(Q.dtype).tiny)), torch.zeros_like(Q))
    return -(Q * logQ).sum(dim=dim)


def slot_energy(Q_W: Tensor, Q_Z: Tensor, Q_c: Tensor, Bkey: Tensor, S: Tensor, b: Tensor) -> Tensor:
    """The per-step energy E_t of §17, every term, nothing removed:

        E_t = - sum_w Q_W(w) b_w
              - sum_{w,a} Q_W(w) Q_Z(a) S[w,a]
              - sum_c sum_j sum_a Q_c(j) Q_Z(a) B^(c)[j,a]
    """
    e_unary = torch.einsum("...v,v->...", Q_W, b)
    e_word_label = torch.einsum("...v,...a,va->...", Q_W, Q_Z, S)
    e_arc = torch.einsum("...hk,...a,...hka->...", Q_c, Q_Z, Bkey)
    return -(e_unary + e_word_label + e_arc)


def slot_free_energy(
    Q_W: Tensor, Q_Z: Tensor, Q_c: Tensor, Bkey: Tensor, S: Tensor, b: Tensor, cfg: PTConfig
) -> Tensor:
    """F = E - sum_x lambda_x H(Q_x).

    The three updates are exactly the coordinate-wise minimisers of this F: with
    ``dF/dQ_x = dE/dQ_x + lambda_x (log Q_x + 1)``, stationarity on the simplex
    gives ``Q_x ∝ exp(-(1/lambda_x) dE/dQ_x)``, which is what §17 writes.

    Check 8 asserts this quantity is non-increasing along the inner loop.  It is
    derived from the *same* energy as the updates on purpose -- deriving it
    independently would only test that two transcriptions agree.
    """
    E = slot_energy(Q_W, Q_Z, Q_c, Bkey, S, b)
    H_Z = _entropy(Q_Z)
    H_c = _entropy(Q_c).sum(dim=-1)
    H_W = _entropy(Q_W)
    return E - cfg.lambda_Z * H_Z - cfg.lambda_H * H_c - cfg.lambda_W * H_W
