"""The exact tree readout (§17.2), which §23.3 makes the mainline.

The slot's factor graph is a star centred at Z: edges W-Z and H^(1)-Z, ..., H^(h)-Z.
A star is a tree, so sum-product is exact and closes in one expression:

    p(W_t = w | w_<t)  proportional to  e^{b_w} sum_a e^{S[w,a]} mu_t(a),
    mu_t(a) = prod_c sum_{j in D_t} e^{B^(c)[j,a]}

This is a mixture of exponentials over labels -- a mixture-of-softmaxes head --
and it is *not* rank-limited the way the mean-field readout is.

Along the sequence, ``log mu_t(a) = sum_c LSE_{j in D_t} B^(c)[j,a]`` is a causal
prefix log-sum-exp, i.e. one ``logcumsumexp`` scan, O(n d h), fully parallel
(§23.3).  The layer-parallel schedule survives.
"""

from __future__ import annotations

import itertools

import torch
from torch import Tensor


def log_mu_slot(Bkey: Tensor) -> Tensor:
    """log mu(a) = sum_c LSE_j B^(c)[j, a] for a single slot.

    Args:
        Bkey: (..., h, K, d) with every key already inside D_t.

    Returns:
        (..., d)
    """
    return torch.logsumexp(Bkey, dim=-2).sum(dim=-2)


def log_mu_sequence(Bkey: Tensor) -> Tensor:
    """The causal prefix scan of §23.3.

    Args:
        Bkey: (batch, h, n + 1, d) with index 0 holding ROOT and index ``j + 1``
            holding prefix position ``j``.

    Returns:
        (batch, n, d) where entry ``t`` is ``log mu_t`` over
        ``D_t = {ROOT, 0, ..., t - 1}`` -- exactly the keys a causal mask allows.
    """
    n = Bkey.shape[-2] - 1
    cum = torch.logcumsumexp(Bkey, dim=-2)  # (batch, h, n + 1, d)
    return cum[..., :n, :].sum(dim=-3)


def exact_logits(log_mu: Tensor, S: Tensor, b: Tensor) -> Tensor:
    """log p_hat(W = w) up to an additive constant: b_w + LSE_a( S[w,a] + log mu(a) ).

    Args:
        log_mu: (..., d)
        S: (V, d), b: (V,)

    Returns:
        (..., V) unnormalised log-probabilities.

    Note on cost (§23.3): this materialises a ``(..., V, d)`` tensor.  It is an
    LSE rather than a matmul -- the same FLOPs with worse hardware constants.  At
    LM scale it must be chunked over the vocabulary with a fused cross-entropy.
    At toy scale it is written plainly on purpose.
    """
    scores = S + log_mu.unsqueeze(-2)  # (..., V, d)
    return b + torch.logsumexp(scores, dim=-1)


def brute_force_logits(Bkey: Tensor, S: Tensor, b: Tensor) -> Tensor:
    """Enumerate the slot's joint distribution explicitly and marginalise.

    Deliberately naive: nested Python loops over every label and every tuple of
    head assignments.  This is the oracle for check 9 -- if it agreed with
    :func:`exact_logits` because it shared its cleverness, it would prove
    nothing.

    The joint over one slot is

        p(W = w, Z = a, H = (j_1, ..., j_h))
            proportional to exp( b_w + S[w, a] + sum_c B^(c)[j_c, a] )

    Args:
        Bkey: (h, K, d) for a single slot, no batch dimension.
        S: (V, d), b: (V,)

    Returns:
        (V,) unnormalised log-probabilities on the same scale as
        :func:`exact_logits`.
    """
    if Bkey.dim() != 3:
        raise ValueError("brute_force_logits takes a single slot: (h, K, d)")
    h, K, d = Bkey.shape
    V = S.shape[0]
    totals = torch.zeros(V, dtype=Bkey.dtype)
    for w in range(V):
        acc = 0.0
        for a in range(d):
            for heads in itertools.product(range(K), repeat=h):
                arc = sum(Bkey[c, heads[c], a] for c in range(h))
                acc = acc + torch.exp(b[w] + S[w, a] + arc)
        totals[w] = torch.log(acc)
    return totals
