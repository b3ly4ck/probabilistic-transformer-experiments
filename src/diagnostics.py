"""Measurement utilities for the causal PT decoder.

This model class has no layer norm and no residual connection — adding either would be a
map rather than a factor, which §22.2 of Part IV names as a tripwire. What bounds the
activations instead is the simplex geometry: every message is a convex combination of
arc scores, so

    |G_i(a)| = |sum_c sum_j Q_c(j) B^(c)_{j,a}| <= h * max_{c,a,b} |T^(c)_{a,b}|

with equality only if every head posterior and every prefix belief is a one-hot. The
activations therefore cannot blow up at fixed parameters; only the *parameters* can grow,
and the original paper's control for that is an explicit L2 penalty on the ternary scores
(Wu & Tu §4.2: "For MLM tasks, we add a small L2 regularization term to the ternary
scores in our model, which we experimentally find beneficial"; Table 2 gives 5e-4 on PTB).

Two quantities are worth watching during training, and neither is expensive:

* ``ratio = ||G_i|| / ||S_{w_i,.}||`` — how much of the label belief is context and how
  much is word identity. It is what replaces "is the residual stream exploding".
* ``rho = sum_c ||B^(c)||^2 / (4 lambda_Z lambda_H)`` — the contraction constant of
  Lemma 23.1 in Part IV. ``rho < 1`` guarantees the inner loop converges to a single
  fixed point and bounds the prior/posterior divergence; ``rho >= 1`` "is precisely the
  regime in which the inner loop admits multistability — prediction and encoding landing
  in different fixed points". It is a boundary worth knowing you are near.
"""

from typing import Optional

import torch

from .pt_decoder import CausalPTDecoder


def contraction_rho(model: CausalPTDecoder, B_full: torch.Tensor) -> torch.Tensor:
    """``rho = sum_c ||B^(c)||_2^2 / (4 lambda_Z lambda_H)`` per sequence, Lemma 23.1.

    ``B_full`` is ``(B, h, 1+t, d)``. The spectral norm is taken on the raw matrix, which
    upper-bounds the centred one the lemma uses, so this is a conservative reading.
    """
    sv = torch.linalg.matrix_norm(B_full, ord=2)  # (B, h)
    return (sv**2).sum(-1) / (4 * model.cfg.lambda_Z * model.cfg.lam_H)


def message_scale_report(model: CausalPTDecoder, idx: torch.Tensor) -> list:
    """Per-iteration message statistics of the content stream."""
    trace: list = []
    model.content_stream(idx, trace=trace)
    return trace


def schedule_divergence(model: CausalPTDecoder, idx: torch.Tensor) -> dict:
    """How far apart the layer-parallel and serial schedules land.

    Both are legal causal schedules of the same updates (Part II §12.3), so they are not
    expected to agree; the question is by how much, since Experiment 0's worked example
    uses one and training uses the other.
    """
    saved = model.cfg.schedule
    try:
        model.cfg.schedule = "parallel"
        qp = model.content_stream(idx)
        lp = model(idx)
        model.cfg.schedule = "serial"
        qs = model.content_stream(idx)
        ls = model(idx)
    finally:
        model.cfg.schedule = saved

    tv = 0.5 * (qp - qs).abs().sum(-1)  # total variation per position
    nll_p = torch.nn.functional.cross_entropy(
        lp.reshape(-1, model.cfg.vocab_size), idx.reshape(-1)
    )
    nll_s = torch.nn.functional.cross_entropy(
        ls.reshape(-1, model.cfg.vocab_size), idx.reshape(-1)
    )
    return {
        "tv_mean": float(tv.mean()),
        "tv_max": float(tv.max()),
        "tv_by_position": tv.mean(0).tolist(),
        "logit_absdiff_max": float((lp - ls).abs().max()),
        "nll_parallel": float(nll_p),
        "nll_serial": float(nll_s),
    }


def global_head_readout_term(model: CausalPTDecoder) -> Optional[torch.Tensor]:
    """The global head's contribution to ``log mu_t``, which §22.2 gives as ``sum_m e^{B'_{m,a}}``.

    It is a single ``d``-vector: independent of position and of the prefix. Under the
    exact readout the B.3.3 global head therefore collapses to ``d`` effective degrees of
    freedom — a fixed per-label prior on the mixture weights — no matter how large ``m``
    is. It does *not* cancel, because the readout takes a log-sum-exp over labels rather
    than a linear combination, but it cannot carry context. Its context-dependent work
    happens in the content stream, where the update gains the GFU term
    ``sigma(q B'^T) B'`` that Wu & Tu identify as the feed-forward analogue.
    """
    if model.B_glob is None:
        return None
    return torch.logsumexp(model.B_glob, dim=0)


def _fmt(trace: list) -> str:
    head = f"{'iter':>4} {'|G|':>9} {'|Sw|':>9} {'|G|/|Sw|':>9} {'max|G|':>9} {'H(attn)':>9} {'H/Hmax':>8} {'H(Qz)':>8}"
    rows = [
        f"{i:>4} {t['G_norm']:>9.4f} {t['Sw_norm']:>9.4f} {t['ratio']:>9.4f} "
        f"{t['G_absmax']:>9.4f} {t['attn_entropy']:>9.4f} {t['attn_entropy_frac']:>8.4f} "
        f"{t['label_entropy']:>8.4f}"
        for i, t in enumerate(trace, 1)
    ]
    return "\n".join([head] + rows)


def main() -> None:
    """Report at the scale of Wu & Tu's PTB masked-LM configuration (their Table 2)."""
    from .config import PTConfig

    torch.manual_seed(0)
    cfg = PTConfig(vocab_size=10000, d=384, h=16, rank=64, gamma=3, n_iters=5)
    m = CausalPTDecoder(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 64))

    print(f"config: d={cfg.d} h={cfg.h} rank={cfg.rank} gamma={cfg.gamma} T={cfg.n_iters}")
    print(f"lambda_Z={cfg.lambda_Z} lambda_H=1/d={cfg.lam_H:.6f} lambda_W={cfg.lambda_W}")
    print(f"init_std={cfg.init_std}  params={m.num_parameters()}")
    print("\ncontent stream, untrained:")
    print(_fmt(message_scale_report(m, idx)))

    B_full = m._slot_keys(m.contract(m.content_stream(idx)), idx.shape[1])
    print("\nrho (Lemma 23.1) at the last slot:", contraction_rho(m, B_full).tolist())


if __name__ == "__main__":
    main()
