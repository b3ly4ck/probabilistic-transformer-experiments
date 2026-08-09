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


def contraction_rho(
    model: CausalPTDecoder, B_full: torch.Tensor, centred: bool = False
) -> torch.Tensor:
    """``rho = sum_c ||B^(c)||_2^2 / (4 lambda_Z lambda_H)`` per sequence, Lemma 23.1.

    ``B_full`` is ``(B, h, 1+t, d)``.

    The lemma specifies "Euclidean norms on centred vectors; shifts are irrelevant by
    softmax invariance", so the operator that actually appears in the recursion is the
    doubly centred one: the input ``delta q`` is a difference of distributions and sums to
    zero, and the output only matters up to a shift because it enters a softmax. Pass
    ``centred=True`` for that; the raw norm is an upper bound on it, hence a conservative
    reading of ``rho``, and is the default so that a reported violation is never an
    artefact of tightening.
    """
    B = B_full
    if centred:
        B = B - B.mean(dim=2, keepdim=True)  # centre over the head domain D_t
        B = B - B.mean(dim=3, keepdim=True)  # centre over the labels
    sv = torch.linalg.matrix_norm(B, ord=2)  # (B, h)
    return (sv**2).sum(-1) / (4 * model.cfg.lambda_Z * model.cfg.lam_H)


def root_attention_mass(model: CausalPTDecoder, idx: torch.Tensor) -> dict:
    """How much attention the ROOT column takes, against what uniform would give.

    ``r^(c)`` reaches the attention in raw ``d``-space while the arc scores arrive
    contracted, so a uniform initialisation starts ROOT far above the rows it competes
    with. Whether that becomes a genuine attention sink is a *measured* variable, not an
    assumption — log it, and if a sink appears, ``PTConfig.root_init_std`` is the knob
    that was already there.
    """
    qbar = model.content_stream(idx)
    _, alpha = model._arc_message(qbar, model.contract(qbar))
    n = idx.shape[1]
    support = torch.arange(1, n + 1, device=idx.device, dtype=alpha.dtype)  # |D_t| per slot
    mass = alpha[..., 0]  # (B, h, n)
    return {
        "root_mass_mean": float(mass.mean()),
        "root_mass_last": float(mass[..., -1].mean()),
        "uniform_last": float(1.0 / support[-1]),
        "excess_over_uniform": float((mass / (1.0 / support)).mean()),
    }


def fixed_point_multiplicity(
    model: CausalPTDecoder,
    B_full: torch.Tensor,
    m_W: torch.Tensor,
    n_starts: int = 48,
    iters: int = 500,
    spread: float = 4.0,
    tol: float = 1e-6,
    seed: int = 0,
) -> dict:
    """Does the slot inner loop have one fixed point, or several?

    This is the direct test of what Lemma 23.1's ``rho >= 1`` *admits*. Setting the word
    message difference to zero in the lemma's recursion leaves ``delta q_s <= rho delta
    q_{s-1}``, i.e. ``rho`` is the contraction factor of the slot map itself: below 1 the
    map is a contraction, so the fixed point is unique and reached from any start. Above
    1 the guarantee is *vacuous* — multistability becomes possible, not certain. Only an
    experiment settles which, so run the loop from many random initialisations of ``Q_Z``
    and count how many distinct fixed points come back.

    ``B_full`` is ``(B, h, 1+t, d)`` and ``m_W`` is ``(B, d)`` — ``S_{w_t,.}`` in observed
    mode, ``s_bar`` in predictive mode.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    Bt, d = m_W.shape
    qz = torch.softmax(
        torch.randn(n_starts, Bt, d, generator=g, dtype=m_W.dtype) * spread, dim=-1
    )
    for _ in range(iters):
        logit = torch.einsum("sba,bcja->sbcj", qz, B_full)
        alpha = torch.softmax(logit / model.cfg.lam_H, dim=-1)
        ctx = torch.einsum("sbcj,bcja->sba", alpha, B_full)
        qz = torch.softmax((m_W + ctx) / model.cfg.lambda_Z, dim=-1)

    counts, examples = [], []
    for b in range(Bt):
        pts = qz[:, b]  # (n_starts, d)
        reps: list = []
        for p in pts:
            if not any(0.5 * (p - r).abs().sum() < tol for r in reps):
                reps.append(p)
        counts.append(len(reps))
        if len(reps) > 1:
            sep = max(
                float(0.5 * (a - b2).abs().sum()) for a in reps for b2 in reps if a is not b2
            )
            examples.append(sep)
    return {
        "n_fixed_points": counts,
        "max_separation": max(examples) if examples else 0.0,
        "n_starts": n_starts,
    }


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
