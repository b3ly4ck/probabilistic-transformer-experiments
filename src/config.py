"""Configuration for the causal PT decoder.

Field names follow the papers rather than transformer convention:

* ``d``  is the size of the *label* set of a word variable ``Z_t`` (Wu & Tu, §2.2).
  It is the model width in the sense that all predictive information passes through
  it, but it is a number of discrete labels, not a hidden dimension.
* ``h``  is the number of *channels*, i.e. the number of head variables ``H_t^(c)``
  per word. It corresponds to attention heads.
* ``rank`` is the Kruskal rank ``r`` of the arc-score decomposition
  ``T^(c) = U^(c) V^(c)^T`` (Wu & Tu, Eqs. 14/21). It corresponds to the head
  dimension. ``None`` keeps the arc score as a full ``d x d`` matrix per channel.
* ``gamma`` is the clip threshold of the distance-sensitive ternary potential
  (Wu & Tu, Eqs. 9/10). Only the ``i - j > 0`` half of the table exists in a causal
  model, so the number of distance buckets is ``gamma + 1``. ``gamma = 0`` makes the
  arc score distance-insensitive (one bucket) — that is the setting of the worked
  example in ``causal_pt_output_note.pdf`` §5.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PTConfig:
    vocab_size: int

    # --- graph shape ---
    # Defaults are Wu & Tu's Table 2 row for PTB masked LM: d = 384, h = 16, T = 5,
    # gamma = 3, Kruskal decomposition with r = 64. They were previously arbitrary
    # choices of mine; taking the source's row wholesale means there is nothing here to
    # defend that the source has not already defended. Note the row is coherent only as a
    # row: rank = 64 saves parameters against a full T only while 2*rank < d, so a small
    # d with rank = 64 would cost *more* than leaving rank at None.
    d: int = 384
    h: int = 16
    rank: Optional[int] = 64
    gamma: int = 3
    n_components: int = 1  # K; §22.2 "factored labels". 1 is the flat model of Part III.
    channel_assignment: str = "roundrobin"  # "roundrobin" | "diagonal"
    # §22.2 replaces the single label variable Z_t by K variables Z_t^(1..K), each over
    # d' = d // K values, so that the *total* label width stays d and the word-label factor
    # S keeps its |V| x d = |V| K d' shape (read as K contiguous blocks of width d'). Each
    # head channel c is assigned a child component k(c) — whose belief queries the channel
    # and receives its message — and a head component k'(c) — whose prefix belief is
    # contracted into the channel's keys. The document does not fix that assignment; the two
    # named here are this project's choices, see `src/factored.py`.
    #   "roundrobin": k(c) = c mod K, k'(c) = (c // K) mod K. At h = K^2 every (child, head)
    #                 component pair occurs exactly once.
    #   "diagonal":   k'(c) = k(c) = c mod K. Components never mix.
    # K = 1 collapses both to the flat model, tensor for tensor.
    n_global: int = 0  # m; B.3.3 single-split global head. 0 disables it.
    allow_exact_global_head: bool = False
    # Under the exact readout G_t's direct contribution to log mu is
    # LSE_k B'[k, .] — a position- and prefix-independent d-vector, measured constant to
    # 1e-12 on 2026-08-09. A *run* in that mode therefore measures a label prior, not a
    # feed-forward analogue, so n_global > 0 with readout="exact" is refused. The flag is
    # the narrow escape hatch for the tests that assert exactly that constancy; it must not
    # be set in an experiment.
    b_glob_init_std: Optional[float] = None
    regularise_global_head: bool = True
    # Both address the mechanism measured on 2026-08-10 that kills G_t at d = 16: the rows of
    # B' collapse (spread 0.0197 -> 0.0037) because under a uniform Q_g every row receives the
    # same gradient 1/m, and the L2 term pulls them together on top of that. Neither the
    # symmetric initialisation nor the penalty is part of the construction — both are our
    # choices — so a fair verdict on §22.2 has to test the head with both removed.
    # b_glob_init_std=None falls back to init_std; regularise_global_head=False drops B' from
    # the L2 term while leaving T and r penalised.
    word_unary: bool = True  # the factor b; §16(c) allows dropping it (b == 0)
    freeze_word_unary: bool = False
    # Clamp b to the corpus log-unigram and never update it. b is a free per-word parameter
    # that reproduces the unigram distribution exactly, and the diagnostics of 2026-08-09
    # showed that solution winning the race inside the first 500 steps. Freezing it removes
    # the cheap descent direction without removing the factor — b stays in the graph, it is
    # simply observed rather than learned. Use CausalPTDecoder.set_word_unary().

    # --- inference ---
    schedule: str = "parallel"  # "parallel" (depth-T shared causal transformer) | "serial"
    n_iters: int = 5  # T, content-stream iterations, parallel schedule; Table 2 PTB value
    tau_obs: int = 1  # inner rounds per observed slot, serial schedule
    tau: int = 2  # predictive inner rounds of the MFVI readout (§17.1 asks for >= 2)
    alpha_Z: float = 1.0
    # Step size of the label update, Wu & Tu Appendix B.1:
    #     Q_i^(t) = alpha_Z * Q_i^*(t) + (1 - alpha_Z) * Q_i^(t-1)
    # 1.0 reproduces the original model exactly. Below 1 it damps the loop
    # q -> B -> G -> q, which is the loop measured to blow up at d = 32: the message
    # exploded from msg/unary 2.88 at step 500 to 20.57 at step 1000 and the run never
    # recovered. B.1 also defines a step size alpha_H on the head posteriors; that one is
    # not implemented, because it requires carrying Q_c across iterations and the Z step
    # alone already breaks the loop in question.
    readout: str = "exact"  # "exact" (mainline, §23.3) | "mfvi" (ablation)

    # --- entropic Frank-Wolfe message weights (Wu & Tu §2.3.3, A.5) ---
    lambda_Z: float = 1.0  # Wu & Tu §2.3.3: "we set lambda_Z = 1"
    lambda_H: Optional[float] = None  # None -> 1/d, the paper's default, App. A.5
    lambda_W: float = 1.0
    # lambda_W is a *mean-field* message weight and has no analogue in the exact readout,
    # which is sum-product in the declared model and admits no temperature. §18 Check 5
    # fixes it to 1 "so that the readout is an untempered conditional"; §22.1 reopens
    # lambda_W < 1 as a capacity lever. Consequence for Experiment 3: the evaluation-time
    # swap between the two readouts is only meaningful at lambda_W = 1. Any other value
    # makes the comparison measure temperature, not the cost of the approximation.
    lambda_G: float = 1.0  # not specified by either paper; chosen to match lambda_Z

    # --- engineering ---
    temp_mode: str = "fixed"  # "fixed" | "qnorm" | "qknorm"
    qk_gain: float = 1.0
    # Attention temperature of the head update. Wu & Tu App. A.5 fixes lambda_H = 1/d, and
    # `"fixed"` reproduces that exactly. The other two modes are this project's proposal and
    # are motivated by where 1/d comes from.
    #
    # The head logit is F_c(i,j) = <q_i, B^(c)_{j,.}> with q_i a *distribution*, so its
    # Euclidean norm is not a constant of the architecture: ||q||_2 runs from 1/sqrt(d)
    # (uniform) to 1 (one-hot), and B^(c)_{j,.} = E_{q_bar_j}[T^(c)_{.,b}] inherits the same
    # sqrt(d) dynamic range from the *key* side. At initialisation both beliefs are near
    # uniform, F ~ sigma_T / d, and dividing by lambda_H = 1/d leaves an O(sigma_T) logit --
    # which is why the source's choice is the right one *there*. Once the beliefs sharpen,
    # F ~ sigma_T and the same division leaves an O(d * sigma_T) logit. The temperature is
    # calibrated for the uniform-belief limit and mis-calibrated by up to a factor d away
    # from it; that mis-calibration is the loop this project measured running away at
    # d = 32 (msg/unary 2.88 -> 20.57 within 500 steps) and damped with alpha_Z.
    #
    #   "qnorm"  divides the logit by ||q_i||_2 * sqrt(d), which is 1 at the uniform belief.
    #            The mode therefore *agrees with Wu & Tu at initialisation* and only differs
    #            as the query sharpens. Query-side normalisation only.
    #   "qknorm" divides by ||q_i||_2 ||B_j||_2 -- the cosine of the two -- and multiplies by
    #            `qk_gain`. This is QK-normalisation as used in modern transformers, and it
    #            removes the key-side range as well. `qk_gain` is a temperature, not a
    #            parameter: nothing learned is added, so the "every matrix is a factor"
    #            constraint is untouched.
    #
    # Caveat, stated because it matters: lambda_H is also the entropy weight of Q_c in the
    # slot free energy (`src/energy.py`). A query-dependent temperature is no longer the
    # stationarity condition of a *fixed* functional, so the free-energy monotonicity check
    # (validation check 8) applies to "fixed" only. That check is asserted on "fixed" and the
    # other two modes are declared as what they are: an approximation deliberately taken
    # outside the variational derivation, kept because it is what makes width transfer.
    vocab_chunk: int = 8192  # chunk width of the exact readout's LSE over the vocabulary
    init_std: float = 0.02  # not from either paper; the nanoGPT convention, see below
    arc_init_std: Optional[float] = None  # None -> init_std; separate scale for T / U,V
    init_dist: str = "normal"  # "normal" | "uniform" | "orthogonal"
    # Prof. Tu, 2026-08-11: "Uniform initialisation (of what? B?) does not sound like a good
    # choice. How does it work with random initialisation?" Every factor here has always been
    # drawn from N(0, init_std^2) -- "uniform" in the 2026-08 report named the *treatment*
    # (one scale and one L2 coefficient for every factor), not the distribution. `init_dist`
    # makes the literal question answerable too: "uniform" draws U(-a, a) with a = sqrt(3)*std
    # (matched variance), "orthogonal" gives semi-orthogonal factors scaled to the same std.
    root_init_std: Optional[float] = None
    # None -> init_std. The root/sink column r^(c) enters the attention in raw d-space,
    # whereas the arc scores reach it contracted, B^(c)_{j,a} = E_{q_bar_j}[T^(c)_{a,.}],
    # which shrinks them by roughly 1/sqrt(d) for a near-uniform prefix belief (and again
    # by the Kruskal product when rank is set). Drawing both from N(0, init_std^2)
    # therefore starts the ROOT row about two orders of magnitude larger than the rows it
    # competes with — measured at 121x for d=384, h=16, rank=64. See exp0's status file.
    detach_prefix: bool = False  # see CausalPTDecoder docstring; the paper says False

    def __post_init__(self):
        if self.schedule not in ("parallel", "serial"):
            raise ValueError(f"unknown schedule {self.schedule!r}")
        if self.readout not in ("exact", "mfvi"):
            raise ValueError(f"unknown readout {self.readout!r}")
        if self.temp_mode not in ("fixed", "qnorm", "qknorm"):
            raise ValueError(f"unknown temp_mode {self.temp_mode!r}")
        if self.init_dist not in ("normal", "uniform", "orthogonal"):
            raise ValueError(f"unknown init_dist {self.init_dist!r}")
        if self.gamma < 0:
            raise ValueError("gamma must be >= 0")
        if self.n_global > 0 and self.readout == "exact" and not self.allow_exact_global_head:
            raise ValueError(
                "n_global > 0 with readout='exact': the global head's contribution to the "
                "exact readout is a position- and prefix-independent constant (§22.2, "
                "measured to 1e-12), so G_t is alive only under MFVI. Use readout='mfvi', "
                "or set allow_exact_global_head=True if you are testing that constancy."
            )
        if self.n_components < 1:
            raise ValueError("n_components must be >= 1")
        if self.d % self.n_components != 0:
            raise ValueError(
                f"n_components {self.n_components} does not divide d {self.d}: the K label "
                "variables partition the total width d into equal blocks d' = d // K, so "
                "that S keeps its |V| x d shape and K = 1 is the flat model"
            )
        if self.channel_assignment not in ("roundrobin", "diagonal"):
            raise ValueError(f"unknown channel_assignment {self.channel_assignment!r}")
        if self.rank is not None and self.rank < 1:
            raise ValueError("rank must be >= 1 or None")
        if self.rank is not None and self.rank > self.d_label:
            raise ValueError(
                f"rank {self.rank} exceeds d' {self.d_label}: the Kruskal form T = U V^T "
                "cannot have rank above the width of one label variable, and costs more "
                "parameters than a full T already at 2*rank >= d'"
            )

    @property
    def n_dist(self) -> int:
        """Number of distance buckets in the causal half of the RPE table."""
        return self.gamma + 1

    @property
    def d_label(self) -> int:
        """``d'`` — the label-set size of **one** label variable ``Z^(k)``.

        ``d`` is the *total* width ``D = K d'`` throughout, so that ``S`` is ``|V| x d``
        whatever ``K`` is (§22.2: "parameters |V|Kd' = |V|D at total width D = Kd' —
        identical to a flat label of width D"). At ``K = 1`` this is ``d``.
        """
        return self.d // self.n_components

    @property
    def lam_H(self) -> float:
        """Wu & Tu App. A.5 default ``1/d`` — read on the *label variable's* own width.

        For the flat model ``d_label == d`` and this is literally the source's value. Under
        factoring the belief entering a head logit is a distribution over ``d'`` labels, so
        ``1/d'`` is the same rule applied to the variable that is actually there.
        """
        return self.lambda_H if self.lambda_H is not None else 1.0 / self.d_label
