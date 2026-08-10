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
    n_global: int = 0  # m; B.3.3 single-split global head. 0 disables it.
    allow_exact_global_head: bool = False
    # Under the exact readout G_t's direct contribution to log mu is
    # LSE_k B'[k, .] — a position- and prefix-independent d-vector, measured constant to
    # 1e-12 on 2026-08-09. A *run* in that mode therefore measures a label prior, not a
    # feed-forward analogue, so n_global > 0 with readout="exact" is refused. The flag is
    # the narrow escape hatch for the tests that assert exactly that constancy; it must not
    # be set in an experiment.
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
    vocab_chunk: int = 8192  # chunk width of the exact readout's LSE over the vocabulary
    init_std: float = 0.02  # not from either paper; the nanoGPT convention, see below
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
        if self.gamma < 0:
            raise ValueError("gamma must be >= 0")
        if self.n_global > 0 and self.readout == "exact" and not self.allow_exact_global_head:
            raise ValueError(
                "n_global > 0 with readout='exact': the global head's contribution to the "
                "exact readout is a position- and prefix-independent constant (§22.2, "
                "measured to 1e-12), so G_t is alive only under MFVI. Use readout='mfvi', "
                "or set allow_exact_global_head=True if you are testing that constancy."
            )
        if self.rank is not None and self.rank < 1:
            raise ValueError("rank must be >= 1 or None")
        if self.rank is not None and self.rank > self.d:
            raise ValueError(
                f"rank {self.rank} exceeds d {self.d}: the Kruskal form T = U V^T cannot "
                "have rank above d, and costs more parameters than a full T already at "
                "2*rank >= d"
            )

    @property
    def n_dist(self) -> int:
        """Number of distance buckets in the causal half of the RPE table."""
        return self.gamma + 1

    @property
    def lam_H(self) -> float:
        return self.lambda_H if self.lambda_H is not None else 1.0 / self.d
