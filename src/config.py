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
    d: int = 64
    h: int = 4
    rank: Optional[int] = None
    gamma: int = 3
    n_global: int = 0  # m; B.3.3 single-split global head. 0 disables it.
    word_unary: bool = True  # the factor b; §16(c) allows dropping it (b == 0)

    # --- inference ---
    schedule: str = "parallel"  # "parallel" (depth-T shared causal transformer) | "serial"
    n_iters: int = 4  # T, content-stream iterations, parallel schedule
    tau_obs: int = 1  # inner rounds per observed slot, serial schedule
    tau: int = 2  # predictive inner rounds of the MFVI readout (§17.1 asks for >= 2)
    readout: str = "exact"  # "exact" (mainline, §23.3) | "mfvi" (ablation)

    # --- entropic Frank-Wolfe message weights (Wu & Tu §2.3.3, A.5) ---
    lambda_Z: float = 1.0
    lambda_H: Optional[float] = None  # None -> 1/d, the paper's default
    lambda_W: float = 1.0  # §18 Check 5 fixes it to 1; §22.1 reopens it as a lever
    lambda_G: float = 1.0

    # --- engineering ---
    vocab_chunk: int = 8192  # chunk width of the exact readout's LSE over the vocabulary
    init_std: float = 0.02
    detach_prefix: bool = False  # see CausalPTDecoder docstring; the paper says False

    def __post_init__(self):
        if self.schedule not in ("parallel", "serial"):
            raise ValueError(f"unknown schedule {self.schedule!r}")
        if self.readout not in ("exact", "mfvi"):
            raise ValueError(f"unknown readout {self.readout!r}")
        if self.gamma < 0:
            raise ValueError("gamma must be >= 0")
        if self.rank is not None and self.rank < 1:
            raise ValueError("rank must be >= 1 or None")

    @property
    def n_dist(self) -> int:
        """Number of distance buckets in the causal half of the RPE table."""
        return self.gamma + 1

    @property
    def lam_H(self) -> float:
        return self.lambda_H if self.lambda_H is not None else 1.0 / self.d
