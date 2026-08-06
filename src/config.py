"""Configuration for the causal Probabilistic Transformer decoder.

Section references are to `developer files/causalprobabilistictransformer_1.pdf`
unless the note is named explicitly.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PTConfig:
    """Model dimensions and the MFVI temperatures.

    The parameter list is exactly the factor list (§18, Check 1): ``S``, ``T^(c)``,
    ``r``, ``b``. Nothing here introduces a matrix that is not a factor.
    """

    vocab_size: int
    d: int
    """Label-set size. Also the width of the label bottleneck (§16, Remark 16.1)."""

    n_channels: int = 1
    """Number of dependency-head channels ``h``."""

    n_rounds: int = 3
    """Content-stream rounds ``T``. Each round is one (H, Z) update pair."""

    lambda_Z: float = 1.0
    """Entropic weight on the label variable. Paper default: 1."""

    lambda_H: float = None  # type: ignore[assignment]
    """Entropic weight on the head variables. Paper default: ``1/d`` (Notation)."""

    lambda_W: float = 1.0
    """Entropic weight on the word variable, used only by the MFVI readout (§17)."""

    use_word_unary: bool = True
    """The word unary ``b`` is the LM-head bias as a factor (§16(c)). Optional but
    principled; setting this False sets ``b ≡ 0``."""

    # -- Appendix B.3.3 single-split global head -------------------------

    use_global_head: bool = False
    """Attach the global-head variable ``G_t`` (§22.2, Wu & Tu Appendix B.3.3).

    This is the flag that separates arm 1.1 from arm 1.2 of Experiment 1. Both
    arms run the same code path: whether the in-graph feed-forward analogue helps
    is measured, not assumed.
    """

    n_global: int = 0
    """``m`` -- the size of the global-head domain. Only read when
    ``use_global_head``."""

    lambda_G: float = 1.0
    """Entropic weight on the global head. **Not pinned by the source** and it
    matters: Wu & Tu's Eq. (44) carries no explicit temperature and §22.2 names
    none.

    Their calibration argument for ``lambda_H = 1/d`` is about the message being
    a sum over ``d`` labels, and the message to ``G`` has the same form, so the
    principled analogue is ``1/d`` -- *not* ``1/m``, which depends on the wrong
    axis. But at toy scale ``lambda_G <= 1`` makes the global softmax saturate
    and collapses the label posterior (see
    tests/test_06_overfit.py::test_global_head_collapses_at_low_lambda_G), while
    ``lambda_G >= 5`` trains cleanly. At realistic ``d`` the calibration may
    matter more and the collapse less. Record the value used in every run."""

    def __post_init__(self) -> None:
        if self.lambda_H is None:
            object.__setattr__(self, "lambda_H", 1.0 / self.d)
        if self.n_rounds < 1:
            raise ValueError("n_rounds must be >= 1")
        names = ["lambda_Z", "lambda_H", "lambda_W"]
        if self.use_global_head:
            if self.n_global < 1:
                raise ValueError("use_global_head requires n_global >= 1")
            names.append("lambda_G")
        for name in names:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive (MFVI requires lambda > 0)")
