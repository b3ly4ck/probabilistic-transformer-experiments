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

    def __post_init__(self) -> None:
        if self.lambda_H is None:
            object.__setattr__(self, "lambda_H", 1.0 / self.d)
        if self.n_rounds < 1:
            raise ValueError("n_rounds must be >= 1")
        for name in ("lambda_Z", "lambda_H", "lambda_W"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive (MFVI requires lambda > 0)")
