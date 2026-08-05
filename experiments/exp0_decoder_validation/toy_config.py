"""Toy configuration for Experiment 0.

Deliberately tiny and CPU-runnable so every intermediate tensor can be printed
and read.  The point is correctness, not learning.
"""

import torch

from src.config import PTConfig

SEED = 0
DEVICE = "cpu"

TOY = PTConfig(
    vocab_size=20,
    d=8,
    n_channels=1,
    n_rounds=3,
    lambda_Z=1.0,
    lambda_H=None,  # -> 1/d, the paper default
    lambda_W=1.0,
)

BATCH = 2
SEQ_LEN = 6


def toy_tokens(generator: torch.Generator | None = None) -> torch.Tensor:
    return torch.randint(0, TOY.vocab_size, (BATCH, SEQ_LEN), generator=generator)


def toy_generator() -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(SEED)
    return g
