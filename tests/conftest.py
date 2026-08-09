"""Shared toy fixtures.

Everything here is deliberately tiny and CPU-only: small ``d``, a vocabulary of a few
tokens, a handful of positions. The point of Experiment 0 is correctness, not learning,
and the causality check in particular must run where the hardware is reproducible.
"""

import pytest
import torch

from src import CausalPTDecoder, PTConfig

torch.use_deterministic_algorithms(True)


def toy_cfg(**over):
    base = dict(
        vocab_size=7,
        d=4,
        h=2,
        rank=None,
        gamma=2,  # 3 distance buckets: exercises both the near band and the far scan
        n_iters=2,
        tau=2,
        tau_obs=1,
        lambda_Z=1.0,
        lambda_H=1.0,
        lambda_W=1.0,
        init_std=0.5,  # large enough that a sign error cannot hide in the noise
    )
    base.update(over)
    return PTConfig(**base)


def toy_model(seed: int = 0, dtype=torch.float64, **over) -> CausalPTDecoder:
    torch.manual_seed(seed)
    return CausalPTDecoder(toy_cfg(**over)).to(dtype)


@pytest.fixture
def model():
    return toy_model()


@pytest.fixture
def idx():
    torch.manual_seed(1)
    return torch.randint(0, 7, (3, 6))


@pytest.fixture(params=["exact", "mfvi"])
def readout(request):
    return request.param


@pytest.fixture(params=["parallel", "serial"])
def schedule(request):
    return request.param
