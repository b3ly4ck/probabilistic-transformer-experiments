import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PTConfig  # noqa: E402
from src.pt_decoder import CausalPTDecoder  # noqa: E402

BATCH, SEQ = 2, 6
M_GLOBAL = 5

BASE = dict(vocab_size=20, d=8, n_channels=1, n_rounds=3)

# Both arms of Experiment 1 run the same code path.  Every check runs in both
# flag states: a check that passes in only one arm is a bug.
CONFIGS = {
    "no_global": PTConfig(**BASE),
    "global": PTConfig(**BASE, use_global_head=True, n_global=M_GLOBAL),
}


@pytest.fixture(params=list(CONFIGS), ids=list(CONFIGS))
def cfg(request):
    return CONFIGS[request.param]


@pytest.fixture
def gen():
    g = torch.Generator()
    g.manual_seed(0)
    return g


@pytest.fixture
def model(cfg, gen):
    return CausalPTDecoder(cfg, generator=gen)


@pytest.fixture
def tokens(gen):
    return torch.randint(0, BASE["vocab_size"], (BATCH, SEQ), generator=gen)


def both_flag_states(**overrides):
    """Config pairs for tests that build their own, kept in one place."""
    return [
        pytest.param(PTConfig(**overrides), id="no_global"),
        pytest.param(
            PTConfig(**overrides, use_global_head=True, n_global=M_GLOBAL), id="global"
        ),
    ]
