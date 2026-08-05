import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PTConfig  # noqa: E402
from src.pt_decoder import CausalPTDecoder  # noqa: E402

TOY = PTConfig(vocab_size=20, d=8, n_channels=1, n_rounds=3)
BATCH, SEQ = 2, 6


@pytest.fixture
def gen():
    g = torch.Generator()
    g.manual_seed(0)
    return g


@pytest.fixture
def model(gen):
    return CausalPTDecoder(TOY, generator=gen)


@pytest.fixture
def tokens(gen):
    return torch.randint(0, TOY.vocab_size, (BATCH, SEQ), generator=gen)
