"""The bigram reference, which is the one that matters once attention sits at distance one.

The unigram says whether a model learned anything; the bigram says whether it learned anything
beyond the previous word. `experiments/interpretability.py` measures the trained model putting
0.976 of its positional attention mass at distance 1, so this reference stops being a formality.
"""

import math

import torch

from src.data import bigram_perplexity, unigram_perplexity


def _toy():
    """A stream where the bigram is exactly right and the unigram is exactly uninformed."""
    # a b a b a b ... : p(b|a) = 1, p(a|b) = 1, but p(a) = p(b) = 1/2
    return torch.tensor([0, 1] * 4096, dtype=torch.long)


def test_bigram_beats_unigram_where_it_must():
    data = _toy()
    uni = unigram_perplexity(data, data, 2, block_size=8, batch_size=4, ignore_first=1)
    big = bigram_perplexity(data, data, 2, block_size=8, batch_size=4, alpha=1e-6)
    assert 1.9 < uni < 2.1, uni          # a fair coin, per token
    assert big < 1.05, big               # the next token is determined
    assert big < uni


def test_bigram_is_a_distribution_and_finite_on_real_data():
    from src.data import load_ptb

    c = load_ptb()
    ppl = bigram_perplexity(c.train, c.valid, c.vocab_size, 64, 16)
    assert math.isfinite(ppl)
    # sanity corridor, not a golden value: better than the unigram, worse than the models
    assert 300.0 < ppl < 600.0, ppl


def test_smoothing_toward_unigram_never_assigns_zero():
    data = _toy()
    # a symbol pair never seen in training must still receive positive probability
    ev = torch.tensor([0, 0, 1, 1] * 8, dtype=torch.long)
    ppl = bigram_perplexity(data, ev, 2, block_size=4, batch_size=2, alpha=1.0)
    assert math.isfinite(ppl)
