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
    """A toy decoder with the word unary ``b`` deliberately broken away from zero.

    ``CausalPTDecoder.reset_parameters`` zeroes ``b``, and a zero ``b`` is *invisible*:
    it makes ``Q_W^(0) = softmax(b)`` uniform in ``_word_prior`` and adds nothing in
    ``_logits_from_log_mu`` / ``mfvi_readout`` / ``slot_mfvi_readout``. The word-unary
    factor of §16(c) would then go unchecked by value everywhere this fixture is used.
    Mutation-tested on 2026-09-08: with ``b`` at zero, deleting the ``+ self.b`` block
    from ``_logits_from_log_mu`` left ``tests/test_09_exact_vs_brute.py`` fully green.

    The values are ``cos(1.3 i) * 0.9`` — non-uniform, and non-monotone so that a test
    which only cares about the *ordering* of the logits cannot pass by accident. This
    mirrors ``tests/test_18_factored.py::_break_the_unary``, which caught the same
    mutation in 9 places.

    ``tests/test_07_worked_example.py`` is not affected: it builds its model from
    ``experiments/exp0_decoder_validation/worked_example.py``, which pins the note's own
    ``b = (1, 0, 0, 0)`` (§5 of ``causal_pt_output_note.pdf`` — the note does *not*
    assume a zero unary; it prints ``Q_W^(0) = (.475, .175, .175, .175)``).
    """
    torch.manual_seed(seed)
    m = CausalPTDecoder(toy_cfg(**over)).to(dtype)
    if m.b is not None:
        with torch.no_grad():
            V = m.cfg.vocab_size
            m.b.copy_(torch.cos(torch.arange(V, dtype=m.b.dtype) * 1.3) * 0.9)
    return m


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
