"""Check 4 -- no anti-causal path from a slot into positions at or after it.

Restated 2026-08-05.  The original form ("inspect .grad on detached prefix
quantities") was vacuous: a detached tensor has .grad is None unconditionally,
so it passed whether or not the code was correct.

"Frozen prefix" is a statement about inference -- qbar_j is a conditioning
constant of the variational problem at later slots (§25.1) -- not about autograd.
Gradients flow through the content stream normally, as in any causal
transformer.  What must hold is that the readout at t is *a function of*
qbar_{<t} alone.  Autograd is the sharp instrument for that: hold qbar as a leaf
and check the gradient of the logits at t is exactly zero at every j >= t.
"""

import torch

from tests.conftest import SEQ, TOY


def _qbar_leaf(model, tokens):
    qbar = model.content_stream(tokens).detach().clone()
    qbar.requires_grad_(True)
    return qbar


def test_logits_at_t_do_not_depend_on_qbar_at_or_after_t(model, tokens):
    for t in range(SEQ):
        qbar = _qbar_leaf(model, tokens)
        model.zero_grad(set_to_none=True)
        model.exact_logits(qbar)[:, t].sum().backward()
        assert qbar.grad is not None
        future = qbar.grad[:, t:]
        assert torch.equal(future, torch.zeros_like(future)), (
            f"logits at {t} carry gradient into qbar at positions >= {t}"
        )


def test_the_past_does_carry_gradient(model, tokens):
    """Guards the test above: the causal half must be non-zero, or the check is empty."""
    t = SEQ - 1
    qbar = _qbar_leaf(model, tokens)
    model.exact_logits(qbar)[:, t].sum().backward()
    assert qbar.grad[:, :t].abs().sum() > 0


def test_content_stream_is_trained_not_frozen(model, tokens):
    """The corrected reading: S and T must receive gradient from the loss.

    A no_grad filtering pass would leave these at zero, which is how the
    retracted design would have failed silently.
    """
    model.zero_grad(set_to_none=True)
    model.loss(tokens).backward()
    for name in ("S", "T", "r"):
        grad = getattr(model, name).grad
        assert grad is not None and grad.abs().sum() > 0, f"{name} received no gradient"
