"""Check 4 — what "frozen prefix" means, tested from the gradient side.

**A conflict, stated rather than hidden.** ``CLAUDE.md`` constraint 3 and check 4 of
``RESEARCH_PLAN.md`` say that *neither gradients nor messages* may flow backwards from
step t into the prefix posteriors. The specification says the opposite about gradients,
twice and explicitly:

  Part II §12.3 Check 2 — "One clarification that pre-empts a standard confusion:
  *frozen* means constant with respect to step-i's inference problem, *not* detached in
  autodiff. Training gradients flow backward through B^(c)_{j,.} into q_bar_j exactly as
  they flow through cached activations in a causal transformer. Forward causality is
  what defines a decoder; backward gradient flow is what trains it."

  Part III §18 Check 5 "Gradients" — same statement, and it adds that this is how the
  tied matrix S is trained in both of its roles.

The paper wins (``CLAUDE.md``: "when code and paper disagree, the paper wins"), so the
mainline is ``detach_prefix=False``. The other reading is available as a config flag and
is tested here too, so that switching between them is one line and not a rewrite.

What is *not* in dispute, and is the substantive content of this file: the loss at slot t
may only reach prefix beliefs q_bar_j with **j < t**. A gradient arriving at j >= t would
be an anti-causal path. That is asserted below to be exactly zero.

Methodology follows the research plan: the prefix beliefs are made leaf tensors with
``requires_grad=True`` and left *attached*. Reading ``.grad`` on a detached tensor would
be green by construction — it is ``None`` whether or not the code is correct.
"""

import pytest
import torch

from conftest import toy_model


def _prefix_grad(detach: bool, slot: int, idx: torch.Tensor, readout: str):
    m = toy_model(readout=readout, detach_prefix=detach)
    qbar = m.content_stream(idx).detach().requires_grad_(True)  # leaf, still attached
    keys = qbar.detach() if detach else qbar
    Bk = m.contract(keys)
    logits = (
        m._logits_from_log_mu(m.exact_log_mu(Bk)) if readout == "exact" else m.mfvi_readout(Bk)
    )
    loss = torch.nn.functional.cross_entropy(logits[:, slot], idx[:, slot])
    loss.backward()
    return qbar.grad


def test_no_gradient_path_from_slot_t_into_the_future(idx, readout):
    """The anti-causal check: slot t's loss must not reach q_bar_j for j >= t."""
    n = idx.shape[1]
    for slot in range(1, n):
        g = _prefix_grad(False, slot, idx, readout)
        assert g is not None
        assert g[:, slot:].abs().max() == 0.0, f"slot {slot} leaked gradient into j >= {slot}"
        assert g[:, :slot].abs().max() > 0.0, f"slot {slot} reached no prefix belief at all"


def test_slot_zero_reaches_no_prefix_at_all(idx, readout):
    """D_0 = {ROOT}: the first slot's prediction touches no word belief."""
    g = _prefix_grad(False, 0, idx, readout)
    assert g is None or g.abs().max() == 0.0


@pytest.mark.parametrize("readout", ["exact", "mfvi"])
def test_detach_prefix_flag_blocks_the_readout_gradient(idx, readout):
    """The non-mainline reading of constraint 3, available and behaving as advertised."""
    g = _prefix_grad(True, idx.shape[1] - 1, idx, readout)
    assert g is None or g.abs().max() == 0.0


def test_mainline_trains_S_through_both_of_its_roles(idx):
    """Gradient reaches S from the unary role and the emission role, on one tensor."""
    m = toy_model()
    m.loss(idx).backward()
    assert m.S.grad is not None and m.S.grad.abs().max() > 0
    assert m.r_root.grad is not None and m.r_root.grad.abs().max() > 0
    assert m.T.grad is not None and m.T.grad.abs().max() > 0
    assert m.b.grad is not None and m.b.grad.abs().max() > 0
