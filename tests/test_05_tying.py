"""Check 5 — weight tying, as a property of the computation and not of the source text.

§16(b): the step conditional contains exactly *one* word-label factor. Two matrices on
the pair (W_t, Z_t) would multiply into one, ``exp(S)exp(S') = exp(S + S')``, so untied
weights are unrepresentable in this model class. The assertion is therefore not
"the two matrices are equal" but "there is only one matrix, and both paths read it".

Both directions are exercised by perturbing ``S`` and observing the two observable
consequences separately: the unary role (it moves the content stream) and the emission
role (it moves the logits with the context held fixed).
"""

import torch

from conftest import toy_model


def test_only_one_vocabulary_sized_matrix_exists():
    m = toy_model()
    V, d = m.cfg.vocab_size, m.cfg.d
    named = dict(m.named_parameters())
    two_d = {k: tuple(p.shape) for k, p in named.items() if p.dim() == 2}
    vocab_shaped = [k for k, s in two_d.items() if V in s]
    assert vocab_shaped == ["S"], f"a second vocabulary-sized matrix exists: {two_d}"
    assert named["S"].shape == (V, d)
    assert all(tuple(p.shape) != (d, V) for p in m.parameters())


def test_emission_role_reads_S(idx):
    """With the context (log mu) held fixed, moving S moves the logits."""
    m = toy_model()
    Bk = m.contract(m.content_stream(idx))
    log_mu = m.exact_log_mu(Bk).detach()
    before = m._logits_from_log_mu(log_mu)
    with torch.no_grad():
        m.S[2] += 1.0
    after = m._logits_from_log_mu(log_mu)
    assert not torch.allclose(before, after)
    assert (before[..., 2] != after[..., 2]).all()


def test_unary_role_reads_S(idx):
    """The same tensor, read in the other direction, moves the filtering marginals.

    The perturbation is a *single entry*, not a whole row: adding a constant to a row of
    S shifts the unary message uniformly across labels, which a softmax over labels
    absorbs. That gauge freedom is real — it is the constant the word unary b carries —
    so a row-wise perturbation would leave the content stream genuinely unchanged and
    the test would be asserting the wrong thing.
    """
    m = toy_model()
    token = int(idx[2, 2])
    before = m.content_stream(idx)
    with torch.no_grad():
        m.S[token, 0] += 1.0
    after = m.content_stream(idx)
    assert not torch.allclose(before, after)


def test_uniform_row_shift_of_S_is_a_gauge_freedom(idx):
    """The complement of the above, stated as a property rather than left implicit."""
    m = toy_model()
    before = m.content_stream(idx)
    with torch.no_grad():
        m.S += 1.0
    assert torch.allclose(before, m.content_stream(idx), atol=1e-12, rtol=0)


def test_the_two_roles_share_one_gradient_accumulator(idx):
    """A single backward deposits both contributions on the same tensor."""
    m = toy_model()
    m.loss(idx).backward()
    g_both = m.S.grad.clone()

    m.zero_grad(set_to_none=True)
    Bk = m.contract(m.content_stream(idx).detach())
    log_mu = m.exact_log_mu(Bk)
    torch.nn.functional.cross_entropy(
        m._logits_from_log_mu(log_mu).reshape(-1, m.cfg.vocab_size), idx.reshape(-1)
    ).backward()
    g_emission_and_arcs = m.S.grad.clone()

    # blocking the unary path changes the gradient on S: it is genuinely fed by both.
    assert not torch.allclose(g_both, g_emission_and_arcs)
