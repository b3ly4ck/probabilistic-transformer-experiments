"""Check 9 — exact readout against brute-force enumeration of the slot joint.

The strongest correctness test available in the project. The per-slot graph is a star
centred on Z_t, hence a tree, so sum-product is exact; at toy scale the same marginal is
also obtainable by explicitly enumerating

    p(W, Z, H^(1), ..., H^(h)) ∝ exp( b_W + S_{W,Z} + sum_c B^(c)_{H^(c), Z} )

over every assignment. Agreement validates the *factorisation* — that the model the
readout implements is the model that was declared — not an approximation to it.

The enumeration below is written with plain Python loops on purpose: a vectorised
reference that reuses the same einsums would be testing the code against itself.
"""

import itertools
import math

import torch

from conftest import toy_model


def brute_force_log_probs(channels, S, b):
    """Enumerate the slot joint.

    ``channels`` is one ``(|domain_c|, d)`` score matrix per leaf hanging off Z_t: the h
    head variables, plus the global head when it is enabled.
    """
    d = S.shape[1]
    domains = [range(c.shape[0]) for c in channels]
    mass = []
    for w in range(S.shape[0]):
        acc = 0.0
        for a in range(d):
            base = float(b[w]) + float(S[w, a])
            for heads in itertools.product(*domains):
                e = base + sum(float(channels[c][heads[c], a]) for c in range(len(channels)))
                acc += math.exp(e)
        mass.append(acc)
    total = sum(mass)
    return torch.tensor([math.log(x / total) for x in mass], dtype=torch.float64)


def test_exact_slot_readout_equals_brute_force(idx):
    m = toy_model(readout="exact")
    Bk = m.contract(m.content_stream(idx))
    for t in range(idx.shape[1] + 1):
        B_full = m._slot_keys(Bk, t).detach()
        got = torch.log_softmax(m.slot_exact_readout(B_full), dim=-1)
        for s in range(idx.shape[0]):
            channels = [B_full[s, c] for c in range(m.cfg.h)]
            want = brute_force_log_probs(channels, m.S.detach(), m.b.detach())
            assert torch.allclose(got[s], want, atol=1e-12, rtol=0), f"slot {t}, sequence {s}"


def test_exact_readout_with_global_head_equals_brute_force(idx):
    """The global head is one more leaf off Z_t, so the star is still a tree."""
    m = toy_model(readout="exact", n_global=3)
    Bk = m.contract(m.content_stream(idx))
    t = idx.shape[1]
    B_full = m._slot_keys(Bk, t).detach()
    got = torch.log_softmax(m.slot_exact_readout(B_full), dim=-1)
    for s in range(idx.shape[0]):
        channels = [B_full[s, c] for c in range(m.cfg.h)] + [m.B_glob.detach()]
        want = brute_force_log_probs(channels, m.S.detach(), m.b.detach())
        assert torch.allclose(got[s], want, atol=1e-12, rtol=0)


def test_vectorised_scan_matches_slot_assembly(idx):
    """``exact_log_mu`` (logcumsumexp over the far bucket + shifts for the near band)
    must agree with assembling D_t position by position. This is where an off-by-one in
    the RPE bucketing would hide."""
    for gamma in (0, 1, 2, 4, 9):
        m = toy_model(readout="exact", gamma=gamma)
        Bk = m.contract(m.content_stream(idx))
        scan = m.exact_log_mu(Bk)
        for t in range(idx.shape[1]):
            B_full = m._slot_keys(Bk, t)
            direct = torch.logsumexp(B_full, dim=2).sum(dim=1)
            assert torch.allclose(scan[:, t], direct, atol=1e-12, rtol=0), f"gamma {gamma}, t {t}"


def test_forward_logits_match_the_slot_path(idx, readout):
    """The vectorised forward and the per-slot path are the same computation."""
    m = toy_model(readout=readout)
    logits = m(idx)
    Bk = m.contract(m.content_stream(idx))
    for t in range(idx.shape[1]):
        B_full = m._slot_keys(Bk, t)
        slot = (
            m.slot_exact_readout(B_full) if readout == "exact" else m.slot_mfvi_readout(B_full)
        )
        assert torch.allclose(logits[:, t], slot, atol=1e-11, rtol=0), f"slot {t}"


def test_mixture_of_softmaxes_form(idx):
    """§17.2: the exact readout is a mixture over labels, not a single softmax.

    p(w) ∝ e^{b_w} sum_a e^{S_{w,a}} mu(a) — check the identity directly against the
    weights mu, which is the form the rank argument of Part IV rests on.
    """
    m = toy_model(readout="exact")
    Bk = m.contract(m.content_stream(idx))
    log_mu = m.exact_log_mu(Bk)
    mu = log_mu.exp()
    manual = m.b + torch.log(torch.einsum("bta,wa->btw", mu, m.S.exp()))
    assert torch.allclose(m(idx), manual, atol=1e-9, rtol=0)
