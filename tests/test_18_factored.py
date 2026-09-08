"""Check 18 — factored labels (Part IV §22.2), the ``K`` label variables per position.

Three of these tests are decisive and the rest are hygiene.

1. ``K = 1`` must *be* the flat decoder. Same config, same state dict, same logits to
   1e-12 in float64, same parameter counts, under both readouts and with ``gamma > 0``
   so the relative-position path is exercised. This is the specification of the tensor
   layout in ``src/factored.py``: if it fails, the layout is wrong and the fix is the
   layout, never the tolerance.

2. The exact readout at ``K = 2`` against a brute-force enumeration of the slot joint.
   The slot graph is ``W_t`` joined to ``K`` stars — still a tree — so sum-product is
   exact, and at toy scale the same marginal is obtainable by explicitly summing

       p(W, Z^(1), Z^(2), H^(1..h)) ∝ exp( b_W + Σ_k S^(k)_{W,Z^(k)}
                                                + Σ_c B^(c)_{H^(c), Z^(k(c))} )

   over every configuration. Following ``tests/test_09_exact_vs_brute.py``: the
   enumeration is written in plain Python loops and does not call a single model helper,
   because a reference built out of the model's own einsums would be testing the code
   against itself. The contracted keys ``B^(c)`` are additionally rebuilt by hand from
   the raw parameters, which is what pins the component routing ``k'(c)``.

3. The content stream and the mean-field readout against §22.2's three update equations,
   written out in plain Python loops (``content_by_hand``, ``slot_mfvi_by_hand``). The
   brute force above takes the contracted keys ``B^(c)`` — hence ``q̄`` — as given, and
   ``K = 1`` cannot see a component permutation, so without these two references the
   ``K`` components are interchangeable as far as this file can tell. Mutation-tested on
   2026-09-08: rolling the component axis of the initial belief, of the word unary, or of
   the prefix belief before contraction; reading ``S`` as interleaved rather than
   contiguous blocks in either the content stream or the word message; and reversing the
   component axis of ``Q_Z`` in the final word message — six real bugs — all left the
   file green before they were added. Every model built here also carries a *non-uniform*
   ``b``: ``reset_parameters`` zeroes it, and with ``b = 0`` deleting the word-unary
   factor from either readout is invisible.

The §22.2 prediction is also asserted numerically (``test_mean_field_readout_is_affine``
/ ``test_exact_readout_escapes_the_affine_family``): under the mean-field readout the
log-conditionals lie in an affine subspace of dimension ``d`` whatever ``K`` is, and
under the exact readout they do not. That is the whole reason for factoring, and the
experiment that follows this file measures the gap it predicts.
"""

import itertools
import math

import pytest
import torch

from src import CausalPTDecoder, FactoredPTDecoder, PTConfig
from src.factored import component_assignment

# --------------------------------------------------------------------- fixtures --


def cfg(**over):
    """A toy config. ``d = 8`` so that K in {1, 2, 4} all divide it."""
    base = dict(
        vocab_size=7,
        d=8,
        h=4,  # = K^2 at K = 2: under "roundrobin" every (child, head) pair occurs once
        rank=None,
        gamma=2,  # 3 distance buckets: both the near band and the far prefix scan
        n_iters=2,
        tau=2,
        tau_obs=1,
        lambda_Z=1.0,
        lambda_H=1.0,
        lambda_W=1.0,
        init_std=0.5,
    )
    base.update(over)
    return PTConfig(**base)


def _break_the_unary(m):
    """Give ``b`` a non-uniform value.

    ``reset_parameters`` zeroes the word unary, and a zero ``b`` is invisible: it makes
    ``Q_W^(0) ∝ exp(b)`` uniform and adds nothing to either readout, so *dropping the
    factor entirely* leaves every logit unchanged. Mutation-tested on 2026-09-08:
    deleting ``+ self.b`` from ``mfvi_readout`` and forcing ``Q_W^(0)`` uniform in
    ``_word_prior`` both left this file green while ``b`` was zero. Every model built
    here therefore carries a non-uniform, non-monotone ``b``.
    """
    if m.b is not None:
        with torch.no_grad():
            V = m.cfg.vocab_size
            m.b.copy_(torch.cos(torch.arange(V, dtype=m.b.dtype) * 1.3) * 0.9)
    return m


def flat(seed=0, **over):
    torch.manual_seed(seed)
    return _break_the_unary(CausalPTDecoder(cfg(**over)).to(torch.float64))


def fac(seed=0, **over):
    torch.manual_seed(seed)
    return _break_the_unary(FactoredPTDecoder(cfg(**over)).to(torch.float64))


def tokens(V=7, shape=(3, 6), seed=1):
    torch.manual_seed(seed)
    return torch.randint(0, V, shape)


# ------------------------------------------------- 1. K = 1 is the flat decoder --

K1_CASES = [
    dict(rank=None, temp_mode="fixed", lambda_H=1.0, schedule="parallel"),
    dict(rank=None, temp_mode="fixed", lambda_H=None, schedule="parallel"),
    dict(rank=3, temp_mode="fixed", lambda_H=None, schedule="parallel"),
    dict(rank=None, temp_mode="qnorm", lambda_H=None, schedule="parallel"),
    dict(rank=None, temp_mode="qknorm", lambda_H=1.0, schedule="parallel"),
    dict(rank=None, temp_mode="fixed", lambda_H=1.0, schedule="serial"),
]


@pytest.mark.parametrize("readout", ["exact", "mfvi"])
@pytest.mark.parametrize("case", K1_CASES, ids=lambda c: "-".join(str(v) for v in c.values()))
def test_k1_reproduces_the_flat_decoder(readout, case):
    """Same state dict in, same logits out — to 1e-12 in float64."""
    a = flat(readout=readout, **case)
    b = fac(readout=readout, n_components=1, **case)
    assert a.cfg.n_dist == 3, "gamma must be > 0 here or the RPE path is not exercised"

    missing = b.load_state_dict(a.state_dict())  # strict: the buffers are non-persistent
    assert missing.missing_keys == [] and missing.unexpected_keys == []

    idx = tokens()
    la, lb = a(idx), b(idx)
    assert la.shape == lb.shape
    assert torch.allclose(la, lb, atol=1e-12, rtol=0), (la - lb).abs().max()
    assert torch.allclose(
        a.next_token_logits(idx), b.next_token_logits(idx), atol=1e-12, rtol=0
    )
    assert torch.allclose(a.arc_regulariser(), b.arc_regulariser(), atol=1e-14, rtol=0)


def test_k1_parameter_counts_are_identical():
    for rank in (None, 3):
        a, b = flat(rank=rank), fac(rank=rank, n_components=1)
        assert a.num_parameters() == b.num_parameters()
        assert sorted(k for k, _ in a.named_parameters()) == sorted(
            k for k, _ in b.named_parameters()
        )
        for (ka, pa), (kb, pb) in zip(
            sorted(a.named_parameters()), sorted(b.named_parameters())
        ):
            assert ka == kb and pa.shape == pb.shape


# ------------------------------------- 2. exact readout vs brute force at K = 2 --

BF = dict(vocab_size=5, d=6, h=4, gamma=1, n_components=2, readout="exact")  # d' = 3


def brute_force_log_probs(channels, k_child, S_blocks, b, K, dp):
    """Enumerate the slot joint and marginalise to ``p(W_t)``.

    ``channels[c]`` is the ``(|D_t|, d')`` score matrix of the ``c``-th head leaf and
    ``k_child[c]`` the label component it hangs off. Plain loops, raw numbers, no model
    helper of any kind — this is the reference the model is judged against.
    """
    V = S_blocks.shape[0]
    domains = [range(c.shape[0]) for c in channels]
    mass = []
    for w in range(V):
        acc = 0.0
        for labels in itertools.product(range(dp), repeat=K):
            base = float(b[w]) + sum(float(S_blocks[w, k, labels[k]]) for k in range(K))
            for heads in itertools.product(*domains):
                e = base + sum(
                    float(channels[c][heads[c], labels[k_child[c]]])
                    for c in range(len(channels))
                )
                acc += math.exp(e)
        mass.append(acc)
    total = sum(mass)
    return torch.tensor([math.log(x / total) for x in mass], dtype=torch.float64)


def _softmax(xs):
    top = max(xs)
    e = [math.exp(x - top) for x in xs]
    z = sum(e)
    return [v / z for v in e]


def _keys_for_slot(m, qbar_s, t, k_head):
    """``B^(c)_{j,a}`` over ``D_t = {ROOT, 0..t-1}`` for one sequence, as nested lists.

    Independent recomputation of ``contract`` + the distance bucketing + the head
    component routing ``k'(c)``, which is the one piece of §22.2 that a brute-force
    enumeration cannot see (it takes the keys as given). ``qbar_s`` is ``[n][K][d']``.
    """
    h, dp, nd = m.cfg.h, m.cfg.d_label, m.cfg.n_dist
    out = []
    for c in range(h):
        rows = [[float(m.r_root[c, a]) for a in range(dp)]]
        for j in range(t):
            bucket = min(t - j, nd) - 1  # Wu & Tu Eq. 10, causal half
            rows.append(
                [
                    sum(
                        qbar_s[j][k_head[c]][e] * float(m.T[bucket, c, a, e])
                        for e in range(dp)
                    )
                    for a in range(dp)
                ]
            )
        out.append(rows)
    return out


def keys_by_hand(m, qbar, t, k_head):
    """:func:`_keys_for_slot` over a batch, as a ``(B, h, 1+t, d')`` tensor."""
    return torch.tensor(
        [_keys_for_slot(m, qbar[s].tolist(), t, k_head) for s in range(qbar.shape[0])],
        dtype=torch.float64,
    )


def _head_round(m, q, rows, k_child):
    """One H-update and the H → Z message, §22.2's first and third bracket, in loops.

        Q_c(j)  ∝ exp( (1/λ_H) Σ_a Q_{Z^(k(c))}(a) B^(c)_{j,a} )
        ctx^(k) = Σ_{c: k(c)=k} Σ_j Q_c(j) B^(c)_{j,a}

    ``q`` is ``[K][d']`` and ``rows`` the output of :func:`_keys_for_slot`.
    """
    K, dp, lam_H = m.cfg.n_components, m.cfg.d_label, m.cfg.lam_H
    ctx = [[0.0] * dp for _ in range(K)]
    for c, kc in enumerate(k_child):
        dom = range(len(rows[c]))
        al = _softmax([sum(q[kc][a] * rows[c][j][a] for a in range(dp)) / lam_H for j in dom])
        for a in range(dp):
            ctx[kc][a] += sum(al[j] * rows[c][j][a] for j in dom)
    return ctx


def content_by_hand(m, idx, k_child, k_head):
    """The filtering marginals ``q̄``, both schedules, from the raw parameters in loops.

    The content stream is the KV cache: every other test here takes ``q̄`` as given, so
    without this reference a permutation of the ``K`` components inside it is invisible.
    Mutation-tested on 2026-09-08 — rolling the component axis of the initial belief, of
    the word unary, or of the prefix belief before contraction all left this file green.
    """
    Bt, n = idx.shape
    K, dp, lamZ = m.cfg.n_components, m.cfg.d_label, m.cfg.lambda_Z
    assert m.cfg.alpha_Z == 1.0 and m.cfg.temp_mode == "fixed"  # what this reference covers
    S = m.S.detach().reshape(m.cfg.vocab_size, K, dp)  # contiguous blocks, spelled out here
    out = []
    for s in range(Bt):
        u = [
            [[float(S[int(idx[s, i]), k, a]) for a in range(dp)] for k in range(K)]
            for i in range(n)
        ]
        def step(i, ctx, u=u):
            """``Q_{Z^(k)}(a) ∝ exp( (S^(k)_{w_i,a} + ctx^(k)(a)) / λ_Z )``."""
            return [
                _softmax([(u[i][k][a] + ctx[k][a]) / lamZ for a in range(dp)])
                for k in range(K)
            ]

        zero = [[0.0] * dp for _ in range(K)]
        if m.cfg.schedule == "serial":
            qbar = []
            for t in range(n):
                rows = _keys_for_slot(m, qbar, t, k_head)
                q = step(t, zero)
                for _ in range(m.cfg.tau_obs):
                    q = step(t, _head_round(m, q, rows, k_child))
                qbar.append(q)
        else:
            qbar = [step(i, zero) for i in range(n)]
            for _ in range(m.cfg.n_iters):
                qbar = [
                    step(i, _head_round(m, qbar[i], _keys_for_slot(m, qbar, i, k_head), k_child))
                    for i in range(n)
                ]
        out.append(qbar)
    return torch.tensor(out, dtype=torch.float64)


def slot_mfvi_by_hand(m, rows, k_child):
    """§22.2's mean-field readout for one slot, from the raw parameters, in loops.

    The three update equations verbatim, including the derived [MASK] of §17.1
    (``Q_W^(0) ∝ exp(b)``, ``s̄^(k)_a = Σ_w Q_W^(0)(w) S^(k)_{w,a}``) and the final word
    message ``Σ_k Σ_a Q_{Z^(k)}(a) S^(k)_{w,a}``. Returns unnormalised logits.
    """
    V, K, dp = m.cfg.vocab_size, m.cfg.n_components, m.cfg.d_label
    lamZ, lamW = m.cfg.lambda_Z, m.cfg.lambda_W
    S = m.S.detach().reshape(V, K, dp)
    b = [float(x) for x in m.b.detach()]
    qw0 = _softmax(b)
    sbar = [[sum(qw0[w] * float(S[w, k, a]) for w in range(V)) for a in range(dp)] for k in range(K)]
    q = [_softmax([sbar[k][a] / lamZ for a in range(dp)]) for k in range(K)]
    for _ in range(m.cfg.tau):
        ctx = _head_round(m, q, rows, k_child)
        q = [_softmax([(sbar[k][a] + ctx[k][a]) / lamZ for a in range(dp)]) for k in range(K)]
    return torch.tensor(
        [
            (b[w] + sum(q[k][a] * float(S[w, k, a]) for k in range(K) for a in range(dp))) / lamW
            for w in range(V)
        ],
        dtype=torch.float64,
    )


def test_contracted_keys_route_the_head_component():
    m = fac(**BF)
    child, head = component_assignment(m.cfg.h, m.cfg.n_components, "roundrobin")
    assert child == [0, 1, 0, 1] and head == [0, 0, 1, 1]  # the documented default
    assert m.k_child.tolist() == child and m.k_head.tolist() == head

    idx = tokens(V=BF["vocab_size"], shape=(2, 3), seed=3)
    qbar = m.content_stream(idx).detach()
    Bk = m.contract(qbar)
    for t in range(idx.shape[1] + 1):
        got = m._slot_keys(Bk, t)
        want = keys_by_hand(m, qbar, t, head)
        assert got.shape == want.shape == (idx.shape[0], m.cfg.h, 1 + t, m.cfg.d_label)
        assert torch.allclose(got, want, atol=1e-12, rtol=0), f"slot {t}"


@pytest.mark.parametrize("h", [4, 1])
def test_exact_readout_equals_brute_force_at_K2(h):
    """The decisive one: a product of K mixtures is the true marginal of the slot joint.

    ``h = 1`` is the degenerate case the module docstring claims is not a special case:
    component 1 then has no channel assigned to it, ``log μ^(1) ≡ 0``, and its factor
    contributes the empty product ``Σ_a exp(S^(1)_{w,a})``. The enumeration sums over
    ``Z^(1)`` regardless, so it says whether that reading is right.
    """
    m = fac(**dict(BF, h=h))
    K, dp = m.cfg.n_components, m.cfg.d_label
    child = component_assignment(m.cfg.h, K, "roundrobin")[0]
    S_blocks = m.S.detach().reshape(m.cfg.vocab_size, K, dp)
    b = m.b.detach()

    idx = tokens(V=BF["vocab_size"], shape=(2, 3), seed=3)
    Bk = m.contract(m.content_stream(idx))
    for t in range(idx.shape[1] + 1):
        B_full = m._slot_keys(Bk, t).detach()
        got = torch.log_softmax(m.slot_exact_readout(B_full), dim=-1)
        for s in range(idx.shape[0]):
            channels = [B_full[s, c] for c in range(m.cfg.h)]
            want = brute_force_log_probs(channels, child, S_blocks, b, K, dp)
            assert torch.allclose(got[s], want, atol=1e-10, rtol=0), f"slot {t}, seq {s}"


@pytest.mark.parametrize("schedule", ["parallel", "serial"])
def test_content_stream_matches_the_update_equations_at_K2(schedule):
    """``q̄`` against §22.2's updates written out in loops — the KV cache, pinned.

    Without this the ``K`` components of the content stream are interchangeable as far as
    the rest of the file can tell: every other test consumes ``q̄`` from the model itself.
    """
    m = fac(schedule=schedule, **BF)
    child, head = component_assignment(m.cfg.h, m.cfg.n_components, "roundrobin")
    idx = tokens(V=BF["vocab_size"], shape=(2, 3), seed=3)
    got = m.content_stream(idx)
    want = content_by_hand(m, idx, child, head)
    assert got.shape == want.shape == (2, 3, m.cfg.n_components, m.cfg.d_label)
    assert torch.allclose(got, want, atol=1e-12, rtol=0), (got - want).abs().max()


def test_mfvi_slot_readout_matches_the_update_equations_at_K2():
    """The mean-field readout against the three update equations, written out in loops.

    The exact readout has its brute force above; this is the corresponding independent
    reference for the ablation readout, and it is what pins the pairing between the
    component axis of ``Q_Z`` and the blocks of ``S`` in the final word message.
    """
    m = fac(**dict(BF, readout="mfvi"))
    child, head = component_assignment(m.cfg.h, m.cfg.n_components, "roundrobin")
    idx = tokens(V=BF["vocab_size"], shape=(2, 3), seed=3)
    qbar = m.content_stream(idx).detach()
    Bk = m.contract(qbar)
    for t in range(idx.shape[1] + 1):
        got = m.slot_mfvi_readout(m._slot_keys(Bk, t))
        for s in range(idx.shape[0]):
            want = slot_mfvi_by_hand(m, _keys_for_slot(m, qbar[s].tolist(), t, head), child)
            assert torch.allclose(got[s], want, atol=1e-10, rtol=0), f"slot {t}, seq {s}"


def test_forward_matches_the_slot_path_at_K2():
    """The vectorised forward and the per-slot assembly are the same computation."""
    for readout in ("exact", "mfvi"):
        m = fac(n_components=2, readout=readout)
        idx = tokens()
        logits = m(idx)
        Bk = m.contract(m.content_stream(idx))
        for t in range(idx.shape[1]):
            B_full = m._slot_keys(Bk, t)
            slot = (
                m.slot_exact_readout(B_full)
                if readout == "exact"
                else m.slot_mfvi_readout(B_full)
            )
            assert torch.allclose(logits[:, t], slot, atol=1e-11, rtol=0), f"{readout} slot {t}"


# ---------------------------------------------- the §22.2 prediction, measured --


def _centred_rank(logits, tol=1e-8):
    """Affine dimension of a set of log-conditionals: rank after removing the mean row."""
    rows = torch.log_softmax(logits.reshape(-1, logits.shape[-1]), dim=-1)
    rows = rows - rows.mean(dim=0, keepdim=True)
    return int(torch.linalg.matrix_rank(rows, atol=tol, rtol=0))


RANK_CFG = dict(vocab_size=12, d=8, h=4, gamma=1)  # |V| - 1 > d, or the bound is vacuous


@pytest.mark.parametrize("K", [1, 2, 4])
def test_mean_field_readout_is_affine_of_dimension_at_most_d(K):
    """§22.2: "logits = b + Σ_k S^(k) q^(k), affine dimension ≤ D" — factoring buys nothing.

    Measured: 8, 7, 5 for K = 1, 2, 4 at d = 8 — never above d, whatever K is.
    """
    m = fac(n_components=K, readout="mfvi", **RANK_CFG)
    logits = m(tokens(V=12, shape=(8, 7), seed=5))
    assert _centred_rank(logits) <= m.cfg.d


@pytest.mark.parametrize("K", [1, 2, 4])
def test_exact_readout_escapes_the_affine_family(K):
    """"the gain exists only through the exact readout" — same contexts, higher rank.

    Measured: 12 (= |V|, the maximum available) for every K, against the mean-field
    readout's <= 8 on the identical parameters and the identical contexts. The escape is
    Proposition 22.1(iii) and it is a property of the readout, not of the factoring —
    which is exactly why §22.2's prediction is that K only pays under ``exact``.
    """
    m = fac(n_components=K, readout="exact", **RANK_CFG)
    logits = m(tokens(V=12, shape=(8, 7), seed=5))
    assert _centred_rank(logits) > m.cfg.d


# ------------------------------------------------------------------- hygiene --


@pytest.mark.parametrize("K", [1, 2, 4])
@pytest.mark.parametrize("readout", ["exact", "mfvi"])
@pytest.mark.parametrize("schedule", ["parallel", "serial"])
def test_shapes_and_finiteness(K, readout, schedule):
    m = fac(n_components=K, readout=readout, schedule=schedule)
    idx = tokens()
    B, n = idx.shape
    h, V, dp, nd = m.cfg.h, m.cfg.vocab_size, m.cfg.d_label, m.cfg.n_dist
    assert dp == m.cfg.d // K

    qbar = m.content_stream(idx)
    assert qbar.shape == (B, n, K, dp)
    assert torch.allclose(qbar.sum(-1), torch.ones(B, n, K, dtype=qbar.dtype), atol=1e-12)

    Bk = m.contract(qbar)
    assert Bk.shape == (nd, B, h, n, dp)
    assert m.exact_log_mu(Bk).shape == (B, n, K, dp)

    G, alpha = m._arc_message(qbar, Bk)
    assert G.shape == (B, n, K, dp)
    assert alpha.shape == (B, h, n, 1 + n)

    logits = m(idx)
    assert logits.shape == (B, n, V)
    assert torch.isfinite(logits).all()
    assert m.next_token_logits(idx).shape == (B, V)

    loss = m.loss(idx)
    assert loss.shape == () and torch.isfinite(loss)
    assert torch.isfinite(m.loss(idx, ignore_first=1))

    # the diagnostics path (src/diagnostics.py, src/train.py) must still execute
    trace = []
    m.content_stream(idx, trace=trace)
    assert trace and all(math.isfinite(v) for row in trace for v in row.values())


@pytest.mark.parametrize("K", [1, 2, 4])
@pytest.mark.parametrize("readout", ["exact", "mfvi"])
def test_a_future_token_cannot_move_the_present_logits(K, readout):
    m = fac(n_components=K, readout=readout)
    idx = tokens()
    n = idx.shape[1]
    base = m(idx)
    for t in range(n):
        alt = idx.clone()
        alt[:, t] = (idx[:, t] + 3) % m.cfg.vocab_size
        other = m(alt)
        assert torch.equal(base[:, : t + 1], other[:, : t + 1]), (
            f"K={K} {readout}: changing position {t} moved a slot <= {t}"
        )
        if t + 1 < n:
            assert not torch.equal(base[:, t + 1 :], other[:, t + 1 :])


@pytest.mark.parametrize("K", [2, 4])
@pytest.mark.parametrize("readout", ["exact", "mfvi"])
def test_no_gradient_path_from_slot_t_into_the_future(K, readout):
    """``CLAUDE.md`` constraint 3, the half that is not in dispute (see test_04's header).

    Slot ``t``'s loss may reach ``q̄_j`` only for ``j < t``; a gradient at ``j >= t`` is
    an anti-causal path. The prefix beliefs are made leaf tensors and left *attached* —
    reading ``.grad`` on a detached tensor would be green whatever the code does.
    """
    m = fac(n_components=K, readout=readout)
    idx = tokens()
    n = idx.shape[1]
    for slot in range(1, n):
        qbar = m.content_stream(idx).detach().requires_grad_(True)  # (B, n, K, d')
        Bk = m.contract(qbar)
        logits = (
            m._logits_from_log_mu(m.exact_log_mu(Bk))
            if readout == "exact"
            else m.mfvi_readout(Bk)
        )
        torch.nn.functional.cross_entropy(logits[:, slot], idx[:, slot]).backward()
        g = qbar.grad
        assert g is not None and g.shape == qbar.shape
        assert g[:, slot:].abs().max() == 0.0, f"slot {slot} leaked gradient into j >= {slot}"
        assert g[:, :slot].abs().max() > 0.0, f"slot {slot} reached no prefix belief at all"


@pytest.mark.parametrize("K", [1, 2, 4])
@pytest.mark.parametrize("readout", ["exact", "mfvi"])
def test_gradients_reach_every_parameter_and_are_finite(K, readout):
    m = fac(n_components=K, readout=readout)
    (m.loss(tokens()) + m.arc_regulariser()).backward()
    for name, p in m.named_parameters():
        assert p.grad is not None, f"no gradient reached {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite gradient on {name}"
        assert p.grad.abs().sum() > 0, f"zero gradient on {name}"


@pytest.mark.parametrize("K", [2, 4])
def test_S_is_one_tensor_read_in_both_roles(K):
    """Tying is a property of the computation: one matrix, two directions (§16(b))."""
    m = fac(n_components=K)
    V, d = m.cfg.vocab_size, m.cfg.d
    named = dict(m.named_parameters())
    vocab_shaped = [k for k, p in named.items() if V in tuple(p.shape)]
    assert vocab_shaped == ["S"] or sorted(vocab_shaped) == ["S", "b"]
    assert named["S"].shape == (V, d), "S must stay (V, d) whatever K is"
    assert all(tuple(p.shape) != (d, V) for p in m.parameters())

    idx = tokens()
    # emission role: with the context held fixed, moving S moves the logits
    log_mu = m.exact_log_mu(m.contract(m.content_stream(idx))).detach()
    before_em = m._logits_from_log_mu(log_mu)
    before_un = m.content_stream(idx)
    with torch.no_grad():
        m.S[2, 0] += 1.0  # a single entry: a whole row is a gauge freedom of the softmax
    assert not torch.allclose(before_em, m._logits_from_log_mu(log_mu))
    # unary role: the same tensor, read the other way, moves the filtering marginals
    assert not torch.allclose(before_un, m.content_stream(idx))
    # and one backward deposits both contributions on the same accumulator
    m.zero_grad(set_to_none=True)
    m.loss(idx).backward()
    assert m.S.grad is not None and m.S.grad.shape == (V, d)


def test_parameter_budget_at_fixed_total_width():
    """§22.2: ``|V| K d' = |V| D``. The embedding block is exactly K-independent.

    The non-embedding block is *not* constant and the docstring of ``src/factored.py``
    says so: the arc factors act ``d' x d'``, so a larger K is strictly cheaper. The
    comparison is fair in the direction that matters — factoring never buys its effect
    with extra parameters.
    """
    counts = {K: fac(n_components=K, d=8, h=4).num_parameters() for K in (1, 2, 4)}
    V, d = 7, 8
    for K, c in counts.items():
        assert c["embedding"] == V * d + V, f"K={K}"
    assert len({c["embedding"] for c in counts.values()}) == 1
    assert (
        counts[1]["non_embedding"] > counts[2]["non_embedding"] > counts[4]["non_embedding"]
    )
    for K in (2, 4):
        assert counts[K]["total"] < counts[1]["total"]
        assert counts[K]["embedding"] + counts[K]["non_embedding"] == counts[K]["total"]


@pytest.mark.parametrize("rank", [None, 2])
def test_arc_regulariser_is_the_mean_square_of_the_arc_scores_at_K2(rank):
    """The L2 term is a *mean* over the ``d' x d'`` arc table, so the coefficient does not
    depend on ``d``, ``h`` or the number of buckets.

    Under the Kruskal form ``T`` is never materialised and the entry count is read off the
    factors; reading it off the *total* width would divide by a further ``K²``. Nothing
    else here asserts the value of ``arc_regulariser`` at ``K > 1`` — the ``K = 1`` test
    compares it against the flat model, where ``d' = d`` and the error cancels.
    """
    m = fac(n_components=2, rank=rank)
    T = m.arc_scores()
    assert T.shape == (m.cfg.n_dist, m.cfg.h, m.cfg.d_label, m.cfg.d_label)
    want = (T**2).mean() + (m.r_root**2).mean()
    assert torch.allclose(m.arc_regulariser(), want, atol=1e-12, rtol=0)


@pytest.mark.parametrize("K", [2, 4])
def test_qnorm_agrees_with_fixed_at_a_uniform_belief(K):
    """``qnorm`` divides by ``||q||_2 · sqrt(d')``, which is exactly 1 at the uniform belief.

    That is the documented property of the mode — it reproduces Wu & Tu's ``λ_H = 1/d``
    at initialisation and only differs as the query sharpens. The querying variable under
    factoring has ``d'`` labels, so reading the *total* width here would leave a residual
    factor ``sqrt(K)`` and the two modes would disagree at initialisation. With ``S = 0``
    every label belief starts uniform, so one content-stream iteration is exactly that
    comparison.
    """
    out = {}
    for mode in ("fixed", "qnorm"):
        m = fac(n_components=K, temp_mode=mode, lambda_H=None, n_iters=1)
        with torch.no_grad():
            m.S.zero_()
        out[mode] = m.content_stream(tokens())
    assert torch.allclose(out["fixed"], out["qnorm"], atol=1e-12, rtol=0), (
        (out["fixed"] - out["qnorm"]).abs().max()
    )
    # ... and the agreement must not be vacuous: the beliefs did move off uniform.
    assert not torch.allclose(
        out["fixed"], torch.full_like(out["fixed"], 1.0 / out["fixed"].shape[-1]), atol=1e-6
    )


@pytest.mark.parametrize("K", [2, 4])
@pytest.mark.parametrize("temp_mode", ["qnorm", "qknorm"])
@pytest.mark.parametrize("readout", ["exact", "mfvi"])
def test_temperature_modes_execute_at_K_above_one(K, temp_mode, readout):
    """``_scaled_logits`` is inherited, and the factored decoder is the only caller that
    hands it a query carrying a channel axis. Both the vectorised path (``_arc_message``)
    and the single-slot path (``_slot_message``) are exercised here; every other test in
    this file runs at ``temp_mode="fixed"``.
    """
    m = fac(n_components=K, temp_mode=temp_mode, readout=readout, lambda_H=None)
    idx = tokens()
    q = m.content_stream(idx)
    assert q.shape == (idx.shape[0], idx.shape[1], K, m.cfg.d_label)
    assert torch.allclose(q.sum(-1), torch.ones_like(q.sum(-1)), atol=1e-12)
    assert torch.isfinite(m(idx)).all()
    assert torch.isfinite(m.next_token_logits(idx)).all()  # the `_slot_message` path
    m.loss(idx).backward()
    for name, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


def test_diagonal_assignment_never_mixes_components():
    child, head = component_assignment(6, 3, "diagonal")
    assert child == head == [0, 1, 2, 0, 1, 2]
    m = fac(n_components=2, channel_assignment="diagonal")
    assert m.k_child.tolist() == m.k_head.tolist()
    assert torch.isfinite(m(tokens())).all()


def test_roundrobin_covers_every_component_pair_at_h_equals_K_squared():
    child, head = component_assignment(9, 3, "roundrobin")
    assert sorted(zip(child, head)) == sorted(itertools.product(range(3), repeat=2))


def test_bad_configurations_raise():
    with pytest.raises(ValueError, match="does not divide"):
        cfg(d=8, n_components=3)
    with pytest.raises(ValueError, match="exceeds d'"):
        cfg(d=8, n_components=4, rank=3)  # d' = 2
    with pytest.raises(ValueError, match="channel_assignment"):
        cfg(n_components=2, channel_assignment="spiral")
    with pytest.raises(ValueError, match="n_components"):
        CausalPTDecoder(cfg(n_components=2))  # the flat class must refuse it
    with pytest.raises(NotImplementedError, match="global head"):
        FactoredPTDecoder(cfg(n_components=2, n_global=3, readout="mfvi"))


# ------------------------------------------------------------ training smoke --


def test_a_few_adam_steps_at_K2_do_not_produce_nan():
    torch.manual_seed(0)
    m = FactoredPTDecoder(cfg(n_components=2, vocab_size=11, d=8, h=4, rank=2))  # float32
    opt = torch.optim.Adam(m.parameters(), lr=3e-2)
    idx = tokens(V=11, shape=(4, 8), seed=7)
    losses = []
    for _ in range(8):
        opt.zero_grad(set_to_none=True)
        loss = m.loss(idx) + 5e-4 * m.arc_regulariser()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    assert all(math.isfinite(x) for x in losses), losses
    assert all(torch.isfinite(p).all() for p in m.parameters())
    assert losses[-1] < losses[0], losses  # 8 steps on 32 tokens: it must be able to fit


# --------------------------------------------------- the update equations themselves --
#
# ``CLAUDE.md`` names this check by name ("MFVI updates: check the free energy is
# non-increasing across iterations on a toy graph") and ``tests/test_08_free_energy.py``
# runs it, but only for the flat model: ``src.energy.slot_free_energy`` takes a single
# ``(B, d)`` label belief and has no factored form. Every other test in this file is
# blind to a *typo inside* the factored update equations — shapes, normalisation,
# causality, the K=1 collapse and even the brute-force agreement of the exact readout all
# survive a wrong lambda or a message routed to the wrong component's belief, because
# none of them exercises the mean-field loop as a descent.
#
# Under §22.2's energy the functional the loop performs coordinate descent on is
#
#     F = E - lambda_W H(Q_W) - lambda_Z sum_k H(Q_Z^(k)) - lambda_H sum_c H(Q_c)
#     E = - sum_w Q_W(w) b_w
#         - sum_k sum_{w,a} Q_W(w) Q_Z^(k)(a) S^(k)_{w,a}
#         - sum_c sum_j sum_a Q_c(j) Q_Z^(k(c))(a) B^(c)_{j,a}
#
# with one entropy term *per component* and the head term reading the belief of the
# channel's own child component. Written out here in the test rather than imported, for
# the same reason the enumerations above are.


def _slot_free_energy(m, qw, qz, alpha, B_full):
    """``F`` above for one slot. ``qz`` is ``(B, K, d')``, ``alpha`` is ``(B, h, 1+t)``."""

    def ent(p):
        return -(p * torch.where(p > 0, p.clamp(min=1e-300).log(), torch.zeros_like(p))).sum(-1)

    if qw.dim() == 1:
        qw = qw.unsqueeze(0).expand(qz.shape[0], -1)
    S_k = m.S.view(m.cfg.vocab_size, m.cfg.n_components, m.cfg.d_label)
    energy = -torch.einsum("bke,bke->b", qz, torch.einsum("bw,wke->bke", qw, S_k))
    # the head term reads component k(c), which is what makes this check sensitive to the
    # routing: contracting against the wrong component is no longer a descent direction.
    energy = energy - torch.einsum(
        "bcj,bcja,bca->b", alpha, B_full, qz.index_select(1, m.k_child)
    )
    if m.b is not None:
        energy = energy - qw @ m.b
    free = energy - m.cfg.lambda_W * ent(qw)
    free = free - m.cfg.lambda_Z * ent(qz).sum(-1)  # one entropy per label component
    return free - m.cfg.lam_H * ent(alpha).sum(-1)  # and one per channel


def _free_energy_curve(m, idx, slot, tau):
    m.cfg.tau = tau
    B_full = m._slot_keys(m.contract(m.content_stream(idx)), slot)
    qw0, _ = m._word_prior()  # Q_W is held at exp(b) through the inner loop (§17.1)
    trace: list = []
    m.slot_mfvi_readout(B_full, trace=trace)
    return torch.stack([_slot_free_energy(m, qw0, qz, a, B_full) for qz, a, _ in trace])


@pytest.mark.parametrize("K", [1, 2, 4])
@pytest.mark.parametrize("assignment", ["roundrobin", "diagonal"])
def test_factored_slot_free_energy_is_non_increasing(K, assignment):
    m = fac(n_components=K, readout="mfvi", channel_assignment=assignment)
    idx = tokens()
    for slot in range(1, idx.shape[1] + 1):
        curve = _free_energy_curve(m, idx, slot, tau=8)
        rise = float((curve[1:] - curve[:-1]).max())
        assert rise <= 1e-10, f"K={K} {assignment} slot {slot}: free energy rose by {rise:.3e}"


def test_the_free_energy_check_has_teeth_at_K2():
    """A sign error in the H-update — the most realistic typo in these equations, and the
    one every other test in this file survives untouched — must break monotonicity."""
    m = fac(n_components=2, readout="mfvi")
    idx = tokens()
    B_full = m._slot_keys(m.contract(m.content_stream(idx)), idx.shape[1])
    qw0, _ = m._word_prior()
    sbar = m._word_message()
    qz = torch.softmax(sbar / m.cfg.lambda_Z, dim=-1).expand(
        idx.shape[0], m.cfg.n_components, m.cfg.d_label
    )
    curve = []
    for _ in range(6):
        qc = qz.index_select(1, m.k_child)
        alpha = torch.softmax(
            -torch.einsum("bca,bcja->bcj", qc, B_full) / m.cfg.lam_H, dim=-1  # the mutation
        )
        curve.append(_slot_free_energy(m, qw0, qz, alpha, B_full))
        ctx = torch.einsum(
            "kc,bca->bka", m.child_onehot, torch.einsum("bcj,bcja->bca", alpha, B_full)
        )
        qz = torch.softmax((sbar + ctx) / m.cfg.lambda_Z, dim=-1)
        curve.append(_slot_free_energy(m, qw0, qz, alpha, B_full))
    rise = float((torch.stack(curve[1:]) - torch.stack(curve[:-1])).max())
    assert rise > 1e-6, f"a sign-flipped update stayed monotone (max rise {rise:.3e})"
