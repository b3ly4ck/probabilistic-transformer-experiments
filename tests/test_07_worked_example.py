"""Check 7 -- reproduce the worked example of causal_pt_output_note.pdf §5.

The reference numbers are independently derived in the note, so agreement tests
the update equations against something that was not written alongside this code.
The note uses the *sequential* per-slot schedule (clamp W_t, one round, freeze
qbar_t, cache B_t), which is why this test drives the slot primitives directly
rather than the layer-parallel content stream.

Setup: V = {the, cat, sat, mat}, d = 2 with labels {N, V}, h = 1,
lambda_Z = lambda_H = 1, tau = 1 observed and tau = 2 predictive.
Sentence: "the cat sat []".
"""

import torch

from src import mfvi
from src.config import PTConfig

CFG = PTConfig(vocab_size=4, d=2, n_channels=1, n_rounds=1, lambda_Z=1.0, lambda_H=1.0)

S = torch.tensor([[1.5, -1.5], [2.0, -2.0], [-2.0, 2.0], [2.0, -2.0]])
B_UNARY = torch.tensor([1.0, 0.0, 0.0, 0.0])
T = torch.tensor([[[0.0, 2.0], [1.0, 0.0]]])  # (h, d, d)
R = torch.tensor([[0.0, 1.5]])  # (h, d)

SENTENCE = [0, 1, 2]  # the, cat, sat

EXPECTED_QBAR = [(0.818, 0.182), (0.957, 0.043), (0.006, 0.994)]
EXPECTED_CACHED_B = [(0.365, 0.818), (0.085, 0.957), (1.987, 0.006)]
EXPECTED_QC = [(1.0,), (0.414, 0.586), (0.476, 0.245, 0.280)]
EXPECTED_QZ_PRED = [(0.951, 0.049), (0.956, 0.044)]
EXPECTED_LOGITS = (2.368, 1.824, -1.824, 1.824)
EXPECTED_PROBS = (0.460, 0.267, 0.007, 0.267)

ATOL = 1e-3  # the note prints three decimals


def _observed_pass():
    """Left to right: clamp the word, one round, freeze qbar, cache B."""
    cached = [R[0]]  # index 0 is ROOT, whose row is r
    qbars, qcs = [], []
    for w in SENTENCE:
        Bkey = torch.stack(cached).unsqueeze(0)  # (h=1, K, d)
        Q_Z, Q_c = mfvi.run_slot_mfvi(S[w], Bkey, CFG, n_rounds=1)
        qbars.append(Q_Z)
        qcs.append(Q_c[0])
        cached.append(torch.einsum("e,ae->a", Q_Z, T[0]))  # B_t = T qbar_t
    return qbars, qcs, cached


def test_observed_pass_matches_the_note():
    qbars, qcs, cached = _observed_pass()
    for t, (got, want) in enumerate(zip(qbars, EXPECTED_QBAR), start=1):
        assert torch.allclose(got, torch.tensor(want), atol=ATOL), f"qbar_{t}: {got}"
    for t, (got, want) in enumerate(zip(qcs, EXPECTED_QC), start=1):
        assert torch.allclose(got, torch.tensor(want), atol=ATOL), f"Q_c at slot {t}: {got}"
    for t, (got, want) in enumerate(zip(cached[1:], EXPECTED_CACHED_B), start=1):
        assert torch.allclose(got, torch.tensor(want), atol=ATOL), f"cached B_{t}: {got}"


def test_predictive_slot_four_matches_the_note():
    _, _, cached = _observed_pass()
    Bkey = torch.stack(cached).unsqueeze(0)  # D_4 = {ROOT, 1, 2, 3}

    # Q_W^(0) proportional to exp(b); s_bar is the derived [MASK] embedding (§17.1)
    Q_W0 = torch.softmax(B_UNARY, dim=-1)
    s_bar = mfvi.word_message(Q_W0, S)
    assert torch.allclose(s_bar, torch.tensor([1.063, -1.063]), atol=ATOL), s_bar

    Q_Z, _, trace = mfvi.run_slot_mfvi(s_bar, Bkey, CFG, n_rounds=2, return_trace=True)
    for i, (want) in enumerate(EXPECTED_QZ_PRED):
        assert torch.allclose(trace[i][0], torch.tensor(want), atol=ATOL), (
            f"Q_Z after round {i + 1}: {trace[i][0]}"
        )

    logits = mfvi.mfvi_readout_logits(Q_Z, S, B_UNARY, CFG)
    assert torch.allclose(logits, torch.tensor(EXPECTED_LOGITS), atol=ATOL), logits
    probs = torch.softmax(logits, dim=-1)
    assert torch.allclose(probs, torch.tensor(EXPECTED_PROBS), atol=ATOL), probs


def test_context_kills_the_verb():
    """The note's reading of the result: .993 of the mass on the noun-like cluster."""
    _, _, cached = _observed_pass()
    Bkey = torch.stack(cached).unsqueeze(0)
    Q_W0 = torch.softmax(B_UNARY, dim=-1)
    s_bar = mfvi.word_message(Q_W0, S)
    Q_Z, _ = mfvi.run_slot_mfvi(s_bar, Bkey, CFG, n_rounds=2)
    probs = torch.softmax(mfvi.mfvi_readout_logits(Q_Z, S, B_UNARY, CFG), dim=-1)
    assert probs[2] < 0.01  # "sat", the verb
    assert probs[0] > probs[1] and probs[0] > probs[3]  # the prior b favours "the"
