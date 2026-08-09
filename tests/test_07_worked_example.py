"""Check 7 — the worked example of ``causal_pt_output_note.pdf`` §5, number for number.

These are not numbers this implementation produced and then enshrined. They are an
independently derived reference: the note works the whole slot through by hand, in both
modes, for V = {the, cat, sat, mat}, d = 2, one channel, lambda_Z = lambda_H = 1.
Reproducing *those* numbers is what makes the check worth anything.

Tolerance is 5e-4 because the note prints three decimals.
"""

import torch

from experiments.exp0_decoder_validation.worked_example import build_model

TOL = 5e-4
IDX = torch.tensor([[0, 1, 2]])  # the cat sat


def _close(got, want):
    return torch.allclose(got, torch.tensor(want, dtype=torch.float64), atol=TOL, rtol=0)


def test_observed_pass_initial_label_beliefs():
    """Q_Z^(0) = sigma(S_{w,.}) for the three observed words."""
    m = build_model()
    Sw = m.S[IDX]
    q0 = torch.softmax(Sw[0], dim=-1)
    assert _close(q0[0], [0.953, 0.047])
    assert _close(q0[1], [0.982, 0.018])
    assert _close(q0[2], [0.018, 0.982])


def test_observed_pass_frozen_filtering_marginals():
    """q_bar after one round of (2)-(3) per slot."""
    qbar = build_model().content_stream(IDX)[0]
    assert _close(qbar[0], [0.818, 0.182])
    assert _close(qbar[1], [0.957, 0.043])
    assert _close(qbar[2], [0.006, 0.994])


def test_cached_contracted_arc_scores():
    """B_{t,.} = sum_b q_bar_t(b) T_{.,b} — the KV cache, as the note prints it."""
    m = build_model()
    Bk = m.contract(m.content_stream(IDX))
    B = Bk[0, 0, 0]  # (bucket 0, batch 0, channel 0) -> (n, d)
    assert _close(B[0], [0.365, 0.818])
    assert _close(B[1], [0.085, 0.957])
    assert _close(B[2], [1.987, 0.006])


def test_predictive_slot_prior_word_message():
    """Q_W^(0) = sigma(b), s_bar = sum_w Q_W^(0)(w) S_{w,.}, Q_Z^(0) = sigma(s_bar)."""
    m = build_model()
    qw0, sbar = m._word_prior()
    assert _close(qw0, [0.475, 0.175, 0.175, 0.175])
    assert _close(sbar, [1.063, -1.063])
    assert _close(torch.softmax(sbar, dim=-1), [0.893, 0.107])


def test_predictive_slot_inner_rounds():
    """Attention logits F, head posterior Q_c and label posterior Q_Z, both rounds."""
    m = build_model()
    B_full = m._slot_keys(m.contract(m.content_stream(IDX)), 3)
    trace: list = []
    m.slot_mfvi_readout(B_full, trace=trace)

    want_F = [[0.160, 0.413, 0.178, 1.776], [0.074, 0.387, 0.128, 1.890]]
    want_Qc = [[0.120, 0.154, 0.122, 0.604], [0.104, 0.143, 0.110, 0.642]]
    want_Qz = [[0.951, 0.049], [0.956, 0.044]]

    for k in range(2):
        qz_in, alpha, _ = trace[2 * k]
        qz_out = trace[2 * k + 1][0]
        F = torch.einsum("ba,bcja->bcj", qz_in, B_full)
        assert _close(F[0, 0], want_F[k]), f"round {k + 1} attention logits"
        assert _close(alpha[0, 0], want_Qc[k]), f"round {k + 1} Q_c"
        assert _close(qz_out[0], want_Qz[k]), f"round {k + 1} Q_Z"


def test_predictive_readout_logits_and_probabilities():
    """Eq. (4) as the LM output layer: logits b_w + sum_a Q_Z(a) S_{w,a}."""
    m = build_model()
    logits = m.next_token_logits(IDX)[0]
    assert _close(logits, [2.368, 1.824, -1.824, 1.824])
    assert _close(torch.softmax(logits, dim=-1), [0.460, 0.267, 0.007, 0.267])


def test_prior_to_posterior_sharpens_the_slot():
    """§5, last paragraph: clamping W_4 = mat sharpens the belief and the attention.

    Entropy of q_bar falls from .180 to .040 nats, and the attention on the verb slot
    rises from .642 to .663.
    """
    m = build_model()
    Bk = m.contract(m.content_stream(IDX))
    B_full = m._slot_keys(Bk, 3)

    trace: list = []
    m.slot_mfvi_readout(B_full, trace=trace)
    qz_pred, alpha_pred, _ = trace[-1]

    # clamp W_4 = mat and run the observed step of the same slot
    S_mat = m.S[3]
    q = torch.softmax(S_mat / m.cfg.lambda_Z, dim=-1).unsqueeze(0)
    ctx, alpha_obs = m._slot_message(q, B_full)
    q_obs = torch.softmax((S_mat + ctx) / m.cfg.lambda_Z, dim=-1)

    ent = lambda p: float(-(p * p.log()).sum())
    assert abs(ent(qz_pred[0]) - 0.180) < 5e-3
    assert abs(ent(q_obs[0]) - 0.040) < 5e-3
    assert abs(float(alpha_pred[0, 0, 3]) - 0.642) < 5e-4
    assert abs(float(alpha_obs[0, 0, 3]) - 0.663) < 5e-4
