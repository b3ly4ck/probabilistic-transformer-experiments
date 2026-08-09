"""Check 10 — the properties the message-scale diagnostics claim, asserted.

This model class has no layer norm and no residual, so what keeps activations bounded is
the simplex geometry rather than a normalisation layer. That is a *provable* bound, and
a bound worth asserting rather than believing:

    |G_i(a)| = |sum_c sum_{j in D_i} Q_c(j) B^(c)_{j,a}|  <=  h * max(max|T|, max|r|)

because Q_c is a distribution over D_i and B^(c)_{j,.} is itself an expectation of a row
of T^(c) under the distribution q_bar_j. Nothing here can diverge at fixed parameters;
only the parameters can grow, which is what the original's L2 penalty on the ternary
scores controls (Wu & Tu §4.2, Table 2: 5e-4 on PTB).
"""

import torch

from src.diagnostics import (
    contraction_rho,
    fixed_point_multiplicity,
    global_head_readout_term,
    message_scale_report,
    root_attention_mass,
    schedule_divergence,
)
from conftest import toy_model


def test_message_is_bounded_by_the_arc_scores(idx, schedule):
    """The convex-combination bound, which is what replaces layer norm here."""
    for std in (0.02, 0.5, 3.0):
        m = toy_model(schedule=schedule, init_std=std)
        bound = m.cfg.h * max(float(m.T.abs().max()), float(m.r_root.abs().max()))
        trace = message_scale_report(m, idx)
        assert trace, "no iterations were traced"
        for t in trace:
            assert t["G_absmax"] <= bound + 1e-9, f"|G| {t['G_absmax']} exceeded {bound}"


def test_trace_has_one_entry_per_iteration(idx):
    m = toy_model(n_iters=3)
    assert len(message_scale_report(m, idx)) == 3
    m.cfg.n_iters = 5
    assert len(message_scale_report(m, idx)) == 5


def test_trace_quantities_are_finite_and_in_range(idx):
    m = toy_model(init_std=1.0)
    for t in message_scale_report(m, idx):
        assert all(v == v and abs(v) < float("inf") for v in t.values())
        assert 0.0 <= t["attn_entropy_frac"] <= 1.0 + 1e-9
        assert t["ratio"] >= 0.0


def test_contraction_constant_is_positive_and_scales_with_the_arc_scores(idx):
    """rho grows quadratically in the arc scores, as Lemma 23.1 says."""
    rhos = []
    for std in (0.1, 0.2):
        m = toy_model(init_std=std)
        B_full = m._slot_keys(m.contract(m.content_stream(idx)), idx.shape[1])
        rhos.append(float(contraction_rho(m, B_full).mean()))
    assert all(r > 0 for r in rhos)
    assert rhos[1] > 3 * rhos[0]  # doubling the scale roughly quadruples rho


def test_global_head_term_is_constant_under_the_exact_readout(idx):
    """§22.2: mu_t gains the multiplicative term sum_m e^{B'_{m,a}} — position-free.

    Consequence, asserted here: however large m is, only d numbers of the m x d global
    matrix reach the exact readout *directly*. The two models are given the identical
    contracted prefix so that the readout contribution is isolated.
    """
    m = toy_model(readout="exact", n_global=9)
    term = global_head_readout_term(m)
    assert term.shape == (m.cfg.d,)

    plain = toy_model(readout="exact", n_global=0)
    plain.load_state_dict({k: v for k, v in m.state_dict().items() if k != "B_glob"})

    Bk = m.contract(m.content_stream(idx))  # one prefix, both readouts
    delta = m.exact_log_mu(Bk) - plain.exact_log_mu(Bk)
    assert torch.allclose(delta, term.expand_as(delta), atol=1e-12, rtol=0)


def test_global_head_still_moves_the_readout_through_the_content_stream(idx):
    """The complement, and the reason "G_t is a constant" would be the wrong summary.

    The direct readout contribution is a fixed d-vector, but the global head also enters
    the content stream, reshapes q_bar, and so moves log mu position by position. Only
    the two effects together describe what it does.
    """
    m = toy_model(readout="exact", n_global=9)
    plain = toy_model(readout="exact", n_global=0)
    plain.load_state_dict({k: v for k, v in m.state_dict().items() if k != "B_glob"})

    end_to_end = m.exact_log_mu(m.contract(m.content_stream(idx))) - plain.exact_log_mu(
        plain.contract(plain.content_stream(idx))
    )
    term = global_head_readout_term(m)
    assert not torch.allclose(end_to_end, term.expand_as(end_to_end), atol=1e-6, rtol=0)


def test_global_head_is_not_constant_in_the_content_stream(idx):
    """The complement: it does carry context where it is supposed to."""
    m = toy_model(n_global=9, init_std=0.5)
    q = m.content_stream(idx)
    msg, _ = m._global_message(q)
    assert msg.std(dim=1).max() > 0, "the global message was identical at every position"


def test_schedules_agree_when_the_model_is_degenerate_and_may_not_otherwise(idx):
    """Both schedules are legal (§12.3); they are not the same computation.

    At tiny parameter scale both land on the same near-uniform fixed point. At large
    scale they need not, and this asserts the measurement is actually sensitive rather
    than always returning zero.
    """
    small = schedule_divergence(toy_model(init_std=0.01), idx)
    assert small["tv_max"] < 1e-3

    large = schedule_divergence(toy_model(init_std=3.0), idx)
    assert large["tv_max"] > 0.1, "the schedule comparison is not sensitive to anything"


def test_rho_below_one_implies_a_unique_fixed_point(idx):
    """Lemma 23.1 with the word-message difference set to zero: rho is the contraction
    factor of the slot map, so rho < 1 makes it a contraction and the fixed point unique
    and reachable from any start. Asserted as an implication, not assumed."""
    m = toy_model(init_std=0.1)
    B_full = m._slot_keys(m.contract(m.content_stream(idx)), idx.shape[1])
    assert float(contraction_rho(m, B_full).max()) < 1.0
    fp = fixed_point_multiplicity(m, B_full, m.S[idx[:, -1]])
    assert fp["n_fixed_points"] == [1] * idx.shape[0]
    assert fp["max_separation"] == 0.0


def test_multistability_appears_only_far_above_the_bound(idx):
    """rho >= 1 *admits* multistability; it does not create it. The measured onset is far
    above 1, which is why the check exists as an experiment and not as an inference."""
    m = toy_model(init_std=6.0)
    B_full = m._slot_keys(m.contract(m.content_stream(idx)), idx.shape[1])
    assert float(contraction_rho(m, B_full).min()) > 1.0
    fp = fixed_point_multiplicity(m, B_full, m.S[idx[:, -1]])
    assert max(fp["n_fixed_points"]) > 1, "the multistability probe found nothing at rho >> 1"
    assert fp["max_separation"] > 0.5


def test_centred_rho_never_exceeds_the_raw_one(idx):
    """The lemma takes norms on centred vectors; the raw norm bounds that, so reporting
    the raw one can only over-state rho, never hide a violation."""
    for std in (0.05, 0.5, 3.0):
        m = toy_model(init_std=std)
        B_full = m._slot_keys(m.contract(m.content_stream(idx)), idx.shape[1])
        raw = contraction_rho(m, B_full)
        centred = contraction_rho(m, B_full, centred=True)
        assert (centred <= raw + 1e-12).all()


def test_root_attention_mass_is_measured(idx):
    m = toy_model()
    stats = root_attention_mass(m, idx)
    assert 0.0 <= stats["root_mass_mean"] <= 1.0
    assert stats["excess_over_uniform"] > 0.0
    for t in message_scale_report(m, idx):
        assert 0.0 <= t["root_mass"] <= 1.0
        assert t["root_mass_over_uniform"] >= 0.0


def test_default_config_is_the_source_table_2_row():
    """The defaults are Wu & Tu Table 2, PTB masked LM. Guard against silent drift."""
    from src import PTConfig

    cfg = PTConfig(vocab_size=10)
    assert (cfg.d, cfg.h, cfg.rank, cfg.gamma, cfg.n_iters) == (384, 16, 64, 3, 5)
    assert cfg.lam_H == 1.0 / 384 and cfg.lambda_Z == 1.0


def test_rank_above_d_is_rejected():
    from src import PTConfig

    with torch.no_grad():
        try:
            PTConfig(vocab_size=10, d=32, rank=64)
        except ValueError as e:
            assert "rank" in str(e)
        else:
            raise AssertionError("a Kruskal rank above d was accepted")


def test_loss_alignment_option_drops_the_leading_slots(idx):
    """PT and GPT do not score the same tokens by default; ignore_first is the fix."""
    m = toy_model()
    full = m.loss(idx)
    aligned = m.loss(idx, ignore_first=1)
    assert not torch.allclose(full, aligned)

    logits = m(idx)
    manual = torch.nn.functional.cross_entropy(
        logits[:, 1:].reshape(-1, m.cfg.vocab_size), idx[:, 1:].reshape(-1)
    )
    assert torch.allclose(aligned, manual, atol=1e-12, rtol=0)
