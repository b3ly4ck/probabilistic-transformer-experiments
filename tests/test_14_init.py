"""Check 14 - initialisation: the distribution, and the per-factor scales.

Prof. Tu, 2026-08-11: "Uniform initialisation (of what? B?) does not sound like a good
choice. How does it work with random initialisation?" The word "uniform" in the 2026-08
report named the *treatment* - one scale and one L2 coefficient for every factor - and not
the distribution, which had always been ``N(0, init_std^2)``. ``PTConfig.init_dist`` makes
the literal reading answerable too, and this file is the evidence for the answer: it shows
which distribution each setting actually draws, by measuring the drawn tensors.

Two families of claim are asserted, both by measurement and never by reading the source:

* **Scale.** All three distributions are matched on the standard deviation, so ``init_dist``
  is a change of *shape* only and a sweep over ``init_std`` stays interpretable across it.
* **Shape.** ``"uniform"`` has bounded support, ``sqrt(3) * std`` by the variance match;
  ``"normal"`` does not, and on a large tensor demonstrably exceeds that bound.
  ``"orthogonal"`` additionally leaves the factor semi-orthogonal, which is checked through
  the Gram matrix rather than through the name of the initialiser that was called.

Then the three per-factor scales - ``arc_init_std`` for ``T`` / ``U`` / ``V``,
``root_init_std`` for the ROOT column ``r``, ``b_glob_init_std`` for the global head ``B'``
- are asserted to move their own factor and nothing else. ``root_init_std`` and
``b_glob_init_std`` predate ``init_dist``; they are re-asserted here because the shared
``_init_`` helper is new underneath all of them, and a scale that silently reverted to
``init_std`` would undo the finding that motivated it (the ROOT row starting 121x larger
than the rows it competes with, exp0's status file).

Tensors are deliberately large enough that a sample standard deviation is a measurement and
not a coin flip: the relative sampling error of an empirical std over N draws is about
1/sqrt(2N), so the smallest tensor asserted on here (``r`` at 512 entries, 3.1%) sits well
inside the 15% tolerance. Nothing in this file runs a forward pass except the last test.
"""

import math

import pytest
import torch

from src import CausalPTDecoder, PTConfig
from src.data import random_batch
from src.train import TrainConfig, train

DISTS = ["normal", "uniform", "orthogonal"]

# Relative tolerance on an empirical standard deviation. At the sizes used below the
# sampling error is 3.1% at worst, so 15% is ~5 sigma and the test is not flaky; it is
# still tight enough to catch a wrong distribution constant (a missing sqrt(3) on the
# uniform bound would show up as a 42% error, a missing rescale on the orthogonal branch
# as a factor of several).
REL = 0.15


def big_model(seed: int = 0, **over) -> CausalPTDecoder:
    """A model built only to be measured - never run - so it can afford real tensor sizes.

    ``S`` is 200x64 = 12800 entries, ``T`` is 3x8x64x64 = 98304, ``r`` is 8x64 = 512.
    """
    base = dict(vocab_size=200, d=64, h=8, rank=None, gamma=2, init_std=0.05)
    base.update(over)
    torch.manual_seed(seed)
    return CausalPTDecoder(PTConfig(**base))


def measured(m):
    """``{name: empirical std}`` for every factor that is drawn rather than zeroed."""
    return {n: float(p.detach().std()) for n, p in m.named_parameters() if n != "b"}


def arc_names(m):
    return [n for n in ("T", "U", "V") if getattr(m, n) is not None]


# ------------------------------------------------------------------------------ scale --


@pytest.mark.parametrize("dist", DISTS)
@pytest.mark.parametrize("rank", [None, 16])
def test_every_distribution_lands_on_init_std(dist, rank):
    """The three settings are matched on the standard deviation, not on the support."""
    std = 0.05
    m = big_model(init_dist=dist, rank=rank, init_std=std)
    for name, got in measured(m).items():
        assert got == pytest.approx(std, rel=REL), f"{dist}/{name}: std {got:.5f}"
    # b is the log-unigram slot and is *not* random: it starts at exactly zero.
    assert float(m.b.abs().max()) == 0.0


@pytest.mark.parametrize("dist", DISTS)
def test_the_scale_is_the_only_thing_init_std_changes(dist):
    """Doubling ``init_std`` doubles every drawn factor. Guards against a hard-coded scale."""
    small = measured(big_model(init_dist=dist, init_std=0.05))
    large = measured(big_model(init_dist=dist, init_std=0.10))
    for name in small:
        assert large[name] / small[name] == pytest.approx(2.0, rel=REL), name


# ------------------------------------------------------------------------------ shape --


def test_uniform_has_bounded_support_and_normal_does_not():
    """This is the assertion that answers "which distribution is it".

    ``U(-a, a)`` with ``a = sqrt(3) * std`` has the right variance and cannot produce a
    value outside ``[-a, a]``; a Gaussian puts ~8.3% of its mass outside the same interval,
    so on 12800 draws exceeding it is certain. The two are therefore separated by an
    observation on the drawn tensor, with no reference to how it was drawn.
    """
    std = 0.05
    bound = math.sqrt(3.0) * std

    u = big_model(init_dist="uniform", init_std=std)
    n = big_model(init_dist="normal", init_std=std)

    for name, p in u.named_parameters():
        if name == "b":
            continue
        assert float(p.abs().max()) <= bound * (1 + 1e-6), f"uniform {name} left its support"
    # ...and it really is U(-a, a) and not a narrower box that happens to fit: on 12800
    # draws the sample maximum sits within a fraction of a percent of the bound.
    assert float(u.S.abs().max()) > 0.99 * bound

    assert float(n.S.abs().max()) > bound, (
        "a 'normal' draw stayed inside the uniform support on 12800 samples: it is not "
        "normal"
    )


def test_orthogonal_leaves_the_factor_semi_orthogonal():
    """The third setting is checked by its defining property, not by its name.

    ``S`` is 200x64, so semi-orthogonal means orthogonal *columns*: ``S^T S`` is a multiple
    of the identity. The multiple is the rescale that puts the factor at ``init_std``.
    """
    std = 0.05
    m = big_model(init_dist="orthogonal", init_std=std)
    S = m.S.detach()
    gram = S.T @ S
    diag = gram.diagonal()
    off = gram - torch.diag(diag)

    scale = float(diag.mean())
    assert float(diag.std()) < 1e-4 * scale, "the columns do not share a norm"
    assert float(off.abs().max()) < 1e-4 * scale, "the columns are not orthogonal"
    # the shared column norm is what makes the std come out at init_std
    assert math.sqrt(scale * S.shape[1] / S.numel()) == pytest.approx(std, rel=1e-3)

    # a normal draw of the same shape is nothing like orthogonal - the control
    ctl = big_model(init_dist="normal", init_std=std).S.detach()
    g = ctl.T @ ctl
    assert float((g - torch.diag(g.diagonal())).abs().max()) > 1e-2 * float(g.diagonal().mean())


@pytest.mark.parametrize("rank", [None, 16])
def test_orthogonal_leaves_every_slice_of_the_arc_factors_semi_orthogonal(rank):
    """The 4-D branch of ``_init_``, which the ``S`` test above never reaches.

    ``S`` and ``r`` are matrices; ``T`` is ``(n_dist, h, d, d)`` and ``U`` / ``V`` are
    ``(n_dist, h, d, rank)``. The helper flattens the leading axes and orthogonalises each
    trailing 2-D slice separately - 24 of them at ``gamma = 2, h = 8`` - and then restores
    the scale with a single global multiply. That last step is why the scale tests cannot
    stand in for this one: a version that orthogonalised the first slice and left the other
    23 at whatever ``torch.empty`` returned would be rescaled to land on ``init_std``
    anyway, and would pass every other assertion in this file. Here each slice is checked
    through its own Gram matrix, and the arc scores are where the setting has to hold -
    they are the factor Prof. Tu's question was about.
    """
    std = 0.05
    m = big_model(init_dist="orthogonal", rank=rank, init_std=std)
    ctl = big_model(init_dist="normal", rank=rank, init_std=std)
    assert arc_names(m), "no arc factor was built: the test would assert nothing"

    for name in arc_names(m):
        p = getattr(m, name).detach()
        flat = p.reshape(-1, p.shape[-2], p.shape[-1])
        gram = torch.einsum("kar,kas->krs", flat, flat)  # per slice, over the columns
        diag = torch.diagonal(gram, dim1=-2, dim2=-1)
        off = gram - torch.diag_embed(diag)
        scale = float(diag.mean())
        assert float(diag.std()) < 1e-4 * scale, f"{name}: the columns do not share a norm"
        assert float(off.abs().max()) < 1e-4 * scale, f"{name}: a slice is not orthogonal"
        # every column having squared norm `scale` over `flat.shape[-2]` rows is exactly
        # what puts the entries at init_std, so the two claims are one measurement apart.
        assert math.sqrt(scale / flat.shape[-2]) == pytest.approx(std, rel=1e-2)

        c = getattr(ctl, name).detach()
        cf = c.reshape(-1, c.shape[-2], c.shape[-1])
        cg = torch.einsum("kar,kas->krs", cf, cf)
        cd = torch.diagonal(cg, dim1=-2, dim2=-1)
        assert float((cg - torch.diag_embed(cd)).abs().max()) > 1e-2 * float(cd.mean()), (
            f"the 'normal' control for {name} came out orthogonal: the check is vacuous"
        )


# ------------------------------------------------------------------- per-factor scales --


@pytest.mark.parametrize("dist", DISTS)
@pytest.mark.parametrize("rank", [None, 16])
def test_arc_init_std_scales_the_arc_scores_and_nothing_else(dist, rank):
    """``arc_init_std`` is the scale of ``T`` (or ``U``, ``V``); ``S`` and ``r`` keep
    ``init_std``.

    The two are separate knobs because the arc scores reach the head message *contracted*
    against a near-uniform prefix belief while ``S`` reaches the label posterior directly,
    so one scale cannot be right for both.
    """
    m = big_model(init_dist=dist, rank=rank, init_std=0.02, arc_init_std=0.5)
    got = measured(m)
    assert got["S"] == pytest.approx(0.02, rel=REL), got
    assert got["r_root"] == pytest.approx(0.02, rel=REL), got
    for name in arc_names(m):
        assert got[name] == pytest.approx(0.5, rel=REL), f"{name}: {got}"


@pytest.mark.parametrize("rank", [None, 16])
def test_arc_init_std_none_falls_back_to_init_std(rank):
    m = big_model(rank=rank, init_std=0.02, arc_init_std=None)
    got = measured(m)
    for name in arc_names(m):
        assert got[name] == pytest.approx(0.02, rel=REL), f"{name}: {got}"


@pytest.mark.parametrize("dist", DISTS)
def test_root_init_std_scales_only_the_root_column(dist):
    """Pre-existing behaviour, re-asserted on top of the shared ``_init_``.

    ``r^(c)`` enters the attention in raw d-space while the arc scores arrive contracted,
    which measured at 121x for d=384, h=16, rank=64 (exp0's status file). If this knob
    silently reverted to ``init_std`` the ROOT sink would come back with it.
    """
    m = big_model(init_dist=dist, init_std=0.02, root_init_std=0.5)
    got = measured(m)
    assert got["r_root"] == pytest.approx(0.5, rel=REL), got
    assert got["S"] == pytest.approx(0.02, rel=REL), got
    assert got["T"] == pytest.approx(0.02, rel=REL), got

    fallback = measured(big_model(init_dist=dist, init_std=0.02, root_init_std=None))
    assert fallback["r_root"] == pytest.approx(0.02, rel=REL), fallback


@pytest.mark.parametrize("dist", DISTS)
def test_b_glob_init_std_scales_only_the_global_head(dist):
    """Pre-existing behaviour of the B.3.3 global head's matrix ``B'``.

    ``readout="mfvi"`` because ``PTConfig`` refuses ``n_global > 0`` with the exact readout:
    ``B'``'s contribution to ``log mu`` there is a position- and prefix-independent constant
    (§22.2, measured to 1e-12).
    """
    m = big_model(
        init_dist=dist, init_std=0.02, n_global=64, readout="mfvi", b_glob_init_std=0.5
    )
    got = measured(m)
    assert got["B_glob"] == pytest.approx(0.5, rel=REL), got
    assert got["S"] == pytest.approx(0.02, rel=REL), got
    assert got["r_root"] == pytest.approx(0.02, rel=REL), got

    fallback = measured(big_model(init_dist=dist, init_std=0.02, n_global=64, readout="mfvi"))
    assert fallback["B_glob"] == pytest.approx(0.02, rel=REL), fallback


def test_the_three_scales_are_independent_of_each_other():
    """All four knobs set at once, all four land where they were asked to."""
    m = big_model(
        init_std=0.02, arc_init_std=0.4, root_init_std=0.1, b_glob_init_std=0.7,
        n_global=64, readout="mfvi",
    )
    got = measured(m)
    want = {"S": 0.02, "T": 0.4, "r_root": 0.1, "B_glob": 0.7}
    for name, target in want.items():
        assert got[name] == pytest.approx(target, rel=REL), got


# ------------------------------------------------------------------ it has to run, too --


@pytest.mark.parametrize("dist", DISTS)
def test_a_model_under_each_init_dist_trains_without_nan(dist):
    """A handful of real optimiser steps through the shared training loop.

    A distribution that initialises to the right *scale* can still be unusable - an exactly
    orthogonal factor is a degenerate starting point in a way a Gaussian is not - and
    nothing above would notice. This is the cheapest statement that each setting produces a
    model the shared loop can actually take steps on: finite loss, finite perplexity, finite
    parameters afterwards. It is not a claim about which setting trains *better*; that is a
    sweep, not a unit test, and the data here is random noise on purpose.
    """
    torch.manual_seed(0)
    m = CausalPTDecoder(
        PTConfig(vocab_size=11, d=8, h=2, rank=None, gamma=2, n_iters=2,
                 init_dist=dist, init_std=0.1)
    )
    before = {n: p.detach().clone() for n, p in m.named_parameters()}

    torch.manual_seed(3)
    data = torch.randint(0, 11, (500,))
    cfg = TrainConfig(
        block_size=8, batch_size=4, max_steps=6, lr=0.05, warmup_steps=2,
        eval_every=3, eval_blocks=4, eval_train_blocks=2,
        log_every=10**6, diagnostics=False,
    )
    hist = train(m, data, data, cfg, random_batch, log=lambda _s: None)

    assert hist.val_ppl, "no evaluation was recorded"
    assert all(math.isfinite(v) for v in hist.val_loss), hist.val_loss
    assert all(math.isfinite(v) for v in hist.val_ppl), hist.val_ppl
    assert all(math.isfinite(v) for v in hist.train_ppl), hist.train_ppl
    for name, p in m.named_parameters():
        assert torch.isfinite(p).all(), f"{name} went non-finite during training"
        assert not torch.equal(p.detach(), before[name]), f"{name} never moved"
