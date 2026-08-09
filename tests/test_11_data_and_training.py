"""Check 11 — the data pipeline and the shared training loop.

The training loop is the one component every model in the comparison shares, so a defect
here is not a bug in one model, it invalidates every number the project will report. The
two properties that matter most are asserted directly: evaluation is deterministic, and the
scored token set is exactly the one the metric claims.
"""

import math
from pathlib import Path

import pytest
import torch

from src import CausalPTDecoder, PTConfig
from src.data import (
    load_ptb,
    random_batch,
    sequential_batches,
    unigram_perplexity,
)
from src.train import TrainConfig, evaluate, lr_at, train

PTB = Path("data/ptb/ptb.train.txt")
needs_ptb = pytest.mark.skipif(not PTB.exists(), reason="PTB corpus not present (data/ is gitignored)")


# ------------------------------------------------------------------ synthetic corpus --


def synthetic(vocab: int = 11, n: int = 4000) -> torch.Tensor:
    """A learnable stream: token t+1 is a deterministic function of token t."""
    out = [0]
    for _ in range(n - 1):
        out.append((out[-1] * 3 + 1) % vocab)
    return torch.tensor(out, dtype=torch.long)


def test_sequential_batches_are_deterministic_and_disjoint():
    data = torch.arange(100)
    first = list(sequential_batches(data, batch_size=3, block_size=10))
    second = list(sequential_batches(data, batch_size=3, block_size=10))
    assert all(torch.equal(a, b) for a, b in zip(first, second))

    seen = torch.cat([b.reshape(-1) for b in first])
    assert seen.numel() == 100 and torch.equal(seen.sort().values, data)


def test_sequential_batches_limit_truncates():
    data = torch.arange(100)
    limited = list(sequential_batches(data, batch_size=3, block_size=10, limit=2))
    assert sum(b.shape[0] for b in limited) == 6


def test_random_batch_shape_and_reproducibility():
    data = torch.arange(500)
    g1 = torch.Generator().manual_seed(0)
    g2 = torch.Generator().manual_seed(0)
    a = random_batch(data, 4, 16, g1)
    b = random_batch(data, 4, 16, g2)
    assert a.shape == (4, 16) and torch.equal(a, b)
    # blocks are contiguous slices of the stream
    assert torch.equal(a[:, 1:] - a[:, :-1], torch.ones(4, 15, dtype=a.dtype))


def test_unigram_baseline_matches_a_hand_computation():
    train_data = torch.tensor([0, 0, 0, 1] * 25)
    ppl = unigram_perplexity(train_data, train_data, 2, block_size=4, batch_size=5)
    p = torch.tensor([0.75, 0.25])
    want = float(torch.exp(-(p * p.log()).sum()))
    assert abs(ppl - want) < 1e-6


# ------------------------------------------------------------------ learning-rate ----


def test_lr_schedule_warms_up_then_decays():
    cfg = TrainConfig(lr=1e-3, warmup_steps=10, max_steps=100, min_lr_frac=0.1)
    assert lr_at(0, cfg) == pytest.approx(1e-4)
    assert lr_at(9, cfg) == pytest.approx(1e-3)
    assert lr_at(10, cfg) == pytest.approx(1e-3, rel=1e-6)
    assert lr_at(99, cfg) < lr_at(50, cfg) < lr_at(10, cfg)
    assert lr_at(99, cfg) >= 1e-4 * 0.999


# ------------------------------------------------------------------ evaluation -------


def test_evaluation_is_deterministic_and_scores_the_declared_tokens():
    torch.manual_seed(0)
    m = CausalPTDecoder(PTConfig(vocab_size=11, d=8, h=2, rank=None, gamma=2, n_iters=2))
    data = synthetic(11, 200)
    cfg = TrainConfig(block_size=10, batch_size=4, ignore_first=1, device="cpu")

    first = evaluate(m, data, cfg)
    second = evaluate(m, data, cfg)
    assert first == second, "evaluation moved between two identical calls"

    n_blocks = data.numel() // cfg.block_size
    assert first["tokens"] == n_blocks * (cfg.block_size - cfg.ignore_first)
    assert first["ppl"] == pytest.approx(math.exp(first["loss"]))


def test_ignore_first_changes_the_token_count_not_the_model():
    torch.manual_seed(0)
    m = CausalPTDecoder(PTConfig(vocab_size=11, d=8, h=2, rank=None, gamma=2, n_iters=1))
    data = synthetic(11, 200)
    with_first = evaluate(m, data, TrainConfig(block_size=10, batch_size=4, ignore_first=0))
    without = evaluate(m, data, TrainConfig(block_size=10, batch_size=4, ignore_first=1))
    assert with_first["tokens"] == 200
    assert without["tokens"] == 180
    assert with_first["loss"] != without["loss"]


# ------------------------------------------------------------------ regulariser ------


def test_arc_regulariser_matches_the_materialised_scores():
    """The Kruskal short-cut must equal the mean square of the T it stands for."""
    for rank in (None, 3):
        torch.manual_seed(0)
        m = CausalPTDecoder(
            PTConfig(vocab_size=5, d=6, h=2, rank=rank, gamma=1, init_std=0.7)
        ).double()
        got = float(m.arc_regulariser())
        assert got == pytest.approx(float((m.arc_scores() ** 2).mean()), rel=1e-9)


def test_regulariser_actually_restrains_the_arc_scores():
    """It is the only mechanism bounding ||T||, so it has to bite."""
    curves = {}
    for l2 in (0.0, 5.0):
        torch.manual_seed(0)
        m = CausalPTDecoder(PTConfig(vocab_size=11, d=8, h=2, rank=None, gamma=2, n_iters=2))
        data = synthetic(11, 600)
        cfg = TrainConfig(
            block_size=8, batch_size=4, max_steps=80, lr=0.05, warmup_steps=5,
            eval_every=80, l2_arc=l2, log_every=1000, diagnostics=False,
        )
        train(m, data, data, cfg, random_batch, log=lambda _s: None)
        curves[l2] = float(m.arc_scores().abs().max())
    assert curves[5.0] < curves[0.0], f"the L2 term did not restrain T: {curves}"


# ------------------------------------------------------------------ the loop ---------


def test_training_reduces_loss_and_perplexity_on_a_learnable_stream():
    torch.manual_seed(0)
    m = CausalPTDecoder(
        PTConfig(vocab_size=11, d=16, h=2, rank=None, gamma=3, n_iters=2, readout="exact")
    )
    data = synthetic(11, 4000)
    cfg = TrainConfig(
        block_size=16, batch_size=8, max_steps=200, lr=0.05, warmup_steps=20,
        eval_every=50, eval_blocks=8, log_every=1000, diagnostics=True,
    )
    hist = train(m, data, data, cfg, random_batch, log=lambda _s: None)

    assert hist.val_ppl, "no evaluation was recorded"
    assert hist.val_ppl[-1] < hist.val_ppl[0], f"perplexity did not fall: {hist.val_ppl}"
    assert hist.val_ppl[-1] < 11.0, f"did not beat uniform over the vocabulary: {hist.val_ppl}"
    assert all(math.isfinite(p) for p in hist.val_ppl)
    for d in hist.diag:
        assert {"msg_over_unary", "rho", "max_abs_T"} <= set(d)


# ------------------------------------------------------------------ real corpus ------


@needs_ptb
def test_ptb_pipeline_matches_the_published_corpus():
    c = load_ptb()
    assert c.vocab_size == 10000
    assert c.sizes() == {"train": 929589, "valid": 73760, "test": 82430}
    assert "<eos>" in c.stoi and "<unk>" in c.stoi


@needs_ptb
def test_ptb_unigram_baseline_is_the_number_the_spec_quotes():
    c = load_ptb()
    ppl = unigram_perplexity(c.train, c.valid, c.vocab_size, 64, 16, ignore_first=1)
    assert ppl == pytest.approx(688.82, abs=0.05)
