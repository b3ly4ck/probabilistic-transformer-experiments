"""Check 12 — the GPT and Looped baselines, and that they are comparable to PT.

The baselines are off-the-shelf architecture, so the tests here are not about the
transformer. They are about the two things that would silently void every comparison the
paper makes: the slot convention, and whether Looped really shares its weights.
"""

import math

import pytest
import torch

from src import CausalPTDecoder, PTConfig
from src.gpt import GPT, GPTConfig
from src.train import TrainConfig, evaluate, train
from src.data import random_batch


def gpt(**over) -> GPT:
    base = dict(vocab_size=17, block_size=12, n_layer=2, n_head=2, n_embd=16, dropout=0.0)
    base.update(over)
    torch.manual_seed(0)
    return GPT(GPTConfig(**base))


def synthetic(vocab: int = 11, n: int = 4000) -> torch.Tensor:
    out = [0]
    for _ in range(n - 1):
        out.append((out[-1] * 3 + 1) % vocab)
    return torch.tensor(out, dtype=torch.long)


# ------------------------------------------------------------------ slot convention --


def test_slot_t_predicts_token_t_from_the_strict_prefix():
    """The whole comparison rests on this: both models answer p(w_t | w_<t).

    Changing token t must leave the logits at every slot <= t untouched, exactly as for PT.
    """
    m = gpt().eval()
    torch.manual_seed(1)
    idx = torch.randint(0, 17, (3, 12))
    base = m(idx)
    for t in range(1, 12):
        alt = idx.clone()
        alt[:, t] = (idx[:, t] + 5) % 17
        other = m(alt)
        assert torch.equal(base[:, : t + 1], other[:, : t + 1]), f"slot <= {t} moved"
        if t + 1 < 12:
            assert not torch.equal(base[:, t + 1 :], other[:, t + 1 :])


def test_slot_zero_is_never_scored():
    """It is returned as zeros, so scoring it would grade the baseline on a non-prediction."""
    m = gpt()
    idx = torch.randint(0, 17, (2, 12))
    assert torch.equal(m(idx)[:, 0], torch.zeros(2, 17))
    with pytest.raises(AssertionError, match="slot 0"):
        m.loss(idx, ignore_first=0)
    assert torch.isfinite(m.loss(idx, ignore_first=1))


def test_pt_and_gpt_score_the_same_tokens_through_the_shared_loop():
    """Same blocks, same count, same metric — otherwise the perplexities are incomparable."""
    data = synthetic(11, 400)
    cfg = TrainConfig(block_size=10, batch_size=4, ignore_first=1)
    torch.manual_seed(0)
    pt = CausalPTDecoder(PTConfig(vocab_size=11, d=8, h=2, rank=None, gamma=2, n_iters=1))
    g = gpt(vocab_size=11, block_size=10)
    a, b = evaluate(pt, data, cfg), evaluate(g, data, cfg)
    assert a["tokens"] == b["tokens"] == (data.numel() // 10) * 9


# ------------------------------------------------------------------------ looped ----


def test_looped_shares_one_block_and_is_smaller_than_the_stack():
    stacked = gpt(n_layer=4, shared_block=False)
    looped = gpt(n_layer=4, shared_block=True)
    assert looped.blocks is None and looped.block is not None
    assert stacked.block is None and len(stacked.blocks) == 4
    assert looped.num_parameters()["non_embedding"] < stacked.num_parameters()["non_embedding"]

    # applying the same block T times is what "weight sharing without structure" means
    ids = {id(p) for p in looped.block.parameters()}
    assert len(ids) > 0
    looped.eval()
    idx = torch.randint(0, 17, (2, 12))
    assert torch.isfinite(looped(idx)).all()


def test_parameter_split_is_reported_for_the_baseline():
    m = gpt()
    p = m.num_parameters()
    assert p["embedding"] == m.wte.weight.numel() + m.wpe.weight.numel()
    assert p["embedding"] + p["non_embedding"] == p["total"]
    assert m.lm_head.weight is m.wte.weight  # tied, as PT's are forced to be


def test_arc_regulariser_is_zero_and_differentiable_free():
    """The shared loop asks every model for it; a transformer has no ternary scores."""
    m = gpt()
    r = m.arc_regulariser()
    assert float(r) == 0.0 and r.shape == ()


# ------------------------------------------------------------------------ learning --


def test_gpt_learns_the_stream_the_shared_loop_is_given():
    """If this fails, the loop or the data is broken, not the model under test."""
    m = gpt(vocab_size=11, block_size=16, n_embd=32, n_layer=2)
    data = synthetic(11, 4000)
    cfg = TrainConfig(
        block_size=16, batch_size=8, max_steps=200, lr=3e-3, warmup_steps=20,
        eval_every=50, eval_blocks=8, log_every=10**9, diagnostics=False,
    )
    hist = train(m, data, data, cfg, random_batch, log=lambda _s: None)
    assert hist.val_ppl[-1] < hist.val_ppl[0]
    assert hist.val_ppl[-1] < 11.0, f"did not beat uniform: {hist.val_ppl}"
    assert all(math.isfinite(p) for p in hist.val_ppl)
