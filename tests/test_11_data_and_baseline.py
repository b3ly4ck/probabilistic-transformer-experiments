"""Data, the GPT baseline, and the shared training loop.

The comparison rests on these being identical across models, so the parity
properties are asserted rather than assumed.
"""

import math
from pathlib import Path

import pytest
import torch

from src import data as data_mod
from src.config import PTConfig
from src.gpt import GPT, GPTConfig, count_parameters
from src.pt_decoder import CausalPTDecoder
from src.train import TrainConfig, evaluate, lr_at, pt_loss, train

HAVE_PTB = (data_mod.DEFAULT_ROOT / "ptb.train.txt").exists()
needs_ptb = pytest.mark.skipif(not HAVE_PTB, reason="PTB not downloaded")

V, CTX = 200, 16


def _corpus(n=4000, period=37):
    """A learnable synthetic stream, not random tokens.

    Uniform noise sits at ln(V) by construction, so a loop that works perfectly
    would still show a flat loss on it -- the test would assert nothing.  A
    periodic stream has structure to find.
    """
    vocab = data_mod.Vocab.from_tokens([str(i) for i in range(V)])
    stream = lambda k, off: (torch.arange(k) + off) % period  # noqa: E731
    return data_mod.Corpus(
        vocab=vocab, train=stream(n, 0), valid=stream(n // 4, 0), test=stream(n // 4, 0)
    )


def _gpt(seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    return GPT(GPTConfig(vocab_size=V, d_model=32, n_layer=2, n_head=2, context=CTX), generator=g)


# -- tokenisation ----------------------------------------------------------


def test_vocab_is_built_from_train_only():
    vocab = data_mod.Vocab.from_tokens(["a", "b", "a"])
    assert set(vocab.itos) == {"a", "b"}
    assert len(vocab) == 2


def test_unknown_words_map_to_unk():
    vocab = data_mod.Vocab.from_tokens(["a", "<unk>"])
    assert vocab.encode(["zzz"]).tolist() == [vocab.stoi["<unk>"]]


def test_batchify_preserves_order_within_a_row():
    flat = torch.arange(20)
    b = data_mod.batchify(flat, 4)
    assert b.shape == (4, 5)
    assert b[0].tolist() == [0, 1, 2, 3, 4]


def test_windows_target_is_input_shifted_by_one():
    b = data_mod.batchify(torch.arange(100), 2)
    x, y = next(iter(data_mod.windows(b, 8)))
    assert torch.equal(x[:, 1:], y[:, :-1])


@needs_ptb
def test_ptb_matches_the_published_token_counts():
    """929,589 / 73,760 / 82,430 with <eos>, vocabulary 10,000 -- the standard
    numbers.  A mismatch means the preprocessing drifted."""
    c = data_mod.load_ptb(download=False)
    assert c.vocab_size == 10_000
    assert (c.train.numel(), c.valid.numel(), c.test.numel()) == (929_589, 73_760, 82_430)
    assert "<unk>" in c.vocab.stoi and "<eos>" in c.vocab.stoi


# -- baseline parity -------------------------------------------------------


def test_gpt_scores_the_same_token_set_as_pt():
    """Both expose logits(t) predicting tokens[t] from tokens[:t].  Without the
    prepended BOS the baseline would be scored on a different set and the
    perplexities would not be comparable."""
    gpt = _gpt()
    g = torch.Generator()
    g.manual_seed(1)
    tokens = torch.randint(0, V, (2, CTX), generator=g)
    pt = CausalPTDecoder(PTConfig(vocab_size=V, d=8, n_channels=1, n_rounds=2), generator=g)
    assert gpt.logits(tokens).shape == pt(tokens).shape == (2, CTX, V)


def test_gpt_is_causal_under_that_convention():
    gpt = _gpt()
    g = torch.Generator()
    g.manual_seed(2)
    tokens = torch.randint(0, V, (2, CTX), generator=g)
    base = gpt.logits(tokens)
    for p in range(CTX):
        other = tokens.clone()
        other[:, p] = (other[:, p] + 3) % V
        assert torch.equal(base[:, : p + 1], gpt.logits(other)[:, : p + 1]), p


def test_gpt_position_zero_sees_no_token():
    gpt = _gpt()
    g = torch.Generator()
    g.manual_seed(3)
    a = torch.randint(0, V, (2, CTX), generator=g)
    b = torch.randint(0, V, (2, CTX), generator=g)
    assert torch.equal(gpt.logits(a)[:, 0], gpt.logits(b)[:, 0])


def test_gpt_embeddings_are_tied():
    """PT's tying is forced by the construction; untying the baseline would hand
    it free parameters."""
    gpt = _gpt()
    names = dict(gpt.named_parameters())
    assert "lm_head.weight" not in names
    assert sum(1 for n in names if n.endswith("wte.weight")) == 1


def test_parameter_count_splits_embedding_from_the_rest():
    counts = count_parameters(_gpt())
    assert counts["total"] == counts["embedding"] + counts["non_embedding"]
    assert counts["embedding"] == V * 32 + CTX * 32

    pt = CausalPTDecoder(PTConfig(vocab_size=V, d=8, n_channels=1, n_rounds=2))
    pt_counts = count_parameters(pt)
    assert pt_counts["embedding"] == V * 8  # S


# -- the shared loop -------------------------------------------------------


def test_schedule_warms_up_then_decays():
    cfg = TrainConfig(max_steps=1000, warmup_steps=100, lr=1.0, min_lr_ratio=0.1)
    assert lr_at(0, cfg) == pytest.approx(0.01)
    assert lr_at(99, cfg) == pytest.approx(1.0)
    assert lr_at(999, cfg) == pytest.approx(0.1, abs=1e-3)
    assert lr_at(500, cfg) < lr_at(150, cfg)


def test_evaluate_reports_perplexity_consistent_with_loss():
    cfg = TrainConfig(context=CTX, batch_size=4, eval_batches=3)
    out = evaluate(_gpt(), _corpus().valid, cfg)
    assert out["ppl"] == pytest.approx(math.exp(out["loss"]))
    assert 1.0 < out["ppl"] < 10 * V
    assert out["batches"] == 3


@pytest.mark.parametrize(
    "build,loss_fn",
    [
        (lambda: _gpt(), None),
        (
            lambda: CausalPTDecoder(PTConfig(vocab_size=V, d=16, n_channels=1, n_rounds=2)),
            pt_loss("mfvi"),
        ),
    ],
    ids=["gpt", "pt_mfvi"],
)
def test_the_same_loop_trains_both_model_families(build, loss_fn):
    """The loop must be model-agnostic: only loss_fn differs, never a branch."""
    cfg = TrainConfig(
        context=CTX, batch_size=4, max_steps=30, eval_every=15, eval_batches=3,
        lr=3e-3, warmup_steps=5,
    )
    kwargs = {"loss_fn": loss_fn} if loss_fn else {}
    result = train(build(), _corpus(), cfg, log=None, **kwargs)
    assert len(result.history) >= 2
    assert result.history[-1]["val_ppl"] < result.history[0]["val_ppl"]
    assert result.best_val_ppl < V  # better than uniform over the vocabulary
    assert result.tokens_seen == 30 * 4 * CTX


# -- generation ------------------------------------------------------------


def test_generate_from_empty_prompt_for_both_families():
    """Position 0 is predicted from a learned constant in both models -- PT's
    root key r, the baseline's BOS -- so an empty prompt needs no special case."""
    from src.generate import decode, generate

    g = torch.Generator()
    g.manual_seed(0)
    pt = CausalPTDecoder(PTConfig(vocab_size=V, d=16, n_channels=1, n_rounds=2), generator=g)
    for model, readout in ((_gpt(), None), (pt, "mfvi"), (pt, "exact")):
        out = generate(model, max_new_tokens=5, batch_size=3, readout=readout, generator=g)
        assert out.shape == (3, 5)
        assert out.min() >= 0 and out.max() < V
        assert len(decode(out, _corpus().vocab)) == 3


def test_generation_continues_a_prompt():
    from src.generate import generate

    g = torch.Generator()
    g.manual_seed(1)
    prompt = torch.randint(0, V, (2, 4), generator=g)
    out = generate(_gpt(), max_new_tokens=3, prompt=prompt, generator=g)
    assert out.shape == (2, 7)
    assert torch.equal(out[:, :4], prompt)


def test_top_k_restricts_the_support():
    """With k=1 sampling is deterministic, so two draws must agree."""
    from src.generate import generate

    g = torch.Generator()
    model = _gpt()
    prompt = torch.randint(0, V, (1, 3), generator=torch.Generator().manual_seed(2))
    a = generate(model, 4, prompt=prompt, top_k=1, generator=g)
    b = generate(model, 4, prompt=prompt, top_k=1, generator=g)
    assert torch.equal(a, b)
