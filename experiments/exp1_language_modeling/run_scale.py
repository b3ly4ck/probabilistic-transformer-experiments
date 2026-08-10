"""Bisect the scale gap on a synthetic stream where the answer is known.

The load-bearing fact from the twelve PTB runs: this decoder *does* learn context at
`|V| = 11, d = 16` (tests/test_11) and abandons it at `|V| = 10^4, d = 256`. This walks the
two axes separately to find where the transition happens.

**The task.** An order-1 Markov chain: every state has `k = 5` possible successors with
probabilities drawn from a Dirichlet. So the conditional entropy is about `log 5 = 1.61`
nats — an oracle reaches perplexity ≈ 5 — while the unigram baseline is of order `|V|`. The
gap between them is the whole signal, and it is present at every vocabulary size, which is
what makes the sweep comparable. A deterministic successor map is deliberately not used: it
would make the stream eventually periodic and trivially memorable.

Both models are run on every cell. GPT is the control: if PT fails where GPT succeeds, the
failure is PT's; if both fail, the cell is too hard for the budget and says nothing.

Run:  python -m experiments.exp1_language_modeling.run_scale --steps 3000
"""

import argparse
import json
import time
from pathlib import Path

import torch

from src import CausalPTDecoder, PTConfig
from src.data import random_batch, sequential_batches, unigram_perplexity
from src.gpt import GPT, GPTConfig
from src.train import TrainConfig, evaluate, train

K_SUCC = 5


def make_chain(vocab: int, n_train: int, n_val: int, seed: int = 0):
    """Order-1 Markov stream with `K_SUCC` successors per state.

    ``torch.manual_seed`` is set here on purpose: ``Dirichlet.sample`` does not accept a
    generator and draws from the *global* RNG, so without this the chain depends on whatever
    ran before it and the task differs between scripts. That is how the oracle came out as
    3.84 in ``lr_probe`` and 1.96 in ``region_probe`` for nominally the same chain — numbers
    stayed comparable *within* a run, since every cell there shares one chain, but not across
    runs. Each run's ``oracle``/``unigram`` pair identifies which chain it used.
    """
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    succ = torch.randint(0, vocab, (vocab, K_SUCC), generator=g)
    probs = torch.distributions.Dirichlet(torch.ones(K_SUCC)).sample((vocab,))
    total = n_train + n_val
    out = torch.empty(total, dtype=torch.long)
    out[0] = 0
    choice = torch.multinomial(probs, total, replacement=True, generator=g)  # (vocab, total)
    for t in range(1, total):
        s = out[t - 1]
        out[t] = succ[s, choice[s, t]]
    # oracle: the stationary-weighted mean conditional entropy, estimated from the visits
    visits = torch.bincount(out[:-1], minlength=vocab).double()
    ent = -(probs * probs.clamp_min(1e-30).log()).sum(-1).double()
    oracle_ppl = float(((visits * ent).sum() / visits.sum()).exp())
    return out[:n_train], out[n_train:], oracle_ppl


def build(model_name: str, vocab: int, d: int, block: int, device: str):
    if model_name == "pt":
        cfg = PTConfig(vocab_size=vocab, d=d, h=8, rank=min(64, d), gamma=3, n_iters=3,
                       readout="mfvi", vocab_chunk=1024)
        return CausalPTDecoder(cfg).to(device), cfg
    cfg = GPTConfig(vocab_size=vocab, block_size=block, n_layer=4, n_head=4, n_embd=d)
    return GPT(cfg).to(device), cfg


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n-train", type=int, default=400000)
    p.add_argument("--n-val", type=int, default=40000)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    cells = [("vocab", v, 256) for v in (11, 100, 1000, 10000)]
    cells += [("d", 1000, dd) for dd in (16, 64, 256)]

    results = []
    for axis, vocab, d in cells:
        train_data, val_data, oracle_ppl = make_chain(vocab, a.n_train, a.n_val)
        base = unigram_perplexity(train_data, val_data, vocab, a.block_size, a.batch_size,
                                  ignore_first=1, add_k=1.0)
        for model_name in ("pt", "gpt"):
            model, cfg = build(model_name, vocab, d, a.block_size, a.device)
            tcfg = TrainConfig(
                block_size=a.block_size, batch_size=a.batch_size, max_steps=a.steps,
                lr=a.lr, eval_every=max(1, a.steps // 4), ignore_first=1,
                device=a.device, log_every=10**9, diagnostics=(model_name == "pt"),
            )
            t0 = time.time()
            hist = train(model, train_data, val_data, tcfg, random_batch, log=lambda _s: None)
            best = min(hist.val_ppl)
            row = {
                "axis": axis, "vocab": vocab, "d": d, "model": model_name,
                "oracle_ppl": round(oracle_ppl, 3),
                "unigram_ppl": round(base, 2), "best_val_ppl": round(best, 2),
                "fraction_of_gap_closed": round(
                    max(0.0, (base - best) / max(base - oracle_ppl, 1e-9)), 4
                ),
                "params": model.num_parameters(), "seconds": round(time.time() - t0, 1),
                "diag": hist.diag[-1] if hist.diag else {},
            }
            results.append(row)
            print(f"[{axis:5s}] V={vocab:<6} d={d:<4} {model_name:3s}  "
                  f"oracle {oracle_ppl:7.2f}  unigram {base:9.2f}  best {best:9.2f}  "
                  f"gap closed {row['fraction_of_gap_closed']:.3f}  "
                  f"({row['seconds']:.0f}s)", flush=True)

    out = Path(a.out or Path(__file__).parent / "scale_bisection.json")
    out.write_text(json.dumps({"args": vars(a), "results": results}, indent=2))
    print(f"written {out}")


if __name__ == "__main__":
    main()
