"""Is the causal PT simply undertrained at `lr = 1e-3`?

The minimal reproducing case found by the scale bisection: an order-1 Markov chain with
`|V| = 11` and 5 successors per state. GPT closes the whole unigram-to-oracle gap on it;
PT closes 0.000. Vocabulary size is not the axis — PT closes 0.000 at 11, 100 and 1000
alike.

One difference between every failing configuration and the one configuration of this decoder
that *is* known to learn a context task — `tests/test_11::test_training_reduces_loss_and_
perplexity_on_a_learnable_stream` — is the learning rate: **0.05 there, 1e-3 in all twelve
PTB runs and the whole bisection**. A factor of fifty. `1e-3` is Wu & Tu's Table 2 value, but
that is for their *encoder* on masked LM, and PT's parameters sit almost entirely in one tied
matrix whose gradient is spread over the vocabulary.

This sweeps the learning rate at two widths on the minimal case. GPT at the same learning
rates is the control: if PT needs a rate at which GPT diverges, that is a finding about the
construction, not a hyperparameter oversight.

Run:  python -m experiments.exp1_language_modeling.lr_probe
"""

import argparse
import json
import time
from pathlib import Path

import torch

from src import CausalPTDecoder, PTConfig
from src.data import random_batch, unigram_perplexity
from src.gpt import GPT, GPTConfig
from src.train import TrainConfig, train

from experiments.exp1_language_modeling.run_scale import make_chain

LRS = (1e-3, 5e-3, 2e-2, 5e-2, 1e-1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--vocab", type=int, default=11)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--n-train", type=int, default=400000)
    p.add_argument("--n-val", type=int, default=40000)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()

    train_data, val_data, oracle = make_chain(a.vocab, a.n_train, a.n_val)
    base = unigram_perplexity(train_data, val_data, a.vocab, a.block_size, a.batch_size,
                              ignore_first=1, add_k=1.0)
    print(f"chain: |V|={a.vocab}  oracle ppl {oracle:.2f}  unigram ppl {base:.2f}\n")

    results = []
    cells = [("pt", 256), ("pt", 16), ("gpt", 256)]
    for model_name, d in cells:
        for lr in LRS:
            torch.manual_seed(0)
            if model_name == "pt":
                cfg = PTConfig(vocab_size=a.vocab, d=d, h=8 if d >= 64 else 2,
                               rank=min(64, d), gamma=3, n_iters=3, readout="mfvi")
                model = CausalPTDecoder(cfg)
            else:
                cfg = GPTConfig(vocab_size=a.vocab, block_size=a.block_size,
                                n_layer=4, n_head=4, n_embd=d)
                model = GPT(cfg)
            tcfg = TrainConfig(
                block_size=a.block_size, batch_size=a.batch_size, max_steps=a.steps,
                lr=lr, eval_every=max(1, a.steps // 4), ignore_first=1,
                device=a.device, log_every=10**9, diagnostics=(model_name == "pt"),
            )
            t0 = time.time()
            hist = train(model, train_data, val_data, tcfg, random_batch, log=lambda _s: None)
            best = min(hist.val_ppl)
            closed = max(0.0, (base - best) / max(base - oracle, 1e-9))
            diag = hist.diag[-1] if hist.diag else {}
            results.append({"model": model_name, "d": d, "lr": lr,
                            "best_val_ppl": round(best, 3), "gap_closed": round(closed, 4),
                            "diag": diag, "seconds": round(time.time() - t0, 1)})
            extra = ""
            if diag:
                extra = (f"  msg/unary {diag.get('msg_over_unary', float('nan')):7.2f}"
                         f"  H(q)/max {diag.get('label_entropy', float('nan')):6.3f}")
            print(f"{model_name:3s} d={d:<4} lr={lr:<7} best {best:9.3f}  "
                  f"gap closed {closed:6.3f}{extra}  ({results[-1]['seconds']:.0f}s)", flush=True)

    out = Path(__file__).parent / "lr_probe.json"
    out.write_text(json.dumps({"args": vars(a), "oracle": oracle, "unigram": base,
                               "results": results}, indent=2))
    print(f"\nwritten {out}")


if __name__ == "__main__":
    main()
