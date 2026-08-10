"""Step 2 — separate width from channels. Mandatory before any claim about width.

`region_probe.py` swept `d` with `h = 8 if d >= 64 else 2`, so the collapse it found at
`d >= 64` could be width, channels, or both. The two are not interchangeable: the message
bound is `|G_i(a)| <= h * max(max|T|, max|r|)`, linear in `h` and independent of `d`, so a
channel effect and a width effect would mean different things for the construction.

Full cross grid on the minimal Markov task, learning rate fixed at the best *replicated*
value from the region probe, one chain shared by every cell, same budget per cell.

Reported per cell: fraction of the unigram→oracle gap closed, `msg_over_unary` and label
entropy at the end of training, and the prefix-ablation KL computed in process on the same
validation stream. The healthy band established by the region probe is `msg/unary ≈ 2-3`
with the label entropy at neither extreme.

Run:  python -m experiments.exp1_language_modeling.deconfound_dh --lr 0.005
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch

from src import CausalPTDecoder, PTConfig
from src.data import random_batch, unigram_perplexity
from src.train import TrainConfig, evaluate, train

from experiments.exp1_language_modeling.ablate_prefix import ablate
from experiments.exp1_language_modeling.run_scale import make_chain

DS = (16, 32, 64, 128)
HS = (2, 4, 8)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--vocab", type=int, default=11)
    p.add_argument("--lr", type=float, default=5e-3, help="best replicated cell of the region probe")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--n-train", type=int, default=400000)
    p.add_argument("--n-val", type=int, default=40000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()

    train_data, val_data, oracle = make_chain(a.vocab, a.n_train, a.n_val)
    base = unigram_perplexity(train_data, val_data, a.vocab, a.block_size, a.batch_size,
                              ignore_first=1, add_k=1.0)
    print(f"minimal task |V|={a.vocab}: oracle {oracle:.3f}  unigram {base:.3f}  "
          f"lr {a.lr}  seed {a.seed}\n", flush=True)
    print(f"{'d':>5} {'h':>3} {'val ppl':>9} {'train ppl':>10} {'gap':>7} "
          f"{'msg/unary':>10} {'H(q)':>7} {'ablate KL':>10}", flush=True)

    results = []
    for d in DS:
        for h in HS:
            torch.manual_seed(a.seed)
            cfg = PTConfig(vocab_size=a.vocab, d=d, h=h, rank=min(64, d), gamma=3,
                           n_iters=3, readout="mfvi", vocab_chunk=1024)
            model = CausalPTDecoder(cfg)
            tcfg = TrainConfig(block_size=a.block_size, batch_size=a.batch_size,
                               max_steps=a.steps, lr=a.lr, eval_every=max(1, a.steps // 3),
                               ignore_first=1, device=a.device, seed=a.seed,
                               log_every=10**9, diagnostics=True)
            t0 = time.time()
            hist = train(model, train_data, val_data, tcfg, random_batch, log=lambda _s: None)
            best = min(hist.val_ppl)
            closed = max(0.0, (base - best) / max(base - oracle, 1e-9))
            diag = hist.diag[-1] if hist.diag else {}
            abl = ablate(model, val_data, a.block_size, a.batch_size, 4, device=a.device)
            row = {
                "d": d, "h": h, "lr": a.lr, "seed": a.seed,
                "best_val_ppl": round(best, 3),
                "train_ppl": round(hist.train_ppl[-1], 3) if hist.train_ppl else float("nan"),
                "gap_closed": round(closed, 4),
                "msg_over_unary": round(diag.get("msg_over_unary", float("nan")), 3),
                "label_entropy": round(diag.get("label_entropy", float("nan")), 4),
                "label_entropy_max": round(math.log(d), 4),
                "ablation_kl": abl["shuffled/kl_mean"],
                "seconds": round(time.time() - t0, 1),
            }
            results.append(row)
            print(f"{d:>5} {h:>3} {row['best_val_ppl']:>9.3f} {row['train_ppl']:>10.3f} "
                  f"{closed:>7.3f} {row['msg_over_unary']:>10.2f} "
                  f"{row['label_entropy']:>7.3f} {row['ablation_kl']:>10.3e}", flush=True)

    out = Path(__file__).parent / "deconfound_dh.json"
    out.write_text(json.dumps({"args": vars(a), "oracle": oracle, "unigram": base,
                               "results": results}, indent=2))
    print(f"\nwritten {out}")


if __name__ == "__main__":
    main()
