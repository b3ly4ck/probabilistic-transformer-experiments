"""Map the region where the causal PT actually learns, and replicate it across seeds.

`lr_probe.py` found one cell where PT closes the entire unigram→oracle gap on the minimal
Markov task: `d = 16`, `lr = 0.02`, gap closed 1.010 against an oracle of 3.84, with the
message-to-unary ratio at 2.48 rather than the 12–79 of every failing configuration.

That is one cell on one seed. The post-mortem of the previous implementation turned on
exactly this mistake — a memorisation check that passed on a single lucky seed and had a
fit rate of 1/5 — so the cell is replicated here across seeds before it is called anything,
and the grid around it is mapped so the shape of the region is known rather than guessed.

Stage 2 then takes the best setting to PTB, which is the question that matters. Note the
label bottleneck: at `d = 16` the readout is a rank-16 mixture over a 10⁴ vocabulary, so a
low perplexity is not expected there — clearing the unigram baseline decisively is the gate.

Run:  python -m experiments.exp1_language_modeling.region_probe
"""

import argparse
import json
import time
from pathlib import Path

import torch

from src import CausalPTDecoder, PTConfig
from src.data import load_ptb, random_batch, unigram_perplexity
from src.train import TrainConfig, evaluate, train

from experiments.exp1_language_modeling.run_scale import make_chain

DS = (8, 16, 32, 64, 128, 256)
LRS = (5e-3, 1e-2, 2e-2, 4e-2)
SEEDS = (0, 1, 2)


def run_cell(train_data, val_data, vocab, d, lr, seed, a, readout="mfvi"):
    torch.manual_seed(seed)
    cfg = PTConfig(vocab_size=vocab, d=d, h=8 if d >= 64 else 2, rank=min(64, d),
                   gamma=3, n_iters=3, readout=readout, vocab_chunk=1024)
    model = CausalPTDecoder(cfg)
    tcfg = TrainConfig(block_size=a.block_size, batch_size=a.batch_size, max_steps=a.steps,
                       lr=lr, eval_every=max(1, a.steps // 3), ignore_first=1,
                       device=a.device, seed=seed, log_every=10**9, diagnostics=True)
    hist = train(model, train_data, val_data, tcfg, random_batch, log=lambda _s: None)
    return model, min(hist.val_ppl), (hist.diag[-1] if hist.diag else {})


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--vocab", type=int, default=11)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--n-train", type=int, default=400000)
    p.add_argument("--n-val", type=int, default=40000)
    p.add_argument("--ptb-steps", type=int, default=6000)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()

    results = []

    # ---- stage 1: the grid on the minimal task, seed 0 -------------------------------
    train_data, val_data, oracle = make_chain(a.vocab, a.n_train, a.n_val)
    base = unigram_perplexity(train_data, val_data, a.vocab, a.block_size, a.batch_size,
                              ignore_first=1, add_k=1.0)
    print(f"=== grid on |V|={a.vocab}: oracle {oracle:.2f}, unigram {base:.2f} ===", flush=True)
    print(f"{'d':>5} " + " ".join(f"lr={lr:<7}" for lr in LRS), flush=True)
    best_cell = None
    for d in DS:
        row = []
        for lr in LRS:
            _, best, diag = run_cell(train_data, val_data, a.vocab, d, lr, 0, a)
            closed = max(0.0, (base - best) / max(base - oracle, 1e-9))
            row.append(closed)
            results.append({"stage": "grid", "d": d, "lr": lr, "seed": 0,
                            "best_val_ppl": round(best, 3), "gap_closed": round(closed, 4),
                            "diag": diag})
            if best_cell is None or closed > best_cell[0]:
                best_cell = (closed, d, lr)
        print(f"{d:>5} " + " ".join(f"{c:<10.3f}" for c in row), flush=True)

    print(f"\nbest cell: d={best_cell[1]} lr={best_cell[2]} gap closed {best_cell[0]:.3f}",
          flush=True)

    # ---- stage 2: replicate the best cell across seeds --------------------------------
    print(f"\n=== replication of d={best_cell[1]} lr={best_cell[2]} over seeds {SEEDS} ===",
          flush=True)
    closed_by_seed = []
    for seed in SEEDS:
        _, best, diag = run_cell(train_data, val_data, a.vocab, best_cell[1], best_cell[2],
                                 seed, a)
        closed = max(0.0, (base - best) / max(base - oracle, 1e-9))
        closed_by_seed.append(closed)
        results.append({"stage": "replicate", "d": best_cell[1], "lr": best_cell[2],
                        "seed": seed, "best_val_ppl": round(best, 3),
                        "gap_closed": round(closed, 4), "diag": diag})
        print(f"  seed {seed}: best {best:.3f}  gap closed {closed:.3f}  "
              f"msg/unary {diag.get('msg_over_unary', float('nan')):.2f}", flush=True)
    print(f"  -> {sum(1 for c in closed_by_seed if c > 0.8)}/{len(SEEDS)} seeds close >80% "
          f"of the gap", flush=True)

    # ---- stage 3: PTB at the settings the grid selected -------------------------------
    corpus = load_ptb()
    ptb_base = unigram_perplexity(corpus.train, corpus.valid, corpus.vocab_size,
                                  a.block_size, a.batch_size, ignore_first=1)
    print(f"\n=== PTB, unigram {ptb_base:.2f}, GPT reference 115.43 ===", flush=True)
    for d in sorted({best_cell[1], 32, 64}):
        for lr in sorted({best_cell[2], 2e-2}):
            torch.manual_seed(0)
            cfg = PTConfig(vocab_size=corpus.vocab_size, d=d, h=8 if d >= 64 else 2,
                           rank=min(64, d), gamma=3, n_iters=3, readout="mfvi",
                           vocab_chunk=1024)
            model = CausalPTDecoder(cfg)
            tcfg = TrainConfig(block_size=a.block_size, batch_size=a.batch_size,
                               max_steps=a.ptb_steps, lr=lr, eval_every=a.ptb_steps // 6,
                               ignore_first=1, device=a.device, log_every=10**9)
            t0 = time.time()
            hist = train(model, corpus.train, corpus.valid, tcfg, random_batch,
                         log=lambda _s: None)
            best = min(hist.val_ppl)
            test = evaluate(model, corpus.test, tcfg)
            diag = hist.diag[-1] if hist.diag else {}
            results.append({"stage": "ptb", "d": d, "lr": lr, "best_val_ppl": round(best, 2),
                            "test_ppl": round(test["ppl"], 2), "diag": diag,
                            "seconds": round(time.time() - t0, 1)})
            print(f"  d={d:<4} lr={lr:<6} val {best:8.2f}  test {test['ppl']:8.2f}  "
                  f"vs unigram {ptb_base:.2f}  msg/unary "
                  f"{diag.get('msg_over_unary', float('nan')):.2f}", flush=True)
            if best < 0.5 * ptb_base:
                torch.save({"cfg": cfg, "state_dict": model.state_dict(),
                            "args": {"block_size": a.block_size}},
                           Path("checkpoints") / f"region_d{d}_lr{lr}_pt.pt")

    out = Path(__file__).parent / "region_probe.json"
    out.write_text(json.dumps({"args": vars(a), "oracle": oracle, "unigram": base,
                               "ptb_unigram": ptb_base, "results": results}, indent=2))
    print(f"\nwritten {out}")


if __name__ == "__main__":
    main()
