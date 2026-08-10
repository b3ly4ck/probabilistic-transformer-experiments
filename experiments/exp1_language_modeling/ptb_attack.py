"""Step 3 — carry the working region to PTB, one change at a time.

What the region probe and the d x h deconfound established:

* the region is small `d` and small `h`; `h = 2` is best at every width, and at `h = 2` the
  collapse over width sits between `d = 32` and `d = 64`;
* `msg_over_unary` predicts the outcome without exception — the healthy band is 2-5, and the
  only PTB run so far to move off the unigram (`d=16, lr=0.02`, val 473.28) sat at 3.41;
* the prefix-ablation KL tracks the gap closed, so it is an independent read on whether
  context is used at all.

**3a** re-runs the single working PTB configuration with per-evaluation logging, which the
region probe did not have — it kept only the final diagnostic. That gives the full report row
and, if the run fails, *which quantity left the healthy band first*.

**3b** then raises `d` one step at a time while watching `msg_over_unary`. The ladder stops
the moment it leaves the 2-5 band or the ablation KL falls to ~0: that `d` is the ceiling of
the construction at this budget, and that number is the one that goes in the paper.

Nothing else moves. `h`, `lr`, the schedule, the optimiser and the data are held at the values
the previous stages selected.

Run:  python -m experiments.exp1_language_modeling.ptb_attack
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch

from src import CausalPTDecoder, PTConfig
from src.data import load_ptb, random_batch, unigram_perplexity
from src.train import TrainConfig, evaluate, train

from experiments.exp1_language_modeling.ablate_prefix import ablate

MSG_CEILING = 5.0
# Only the UPPER edge is a failure condition. The band was first written as (2, 5) from two
# points on the minimal task; the 15000-step run of 2026-08-10 then reached val 336.89 — the
# best result in the project and the first to pass the gate — while sitting at msg/unary 1.09,
# and the ladder stopped it as "out of band". A small message relative to the word unary is
# not a pathology; a large one is. Recorded rather than quietly retuned.


def run(corpus, d, a, log):
    torch.manual_seed(a.seed)
    cfg = PTConfig(vocab_size=corpus.vocab_size, d=d, h=a.h, rank=min(a.rank, d), gamma=3,
                   n_iters=3, readout="mfvi", vocab_chunk=1024,
                   freeze_word_unary=a.freeze_b, n_global=a.n_global)
    model = CausalPTDecoder(cfg)
    if a.freeze_b:
        counts = torch.bincount(corpus.train, minlength=corpus.vocab_size).double()
        model.set_word_unary((counts / counts.sum()).clamp_min(1e-12).log())
    tcfg = TrainConfig(block_size=a.block_size, batch_size=a.batch_size, max_steps=a.steps,
                       lr=a.lr, warmup_steps=a.warmup, eval_every=a.eval_every,
                       eval_train_blocks=20, ignore_first=1, device=a.device, seed=a.seed,
                       log_every=10**9, diagnostics=True)
    t0 = time.time()
    hist = train(model, corpus.train, corpus.valid, tcfg, random_batch, log=log)
    best = min(hist.val_ppl)
    final = hist.val_ppl[-1]
    # Report both. `best` is the minimum over the trace and is the right number for a run
    # that converges; it lies when a run diverges after an early transient. Measured at
    # d=32 on 2026-08-10: best 523.08 was the evaluation at step 500, before the message
    # exploded, while the run actually ended at 641.5 — near the unigram. The verdict below
    # uses `final`.
    test = evaluate(model, corpus.test, tcfg)
    abl = ablate(model, corpus.valid, a.block_size, a.batch_size, 6, device=a.device)
    diag = hist.diag[-1] if hist.diag else {}

    # which quantity left the healthy band first, and when
    first_out = None
    for step, dg in zip(hist.step, hist.diag):
        m = dg.get("msg_over_unary", float("nan"))
        if m > MSG_CEILING and first_out is None:
            first_out = ("msg_over_unary", step, round(m, 2))
    ck = Path("checkpoints") / f"attack_d{d}_pt.pt"
    torch.save({"cfg": cfg, "state_dict": model.state_dict(),
                "args": {"block_size": a.block_size}}, ck)
    return {
        "d": d, "h": a.h, "lr": a.lr,
        "val_ppl": round(final, 2), "val_ppl_min": round(best, 2),
        "test_ppl": round(test["ppl"], 2),
        "train_ppl": round(hist.train_ppl[-1], 2) if hist.train_ppl else float("nan"),
        "n_global": a.n_global,
        "msg_over_unary": round(diag.get("msg_over_unary", float("nan")), 2),
        "arc_msg_norm": round(diag.get("arc_msg_norm", float("nan")), 3),
        "glob_msg_norm": round(diag.get("glob_msg_norm", float("nan")), 3),
        "glob_over_unary": round(diag.get("glob_over_unary", float("nan")), 3),
        "qg_entropy_frac": round(diag.get("qg_entropy_frac", float("nan")), 4),
        "max_abs_B_glob": round(diag.get("max_abs_B_glob", float("nan")), 3),
        "label_entropy": round(diag.get("label_entropy", float("nan")), 3),
        "label_entropy_max": round(math.log(d), 3),
        "ablation_kl": abl["shuffled/kl_mean"],
        "first_out_of_band": first_out,
        "msg_trace": [round(dg.get("msg_over_unary", float("nan")), 2) for dg in hist.diag],
        "val_trace": [round(v, 1) for v in hist.val_ppl],
        "seconds": round(time.time() - t0, 1),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ladder", type=int, nargs="+", default=[16, 24, 32, 48])
    p.add_argument("--h", type=int, default=2, help="from the d x h deconfound: 2 wins at every d")
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--n-global", type=int, default=0,
                   help="m of the B.3.3 single-split global head. MFVI only — the "
                        "exact readout's contribution is a measured constant.")
    p.add_argument("--lr", type=float, default=2e-2)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--freeze-b", action="store_true", help="step 3c")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--tag", default="attack")
    a = p.parse_args()

    corpus = load_ptb()
    base = unigram_perplexity(corpus.train, corpus.valid, corpus.vocab_size,
                              a.block_size, a.batch_size, ignore_first=1)
    print(f"PTB unigram {base:.2f}  gate {0.5 * base:.2f}  GPT reference 115.43  "
          f"h={a.h} lr={a.lr} freeze_b={a.freeze_b}\n", flush=True)

    results = []
    for d in a.ladder:
        print(f"--- d={d} ---", flush=True)
        row = run(corpus, d, a, log=lambda s: print("   " + s, flush=True))
        results.append(row)
        print(f"  d={d:<4} val {row['val_ppl']:8.2f} (min {row['val_ppl_min']:.2f})  "
              f"train {row['train_ppl']:8.2f}  "
              f"test {row['test_ppl']:8.2f}  msg/unary {row['msg_over_unary']:7.2f}  "
              f"H(q) {row['label_entropy']:.3f}/{row['label_entropy_max']:.2f}  "
              f"ablate KL {row['ablation_kl']:.3e}", flush=True)
        if a.n_global:
            print(f"       arc msg {row['arc_msg_norm']:.3f}  glob msg {row['glob_msg_norm']:.3f}"
                  f"  glob/unary {row['glob_over_unary']:.3f}"
                  f"  H(Q_g)/max {row['qg_entropy_frac']:.4f}"
                  f"  max|B'| {row['max_abs_B_glob']:.3f}", flush=True)

        in_band = row["msg_over_unary"] <= MSG_CEILING
        alive = row["ablation_kl"] > 1e-3
        if not in_band or not alive:
            why = []
            if not in_band:
                why.append(f"msg/unary {row['msg_over_unary']} above the ceiling {MSG_CEILING}")
            if not alive:
                why.append(f"ablation KL {row['ablation_kl']:.2e} at zero")
            print(f"  STOP: {'; '.join(why)} -> d={d} is the ceiling at this budget",
                  flush=True)
            break

    out = Path(__file__).parent / f"{a.tag}.json"
    out.write_text(json.dumps({"args": vars(a), "unigram": base, "results": results}, indent=2))
    print(f"\nwritten {out}")


if __name__ == "__main__":
    main()
