"""Stage 0 of Experiment 1 — does the causal PT decoder beat the unigram model on PTB?

The gate, not the comparison. See EXPERIMENT_STATUS.md in this folder: a previous
implementation reached val ppl 664 against a unigram baseline of 687 and that was not
learning, so "perplexity went down" is not the criterion. Train perplexity falling far
below the baseline is.

Run:
    python -m experiments.exp1_language_modeling.run_pilot --steps 2000 --d 256
    python -m experiments.exp1_language_modeling.run_pilot --readout mfvi --device cuda
"""

import argparse
import json
import time
from pathlib import Path

import torch

from src import CausalPTDecoder, PTConfig
from src.data import load_ptb, random_batch, unigram_perplexity
from src.train import TrainConfig, evaluate, train


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--d", type=int, default=256)
    p.add_argument("--h", type=int, default=8)
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--gamma", type=int, default=3)
    p.add_argument("--n-iters", type=int, default=3)
    p.add_argument("--tau", type=int, default=2,
                   help="predictive inner rounds of the MFVI readout. 17.1 asks for >= 2 so the\nattention query is context-dependent; at tau=1 it is the fixed global probe\nsigma(s_bar/lambda_Z) and the readout sees the prefix only through the mask.")
    p.add_argument("--readout", default="exact", choices=("exact", "mfvi"))
    p.add_argument("--n-global", type=int, default=0)
    p.add_argument(
        "--no-word-unary",
        action="store_true",
        help="drop the factor b (§16(c): 'Set b ≡ 0 to drop it'). Note this does NOT make "
        "the unigram distribution unreachable — logits(w) = LSE_a(S_{w,a} + log mu(a)) can "
        "represent an arbitrary unigram through S alone when mu is constant. What it "
        "removes is the *cheapest* route to it, and it forces that route through the same "
        "tensor the context path uses.",
    )
    p.add_argument("--save-ckpt", action="store_true", help="write a checkpoint for the ablation step")
    p.add_argument("--init-std", type=float, default=0.02,
                   help="not from either paper; the nanoGPT convention. Measured 2026-08-09: at\n0.02 an *untrained* model is already prefix-blind (ablation KL 9.5e-11), at 0.5 it\nis not (KL 6.9e-2).")
    p.add_argument("--lambda-z", type=float, default=1.0)
    p.add_argument("--lambda-h", type=float, default=None, help="default None -> 1/d")
    p.add_argument("--vocab-chunk", type=int, default=512)
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--l2-arc", type=float, default=5e-4)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-blocks", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--tag", default="pilot")
    return p.parse_args()


def main() -> None:
    a = build_args()
    out = Path(__file__).parent / f"{a.tag}_{a.readout}.json"

    corpus = load_ptb()
    tcfg = TrainConfig(
        block_size=a.block_size,
        batch_size=a.batch_size,
        max_steps=a.steps,
        lr=a.lr,
        l2_arc=a.l2_arc,
        eval_every=a.eval_every,
        eval_blocks=a.eval_blocks,
        ignore_first=1,
        seed=a.seed,
        device=a.device,
    )
    base = unigram_perplexity(
        corpus.train, corpus.valid, corpus.vocab_size,
        a.block_size, a.batch_size, ignore_first=1, limit=a.eval_blocks,
    )

    cfg = PTConfig(
        vocab_size=corpus.vocab_size,
        d=a.d,
        h=a.h,
        rank=a.rank,
        gamma=a.gamma,
        n_iters=a.n_iters,
        tau=a.tau,
        readout=a.readout,
        n_global=a.n_global,
        vocab_chunk=a.vocab_chunk,
        lambda_Z=a.lambda_z,
        lambda_H=a.lambda_h,
        word_unary=not a.no_word_unary,
        init_std=a.init_std,
    )
    model = CausalPTDecoder(cfg)

    print(f"corpus: vocab {corpus.vocab_size}, tokens {corpus.sizes()}")
    print(f"unigram val ppl on the identical token set: {base:.2f}")
    print(f"model: {cfg}")

    t0 = time.time()
    hist = train(model, corpus.train, corpus.valid, tcfg, random_batch)
    wall = time.time() - t0

    if a.save_ckpt:
        ck = Path("checkpoints") / f"{a.tag}_{a.readout}.pt"
        ck.parent.mkdir(exist_ok=True)
        torch.save({"cfg": cfg, "state_dict": model.state_dict(), "args": vars(a)}, ck)
        print(f"checkpoint {ck}")

    test = evaluate(model, corpus.test, tcfg)
    best = min(hist.val_ppl) if hist.val_ppl else float("nan")
    print(f"\nunigram baseline val ppl {base:.2f}")
    print(f"best val ppl {best:.2f}   test ppl {test['ppl']:.2f}")
    print(f"gate: {'PASS' if best < 0.5 * base else 'FAIL'} (val ppl below half the baseline)")

    out.write_text(
        json.dumps(
            {
                "args": vars(a),
                "unigram_val_ppl": base,
                "best_val_ppl": best,
                "test_ppl": test["ppl"],
                "wall_clock_s": wall,
                "params": model.num_parameters(),
                "history": hist.__dict__,
            },
            indent=2,
        )
    )
    print(f"written {out}")


if __name__ == "__main__":
    main()
