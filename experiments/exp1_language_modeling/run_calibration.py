"""Calibration run for Experiment 1 -- operational, not scientific.

Purpose: measure wall-clock per run, peak GPU memory, and the order of magnitude
and sign of the PT-vs-GPT gap, at a small budget where ``d`` stays in the
hundreds.  The science is the d sweep that follows; this run exists so the sweep
can be planned with numbers instead of guesses.

Every model runs through the same training loop with the same TrainConfig.  Only
``loss_fn`` differs, and it selects the PT readout and nothing else.

Usage:
    python experiments/exp1_language_modeling/run_calibration.py --steps 2000
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import PTConfig  # noqa: E402
from src.data import load_ptb  # noqa: E402
from src.gpt import GPT, GPTConfig, count_parameters  # noqa: E402
from src.pt_decoder import CausalPTDecoder  # noqa: E402
from src.train import TrainConfig, default_loss, evaluate, pt_loss, train  # noqa: E402


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def build_models(vocab_size: int, d: int, gpt_d: int, gpt_layers: int, ctx: int, chunk: int):
    """The calibration matrix.

    Arm 1.1 runs both readouts; arm 1.2 runs MFVI only, because G_t is degenerate
    under exact inference by construction -- it is a leaf on Z_t, so exact
    inference integrates it out into a term that cannot depend on position.
    """
    pt_base = dict(vocab_size=vocab_size, d=d, n_channels=4, n_rounds=3, readout_chunk=chunk)
    return [
        ("gpt", lambda: GPT(GPTConfig(vocab_size, gpt_d, gpt_layers, 4, ctx)), None),
        ("pt_exact_noG", lambda: CausalPTDecoder(PTConfig(**pt_base)), pt_loss("exact")),
        ("pt_mfvi_noG", lambda: CausalPTDecoder(PTConfig(**pt_base)), pt_loss("mfvi")),
        (
            "pt_mfvi_G",
            lambda: CausalPTDecoder(
                PTConfig(**pt_base, use_global_head=True, n_global=64, lambda_G=5.0)
            ),
            pt_loss("mfvi"),
        ),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--d", type=int, default=256, help="PT label dimension")
    ap.add_argument("--gpt-d", type=int, default=160)
    ap.add_argument("--gpt-layers", type=int, default=4)
    ap.add_argument("--context", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--readout-chunk", type=int, default=250)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--only", default=None, help="run a single model by name")
    ap.add_argument("--out", default=None)
    ap.add_argument("--save-ckpt", default=None, help="directory to write <model>.pt into")
    args = ap.parse_args()

    corpus = load_ptb()
    train_cfg = TrainConfig(
        context=args.context,
        batch_size=args.batch_size,
        max_steps=args.steps,
        lr=args.lr,
        eval_every=args.eval_every,
        seed=args.seed,
        device=args.device,
    )

    env = {
        "commit": git_commit(),
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "device": args.device,
        "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "torch": torch.__version__,
        "host": platform.node(),
        "train_config": train_cfg.__dict__,
        "args": vars(args),
    }
    print(json.dumps(env, indent=2, default=str), flush=True)

    rows = []
    for name, build, loss_fn in build_models(
        corpus.vocab_size, args.d, args.gpt_d, args.gpt_layers, args.context, args.readout_chunk
    ):
        if args.only and name != args.only:
            continue
        print(f"\n{'=' * 70}\n{name}\n{'=' * 70}", flush=True)
        torch.manual_seed(args.seed)
        model = build()
        params = count_parameters(model)
        print(f"params {params}", flush=True)

        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        started = time.time()
        try:
            result = train(model, corpus, train_cfg, loss_fn=loss_fn or default_loss)
            test = evaluate(model, corpus.test, train_cfg, loss_fn or default_loss)
            status, error = "ok", None
        except RuntimeError as exc:  # OOM is the expected failure mode here
            result, test = None, None
            status, error = "failed", str(exc)[:400]
            print(f"FAILED: {error}", flush=True)

        peak = (
            torch.cuda.max_memory_allocated() / 2**30 if args.device == "cuda" else float("nan")
        )
        row = {
            "model": name,
            "status": status,
            "error": error,
            "params": params,
            "wall_clock_s": round(time.time() - started, 1),
            "peak_mem_gib": round(peak, 2),
            "best_val_ppl": round(result.best_val_ppl, 2) if result else None,
            "final_val_ppl": round(result.final_val_ppl, 2) if result else None,
            "test_ppl": round(test["ppl"], 2) if test else None,
            "tokens_seen": result.tokens_seen if result else 0,
            "history": result.history if result else [],
        }
        rows.append(row)
        print(f"-> {json.dumps({k: v for k, v in row.items() if k != 'history'})}", flush=True)

        if args.save_ckpt and status == "ok":
            ckpt_dir = Path(args.save_ckpt)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"model": name, "state_dict": model.state_dict(), "config": model.cfg,
                 "params": params, "val_ppl": result.best_val_ppl},
                ckpt_dir / f"{name}.pt",
            )
            print(f"saved {ckpt_dir / (name + '.pt')}", flush=True)

        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()

    suffix = f"_{args.only}" if args.only else ""
    out = Path(args.out or Path(__file__).parent / f"calibration_results{suffix}.json")
    out.write_text(json.dumps({"env": env, "rows": rows}, indent=2, default=str))
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
