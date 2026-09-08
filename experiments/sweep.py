"""One sweep runner for every model in the project.

Why this file exists
--------------------
Every experiment from 2026-08 onward is a *grid*: a set of cells that differ in a few named
fields and are otherwise identical. Writing a bespoke driver per experiment is how the
2026-08 session ended up with five scripts that each re-implement configuration, evaluation
and result serialisation slightly differently, and it is how a comparison silently stops
being a comparison. `CLAUDE.md`: "The training loop is written once and shared by all three
models — if a change to the loop helps one model and not the others, the comparison is void."
The same argument applies one level up, to the harness around the loop.

A sweep is a list of **cells**. A cell is a flat dict of overrides; everything not named
takes the default in :class:`Cell`. The runner:

* builds the model from the cell (`pt`, `gpt`, `looped`, or a `switch` ladder rung),
* trains it through `src.train.train` — the one shared loop, no branches on model type,
* evaluates deterministically on valid and test, runs the prefix ablation,
* appends one fully-specified row to a JSON file **as soon as the cell finishes**.

Appending immediately is not a convenience. A sweep that writes only at the end loses every
completed cell when the allocation expires, and a preempted 30-cell grid then reports
nothing. Rows carry their own cell dict, so a partially finished grid is still a usable
result and a re-run skips what is already there (`--resume`, on by default).

Sharding: `--shard i/n` runs only the cells with `index % n == i`, so one grid can be spread
over a slurm array without splitting the spec.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import socket
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from src import CausalPTDecoder, PTConfig
from src.data import Corpus, load_ptb, random_batch, unigram_perplexity
from src.gpt import GPT, GPTConfig
from src.train import TrainConfig, evaluate, train

from experiments.exp1_language_modeling.ablate_prefix import ablate

# --------------------------------------------------------------------------- cells --


@dataclass
class Cell:
    """One point of a sweep. Flat on purpose: a cell is what goes in the results row."""

    name: str = "cell"
    model: str = "pt"  # pt | gpt | looped | switch

    # --- PT graph shape ---
    d: int = 32
    h: int = 2
    rank: Optional[int] = None  # None -> min(64, d) is applied by build(); -1 -> full T
    gamma: int = 3
    n_iters: int = 3
    tau: int = 2
    readout: str = "mfvi"
    n_global: int = 0
    b_glob_init_std: Optional[float] = None
    regularise_global_head: bool = True
    word_unary: bool = True
    freeze_b: bool = True

    # --- PT inference constants ---
    alpha_Z: float = 1.0
    lambda_Z: float = 1.0
    lambda_H: Optional[float] = None  # None -> the config default 1/d
    lambda_H_rule: Optional[str] = None  # "1/d" | "1/sqrt(d)" | "const" — sets lambda_H
    lambda_H_coeff: float = 1.0  # multiplier applied to the rule
    lambda_W: float = 1.0
    lambda_G: float = 1.0
    temp_mode: str = "fixed"  # fixed | qnorm | qknorm  (see PTConfig)
    qk_gain: float = 1.0
    init_std: float = 0.02
    arc_init_std: Optional[float] = None
    root_init_std: Optional[float] = None
    init_dist: str = "normal"  # normal | uniform | orthogonal

    # --- GPT / Looped shape ---
    n_embd: int = 160
    n_layer: int = 4
    n_head: int = 4
    dropout: float = 0.0

    # --- switch ladder (model="switch") ---
    switches: Dict[str, Any] = field(default_factory=dict)

    # --- optimisation ---
    lr: float = 2e-2
    steps: int = 6000
    warmup: int = 100
    min_lr_frac: float = 0.1
    l2_arc: float = 5e-4
    weight_decay: float = 1.4e-6
    grad_clip: float = 1.0
    batch_size: int = 16
    block_size: int = 64
    eval_every: int = 500
    seed: int = 0
    keep_best: bool = True  # early stopping in the shared loop; see TrainConfig.keep_best

    # --- data ---
    corpus: str = "ptb"

    # --- bookkeeping ---
    ablate_batches: int = 6
    save_ckpt: bool = False
    tags: Dict[str, Any] = field(default_factory=dict)  # free-form, copied into the row


def cell_key(c: Cell) -> str:
    """Identity of a cell for resume purposes: everything that can change the result."""
    d = asdict(c)
    d.pop("save_ckpt", None)
    d.pop("ablate_batches", None)
    d.pop("tags", None)
    return json.dumps(d, sort_keys=True, default=str)


# ---------------------------------------------------------------------- construction --


def resolve_lambda_H(c: Cell) -> Optional[float]:
    """Turn a width rule into the scalar the config takes.

    ``None`` leaves `PTConfig` at its own default, which is Wu & Tu's ``1/d`` (App. A.5).
    The rules exist so a grid can say *how* the temperature scales with width instead of
    hard-coding a number per cell — that is the object of the width-transfer experiment.
    """
    if c.lambda_H is not None:
        return c.lambda_H
    if c.lambda_H_rule in (None, "1/d"):
        return None if c.lambda_H_coeff == 1.0 else c.lambda_H_coeff / c.d
    if c.lambda_H_rule == "1/sqrt(d)":
        return c.lambda_H_coeff / math.sqrt(c.d)
    if c.lambda_H_rule == "const":
        return c.lambda_H_coeff
    raise ValueError(f"unknown lambda_H_rule {c.lambda_H_rule!r}")


def build_model(c: Cell, vocab_size: int):
    if c.model in ("gpt", "looped"):
        cfg = GPTConfig(
            vocab_size=vocab_size,
            block_size=c.block_size,
            n_layer=c.n_layer,
            n_head=c.n_head,
            n_embd=c.n_embd,
            dropout=c.dropout,
            shared_block=(c.model == "looped"),
        )
        return GPT(cfg), cfg
    if c.model == "switch":
        from src.switches import SwitchConfig, SwitchModel

        cfg = SwitchConfig(
            vocab_size=vocab_size,
            block_size=c.block_size,
            n_layer=c.n_layer,
            n_head=c.n_head,
            n_embd=c.n_embd,
            d_label=c.d,
            dropout=c.dropout,
            **c.switches,
        )
        return SwitchModel(cfg), cfg
    rank = c.rank
    if rank is None:
        rank = min(64, c.d)
    elif rank < 0:
        rank = None
    cfg = PTConfig(
        vocab_size=vocab_size,
        d=c.d,
        h=c.h,
        rank=rank,
        gamma=c.gamma,
        n_iters=c.n_iters,
        tau=c.tau,
        readout=c.readout,
        n_global=c.n_global,
        b_glob_init_std=c.b_glob_init_std,
        regularise_global_head=c.regularise_global_head,
        word_unary=c.word_unary,
        freeze_word_unary=c.freeze_b,
        alpha_Z=c.alpha_Z,
        lambda_Z=c.lambda_Z,
        lambda_H=resolve_lambda_H(c),
        lambda_W=c.lambda_W,
        lambda_G=c.lambda_G,
        temp_mode=c.temp_mode,
        qk_gain=c.qk_gain,
        init_std=c.init_std,
        arc_init_std=c.arc_init_std,
        root_init_std=c.root_init_std,
        init_dist=c.init_dist,
        vocab_chunk=1024,
    )
    return CausalPTDecoder(cfg), cfg


# ------------------------------------------------------------------------ execution --


def run_cell(c: Cell, corpus: Corpus, unigram: float, device: str, log) -> Dict[str, Any]:
    torch.manual_seed(c.seed)
    model, mcfg = build_model(c, corpus.vocab_size)

    if c.model == "pt" and c.freeze_b and c.word_unary:
        counts = torch.bincount(corpus.train, minlength=corpus.vocab_size).double()
        model.set_word_unary((counts / counts.sum()).clamp_min(1e-12).log())

    tcfg = TrainConfig(
        block_size=c.block_size,
        batch_size=c.batch_size,
        max_steps=c.steps,
        lr=c.lr,
        weight_decay=c.weight_decay,
        grad_clip=c.grad_clip,
        warmup_steps=c.warmup,
        min_lr_frac=c.min_lr_frac,
        l2_arc=c.l2_arc,
        eval_every=c.eval_every,
        eval_train_blocks=20,
        ignore_first=1,
        device=device,
        seed=c.seed,
        log_every=10**9,
        diagnostics=(c.model in ("pt", "switch")),
        keep_best=c.keep_best,
    )

    t0 = time.time()
    hist = train(model, corpus.train, corpus.valid, tcfg, random_batch, log=log)
    test = evaluate(model, corpus.test, tcfg)
    abl = ablate(
        model, corpus.valid, c.block_size, c.batch_size, c.ablate_batches, device=device
    )
    diag = hist.diag[-1] if hist.diag else {}
    params = model.num_parameters()

    if c.save_ckpt:
        Path("checkpoints").mkdir(exist_ok=True)
        torch.save(
            {"cfg": mcfg, "state_dict": model.state_dict(), "cell": asdict(c)},
            Path("checkpoints") / f"{c.name}_s{c.seed}.pt",
        )

    row: Dict[str, Any] = {
        "cell": asdict(c),
        "unigram": round(unigram, 3),
        # `val_ppl` is the number of the *selected* checkpoint -- the best-validation one
        # when keep_best is on, the last one otherwise -- and `test_ppl` is measured on that
        # same checkpoint. `val_ppl_final` keeps the end-of-schedule value alongside, so a
        # run that diverged after an early transient is still visible as such.
        "val_ppl": round(hist.best_val_ppl if hist.best_val_ppl else hist.val_ppl[-1], 3),
        "val_ppl_final": round(hist.val_ppl[-1], 3),
        "val_ppl_min": round(min(hist.val_ppl), 3),
        "best_step": hist.best_step,
        "test_ppl": round(test["ppl"], 3),
        "train_ppl": round(hist.train_ppl[-1], 3) if hist.train_ppl else float("nan"),
        "val_loss": round(hist.val_loss[-1], 5),
        "params": params,
        "ablation_kl": abl["shuffled/kl_mean"],
        "ablation_argmax_same": abl["shuffled/argmax_agrees"],
        "seconds": round(time.time() - t0, 1),
        "steps_logged": hist.step,
        "val_trace": [round(v, 2) for v in hist.val_ppl],
        "train_trace": [round(v, 2) for v in hist.train_ppl],
        "diag_final": {k: (round(v, 5) if isinstance(v, float) else v) for k, v in diag.items()},
        "diag_trace": hist.diag,
        "host": socket.gethostname(),
        "device": device,
    }
    return row


# ----------------------------------------------------------------------- the runner --


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:  # pragma: no cover - environment dependent
        return "unknown"


def load_corpus(name: str) -> Corpus:
    if name == "ptb":
        return load_ptb()
    if name in ("wikitext2", "wt2"):
        from src.data import load_wikitext2

        return load_wikitext2()
    raise ValueError(f"unknown corpus {name!r}")


def run_sweep(
    cells: List[Cell],
    out_path: Path,
    device: str,
    resume: bool = True,
    shard: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if shard:
        i, n = (int(x) for x in shard.split("/"))
        cells = [c for k, c in enumerate(cells) if k % n == i]
        print(f"shard {i}/{n}: {len(cells)} cells", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing: List[Dict[str, Any]] = []
    if resume and out_path.exists():
        try:
            existing = json.loads(out_path.read_text())["rows"]
            print(f"resume: {len(existing)} rows already in {out_path}", flush=True)
        except Exception:
            existing = []
    done = {json.dumps(r["cell"], sort_keys=True, default=str) for r in existing}

    corpora: Dict[str, Corpus] = {}
    unigrams: Dict[str, float] = {}
    rows = list(existing)
    meta = {
        "commit": git_commit(),
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "slurm_job": os.environ.get("SLURM_JOB_ID", ""),
    }
    print(f"meta: {meta}", flush=True)

    for k, c in enumerate(cells):
        key = json.dumps(asdict(c), sort_keys=True, default=str)
        if key in done:
            print(f"[{k + 1}/{len(cells)}] skip (done) {c.name}", flush=True)
            continue
        if c.corpus not in corpora:
            corpora[c.corpus] = load_corpus(c.corpus)
            cc = corpora[c.corpus]
            unigrams[c.corpus] = unigram_perplexity(
                cc.train, cc.valid, cc.vocab_size, c.block_size, c.batch_size, ignore_first=1
            )
            print(
                f"corpus {c.corpus}: vocab {cc.vocab_size} sizes {cc.sizes()} "
                f"unigram {unigrams[c.corpus]:.2f}",
                flush=True,
            )
        print(f"\n[{k + 1}/{len(cells)}] {c.name}  {asdict(c)}", flush=True)
        try:
            row = run_cell(
                c, corpora[c.corpus], unigrams[c.corpus], device,
                log=lambda s: print("    " + s, flush=True),
            )
        except Exception as exc:  # a dead cell must not kill the grid
            import traceback

            traceback.print_exc()
            row = {"cell": asdict(c), "error": f"{type(exc).__name__}: {exc}"}
        rows.append(row)
        out_path.write_text(json.dumps({"meta": meta, "rows": rows}, indent=1))
        if "error" not in row:
            print(
                f"  -> {c.name}: val {row['val_ppl']:9.2f}  test {row['test_ppl']:9.2f}  "
                f"train {row['train_ppl']:9.2f}  "
                f"non-emb {row['params']['non_embedding']:>9}  "
                f"tot {row['params']['total']:>9}  "
                f"KL {row['ablation_kl']:.3e}  {row['seconds']:.0f}s",
                flush=True,
            )
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--spec", required=True, help="python module path with build_cells()")
    p.add_argument("--out", default=None)
    p.add_argument("--shard", default=None, help="i/n")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit", type=int, default=None, help="run only the first N cells")
    p.add_argument("rest", nargs="*", help="extra args forwarded to build_cells()")
    a = p.parse_args()

    import importlib

    mod = importlib.import_module(a.spec)
    cells = mod.build_cells(*a.rest)
    if a.limit:
        cells = cells[: a.limit]
    out = Path(a.out) if a.out else Path(mod.__file__).with_suffix(".json")
    if a.shard:
        i, n = a.shard.split("/")
        out = out.with_name(f"{out.stem}.shard{i}of{n}.json")
    print(f"{len(cells)} cells -> {out}", flush=True)
    run_sweep(cells, out, a.device, resume=not a.no_resume, shard=a.shard)
    print("\nSWEEP DONE", flush=True)


if __name__ == "__main__":
    main()
