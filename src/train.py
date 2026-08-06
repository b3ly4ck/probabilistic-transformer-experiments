"""The shared training loop.

**One implementation for every model in the comparison.**  PT with the global
head, PT without it, and the GPT baseline all run through this function with the
same ``TrainConfig``.  A change that helps one model and not the others voids the
comparison, so there is deliberately no per-model branch here: the only thing
that varies is ``loss_fn``, which selects PT's readout and nothing else.

Scoring convention, identical for both model families: position ``t`` is scored
on predicting ``tokens[t]`` from ``tokens[:t]``.  See ``src/gpt.py`` for how the
baseline is aligned to it.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor

from .data import batchify, windows


@dataclass(frozen=True)
class TrainConfig:
    """Held identical across every model in an experiment arm."""

    context: int = 128
    batch_size: int = 32
    max_steps: int = 4000
    lr: float = 1e-3
    min_lr_ratio: float = 0.1
    warmup_steps: int = 200
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    betas: tuple[float, float] = (0.9, 0.95)
    eval_every: int = 250
    eval_batches: int = 0  # 0 = the whole split
    seed: int = 0
    device: str = "cpu"


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Linear warmup then cosine decay -- one schedule for all models."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    progress = min(1.0, progress)
    scale = cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    return cfg.lr * scale


def default_loss(model: nn.Module, batch: Tensor) -> Tensor:
    return model.loss(batch)


def pt_loss(readout: str) -> Callable[[nn.Module, Tensor], Tensor]:
    """Loss adapter selecting a PT readout, so the loop itself stays generic."""

    def fn(model: nn.Module, batch: Tensor) -> Tensor:
        return model.loss(batch, readout=readout)

    return fn


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data: Tensor,
    cfg: TrainConfig,
    loss_fn: Callable[[nn.Module, Tensor], Tensor] = default_loss,
) -> dict[str, float]:
    """Mean token cross-entropy and perplexity over a split."""
    was_training = model.training
    model.eval()
    batched = batchify(data, cfg.batch_size).to(cfg.device)
    total, n_batches = 0.0, 0
    for inputs, _ in windows(batched, cfg.context):
        total += float(loss_fn(model, inputs))
        n_batches += 1
        if cfg.eval_batches and n_batches >= cfg.eval_batches:
            break
    model.train(was_training)
    mean = total / max(1, n_batches)
    return {"loss": mean, "ppl": math.exp(mean), "batches": n_batches}


@dataclass
class TrainResult:
    history: list[dict] = field(default_factory=list)
    best_val_ppl: float = float("inf")
    best_step: int = -1
    final_val_ppl: float = float("inf")
    wall_clock_s: float = 0.0
    tokens_seen: int = 0

    def row(self) -> str:
        return (
            f"best val ppl {self.best_val_ppl:.2f} @ step {self.best_step} | "
            f"final {self.final_val_ppl:.2f} | {self.wall_clock_s:.0f} s | "
            f"{self.tokens_seen:,} tokens"
        )


def _flushing_print(message: str) -> None:
    """Slurm block-buffers redirected stdout, so an unflushed progress line is
    invisible until the process exits -- which makes a running job look hung."""
    print(message, flush=True)


def train(
    model: nn.Module,
    corpus,
    cfg: TrainConfig,
    loss_fn: Callable[[nn.Module, Tensor], Tensor] = default_loss,
    log: Callable[[str], None] | None = _flushing_print,
) -> TrainResult:
    torch.manual_seed(cfg.seed)
    model.to(cfg.device).train()

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay
    )
    train_batched = batchify(corpus.train, cfg.batch_size).to(cfg.device)
    result = TrainResult()

    stream = _endless(train_batched, cfg)
    start = time.time()
    for step in range(cfg.max_steps):
        for group in opt.param_groups:
            group["lr"] = lr_at(step, cfg)
        inputs = next(stream)
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model, inputs)
        loss.backward()
        if cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        result.tokens_seen += inputs.numel()

        last = step == cfg.max_steps - 1
        if cfg.eval_every and (step + 1) % cfg.eval_every == 0 or last:
            val = evaluate(model, corpus.valid, cfg, loss_fn)
            entry = {
                "step": step + 1,
                "train_loss": float(loss.detach()),
                "val_loss": val["loss"],
                "val_ppl": val["ppl"],
                "lr": lr_at(step, cfg),
                "elapsed_s": time.time() - start,
            }
            result.history.append(entry)
            if val["ppl"] < result.best_val_ppl:
                result.best_val_ppl, result.best_step = val["ppl"], step + 1
            result.final_val_ppl = val["ppl"]
            if log:
                log(
                    f"step {step + 1:>6} | train {entry['train_loss']:.4f} | "
                    f"val ppl {val['ppl']:.2f} | {entry['elapsed_s']:.0f}s"
                )
    result.wall_clock_s = time.time() - start
    return result


def _endless(batched: Tensor, cfg: TrainConfig):
    """Cycle over the training stream, offsetting each epoch so the window
    boundaries do not fall in the same places every time."""
    epoch = 0
    while True:
        shift = (epoch * (cfg.context // 2)) % max(1, cfg.context)
        any_yielded = False
        for inputs, _ in windows(batched, cfg.context, shift=shift):
            any_yielded = True
            yield inputs
        if not any_yielded:
            raise RuntimeError("training split shorter than one context window")
        epoch += 1
