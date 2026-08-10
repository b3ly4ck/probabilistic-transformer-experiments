"""The shared training loop.

Written once and used by every model in the comparison. If a change here helps one model
and not the others, the comparison is void — so anything model-specific is reached through
a narrow interface the model implements, not through a branch on the model's type:

* ``model.loss(idx, ignore_first)`` — mean NLL over the scored slots;
* ``model.arc_regulariser()`` — the L2 term on the ternary scores of Wu & Tu §4.2, which a
  model without ternary scores returns as zero;
* ``model.num_parameters()`` — the embedding / non-embedding split, which the research plan
  requires in every table.

Evaluation is deterministic: fixed, non-overlapping blocks in corpus order. A perplexity
that moves with the seed cannot be compared across runs or against the unigram baseline.

``ignore_first`` defaults to 1 and should stay there. PT scores every ``w_t`` from ``w_{<t}``
including ``w_0`` from ROOT alone; a GPT trained the usual way scores ``w_1..w_{n-1}``.
Dropping the first slot makes the two token sets identical, which is the difference between
a comparison and a coincidence.
"""

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional

import torch

from .data import sequential_batches


@dataclass
class TrainConfig:
    # data
    block_size: int = 64
    batch_size: int = 16

    # optimisation — Wu & Tu Table 2, PTB masked LM
    max_steps: int = 2000
    lr: float = 1e-3
    weight_decay: float = 1.4e-6
    betas: tuple = (0.9, 0.999)
    grad_clip: float = 1.0
    warmup_steps: int = 100
    min_lr_frac: float = 0.1
    l2_arc: float = 5e-4  # coefficient of model.arc_regulariser()

    # evaluation
    eval_every: int = 100
    eval_blocks: Optional[int] = None  # None -> the whole split
    eval_train_blocks: Optional[int] = 20
    # Training perplexity on a fixed slice of the train split, measured the same
    # deterministic way as validation. It is the sharper of the two numbers: a model that
    # cannot fit its own training data is not being regularised, it is failing to represent
    # the data. The previous implementation of this project sat at train ppl 611 against a
    # unigram baseline of 687 and that, not the validation number, was the finding.
    ignore_first: int = 1

    seed: int = 0
    device: str = "cpu"
    log_every: int = 20
    diagnostics: bool = True


@dataclass
class History:
    step: List[int] = field(default_factory=list)
    train_loss: List[float] = field(default_factory=list)
    train_ppl: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    val_ppl: List[float] = field(default_factory=list)
    diag: List[dict] = field(default_factory=list)


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Linear warmup then cosine decay to ``min_lr_frac * lr``."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / max(1, cfg.warmup_steps)
    if cfg.max_steps <= cfg.warmup_steps:
        return cfg.lr
    t = (step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps)
    return cfg.lr * (cfg.min_lr_frac + (1 - cfg.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * t)))


@torch.no_grad()
def evaluate(model, data: torch.Tensor, cfg: TrainConfig,
             limit: Optional[int] = -1) -> Dict[str, float]:
    """Mean NLL and perplexity over deterministic blocks, on the scored token set.

    ``limit`` defaults to ``cfg.eval_blocks``; pass an explicit value to score a slice.
    """
    if limit == -1:
        limit = cfg.eval_blocks
    was_training = model.training
    model.eval()
    total, n = 0.0, 0
    for block in sequential_batches(
        data, cfg.batch_size, cfg.block_size, limit, cfg.device
    ):
        logits = model(block)[:, cfg.ignore_first :]
        target = block[:, cfg.ignore_first :]
        nll = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), target.reshape(-1), reduction="sum"
        )
        total += float(nll)
        n += target.numel()
    model.train(was_training)
    mean = total / max(1, n)
    return {"loss": mean, "ppl": math.exp(min(mean, 700.0)), "tokens": n}


@torch.no_grad()
def _diagnostics(model, block: torch.Tensor) -> dict:
    """Message-scale readings, so a collapse is visible while it happens.

    The previous implementation of this project trained to the unigram baseline and the
    cause — the message dominating the word unary and the attention going hard — was only
    found afterwards, by instrumenting a checkpoint. These are the same readings, taken
    every evaluation instead.
    """
    from .diagnostics import contraction_rho

    out: dict = {}
    trace: list = []
    try:
        model.content_stream(block, trace=trace)
    except (AttributeError, TypeError):
        return out
    if trace:
        last = trace[-1]
        out.update(
            {
                "msg_over_unary": last["ratio"],
                "attn_entropy_frac": last["attn_entropy_frac"],
                "label_entropy": last["label_entropy"],
                "root_mass_over_uniform": last["root_mass_over_uniform"],
            }
        )
    if hasattr(model, "arc_scores"):
        out["max_abs_T"] = float(model.arc_scores().abs().max())
        qbar = model.content_stream(block)
        out["qbar_std_over_positions"] = float(qbar.std(dim=1).mean())
        B_full = model._slot_keys(model.contract(qbar), block.shape[1])
        out["rho"] = float(contraction_rho(model, B_full).mean())
    return out


def train(
    model,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    cfg: TrainConfig,
    batch_fn: Callable[[torch.Tensor, int, int, torch.Generator, str], torch.Tensor],
    log: Callable[[str], None] = print,
) -> History:
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    model.to(cfg.device).train()

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay
    )
    hist = History()
    log(f"config: {asdict(cfg)}")
    log(f"parameters: {model.num_parameters()}")

    t0 = time.time()
    running: List[float] = []
    for step in range(cfg.max_steps):
        for group in opt.param_groups:
            group["lr"] = lr_at(step, cfg)

        block = batch_fn(train_data, cfg.batch_size, cfg.block_size, gen, cfg.device)
        nll = model.loss(block, ignore_first=cfg.ignore_first)
        loss = nll + cfg.l2_arc * model.arc_regulariser()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        running.append(float(nll.detach()))

        if (step + 1) % cfg.log_every == 0:
            log(
                f"step {step + 1:>6}  train nll {sum(running) / len(running):.4f}"
                f"  ppl {math.exp(min(sum(running) / len(running), 700)):8.1f}"
                f"  lr {lr_at(step, cfg):.2e}  {time.time() - t0:6.0f}s"
            )
            running = []

        if (step + 1) % cfg.eval_every == 0 or step + 1 == cfg.max_steps:
            ev = evaluate(model, val_data, cfg)
            tr = (evaluate(model, train_data, cfg, limit=cfg.eval_train_blocks)
                  if cfg.eval_train_blocks else {"ppl": float("nan")})
            hist.train_ppl.append(tr["ppl"])
            diag = _diagnostics(model, block) if cfg.diagnostics else {}
            hist.step.append(step + 1)
            hist.val_loss.append(ev["loss"])
            hist.val_ppl.append(ev["ppl"])
            hist.diag.append(diag)
            hist.train_loss.append(float(nll.detach()))
            extra = "  ".join(f"{k} {v:.4f}" for k, v in diag.items())
            log(f"  eval @ {step + 1:>6}  val ppl {ev['ppl']:8.2f}  train ppl {tr['ppl']:8.2f}"
                f"  ({ev['tokens']} tokens)  {extra}")

    log(f"done in {time.time() - t0:.0f}s")
    return hist
