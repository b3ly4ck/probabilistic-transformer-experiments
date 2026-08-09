"""Does context reach the output at all? Prefix ablation on a trained checkpoint.

Step 2 of the diagnosis in EXPERIMENT_STATUS.md, and the decisive read: the diagnostics
logged during training describe the *content stream*, but they do not directly measure
whether the prediction at slot t moves when the prefix moves. This does.

Slot ``t`` predicts ``w_t`` from ``w_{<t}``, so the last slot of a block is scored against
its whole prefix. Three conditions on the same blocks:

* **true**     — the real prefix;
* **shuffled** — the same tokens in a random order. A model that uses word order but not
  identity is unaffected; a model that uses neither is unaffected by both this and the next;
* **constant** — every prefix token replaced by one repeated token: no content at all.

Reported per condition: KL from the true predictive distribution, the largest logit
movement, and how often the argmax is unchanged.

For scale, the previous implementation of this project measured, under shuffling, KL 0.0115
for PT against 12.26 for GPT, with 75 % of PT's argmax predictions unchanged.

Run:  python -m experiments.exp1_language_modeling.ablate_prefix checkpoints/<name>.pt
"""

import argparse
from pathlib import Path

import torch

from src import CausalPTDecoder
from src.data import load_ptb, sequential_batches


def _stats(base_logits: torch.Tensor, alt_logits: torch.Tensor) -> dict:
    p = torch.log_softmax(base_logits, dim=-1)
    q = torch.log_softmax(alt_logits, dim=-1)
    kl = (p.exp() * (p - q)).sum(-1)
    return {
        "kl_mean": float(kl.mean()),
        "kl_max": float(kl.max()),
        "max_abs_dlogit": float((base_logits - alt_logits).abs().max()),
        "argmax_agrees": float((base_logits.argmax(-1) == alt_logits.argmax(-1)).double().mean()),
    }


@torch.no_grad()
def ablate(model: CausalPTDecoder, data: torch.Tensor, block_size: int, batch_size: int,
           n_batches: int, seed: int = 0, device: str = "cpu") -> dict:
    model.eval().to(device)
    g = torch.Generator().manual_seed(seed)
    acc: dict = {}
    n = 0
    for i, block in enumerate(sequential_batches(data, batch_size, block_size, device=device)):
        if i >= n_batches:
            break
        last = block.shape[1] - 1
        base = model(block)[:, last]

        shuffled = block.clone()
        perm = torch.randperm(last, generator=g)
        shuffled[:, :last] = block[:, perm]

        constant = block.clone()
        constant[:, :last] = block[:, 0:1]

        for name, alt in (("shuffled", shuffled), ("constant", constant)):
            s = _stats(base, model(alt)[:, last])
            for k, v in s.items():
                acc[f"{name}/{k}"] = acc.get(f"{name}/{k}", 0.0) + v
        n += 1

    return {k: v / max(1, n) for k, v in acc.items()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--batches", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()

    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    model = CausalPTDecoder(cfg)
    model.load_state_dict(ck["state_dict"])

    corpus = load_ptb()
    block_size = ck["args"]["block_size"]
    out = ablate(model, corpus.valid, block_size, a.batch_size, a.batches, device=a.device)

    print(f"{Path(a.checkpoint).name}  (d={cfg.d} h={cfg.h} T={cfg.n_iters} "
          f"readout={cfg.readout} word_unary={cfg.word_unary})")
    for cond in ("shuffled", "constant"):
        print(f"  {cond:>9}: KL {out[f'{cond}/kl_mean']:.4f} (max {out[f'{cond}/kl_max']:.4f})"
              f"   max|dlogit| {out[f'{cond}/max_abs_dlogit']:.4f}"
              f"   argmax unchanged {out[f'{cond}/argmax_agrees']:.3f}")


if __name__ == "__main__":
    main()
