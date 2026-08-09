"""Stage-by-stage: at which point does the prediction stop depending on *which words*?

Every diagnostic before this one measured variation across *positions* — which is exactly the
axis the trained model still uses, since it learned `p(w | position in block)`. The quantity
that matters is variation across **sequences at a fixed slot**: two different prefixes of the
same length, same slot index, and the question is whether the model's internals differ.

Reported per stage is the *content fraction*

    std over sequences at a fixed slot  /  overall std

which is scale-free, so stages with very different magnitudes are comparable. A stage where
the fraction drops sharply is where the word identity is discarded.

An untrained model at ``init_std = 0.5`` is measured alongside as the control: its prefix
ablation gives KL 6.9e-2 with the argmax changing on 62 % of slots, so its readout demonstrably
works. Whatever stage differs between it and a trained model is the learned degeneracy.

Run:  python -m experiments.exp1_language_modeling.where_content_dies
"""

from typing import List, Tuple

import torch

from src import CausalPTDecoder, PTConfig
from src.data import load_ptb, sequential_batches


def content_fraction(x: torch.Tensor) -> Tuple[float, float]:
    """``x`` is ``(B, n, ...)``. Returns (std across sequences at fixed slot, overall std)."""
    return float(x.std(dim=0).mean()), float(x.std())


def trace(model: CausalPTDecoder, block: torch.Tensor) -> List[Tuple[str, float, float]]:
    cfg = model.cfg
    out: List[Tuple[str, float, float]] = []

    qbar = model.content_stream(block)
    out.append(("q_bar (content stream)",) + content_fraction(qbar))

    Bk = model.contract(qbar)
    out.append(("B (contracted arcs)",) + content_fraction(Bk[-1][:, 0]))

    _, sbar = model._word_prior()
    qz = torch.softmax(sbar / cfg.lambda_Z, dim=-1).expand(block.shape[0], block.shape[1], cfg.d)
    for r in range(cfg.tau):
        G, alpha = model._arc_message(qz, Bk)
        out.append((f"alpha  (attention, round {r + 1})",) + content_fraction(alpha[:, 0]))
        out.append((f"G      (message,   round {r + 1})",) + content_fraction(G))
        qz = torch.softmax((sbar + G) / cfg.lambda_Z, dim=-1)
        out.append((f"Q_Z    (predictive, round {r + 1})",) + content_fraction(qz))

    logits = qz @ model.S.T
    if model.b is not None:
        logits = logits + model.b
    out.append(("logits",) + content_fraction(logits / cfg.lambda_W))
    return out


def report(name: str, model: CausalPTDecoder, block: torch.Tensor) -> None:
    print(f"\n{name}")
    print(f"  {'stage':34s} {'std over seq':>13} {'overall std':>13} {'content frac':>13}")
    for stage, across, overall in trace(model, block):
        frac = across / max(overall, 1e-30)
        print(f"  {stage:34s} {across:13.4e} {overall:13.4e} {frac:13.4e}")


def main() -> None:
    torch.set_grad_enabled(False)
    corpus = load_ptb()
    block = next(iter(sequential_batches(corpus.valid, 8, 64)))

    torch.manual_seed(0)
    untrained = CausalPTDecoder(
        PTConfig(vocab_size=corpus.vocab_size, d=256, h=8, rank=64, gamma=3, n_iters=3,
                 readout="mfvi", init_std=0.5)
    ).eval()
    report("UNTRAINED, init_std=0.5 (control — its readout demonstrably works)", untrained, block)

    for tag in ("init0.5_mfvi", "step1_withb_mfvi"):
        try:
            ck = torch.load(f"checkpoints/{tag}.pt", map_location="cpu", weights_only=False)
        except FileNotFoundError:
            print(f"\n(no checkpoint {tag})")
            continue
        m = CausalPTDecoder(ck["cfg"])
        m.load_state_dict(ck["state_dict"])
        report(f"TRAINED {tag} (val ppl at the unigram baseline)", m.eval(), block)


if __name__ == "__main__":
    main()
