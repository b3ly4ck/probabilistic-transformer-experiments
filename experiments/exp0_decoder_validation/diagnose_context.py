"""Diagnostic: does context reach the output?

Steps 1-3 of the diagnostic ladder.  Read-only on trained checkpoints.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import exact, mfvi  # noqa: E402
from src.config import PTConfig  # noqa: E402
from src.data import batchify, load_ptb  # noqa: E402
from src.gpt import GPT  # noqa: E402
from src.pt_decoder import CausalPTDecoder, causal_key_mask  # noqa: E402

torch.manual_seed(0)


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = CausalPTDecoder(cfg) if isinstance(cfg, PTConfig) else GPT(cfg)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, cfg, ck


def kl(p_logits, q_logits):
    p = torch.log_softmax(p_logits, -1)
    q = torch.log_softmax(q_logits, -1)
    return (p.exp() * (p - q)).sum(-1)


@torch.no_grad()
def step1_prefix_ablation(model, tokens, readout, label):
    """Zero and shuffle the prefix; measure how far the logits move."""
    call = (lambda t: model(t, readout=readout)) if readout else (lambda t: model(t))
    base = call(tokens)

    zeroed = torch.zeros_like(tokens)
    zeroed[:, -1] = tokens[:, -1]
    perm = torch.randperm(tokens.shape[1] - 1)
    shuffled = tokens.clone()
    shuffled[:, :-1] = tokens[:, perm]

    print(f"\n--- step 1: prefix ablation, {label} ---")
    for name, alt in (("zeroed", zeroed), ("shuffled", shuffled)):
        other = call(alt)
        # only the last position is comparable: same own-token, different prefix
        d = (base[:, -1] - other[:, -1]).abs().max()
        k = kl(base[:, -1], other[:, -1]).mean()
        agree = (base[:, -1].argmax(-1) == other[:, -1].argmax(-1)).float().mean()
        print(f"  {name:9s} max|dlogit| {float(d):9.4f}   KL {float(k):9.4f}   argmax agrees {float(agree):.3f}")


@torch.no_grad()
def step2_logit_decomposition(model, tokens, cfg):
    """logit_w = b_w + sum_a Q_Z(a) S[w,a].  Which term carries the variance?"""
    qbar = model.content_stream(tokens)
    n = tokens.shape[1]
    mask = causal_key_mask(n)
    Bkey = mfvi.contract_prefix(qbar, model.T, model.r)
    Q_W0 = torch.softmax(model.b, -1)
    s_bar = mfvi.word_message(Q_W0, model.S)
    Q_Z = torch.softmax(s_bar / cfg.lambda_Z, -1).expand(tokens.shape[0], n, cfg.d)
    for _ in range(cfg.n_rounds):
        sc = torch.einsum("bta,bcka->bctk", Q_Z, Bkey) / cfg.lambda_H
        Q_c = torch.softmax(sc.masked_fill(~mask, float("-inf")), -1)
        msg = torch.einsum("bctk,bcka->bta", Q_c, Bkey)
        Q_Z = torch.softmax((s_bar + msg) / cfg.lambda_Z, -1)

    context_term = torch.einsum("bta,va->btv", Q_Z, model.S)
    unary = model.b

    print("\n--- step 2: logit decomposition (MFVI readout) ---")
    print(f"  b_w            : std over vocab {float(unary.std()):8.4f}  range {float(unary.max()-unary.min()):8.4f}")
    print(f"  Q_Z . S        : std over vocab {float(context_term.std(-1).mean()):8.4f}  "
          f"range {float((context_term.max(-1).values - context_term.min(-1).values).mean()):8.4f}")
    print(f"  ratio std(b)/std(context)      : {float(unary.std()/context_term.std(-1).mean()):8.2f}")
    spread = context_term.std(dim=1).mean()
    print(f"  context term variation ACROSS positions (std over t): {float(spread):8.5f}")
    return Q_Z, Q_c, Bkey


@torch.no_grad()
def step3_entropies(model, cfg, Q_Z, Q_c, tokens):
    import math
    n = tokens.shape[1]
    ent_c = -(Q_c.clamp_min(1e-12).log() * Q_c).sum(-1)
    ent_z = -(Q_Z.clamp_min(1e-12).log() * Q_Z).sum(-1)
    print("\n--- step 3: trained posteriors ---")
    print(f"  Q_c entropy  mean {float(ent_c.mean()):7.4f} nats   max possible {math.log(n):7.4f}   "
          f"mean max Q_c {float(Q_c.max(-1).values.mean()):.4f}")
    print(f"  Q_Z entropy  mean {float(ent_z.mean()):7.4f} nats   max possible {math.log(cfg.d):7.4f}   "
          f"mean max Q_Z {float(Q_Z.max(-1).values.mean()):.4f}")
    print(f"  Q_Z spread across positions (std over t, mean over dims): {float(Q_Z.std(dim=1).mean()):.6f}")
    print("  parameter norms:")
    for nm in ("S", "T", "r", "b"):
        p = getattr(model, nm)
        print(f"    {nm:2s} |.|={float(p.norm()):9.4f}  std={float(p.std()):8.5f}  max|.|={float(p.abs().max()):8.4f}")


def main():
    corpus = load_ptb(download=False)
    data = batchify(corpus.valid, 8)[:, :64]

    for ck_path, readout in (("checkpoints/long/pt_mfvi_noG.pt", "mfvi"),):
        model, cfg, ck = load(ck_path)
        print(f"\n{'=' * 72}\n{ck['model']}  val ppl {ck['val_ppl']:.2f}  d={cfg.d} h={cfg.n_channels} "
              f"rounds={cfg.n_rounds} lambda_H={cfg.lambda_H:.5f}\n{'=' * 72}")
        step1_prefix_ablation(model, data, readout, ck["model"])
        Q_Z, Q_c, _ = step2_logit_decomposition(model, data, cfg)
        step3_entropies(model, cfg, Q_Z, Q_c, data)
        step3b_where_does_attention_point(model, cfg, data)

    gpt, _, ck = load("checkpoints/long/gpt.pt")
    print(f"\n{'=' * 72}\nCONTROL: {ck['model']}  val ppl {ck['val_ppl']:.2f}\n{'=' * 72}")
    step1_prefix_ablation(gpt, data, None, "gpt")

    # untrained PT, to separate "training destroyed it" from "it was never there"
    torch.manual_seed(0)
    fresh = CausalPTDecoder(PTConfig(vocab_size=corpus.vocab_size, d=256, n_channels=4, n_rounds=3))
    fresh.eval()
    print(f"\n{'=' * 72}\nCONTROL: untrained PT (same shape)\n{'=' * 72}")
    step1_prefix_ablation(fresh, data, "mfvi", "pt_untrained")




@torch.no_grad()
def step3b_where_does_attention_point(model, cfg, tokens):
    """If attention always selects the same key, the message is constant and so is Q_Z."""
    n = tokens.shape[1]
    mask = causal_key_mask(n)
    qbar = model.content_stream(tokens)
    Bkey = mfvi.contract_prefix(qbar, model.T, model.r)
    Q_W0 = torch.softmax(model.b, -1)
    s_bar = mfvi.word_message(Q_W0, model.S)
    Q_Z = torch.softmax(s_bar / cfg.lambda_Z, -1).expand(tokens.shape[0], n, cfg.d)
    for _ in range(cfg.n_rounds):
        sc = torch.einsum("bta,bcka->bctk", Q_Z, Bkey) / cfg.lambda_H
        Q_c = torch.softmax(sc.masked_fill(~mask, float("-inf")), -1)
        msg = torch.einsum("bctk,bcka->bta", Q_c, Bkey)
        Q_Z = torch.softmax((s_bar + msg) / cfg.lambda_Z, -1)

    root_mass = Q_c[..., 0].mean()
    argmax_is_root = (Q_c.argmax(-1) == 0).float().mean()
    print("\n--- step 3b: what does attention select? ---")
    print(f"  mean mass on ROOT (key 0)   : {float(root_mass):.4f}")
    print(f"  argmax == ROOT              : {float(argmax_is_root):.4f}")
    print(f"  message std across positions: {float(msg.std(dim=1).mean()):.6f}  "
          f"vs across label dims {float(msg.std(dim=-1).mean()):.6f}")
    # Also: the content stream's own qbar -- does IT vary with position?
    print(f"  qbar std across positions   : {float(qbar.std(dim=1).mean()):.6f}")
    print(f"  qbar entropy                : {float(-(qbar.clamp_min(1e-12).log()*qbar).sum(-1).mean()):.4f} nats")

if __name__ == "__main__":
    main()
