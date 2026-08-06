"""Where the collapse happens: inside the content stream's Z-update.

q_t = softmax( (S[w_t] + message_t) / lambda_Z ).  If the message swamps the
word unary, q_t forgets which word it is, and nothing downstream can recover it.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import mfvi  # noqa: E402
from src.config import PTConfig  # noqa: E402
from src.data import batchify, load_ptb  # noqa: E402
from src.pt_decoder import CausalPTDecoder, causal_key_mask  # noqa: E402


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = CausalPTDecoder(ck["config"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck["config"], ck


@torch.no_grad()
def trace_content_stream(model, cfg, tokens, label):
    n = tokens.shape[1]
    mask = causal_key_mask(n)
    m_W = model.S[tokens]
    q = torch.softmax(m_W / cfg.lambda_Z, dim=-1)
    print(f"\n=== {label}: d={cfg.d} h={cfg.n_channels} lambda_H={cfg.lambda_H:.5f} ===")
    print(f"{'round':>6} {'|word unary|':>13} {'|message|':>11} {'msg/word':>9} "
          f"{'q std over t':>13} {'q entropy':>10} {'Qc entropy':>11}")
    print(f"{'init':>6} {float(m_W.norm(dim=-1).mean()):>13.4f} {'-':>11} {'-':>9} "
          f"{float(q.std(dim=1).mean()):>13.6f} "
          f"{float(-(q.clamp_min(1e-12).log()*q).sum(-1).mean()):>10.4f} {'-':>11}")

    for rnd in range(cfg.n_rounds):
        Bkey = mfvi.contract_prefix(q, model.T, model.r)
        sc = torch.einsum("bta,bcka->bctk", q, Bkey) / cfg.lambda_H
        Q_c = torch.softmax(sc.masked_fill(~mask, float("-inf")), -1)
        msg = torch.einsum("bctk,bcka->bta", Q_c, Bkey)
        q = torch.softmax((m_W + msg) / cfg.lambda_Z, dim=-1)
        ent_c = -(Q_c.clamp_min(1e-12).log() * Q_c).sum(-1).mean()
        print(f"{rnd + 1:>6} {float(m_W.norm(dim=-1).mean()):>13.4f} "
              f"{float(msg.norm(dim=-1).mean()):>11.4f} "
              f"{float(msg.norm(dim=-1).mean() / m_W.norm(dim=-1).mean()):>9.2f} "
              f"{float(q.std(dim=1).mean()):>13.6f} "
              f"{float(-(q.clamp_min(1e-12).log()*q).sum(-1).mean()):>10.4f} "
              f"{float(ent_c):>11.4f}")

    # does q still know its own word?  compare q for the same word across contexts
    flat_tok, flat_q = tokens.reshape(-1), q.reshape(-1, cfg.d)
    uniq = flat_tok.unique()[:50]
    within, between = [], []
    for w in uniq:
        rows = flat_q[flat_tok == w]
        if rows.shape[0] > 1:
            within.append((rows - rows.mean(0)).norm(dim=-1).mean())
    centroids = torch.stack([flat_q[flat_tok == w].mean(0) for w in uniq])
    between = (centroids - centroids.mean(0)).norm(dim=-1).mean()
    if within:
        w = torch.stack(within).mean()
        print(f"\n  q variation WITHIN a word type (different contexts): {float(w):.6f}")
        print(f"  q variation BETWEEN word types                     : {float(between):.6f}")
        print(f"  ratio between/within                               : {float(between / w):.3f}")
        print("  (>1 means q still identifies the word; ~0 means the word is gone)")


corpus = load_ptb(download=False)
data = batchify(corpus.valid, 8)[:, :64]
model, cfg, ck = load("checkpoints/long/pt_mfvi_noG.pt")
trace_content_stream(model, cfg, data, f"TRAINED {ck['model']} (val ppl {ck['val_ppl']:.1f})")

torch.manual_seed(0)
fresh = CausalPTDecoder(PTConfig(vocab_size=corpus.vocab_size, d=256, n_channels=4, n_rounds=3))
fresh.eval()
trace_content_stream(fresh, fresh.cfg, data, "UNTRAINED, same shape")
