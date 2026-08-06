"""Step 4: where between d=8 and d=256 does the toy memorisation task stop fitting?

Check 6 passes at d=8 (MFVI memorises a sequence to ~0.003), so context
demonstrably works there.  Running the identical task across d localises the
failure to a d-dependent quantity.  lambda_H = 1/d is the obvious candidate, so
each d is run twice: with the coupled default and with lambda_H pinned to 1/8,
its value at the d where the task is known to pass.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import mfvi  # noqa: E402
from src.config import PTConfig  # noqa: E402
from src.pt_decoder import CausalPTDecoder, causal_key_mask  # noqa: E402

V, N, STEPS, LR = 12, 6, 1200, 0.05


def run(d, lambda_H=None, seed=0, steps=STEPS):
    cfg = PTConfig(vocab_size=V, d=d, n_channels=2, n_rounds=3, lambda_H=lambda_H)
    g = torch.Generator()
    g.manual_seed(seed)
    model = CausalPTDecoder(cfg, generator=g)
    tokens = torch.randint(0, V, (1, N), generator=g)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = model.loss(tokens, readout="mfvi")
        loss.backward()
        opt.step()

    with torch.no_grad():
        acc = (model(tokens, readout="mfvi").argmax(-1) == tokens).float().mean()
        m_W = model.S[tokens]
        q = torch.softmax(m_W / cfg.lambda_Z, -1)
        mask = causal_key_mask(N)
        for _ in range(cfg.n_rounds):
            Bkey = mfvi.contract_prefix(q, model.T, model.r)
            sc = torch.einsum("bta,bcka->bctk", q, Bkey) / cfg.lambda_H
            Q_c = torch.softmax(sc.masked_fill(~mask, float("-inf")), -1)
            msg = torch.einsum("bctk,bcka->bta", Q_c, Bkey)
            q = torch.softmax((m_W + msg) / cfg.lambda_Z, -1)
        ratio = float(msg.norm(dim=-1).mean() / m_W.norm(dim=-1).mean())
        ent_c = float(-(Q_c.clamp_min(1e-12).log() * Q_c).sum(-1).mean())
    return float(loss), float(acc), ratio, ent_c


print(f"{'d':>6} {'lambda_H':>10} {'final loss':>11} {'token acc':>10} {'msg/word':>9} {'Qc entropy':>11}")
print("-" * 62)
for d in (8, 16, 32, 64, 128, 256):
    loss, acc, ratio, ent = run(d)
    print(f"{d:>6} {1 / d:>10.5f} {loss:>11.4f} {acc:>10.2f} {ratio:>9.2f} {ent:>11.4f}", flush=True)

print("\nsame, with lambda_H decoupled and pinned to 1/8 (its value where the task passes)")
print(f"{'d':>6} {'lambda_H':>10} {'final loss':>11} {'token acc':>10} {'msg/word':>9} {'Qc entropy':>11}")
print("-" * 62)
for d in (8, 16, 32, 64, 128, 256):
    loss, acc, ratio, ent = run(d, lambda_H=1 / 8)
    print(f"{d:>6} {1 / 8:>10.5f} {loss:>11.4f} {acc:>10.2f} {ratio:>9.2f} {ent:>11.4f}", flush=True)
