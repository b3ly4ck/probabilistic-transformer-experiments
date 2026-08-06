"""nanoGPT-style causal decoder baseline.

Used off the shelf in structure -- pre-norm blocks, causal self-attention, a 4x
MLP, learned positional embeddings, tied input/output embeddings.  Nothing here
is a contribution; it exists so the PT numbers have something to sit next to.

**Interface parity.**  Both models expose ``logits(tokens) -> (B, n, V)`` where
entry ``t`` is the prediction for ``tokens[t]`` given ``tokens[:t]`` and nothing
else.  That is what the PT decoder produces natively, since slot ``t`` reads
``D_t = {ROOT, 0..t-1}``.  A standard GPT instead predicts ``t+1`` from ``t``, so
here a **learned BOS vector is prepended internally** and the last position is
dropped.  This is the direct analogue of PT's root key ``r``: position 0 is
predicted from a learned constant and no context, in both models.  Without it the
two would be scored on different token sets and the perplexities would not be
comparable.

Embeddings are tied, matching PT -- where tying is forced by the construction
rather than chosen (§16(b)).  Untying would hand the baseline free parameters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int
    d_model: int
    n_layer: int
    n_head: int
    context: int
    dropout: float = 0.0
    mlp_ratio: int = 4


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        shape = (B, T, self.n_head, C // self.n_head)
        q, k, v = (t.view(*shape).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        hidden = cfg.mlp_ratio * cfg.d_model
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, cfg.d_model, bias=False),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig, generator: torch.Generator | None = None):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.wpe = nn.Embedding(cfg.context, cfg.d_model)
        self.bos = nn.Parameter(torch.zeros(cfg.d_model))
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.apply(self._init)
        with torch.no_grad():
            self.bos.normal_(0.0, 0.02, generator=generator)

    def _init(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def logits(self, tokens: Tensor) -> Tensor:
        """(B, n, V); entry t predicts tokens[t] from tokens[:t]."""
        B, n = tokens.shape
        if n > self.cfg.context:
            raise ValueError(f"sequence of {n} exceeds context {self.cfg.context}")
        # prepend the learned BOS and drop the last position, so position t sees
        # exactly tokens[:t] -- the same conditioning set the PT decoder uses
        emb = self.wte(tokens)
        x = torch.cat([self.bos.expand(B, 1, -1), emb[:, :-1]], dim=1)
        x = x + self.wpe(torch.arange(n, device=tokens.device))
        for block in self.blocks:
            x = block(x)
        return self.ln_f(x) @ self.wte.weight.T  # tied

    def forward(self, tokens: Tensor) -> Tensor:
        return self.logits(tokens)

    def loss(self, tokens: Tensor) -> Tensor:
        lg = self.logits(tokens)
        return F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tokens.reshape(-1))


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Total, embedding and non-embedding parameter counts.

    Reported separately in every table: with tied embeddings PT's budget sits
    almost entirely in the |V| x d matrix S, while a GPT spends much of its in
    the blocks.  A single total hides exactly the difference under test.
    """
    embed = 0
    for name, p in model.named_parameters():
        if name in {"S", "wte.weight", "wpe.weight"} or name.endswith("wte.weight"):
            embed += p.numel()
    total = sum(p.numel() for p in model.parameters())
    return {"total": total, "embedding": embed, "non_embedding": total - embed}
