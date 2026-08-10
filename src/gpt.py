"""GPT baseline — nanoGPT, off the shelf.

Only the PT decoder forward pass is written from scratch in this project (`CLAUDE.md`).
This is a standard pre-LayerNorm causal transformer: learned positional embeddings,
multi-head causal self-attention, a 4× GELU MLP, tied input/output embeddings.

**Slot convention.** PT's `logits[:, t]` predicts `idx[:, t]` from `idx[:, <t]`. A GPT
trained the usual way emits `logits[:, t]` predicting `idx[:, t+1]`. To make the two score
the *identical* token set, this module runs the transformer on `idx[:, :-1]` and returns its
logits shifted right by one slot, with slot 0 filled with zeros. Slot 0 is never scored
because the shared loop uses `ignore_first >= 1`, and that is asserted rather than assumed —
if it were ever scored, the baseline would be graded on a slot it does not predict.

This is the honest alignment: both models are asked for `p(w_t | w_{<t})` for `t = 1..n-1`,
over the same blocks, through the same loop, against the same unigram baseline.
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 64
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 160
    dropout: float = 0.0
    tie_weights: bool = True
    shared_block: bool = False  # Looped Transformer (Experiment 2): one block applied n_layer times


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head, self.n_embd = cfg.n_head, cfg.n_embd
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(
                1, 1, cfg.block_size, cfg.block_size
            ),
            persistent=False,
        )

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = self.attn_dropout(F.softmax(att, dim=-1))
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class GPT(nn.Module):
    """nanoGPT with PT's slot convention. ``shared_block=True`` gives the Looped baseline."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        if cfg.shared_block:
            self.block = Block(cfg)
            self.blocks = None
        else:
            self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
            self.block = None
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.lm_head.weight = self.wte.weight

        self.apply(self._init)
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _backbone(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        h = self.drop(self.wte(x) + self.wpe(pos))
        if self.blocks is not None:
            for blk in self.blocks:
                h = blk(h)
        else:
            for _ in range(self.cfg.n_layer):
                h = self.block(h)
        return self.lm_head(self.ln_f(h))

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """``logits[:, t]`` predicts ``idx[:, t]`` from ``idx[:, <t]`` — PT's convention.

        Slot 0 is returned as zeros and must never be scored; see the module docstring.
        """
        B, n = idx.shape
        shifted = self._backbone(idx[:, :-1])  # (B, n-1, V): slot i predicts idx[:, i+1]
        pad = shifted.new_zeros(B, 1, shifted.shape[-1])
        return torch.cat([pad, shifted], dim=1)

    def loss(self, idx: torch.Tensor, ignore_first: int = 1) -> torch.Tensor:
        assert ignore_first >= 1, (
            "the GPT baseline does not predict slot 0 (it has no prefix to condition on); "
            "scoring it would grade the baseline on a slot it never emits"
        )
        logits = self(idx)[:, ignore_first:]
        target = idx[:, ignore_first:]
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))

    def arc_regulariser(self) -> torch.Tensor:
        """No ternary scores in a transformer; the shared loop still asks every model."""
        return self.wte.weight.new_zeros(())

    def num_parameters(self) -> dict:
        emb = self.wte.weight.numel() + self.wpe.weight.numel()
        total = sum(p.numel() for p in self.parameters())
        if self.cfg.tie_weights:
            non_emb = total - emb
        else:
            non_emb = total - emb - self.lm_head.weight.numel()
        return {"embedding": emb, "non_embedding": non_emb, "total": total}
