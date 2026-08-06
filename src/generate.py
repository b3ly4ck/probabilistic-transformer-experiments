"""Autoregressive sampling from either model family.

The causal PT is a decoder, so it generates -- that is the whole point of the
construction.  This module exists as a qualitative check: perplexity says the
model assigns probability well, samples say what it actually learned.

Both families expose ``logits(tokens) -> (B, n, V)`` where entry ``t`` predicts
``tokens[t]`` from ``tokens[:t]``, so one sampler serves both.  Position 0 is
predicted from a learned constant and no context -- PT's root key ``r``, the
baseline's BOS -- which means **generation from an empty prompt is well defined
in both**, with no special case.

Implementation note: this recomputes the full prefix at every step, `O(n^2)`
over a sample.  §23.3 describes the incremental form -- the content stream's KV
cache plus a running `d x h` mu-cache updated by one `logaddexp` per token,
`O(dh)` -- which is what a serious sampler would use.  At the lengths used for
inspection the naive form is not worth optimising; if sampling ever lands on a
critical path, that is the thing to write.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


def _next_token_logits(model: nn.Module, tokens: Tensor, readout: str | None) -> Tensor:
    """Logits for the position after ``tokens``.

    A model that scores position ``t`` from ``tokens[:t]`` predicts the *next*
    token at the slot one past the end, so the prefix is padded by one and the
    final row read off.  The padding value is never looked at: slot ``n`` reads
    ``D_n = {ROOT, 0..n-1}`` and its own word only through the unary, which is
    exactly the word being predicted.
    """
    pad = torch.zeros(tokens.shape[0], 1, dtype=torch.long, device=tokens.device)
    padded = torch.cat([tokens, pad], dim=1)
    logits = model(padded, readout=readout) if readout else model(padded)
    return logits[:, -1]


@torch.no_grad()
def generate(
    model: nn.Module,
    max_new_tokens: int,
    prompt: Tensor | None = None,
    batch_size: int = 1,
    temperature: float = 1.0,
    top_k: int | None = None,
    readout: str | None = None,
    context: int | None = None,
    generator: torch.Generator | None = None,
    device: str | torch.device = "cpu",
) -> Tensor:
    """Sample ``max_new_tokens`` continuations.

    Args:
        prompt: (batch, n) token ids, or None to start from nothing at all.
        top_k: keep only the k most likely tokens before sampling.
        readout: PT only -- "exact" or "mfvi".
        context: crop the conditioning prefix to the last ``context`` tokens.

    Returns:
        (batch, prompt_len + max_new_tokens) token ids.
    """
    was_training = model.training
    model.eval()

    if prompt is None:
        tokens = torch.zeros(batch_size, 0, dtype=torch.long, device=device)
    else:
        tokens = prompt.to(device)

    for _ in range(max_new_tokens):
        window = tokens if context is None else tokens[:, -context:]
        logits = _next_token_logits(model, window, readout)
        logits = logits / max(temperature, 1e-6)
        if top_k:
            k = min(top_k, logits.shape[-1])
            cutoff = torch.topk(logits, k, dim=-1).values[:, -1:]
            logits = logits.masked_fill(logits < cutoff, float("-inf"))
        probs = torch.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1, generator=generator)
        tokens = torch.cat([tokens, nxt], dim=1)

    model.train(was_training)
    return tokens


def decode(tokens: Tensor, vocab) -> list[str]:
    """Token ids back to text, with <eos> rendered as a line break."""
    out = []
    for row in tokens.tolist():
        words = [vocab.itos[i] for i in row]
        out.append(" ".join(words).replace(" <eos> ", "\n").strip())
    return out
