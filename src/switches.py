"""The ten-switch ladder between a transformer and the causal PT decoder.

Why this file exists
--------------------
Penghao Kuang, reviewing the project in 2026-09, relayed that Haoyi Wu -- the first author
of the Probabilistic Transformer paper -- had run exactly this experiment for the *encoder*:
enumerate the differences between PT and a transformer, make each one an independent switch,
and interpolate. All switches on gives the transformer; all off gives PT; the rungs in
between price each difference on its own. This module builds that ladder for the **causal
decoder**, which is the object this repository actually studies.

The motivating number is in ``REPORT_2026-08.md`` §1: on PTB the causal PT decoder reaches
250.69 +- 2.32 validation perplexity against 115.43 for nanoGPT and 126.46 for the Looped
Transformer -- a factor 2.17, at 75x fewer non-embedding parameters. "PT is worse than GPT"
is not a finding; *which* of the ten structural differences buys back how much of that gap
is a finding, and it is the only way to answer the reviewer's question without hand-waving.

THIS IS ANALYSIS CODE AND IT DELIBERATELY BREAKS THE PROJECT'S MODELLING DISCIPLINE
-----------------------------------------------------------------------------------
``CLAUDE.md`` constraint 1: "Every learned matrix must correspond to a factor in the graph.
No parameter may exist that names no factor." Almost every rung of this ladder violates that
on purpose:

* ``W_Q``, ``W_V`` and ``W_O`` in the attention name no factor -- in PT the attention logit
  and the message back to ``Z`` are the *same* contracted arc score ``B^(c)_{j,a}``, and
  there is exactly one matrix (``T^(c)``) behind both;
* the feed-forward block names no factor -- PT's only analogue is the B.3.3 global head,
  which is a leaf variable with a score matrix, not an MLP;
* ``mu_head`` in the mixture readout is a free ``n_embd x d_label`` projection, and a free
  projection into the label space is precisely the output mechanism that was **rejected**
  in ``causal_pt_output_note.pdf``;
* LayerNorm and the residual stream are maps, not messages; Part IV §22.2 names both as
  tripwires.

That is the point. You cannot price a constraint without building the model that violates
it. **Nothing in this file is the model.** ``src/pt_decoder.py`` is the model, and it stays
inside the discipline. If a rung of this ladder wins, the result to report is "the factor
graph costs X perplexity here", not "we should add W_O".

Fidelity of the two endpoints
-----------------------------
* **All-transformer** (``TRANSFORMER_SWITCHES``, the dataclass defaults) is *numerically
  identical* to :class:`src.gpt.GPT` at the same config. The module names are deliberately
  the same as ``src/gpt.py`` so that ``SwitchModel.load_state_dict(gpt.state_dict())``
  succeeds with ``strict=True``; ``tests/test_15_switches.py`` asserts the logits agree.
  Adding ``weight_sharing=True`` on top reproduces the Looped baseline the same way.
* **All-PT** (``PT_SWITCHES``) *approximates* the causal PT decoder. It does not reproduce
  it, and the residue is worth stating plainly:
  - ``mu_head`` survives; PT reads ``mu`` straight off the label posterior. Removing it
    would couple two switches, which would defeat the whole design.
  - PT's distance dependence lives in a full arc-score matrix per bucket, ``T^(c)_k``;
    switch 8 gives the transformer analogue, a scalar bias per (head, bucket) added to the
    logit. Content-times-distance is not reproduced.
  - PT sums ``h`` channel messages that are each ``d`` wide; a transformer concatenates
    ``n_head`` heads that are each ``n_embd / n_head`` wide. Switch 5 removes ``W_O`` but
    cannot turn the concatenation into a sum without changing the parameter budget, which
    would confound the switch it is supposed to isolate.
  - PT's inference is MFVI with a step size and a temperature (``alpha_Z``, ``lambda_H``);
    there are no switches for those, because they are already sweepable in ``PTConfig``.
  ``d_label`` and ``n_embd`` name the same quantity in PT (the number of labels of ``Z_t``
  *is* the width). A coherent all-PT rung therefore sets ``n_embd == d_label``; that is also
  the only setting in which the word-label factor ``S`` can be tied to the input embedding,
  as constraint 2 of ``CLAUDE.md`` requires. When they differ the tie is dropped and
  :meth:`SwitchModel.num_parameters` accounts for ``S`` as a second embedding matrix.

Interface
---------
:class:`SwitchModel` exposes exactly the four calls the shared training loop makes --
``forward``, ``loss``, ``arc_regulariser``, ``num_parameters`` -- with ``src/gpt.py``'s slot
convention, so ``src/train.py`` and ``experiments/sweep.py`` need no branch on model type.
It has no ``content_stream``, so ``src.train._diagnostics`` catches the ``AttributeError``
and logs no PT message statistics for a switch cell; that is intentional, the statistics are
defined against the factor graph and a switch rung generally has none.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

# Clip threshold of the distance-sensitive ternary potential, Wu & Tu Eqs. 9/10. 3 is the
# PTB row of their Table 2 and the default of `PTConfig.gamma`. Only the i - j > 0 half of
# the table exists in a causal model, so there are gamma + 1 buckets: distances 1..gamma get
# their own and everything beyond shares the last. It is a module constant rather than a
# config field because `SwitchConfig`'s field list is fixed by `experiments/sweep.py`.
RPE_GAMMA = 3

# Chunk width of the mixture readout's log-sum-exp over the vocabulary; mirrors
# `PTConfig.vocab_chunk`. The intermediate of that reduction is (B, n, chunk, d_label),
# which at a full 10k vocabulary is what makes the readout the memory peak of the forward
# pass. Read from module scope at call time so a test can shrink it.
VOCAB_CHUNK = 8192

#: The ten differences, in a fixed order. Used to validate `ladder` orderings.
SWITCH_NAMES = (
    "readout",
    "norm",
    "residual",
    "attn_value",
    "attn_out_proj",
    "attn_query_proj",
    "ffn",
    "position",
    "weight_sharing",
    "state",
)

#: All switches at the transformer setting. Equal to the `SwitchConfig` defaults.
TRANSFORMER_SWITCHES: Dict[str, Any] = {
    "readout": "linear",
    "norm": "layernorm",
    "residual": True,
    "attn_value": "separate",
    "attn_out_proj": True,
    "attn_query_proj": True,
    "ffn": True,
    "position": "absolute",
    "weight_sharing": False,
    "state": "unconstrained",
}

#: All switches at the PT setting.
PT_SWITCHES: Dict[str, Any] = {
    "readout": "mixture",
    "norm": "none",
    "residual": False,
    "attn_value": "tied",
    "attn_out_proj": False,
    "attn_query_proj": False,
    "ffn": False,
    "position": "rpe_bucket",
    "weight_sharing": True,
    "state": "simplex",
}


@dataclass
class SwitchConfig:
    """Shape plus the ten switches. Every switch defaults to the transformer setting.

    The field list is fixed by ``experiments/sweep.py``, which constructs this as
    ``SwitchConfig(vocab_size=, block_size=, n_layer=, n_head=, n_embd=, d_label=,
    dropout=, **cell.switches)``. Do not add required fields.

    ``d_label`` is PT's ``d``: the size of the label set of ``Z_t`` (Wu & Tu §2.2), through
    which *all* predictive information has to pass under ``readout="mixture"``. It is only
    read by the mixture readout; ``n_embd`` remains the width of the stream.
    """

    vocab_size: int
    block_size: int = 64
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 160
    d_label: int = 32
    dropout: float = 0.0

    # --- the ten switches ---
    readout: str = "linear"  # 1. "linear" (W_lm h, tied) | "mixture" (rank-d_label simplex)
    norm: str = "layernorm"  # 2. "layernorm" | "none"
    residual: bool = True  # 3. True | False (block output replaces the stream)
    attn_value: str = "separate"  # 4. "separate" (W_V) | "tied" (value == key)
    attn_out_proj: bool = True  # 5. True (W_O) | False
    attn_query_proj: bool = True  # 6. True (W_Q) | False (query with the state)
    ffn: bool = True  # 7. True (4x GELU MLP) | False
    position: str = "absolute"  # 8. "absolute" (wpe) | "rpe_bucket" (clipped distance)
    weight_sharing: bool = False  # 9. False (independent blocks) | True (one block, n_layer times)
    state: str = "unconstrained"  # 10. "unconstrained" | "simplex" (softmax after each block)

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd {self.n_embd} not divisible by n_head {self.n_head}")
        if self.readout not in ("linear", "mixture"):
            raise ValueError(f"unknown readout {self.readout!r}")
        if self.norm not in ("layernorm", "none"):
            raise ValueError(f"unknown norm {self.norm!r}")
        if self.attn_value not in ("separate", "tied"):
            raise ValueError(f"unknown attn_value {self.attn_value!r}")
        if self.position not in ("absolute", "rpe_bucket"):
            raise ValueError(f"unknown position {self.position!r}")
        if self.state not in ("unconstrained", "simplex"):
            raise ValueError(f"unknown state {self.state!r}")
        if self.d_label < 1:
            raise ValueError("d_label must be >= 1")

    def switches(self) -> Dict[str, Any]:
        """The ten switch values, without the shape fields."""
        return {name: getattr(self, name) for name in SWITCH_NAMES}

    def distance_from_transformer(self) -> int:
        """How many switches are at their PT setting. 0 is nanoGPT, 10 is the PT rung."""
        return sum(1 for n in SWITCH_NAMES if getattr(self, n) != TRANSFORMER_SWITCHES[n])


def ladder(order: Sequence[str], reverse: bool = False) -> Iterator[Dict[str, Any]]:
    """Cumulative switch dicts from one endpoint to the other, one flip per step.

    ``ladder(order)`` yields ``len(order) + 1`` dicts: rung ``i`` has the first ``i`` names
    of ``order`` at their PT setting and everything else at the transformer setting. Rung 0
    is :data:`TRANSFORMER_SWITCHES`; if ``order`` names all ten switches the last rung is
    :data:`PT_SWITCHES`.

    ``reverse=True`` walks the other way -- rung 0 is :data:`PT_SWITCHES` and each step puts
    one more switch back to the transformer setting. Both directions are needed because a
    ladder measures two different things: forward it asks "what does adding this PT
    constraint cost", backward it asks "what does relaxing it buy", and with ten interacting
    switches those are not the same question.

    Each dict is fresh and directly usable as ``Cell.switches`` in ``experiments/sweep.py``.
    """
    unknown = [n for n in order if n not in SWITCH_NAMES]
    if unknown:
        raise ValueError(f"unknown switch name(s) {unknown}; known: {list(SWITCH_NAMES)}")
    if len(set(order)) != len(order):
        raise ValueError(f"repeated switch name in order {list(order)}")

    end = TRANSFORMER_SWITCHES if reverse else PT_SWITCHES
    current = dict(PT_SWITCHES if reverse else TRANSFORMER_SWITCHES)
    yield dict(current)
    for name in order:
        current[name] = end[name]
        yield dict(current)


class SwitchSelfAttention(nn.Module):
    """Causal self-attention with switches 4, 5, 6 and 8.

    At the transformer settings this is byte-for-byte ``src.gpt.CausalSelfAttention``,
    including the name and layout of ``c_attn`` (one fused ``3 * n_embd`` projection, split
    ``q, k, v``) and ``c_proj``, so a GPT state dict loads straight in.

    The projections that survive are packed into the *same* ``c_attn`` module in the order
    ``q, k, v``, with the absent ones simply not allocated. Keeping one fused Linear rather
    than three named ones is what makes the default case transfer without slicing.
    """

    def __init__(self, cfg: SwitchConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.cfg = cfg
        self.n_head, self.n_embd = cfg.n_head, cfg.n_embd
        # Switch 6 drops W_Q: PT's attention logit is <q_i, B^(c)_{j,.}> with q_i the label
        # belief itself, so there is no query map to drop *to* -- the state is the query.
        # Switch 4 drops W_V: PT's message back to Z contracts the same B^(c)_{j,.} that
        # produced the logit, so value and key are literally one tensor.
        n_proj = 1 + int(cfg.attn_query_proj) + int(cfg.attn_value == "separate")
        self.c_attn = nn.Linear(cfg.n_embd, n_proj * cfg.n_embd)
        # Switch 5 drops W_O: PT sums its per-channel messages and hands the sum straight
        # back to Z. See the module docstring for why concatenation, not summation, is the
        # width-preserving analogue here.
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd) if cfg.attn_out_proj else None
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        # Switch 8: a clipped relative-distance bucket table, one scalar per (bucket, head),
        # added to the logit. This is the transformer's RPE bias, i.e. the additive shadow of
        # PT's per-bucket arc-score matrices T^(c)_k. It lives inside the attention rather
        # than at the input because that is where PT's distance dependence lives; whether it
        # is shared across depth is then decided by switch 9 alone, not by this one.
        if cfg.position == "rpe_bucket":
            self.rpe = nn.Embedding(RPE_GAMMA + 1, cfg.n_head)
            ar = torch.arange(cfg.block_size)
            delta = ar.view(-1, 1) - ar.view(1, -1)  # i - j
            bucket = delta.clamp(min=1, max=RPE_GAMMA + 1) - 1  # 0..gamma; j >= i is masked
            self.register_buffer("rpe_bucket", bucket, persistent=False)
        else:
            self.rpe = None

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(
                1, 1, cfg.block_size, cfg.block_size
            ),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        parts: List[torch.Tensor] = list(self.c_attn(x).split(C, dim=2))
        i = 0
        if self.cfg.attn_query_proj:
            q = parts[i]
            i += 1
        else:
            q = x
        k = parts[i]
        i += 1
        v = parts[i] if self.cfg.attn_value == "separate" else k

        hd = C // self.n_head
        q = q.view(B, T, self.n_head, hd).transpose(1, 2)
        k = k.view(B, T, self.n_head, hd).transpose(1, 2)
        v = v.view(B, T, self.n_head, hd).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
        if self.rpe is not None:
            # (T, T, n_head) -> (1, n_head, T, T)
            bias = self.rpe(self.rpe_bucket[:T, :T]).permute(2, 0, 1).unsqueeze(0)
            att = att + bias
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = self.attn_dropout(F.softmax(att, dim=-1))
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        if self.c_proj is not None:
            y = self.c_proj(y)
        return self.resid_dropout(y)


class SwitchBlock(nn.Module):
    """One block, carrying switches 2, 3 and 7 on top of the attention's.

    Module names ``ln_1``, ``attn``, ``ln_2``, ``mlp`` are ``src.gpt.Block``'s. When a
    switch removes a submodule the parameter simply does not exist, so the state dict of a
    non-default rung is a strict subset -- which is exactly what makes the default rung's
    state dict transfer exact.
    """

    def __init__(self, cfg: SwitchConfig):
        super().__init__()
        self.cfg = cfg
        self.ln_1 = nn.LayerNorm(cfg.n_embd) if cfg.norm == "layernorm" else nn.Identity()
        self.attn = SwitchSelfAttention(cfg)
        if cfg.ffn:
            self.ln_2 = nn.LayerNorm(cfg.n_embd) if cfg.norm == "layernorm" else nn.Identity()
            self.mlp = nn.Sequential(
                nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
                nn.GELU(),
                nn.Linear(4 * cfg.n_embd, cfg.n_embd),
                nn.Dropout(cfg.dropout),
            )
        else:
            self.ln_2 = None
            self.mlp = None

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """``emb`` is the input unary; it is only read when ``residual=False``.

        With ``residual=True`` this is the standard pre-LN transformer block: two additive
        skips, applied sequentially.

        With ``residual=False`` the stream is *replaced* rather than accumulated, and the
        only skip left is from the input embedding::

            h_{l+1} = attn(ln_1(h_l)) + mlp(ln_2(h_l)) + emb

        which is PT's update ``q^(l) = softmax((S_w + G^(l)) / lambda_Z)`` with the softmax
        supplied by switch 10: the word unary ``S_w`` is re-added at every iteration and the
        previous state reaches the next one only through the messages. Two details are
        deliberate. The message terms are computed *in parallel* from the same ``h_l``, not
        in sequence, because PT's arc message and its B.3.3 global-head message are both
        functions of the same ``q`` and are summed into one ``G``. And ``emb`` is the raw,
        un-normalised embedding, because in PT the unary is added in log space *before* the
        softmax, not to the distribution afterwards.
        """
        if self.cfg.residual:
            x = x + self.attn(self.ln_1(x))
            if self.mlp is not None:
                x = x + self.mlp(self.ln_2(x))
            return x
        out = self.attn(self.ln_1(x)) + emb
        if self.mlp is not None:
            out = out + self.mlp(self.ln_2(x))
        return out


class SwitchModel(nn.Module):
    """A rung of the transformer-to-PT ladder.

    Defaults reproduce :class:`src.gpt.GPT` exactly (see the module docstring); the module
    names are ``wte``, ``wpe``, ``drop``, ``blocks`` / ``block``, ``ln_f``, ``lm_head``,
    identical to ``src/gpt.py``, so ``load_state_dict`` between the two is strict-clean.

    Slot convention, inherited verbatim from ``src/gpt.py``: ``logits[:, t]`` predicts
    ``idx[:, t]`` from ``idx[:, <t]``, obtained by running the backbone on ``idx[:, :-1]``
    and shifting right one slot with zeros in slot 0. Slot 0 is never scored and
    :meth:`loss` asserts that rather than assuming it.
    """

    def __init__(self, cfg: SwitchConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        # Switch 8: PT has no absolute position table at all -- position enters only through
        # the distance buckets of the ternary potential -- so `wpe` is not merely unused
        # under "rpe_bucket", it is absent, and the parameter count says so.
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd) if cfg.position == "absolute" else None
        self.drop = nn.Dropout(cfg.dropout)

        # Switch 9: one block applied n_layer times is PT's parameter sharing across MFVI
        # iterations, and is also exactly the Looped Transformer baseline of Experiment 2.
        if cfg.weight_sharing:
            self.block = SwitchBlock(cfg)
            self.blocks = None
        else:
            self.blocks = nn.ModuleList([SwitchBlock(cfg) for _ in range(cfg.n_layer)])
            self.block = None

        self.ln_f = nn.LayerNorm(cfg.n_embd) if cfg.norm == "layernorm" else nn.Identity()

        # Switch 1, the rung that matters most. "linear" is GPT's tied W_lm.
        # "mixture" is PT's readout: the state is projected to a distribution mu over
        # d_label labels and the word distribution is reached only through the word-label
        # factor S, logits_w = LSE_a (log mu_a + S_{w,a}) + b_w. Every word score is a
        # convex-combination-in-log-space of d_label columns, so the logit matrix has rank
        # at most d_label + 1 -- the honest softmax bottleneck named in CLAUDE.md, and the
        # thing this ladder is built to price.
        if cfg.readout == "mixture":
            # `mu_head` is the one map with no factor behind it that this readout cannot do
            # without: PT reads mu off the label posterior, which exists only because the
            # whole stream is a label posterior. See the module docstring.
            self.mu_head = nn.Linear(cfg.n_embd, cfg.d_label, bias=False)
            # weight plays the role of S (V, d_label); bias plays the role of the word unary
            # b (V), which PT initialises at zero and can freeze to the log-unigram.
            self.lm_head = nn.Linear(cfg.d_label, cfg.vocab_size, bias=True)
        else:
            self.mu_head = None
            self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # Tying is forced by PT's construction, not chosen (CLAUDE.md constraint 2): one
        # word-label factor S mediates both directions. It is only expressible when the
        # stream and the label set have the same width.
        if self.lm_head.weight.shape == self.wte.weight.shape:
            self.lm_head.weight = self.wte.weight

        self.apply(self._init)
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init(module: nn.Module) -> None:
        """nanoGPT's initialisation, unchanged, so the default rung matches ``src/gpt.py``."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------- internals --

    def tied(self) -> bool:
        """Whether the output word-label matrix is the *same object* as the input embedding."""
        return self.lm_head.weight is self.wte.weight

    def _constrain(self, h: torch.Tensor) -> torch.Tensor:
        """Switch 10. ``simplex`` makes the state a distribution, as PT's ``q`` always is.

        In PT the state is not a free vector: it is ``q_i = softmax((S_w + G) / lambda_Z)``,
        a point of the ``d``-simplex. That single constraint is what removes the need for
        LayerNorm (``src/diagnostics.py``: "what bounds the activations instead is the
        simplex geometry") and it is what makes ``||q||_2`` a *bounded* query norm, in
        ``[1/sqrt(d), 1]``, rather than a free activation scale.

        Applied after every block **and to the embedding**, because "the state is a
        distribution" has to hold at iteration 0 too -- that is PT's
        ``q^(0) = softmax(S_w / lambda_Z)``. Composed with ``residual=False``, the pair
        reproduces PT's update exactly: ``softmax(emb + messages)``.
        """
        if self.cfg.state == "simplex":
            return torch.softmax(h, dim=-1)
        return h

    def _mixture_logits(self, h: torch.Tensor) -> torch.Tensor:
        """``logits_w = LSE_a (log mu_a + S_{w,a}) + b_w`` with ``mu = softmax(mu_head h)``.

        Done in log space throughout -- ``log sum_a mu_a exp(S_{w,a})`` is evaluated as a
        ``logsumexp`` over the label axis, never as a sum of exponentials -- because
        ``S`` is unbounded and a direct product underflows to zero the moment training
        sharpens it. Chunked over the vocabulary for the same reason
        ``CausalPTDecoder._logits_from_log_mu`` is: the intermediate is
        ``(B, n, chunk, d_label)`` and at a full vocabulary it is the memory peak.
        """
        log_mu = torch.log_softmax(self.mu_head(h), dim=-1)  # (B, T, d_label)
        S = self.lm_head.weight  # (V, d_label)
        out = []
        for s in range(0, self.cfg.vocab_size, VOCAB_CHUNK):
            S_c = S[s : s + VOCAB_CHUNK]  # (c, d_label)
            out.append(torch.logsumexp(log_mu.unsqueeze(-2) + S_c, dim=-1))
        return torch.cat(out, dim=-1) + self.lm_head.bias

    def _backbone(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        tok = self.wte(x)
        if self.wpe is not None:
            tok = tok + self.wpe(torch.arange(T, device=x.device))
        emb = self.drop(tok)
        h = self._constrain(emb)
        blocks = self.blocks if self.blocks is not None else [self.block] * self.cfg.n_layer
        for blk in blocks:
            h = self._constrain(blk(h, emb))
        h = self.ln_f(h)
        if self.cfg.readout == "mixture":
            return self._mixture_logits(h)
        return self.lm_head(h)

    # ---------------------------------------------------------------------- public --

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """``logits[:, t]`` predicts ``idx[:, t]`` from ``idx[:, <t]`` -- PT's convention.

        Slot 0 is returned as zeros and must never be scored; see ``src/gpt.py``.
        """
        B, n = idx.shape
        shifted = self._backbone(idx[:, :-1])  # (B, n-1, V): slot i predicts idx[:, i+1]
        pad = shifted.new_zeros(B, 1, shifted.shape[-1])
        return torch.cat([pad, shifted], dim=1)

    def loss(self, idx: torch.Tensor, ignore_first: int = 1) -> torch.Tensor:
        assert ignore_first >= 1, (
            "a switch rung does not predict slot 0 (it has no prefix to condition on); "
            "scoring it would grade the model on a slot it never emits"
        )
        logits = self(idx)[:, ignore_first:]
        target = idx[:, ignore_first:]
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))

    def arc_regulariser(self) -> torch.Tensor:
        """Zero: no rung of this ladder has ternary scores. The shared loop asks anyway.

        Even the mixture readout's ``S`` is not an arc score -- Wu & Tu's L2 term of §4.2 is
        on ``T^(c)``, and ``src/pt_decoder.py`` extends it to the root column ``r`` because
        that is the other carrier of the same message. Neither exists here.
        """
        return self.wte.weight.new_zeros(())

    def num_parameters(self) -> dict:
        """Embedding / non-embedding split, on ``src/gpt.py``'s accounting.

        Embedding means "grows with the vocabulary or with the number of positions":
        ``wte``, ``wpe`` when it exists, the word-label matrix ``S`` when it is *not* tied,
        and the word unary ``b``. Everything else -- including the relative-distance bucket
        table, which is ``n_head * (gamma + 1)`` scalars and structurally an attention
        parameter -- lands in non-embedding, which is the column the trade curve of
        ``REPORT_2026-08.md`` Fig. 6 plots against.

        ``parameters()`` de-duplicates tied tensors, so a tied ``lm_head`` is counted once,
        exactly as in the baseline.
        """
        emb = self.wte.weight.numel()
        if self.wpe is not None:
            emb += self.wpe.weight.numel()
        if not self.tied():
            emb += self.lm_head.weight.numel()
        if self.lm_head.bias is not None:
            emb += self.lm_head.bias.numel()
        total = sum(p.numel() for p in self.parameters())
        return {"embedding": emb, "non_embedding": total - emb, "total": total}
