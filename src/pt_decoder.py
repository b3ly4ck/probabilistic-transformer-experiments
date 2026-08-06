"""The causal Probabilistic Transformer decoder.

The parameter list is exactly the factor list -- ``S``, ``T^(c)``, ``r``, ``b``
(§18, Check 1).  There is no matrix in the computation that is not a factor and
no operation that is not a message.

Schedule (§18, Check 4 and §23.3):

  * **Content stream.**  All positions in parallel, ``T`` rounds, under one
    strictly causal mask: query ``t`` attends ROOT and prefix keys ``j < t``; its
    own word enters through the unary, not the attention.  Positions are
    parallel; rounds are sequential, exactly as layers are in a transformer.
  * **Readout.**  The exact tree readout on top of the final ``qbar``'s -- a
    causal ``logcumsumexp`` scan.  Under §23.3 this is the mainline and the
    query stream does not exist.  The mean-field two-stream readout is kept as
    the ablation and as Experiment 3's comparison object.

On "frozen": ``qbar_j`` is a conditioning *constant of the variational problem*
at later slots -- it is not re-optimised there (§25.1, dropped evidence).  It is
**not** detached from autograd.  Causality is enforced by the mask, as in a
transformer, and gradients flow through the content stream normally.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from . import exact, mfvi
from .config import PTConfig


def causal_key_mask(n: int, device=None) -> Tensor:
    """(n, n + 1) boolean mask over the head domain.

    Key 0 is ROOT and is always available.  Key ``j + 1`` is prefix position
    ``j``, available to query ``t`` when ``j < t``.  Both conditions collapse to
    ``k <= t``.
    """
    t = torch.arange(n, device=device).unsqueeze(1)
    k = torch.arange(n + 1, device=device).unsqueeze(0)
    return k <= t


class CausalPTDecoder(nn.Module):
    def __init__(self, cfg: PTConfig, init_scale: float = 0.1, generator: torch.Generator | None = None):
        super().__init__()
        self.cfg = cfg
        g = generator

        def randn(*shape):
            return torch.randn(*shape, generator=g) * init_scale

        self.S = nn.Parameter(randn(cfg.vocab_size, cfg.d))
        """Word-label factor.  Used clamped on input and free on output -- one
        factor, both directions.  Tying is forced, not chosen (§16(b))."""

        self.T = nn.Parameter(randn(cfg.n_channels, cfg.d, cfg.d))
        self.r = nn.Parameter(randn(cfg.n_channels, cfg.d))

        if cfg.use_word_unary:
            self.b = nn.Parameter(torch.zeros(cfg.vocab_size))
        else:
            self.register_buffer("b", torch.zeros(cfg.vocab_size))

        if cfg.use_global_head:
            self.B_global = nn.Parameter(randn(cfg.n_global, cfg.d))
            """Global-head factor B' of Wu & Tu Eq. (46), single-split form
            (Appendix B.3.3): one score matrix (m, d) shared across channels,
            scoring global feature k against label a.  A factor between two
            variables, trained like S, T, r and b -- not a map."""
        else:
            self.B_global = None

    # -- content stream ---------------------------------------------------

    def content_stream(self, tokens: Tensor, n_rounds: int | None = None) -> Tensor:
        """Run the observed (filtering) stream over a whole sequence.

        Args:
            tokens: (batch, n) int64.
            n_rounds: overrides ``cfg.n_rounds``.

        Returns:
            (batch, n, d) filtering marginals ``qbar_t``, each a function of
            ``w_{1:t}`` only.
        """
        cfg = self.cfg
        rounds = cfg.n_rounds if n_rounds is None else n_rounds
        n = tokens.shape[1]
        mask = causal_key_mask(n, tokens.device)

        m_W = self.S[tokens]  # (batch, n, d) -- the clamped word message S[w_t, :]
        q = torch.softmax(m_W / cfg.lambda_Z, dim=-1)  # §17.1 initialisation

        for _ in range(rounds):
            Bkey = mfvi.contract_prefix(q, self.T, self.r)  # (batch, h, n+1, d)
            scores = torch.einsum("bta,bcka->bctk", q, Bkey) / cfg.lambda_H
            scores = scores.masked_fill(~mask, mfvi.NEG_INF)
            Q_c = torch.softmax(scores, dim=-1)
            message = torch.einsum("bctk,bcka->bta", Q_c, Bkey)
            if self.B_global is not None:
                # G_t is position-local: it reads q at t only, never the prefix,
                # so it opens no path to the future.
                message = message + mfvi.gfu(q, self.B_global, cfg)
            q = torch.softmax((m_W + message) / cfg.lambda_Z, dim=-1)
        return q

    # -- readouts ---------------------------------------------------------

    def exact_logits(self, qbar: Tensor) -> Tensor:
        """Mainline readout (§17.2, §23.3): (batch, n, V) logits for ``w_t``."""
        Bkey = mfvi.contract_prefix(qbar, self.T, self.r)
        log_mu = exact.log_mu_sequence(Bkey, self.B_global)  # (batch, n, d)
        return exact.exact_logits(log_mu, self.S, self.b)

    def mfvi_logits(self, qbar: Tensor, n_rounds: int | None = None) -> Tensor:
        """Ablation readout: the query stream of §17.1.

        ``tau >= 2`` so the attention query is context-dependent; at ``tau = 1``
        it is the fixed probe ``sigma(s_bar / lambda_Z)``.  The readout is taken
        *before* feeding ``Q_W`` back, per §17.1; extra W-rounds are an ablation.
        """
        cfg = self.cfg
        rounds = cfg.n_rounds if n_rounds is None else n_rounds
        n = qbar.shape[1]
        mask = causal_key_mask(n, qbar.device)

        Bkey = mfvi.contract_prefix(qbar, self.T, self.r)
        Q_W0 = torch.softmax(self.b, dim=-1)  # Q_W^(0) proportional to exp(b)
        s_bar = mfvi.word_message(Q_W0, self.S)  # the derived [MASK] embedding
        Q_Z = torch.softmax(s_bar / cfg.lambda_Z, dim=-1)
        Q_Z = Q_Z.expand(qbar.shape[0], n, cfg.d)

        for _ in range(rounds):
            scores = torch.einsum("bta,bcka->bctk", Q_Z, Bkey) / cfg.lambda_H
            scores = scores.masked_fill(~mask, mfvi.NEG_INF)
            Q_c = torch.softmax(scores, dim=-1)
            message = torch.einsum("bctk,bcka->bta", Q_c, Bkey)
            if self.B_global is not None:
                message = message + mfvi.gfu(Q_Z, self.B_global, cfg)
            Q_Z = torch.softmax((s_bar + message) / cfg.lambda_Z, dim=-1)

        return mfvi.mfvi_readout_logits(Q_Z, self.S, self.b, cfg)

    # -- forward / loss ---------------------------------------------------

    def forward(self, tokens: Tensor, readout: str = "exact") -> Tensor:
        qbar = self.content_stream(tokens)
        if readout == "exact":
            return self.exact_logits(qbar)
        if readout == "mfvi":
            return self.mfvi_logits(qbar)
        raise ValueError(f"unknown readout {readout!r}; expected 'exact' or 'mfvi'")

    def loss(self, tokens: Tensor, readout: str = "exact") -> Tensor:
        """L = - sum_t log p_hat(w_t | w_<t).  The model's own NLL, nothing else
        (§18, Check 5).  Position ``t`` is scored from ``qbar_{<t}`` alone."""
        logits = self.forward(tokens, readout=readout)
        return torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), tokens.reshape(-1)
        )
