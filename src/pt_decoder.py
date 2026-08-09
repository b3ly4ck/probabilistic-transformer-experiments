"""The causal Probabilistic Transformer decoder.

Specification: ``developer files/causalprobabilistictransformer_1.pdf`` Part II §12
(the directed chain of conditional CRFs) and Part III §15-§18 (the output mechanism),
with the verdict of Part IV §23.3 (exact readout is the mainline, mean-field is the
ablation). ``causal_pt_output_note.pdf`` is the same construction self-contained, and
its §5 worked example is reproduced verbatim by ``tests/test_07_worked_example.py``.

The parameter list is exactly the factor list — ``S``, ``{T^(c)}`` (or ``U``, ``V``),
``r``, ``b``, and optionally the global-head matrix ``B'``. There is no matrix in the
computation that is not a factor and no operation that is not a message.

Model
-----
Per slot ``t``, conditioned on the frozen prefix labels::

    p(W_t, Z_t, H_t | z_<t) ∝ exp( b_{W_t} + S_{W_t,Z_t}
                                   + Σ_c Σ_{j∈D_t} 1[H_t^(c)=j] · T^(c)_{Z_t, z_j} )

with head domain ``D_t = {ROOT, 1, ..., t-1}``. The prefix enters only through the
contracted arc scores ``B^(c)_{j,a} = Σ_b q̄_j(b) T^(c)_{a,b}``, seeded at ROOT with
``B^(c)_{ROOT,a} = r^(c)_a`` (Part II §12.2; Wu & Tu Appendix B.3.1).

Two streams, one energy
-----------------------
*Content stream* (observed mode, ``W_t`` clamped): MFVI on the chain, producing the
filtering marginals ``q̄_t ≈ p(Z_t | w_{1:t})``. This is what the KV cache is.

*Readout* (predictive mode, ``W_t`` free): the slot graph is a star centred on ``Z_t``,
hence a tree, so sum-product is exact (§17.2)::

    p̂(W_t = w | w_<t) ∝ exp(b_w) Σ_a exp(S_{w,a}) μ_t(a),
    log μ_t(a) = Σ_c LSE_{j ∈ D_t} B^(c)_{j,a}

which along the sequence is a causal prefix log-sum-exp (§23.3). The mean-field
readout of §17.1 is kept as ``readout="mfvi"``; it is Experiment 3's comparison object.

On "frozen"
-----------
The paper is explicit (Part II §12.3 Check 2, and §18 Check 5 "Gradients"): *frozen*
means constant with respect to slot ``t``'s inference problem, **not** detached in
autodiff. Training gradients flow backwards through ``B^(c)_{j,a}`` into ``q̄_j`` and
through the whole prefix computation, exactly as they do through cached activations in
a causal transformer. Forward causality defines the decoder; backward gradient flow
trains it. ``PTConfig.detach_prefix`` implements the other reading (stop-gradient at
the readout's keys) and is **not** the mainline — see EXPERIMENT_STATUS of exp0.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .config import PTConfig

NEG_INF = float("-inf")


class CausalPTDecoder(nn.Module):
    def __init__(self, cfg: PTConfig):
        super().__init__()
        self.cfg = cfg
        d, h, V, K = cfg.d, cfg.h, cfg.vocab_size, cfg.n_dist

        # --- the factor list ---
        # word-label factor S: read as unary when W_t is observed, as emission when it
        # is free. One tensor, both roles: tying is forced (§16(b)), not chosen.
        self.S = nn.Parameter(torch.empty(V, d))
        # word unary b: the bias of the LM head, as a factor (§16(c)).
        self.b = nn.Parameter(torch.zeros(V)) if cfg.word_unary else None
        # root/sink column r^(c): the ROOT entry of the contracted arc score.
        self.r_root = nn.Parameter(torch.empty(h, d))
        # arc score T^(c) per distance bucket, full or Kruskal-decomposed.
        if cfg.rank is None:
            self.T = nn.Parameter(torch.empty(K, h, d, d))
            self.U = None
            self.V = None
        else:
            self.T = None
            self.U = nn.Parameter(torch.empty(K, h, d, cfg.rank))
            self.V = nn.Parameter(torch.empty(K, h, d, cfg.rank))
        # B.3.3 single-split global head (optional; §22.2 of Part IV).
        self.B_glob = nn.Parameter(torch.empty(cfg.n_global, d)) if cfg.n_global > 0 else None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = self.cfg.init_std
        root_std = self.cfg.root_init_std if self.cfg.root_init_std is not None else std
        with torch.no_grad():
            self.S.normal_(0.0, std)
            self.r_root.normal_(0.0, root_std)
            if self.b is not None:
                self.b.zero_()
            for p in (self.T, self.U, self.V, self.B_glob):
                if p is not None:
                    p.normal_(0.0, std)

    # ------------------------------------------------------------------ factors --

    def arc_scores(self) -> torch.Tensor:
        """``T[k, c, a, b]`` — arc score of dependent label ``a`` under head label ``b``.

        Shape ``(n_dist, h, d, d)``. ``k`` indexes the clipped distance bucket.
        """
        if self.T is not None:
            return self.T
        return torch.einsum("khar,khbr->khab", self.U, self.V)

    def bucket_of(self, delta: torch.Tensor) -> torch.Tensor:
        """Clipped-distance bucket of a positive distance ``delta = i - j >= 1``.

        Wu & Tu Eq. 10 restricted to its causal half: distances ``1..gamma`` get their
        own bucket, everything beyond shares the last one.
        """
        return delta.clamp(max=self.cfg.n_dist) - 1

    def contract(self, q: torch.Tensor, T: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Contract the prefix beliefs into arc scores.

        ``B[k, n, c, j, a] = Σ_b q[n, j, b] · T[k, c, a, b]`` — the manoeuvre of
        Wu & Tu Appendix B.3.1, exact at the log-potential level because the
        log-potential is linear in the one-hot encoding of ``z_j``.

        ``q`` is ``(B, n, d)``; the result is ``(n_dist, B, h, n, d)``. ``T`` may be passed
        in when it has already been materialised for this forward pass — under the Kruskal
        form rebuilding it costs ``K h d² r`` and the content stream calls this once per
        iteration.
        """
        return torch.einsum("bje,kcae->kbcja", q, self.arc_scores() if T is None else T)

    def arc_regulariser(self) -> torch.Tensor:
        """Mean square of the arc scores — the L2 term of Wu & Tu §4.2.

        "For MLM tasks, we add a small L2 regularization term to the ternary scores in our
        model, which we experimentally find beneficial" (Table 2 gives 5e-4 on PTB). It is
        the only mechanism restraining the size of the head message, and the message is
        what saturates the label posterior. The source does not say whether the term is a
        sum or a mean of squares; a mean is used here so the coefficient does not depend on
        ``d``, ``h`` or the number of distance buckets.

        **Both carriers are covered, not just ``T``.** The contracted arc score has a ROOT
        column, ``B^(c)_{ROOT,a} = r^(c)_a``, so ``r`` is part of the same score table and
        reaches the message *undiluted* — where ``T`` arrives contracted against ``q̄_j``,
        ``r`` does not. Penalising ``T`` alone was measured to do nothing but move the
        message onto the other carrier: at ``l2_arc = 5.0`` the run of 2026-08-09 drove
        ``max|T|`` from 4.2 to 0.46 while ROOT attention mass rose from 0.20 to 4.78 times
        uniform, and perplexity did not move. The two blocks are averaged separately and
        summed so that one coefficient bounds both regardless of their very different sizes
        (``T`` has ``K·h·d²`` entries against ``r``'s ``h·d``).

        ``T`` is not materialised under the Kruskal form:
        ``‖U Vᵀ‖_F² = Σ_lm (UᵀU)_lm (VᵀV)_lm``.
        """
        if self.T is not None:
            arc = (self.T**2).mean()
        else:
            UtU = torch.einsum("khar,khas->khrs", self.U, self.U)
            VtV = torch.einsum("khbr,khbs->khrs", self.V, self.V)
            n_entries = self.U.shape[0] * self.U.shape[1] * self.cfg.d * self.cfg.d
            arc = (UtU * VtV).sum() / n_entries
        return arc + (self.r_root**2).mean()

    # ------------------------------------------------------- messages, vectorised --

    def _causal_masks(self, n: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
        """``allowed`` over ``{ROOT} ∪ positions`` and the far-bucket pair mask."""
        ar = torch.arange(n, device=device)
        delta = ar.view(-1, 1) - ar.view(1, -1)  # i - j
        causal = delta > 0
        allowed = torch.cat([torch.ones(n, 1, dtype=torch.bool, device=device), causal], dim=1)
        far = causal & (delta >= self.cfg.n_dist)
        return allowed, far

    def _arc_message(
        self, query: torch.Tensor, Bk: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One H-update followed by the H → Z message, for every slot in parallel.

        ``query`` is ``(B, n, d)`` — the current belief of the *querying* variable:
        ``q^(l-1)`` in the content stream, ``Q_Z^pred`` in the mean-field readout.
        ``Bk`` is the output of :meth:`contract` on the frozen prefix beliefs.

        Returns ``(G, alpha)`` with ``G`` of shape ``(B, n, d)``::

            F_c(i, j) = Σ_a query_i(a) B^(c)_{j,a},   Q_c = softmax_{D_i}(F_c / λ_H)
            G_i(a)    = Σ_c Σ_{j∈D_i} Q_c(j) B^(c)_{j,a}

        and ``alpha`` of shape ``(B, h, n, 1 + n)``, column 0 being ROOT.
        """
        Bt, n, d = query.shape
        K = self.cfg.n_dist
        far_B = Bk[-1]  # (B, h, n, d)

        # --- attention logits ---
        logit = torch.einsum("bia,bcja->bcij", query, far_B)
        if K > 1:
            logit = logit.clone()
            for k in range(K - 1):
                delta = k + 1
                if delta >= n:
                    break
                ii = torch.arange(delta, n, device=query.device)
                near = (query[:, delta:, :].unsqueeze(1) * Bk[k][:, :, : n - delta, :]).sum(-1)
                logit[:, :, ii, ii - delta] = near
        root_logit = torch.einsum("bia,ca->bci", query, self.r_root)  # (B, h, n)
        full = torch.cat([root_logit.unsqueeze(-1), logit], dim=-1)  # (B, h, n, 1+n)

        allowed, far_mask = self._causal_masks(n, query.device)
        full = full.masked_fill(~allowed, NEG_INF)
        alpha = torch.softmax(full / self.cfg.lam_H, dim=-1)

        # --- message back to Z ---
        a_root, a_pos = alpha[..., 0], alpha[..., 1:]
        G = torch.einsum("bcij,bcja->bia", a_pos * far_mask, far_B)
        for k in range(K - 1):
            delta = k + 1
            if delta >= n:
                break
            ii = torch.arange(delta, n, device=query.device)
            a_k = a_pos[:, :, ii, ii - delta]  # (B, h, n-delta)
            contrib = (a_k.unsqueeze(-1) * Bk[k][:, :, : n - delta, :]).sum(1)
            G = G.index_add(1, ii, contrib)
        G = G + torch.einsum("bci,ca->bia", a_root, self.r_root)
        return G, alpha

    def _global_message(self, query: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """B.3.3 single-split global head: one extra leaf ``G_t`` off ``Z_t``.

        ``Q_g(k) ∝ exp(Σ_a Q_Z(a) B'_{k,a} / λ_G)``, message back ``Σ_k Q_g(k) B'_{k,a}``.
        """
        if self.B_glob is None:
            return query.new_zeros(query.shape), None
        logit = torch.einsum("...a,ka->...k", query, self.B_glob) / self.cfg.lambda_G
        qg = torch.softmax(logit, dim=-1)
        return torch.einsum("...k,ka->...a", qg, self.B_glob), qg

    # -------------------------------------------------------------- content stream --

    def content_stream(self, idx: torch.Tensor, trace: Optional[list] = None) -> torch.Tensor:
        """Filtering marginals ``q̄`` of the observed words. ``idx`` is ``(B, n)``.

        ``trace``, when given, receives one dict of per-iteration message statistics.
        There is no layer norm and no residual in this model class, so the size of the
        message ``G`` relative to the unary ``S_{w,.}`` is the quantity that says whether
        the inference is being driven by context or by the word identity; see
        :mod:`src.diagnostics`.
        """
        if self.cfg.schedule == "serial":
            return self._content_serial(idx, trace=trace)
        return self._content_parallel(idx, trace=trace)

    def _iteration_stats(self, q, Sw, G, alpha) -> dict:
        """Message-scale diagnostics for one content-stream iteration."""
        allowed = alpha > 0
        ent = -(alpha * torch.where(allowed, alpha.log(), torch.zeros_like(alpha))).sum(-1)
        support = allowed.sum(-1).clamp(min=1).to(alpha.dtype)
        g_norm = G.norm(dim=-1)
        s_norm = Sw.norm(dim=-1)
        return {
            "G_norm": float(g_norm.mean()),
            "Sw_norm": float(s_norm.mean()),
            "ratio": float((g_norm / s_norm.clamp(min=1e-12)).mean()),
            "G_absmax": float(G.abs().max()),
            "attn_entropy": float(ent.mean()),
            "attn_entropy_frac": float((ent / support.log().clamp(min=1e-12)).mean()),
            "label_entropy": float(-(q * q.clamp(min=1e-30).log()).sum(-1).mean()),
            # ROOT is column 0 of D_t. r^(c) reaches the attention in raw d-space while
            # the arc scores arrive contracted, so the sink is a *measured* variable:
            # if it appears, PTConfig.root_init_std is the knob that was already there.
            "root_mass": float(alpha[..., 0].mean()),
            "root_mass_over_uniform": float((alpha[..., 0] * support).mean()),
        }

    def _content_parallel(self, idx: torch.Tensor, trace: Optional[list] = None) -> torch.Tensor:
        """Layer-parallel schedule (Part II §12.3 "Scheduling").

        Iteration ``l`` computes all ``Q_i^(l)`` from ``{Q_j^(l-1)}_{j<i}`` under one
        strict lower-triangular mask — the computation graph of a depth-``T``,
        parameter-shared causal transformer.
        """
        Sw = self.S[idx]  # (B, n, d)
        T = self.arc_scores()  # materialised once for the whole pass
        q = torch.softmax(Sw / self.cfg.lambda_Z, dim=-1)  # Wu & Tu Eq. 7
        for _ in range(self.cfg.n_iters):
            G, alpha = self._arc_message(q, self.contract(q, T))
            G = G + self._global_message(q)[0]
            if trace is not None:
                trace.append(self._iteration_stats(q, Sw, G, alpha))
            q = torch.softmax((Sw + G) / self.cfg.lambda_Z, dim=-1)
        return q

    def _content_serial(self, idx: torch.Tensor, trace: Optional[list] = None) -> torch.Tensor:
        """Serial left-to-right filtering (Part II §12.3, and §12.2's "freeze and advance").

        Each slot is a well-posed two-block MFVI problem against a frozen prefix; this
        is the schedule the worked example of ``causal_pt_output_note.pdf`` §5 uses.
        """
        Sw = self.S[idx]
        n = idx.shape[1]
        qs: List[torch.Tensor] = []
        keys: List[torch.Tensor] = []  # per-position contracted scores, (n_dist,B,h,1,d)
        for t in range(n):
            B_full = self._slot_keys_from(keys, t, idx.shape[0])
            q_t = torch.softmax(Sw[:, t] / self.cfg.lambda_Z, dim=-1)
            for _ in range(self.cfg.tau_obs):
                arc, alpha = self._slot_message(q_t, B_full)
                ctx = arc + self._global_message(q_t)[0]
                if trace is not None:
                    trace.append(
                        self._iteration_stats(
                            q_t.unsqueeze(1), Sw[:, t : t + 1], ctx.unsqueeze(1), alpha.unsqueeze(2)
                        )
                    )
                q_t = torch.softmax((Sw[:, t] + ctx) / self.cfg.lambda_Z, dim=-1)
            qs.append(q_t)
            keys.append(self.contract(q_t.unsqueeze(1)))
        return torch.stack(qs, dim=1)

    # ----------------------------------------------------------- messages, one slot --

    def _slot_keys_from(self, keys: List[torch.Tensor], t: int, batch: int) -> torch.Tensor:
        """Assemble ``B^(c)_{j,·}`` for ``j ∈ D_t`` from per-position contractions.

        Returns ``(B, h, 1 + t, d)`` with ROOT at index 0.
        """
        h, d = self.cfg.h, self.cfg.d
        if t == 0:
            return self.r_root.view(1, h, 1, d).expand(batch, h, 1, d)
        return self._slot_keys(torch.cat(keys[:t], dim=3), t)

    def _slot_keys(self, Bk: torch.Tensor, t: int) -> torch.Tensor:
        """Select the distance bucket for every ``j < t`` and prepend ROOT.

        ``Bk`` is ``(n_dist, B, h, m, d)`` with ``m >= t``; returns ``(B, h, 1+t, d)``.
        """
        device = Bk.device
        h, d = self.cfg.h, self.cfg.d
        j = torch.arange(t, device=device)
        bucket = self.bucket_of(t - j)  # (t,)
        sel = Bk[bucket, :, :, j, :]  # (t, B, h, d)
        sel = sel.permute(1, 2, 0, 3)  # (B, h, t, d)
        root = self.r_root.view(1, h, 1, d).expand(sel.shape[0], h, 1, d)
        return torch.cat([root, sel], dim=2)

    def _slot_message(
        self, query: torch.Tensor, B_full: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The same H-update and H → Z message for a single slot.

        ``query`` is ``(B, d)``, ``B_full`` is ``(B, h, 1+t, d)``. Returns ``(ctx, Q_c)``.
        """
        logit = torch.einsum("ba,bcja->bcj", query, B_full)
        alpha = torch.softmax(logit / self.cfg.lam_H, dim=-1)
        ctx = torch.einsum("bcj,bcja->ba", alpha, B_full)
        return ctx, alpha

    # ------------------------------------------------------------------- readouts --

    def exact_log_mu(self, Bk: torch.Tensor) -> torch.Tensor:
        """``log μ_t(a) = Σ_c LSE_{j ∈ D_t} B^(c)_{j,a}`` for every slot, ``(B, n, d)``.

        §23.3: seeded with ``r^(c)_a`` and, in the far bucket, a causal prefix
        log-sum-exp — one ``logcumsumexp`` scan, ``O(n d h)``, fully parallel. The
        near buckets of the RPE table each contain exactly one position, so they are
        plain shifts.
        """
        K = self.cfg.n_dist
        far_B = Bk[-1]
        Bt, h, n, d = far_B.shape
        terms = [self.r_root.view(1, h, 1, d).expand(Bt, h, n, d)]

        far = far_B.new_full((Bt, h, n, d), NEG_INF)
        if n > K:
            far[:, :, K:, :] = torch.logcumsumexp(far_B, dim=2)[:, :, : n - K, :]
        terms.append(far)

        for k in range(K - 1):
            delta = k + 1
            near = far_B.new_full((Bt, h, n, d), NEG_INF)
            if n > delta:
                near[:, :, delta:, :] = Bk[k][:, :, : n - delta, :]
            terms.append(near)

        log_mu = torch.logsumexp(torch.stack(terms, dim=0), dim=0).sum(dim=1)  # (B, n, d)
        if self.B_glob is not None:
            log_mu = log_mu + torch.logsumexp(self.B_glob, dim=0)
        return log_mu

    def _logits_from_log_mu(self, log_mu: torch.Tensor) -> torch.Tensor:
        """``logits(w) = b_w + LSE_a ( S_{w,a} + log μ(a) )`` — §17.2, chunked over ``V``."""
        d = log_mu.shape[-1]
        lead = log_mu.shape[:-1]
        out = []
        for s in range(0, self.cfg.vocab_size, self.cfg.vocab_chunk):
            S_c = self.S[s : s + self.cfg.vocab_chunk]  # (c, d)
            view = (1,) * len(lead) + S_c.shape
            out.append(torch.logsumexp(log_mu.unsqueeze(-2) + S_c.view(view), dim=-1))
        logits = torch.cat(out, dim=-1)
        if self.b is not None:
            logits = logits + self.b
        return logits

    def _word_prior(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """``Q_W^(0) ∝ exp(b)`` and the prior word message ``s̄_a = Σ_w Q_W^(0)(w) S_{w,a}``.

        §17.1: this is the [MASK] token of MLM / the query embedding of XLNet, derived
        rather than invented.
        """
        b = self.b if self.b is not None else self.S.new_zeros(self.cfg.vocab_size)
        qw0 = torch.softmax(b, dim=-1)
        return qw0, qw0 @ self.S

    def mfvi_readout(self, Bk: torch.Tensor) -> torch.Tensor:
        """Mean-field readout of §17.1, every slot in parallel. ``(B, n, V)``."""
        Bt, n = Bk.shape[1], Bk.shape[3]
        _, sbar = self._word_prior()
        qz = torch.softmax(sbar / self.cfg.lambda_Z, dim=-1).expand(Bt, n, self.cfg.d)
        for _ in range(self.cfg.tau):
            G, _ = self._arc_message(qz, Bk)
            G = G + self._global_message(qz)[0]
            qz = torch.softmax((sbar + G) / self.cfg.lambda_Z, dim=-1)
        logits = qz @ self.S.T
        if self.b is not None:
            logits = logits + self.b
        return logits / self.cfg.lambda_W

    # --------------------------------------------------------------- slot readouts --

    def slot_exact_readout(self, B_full: torch.Tensor) -> torch.Tensor:
        """Exact readout for one slot from its assembled keys ``(B, h, 1+t, d)``."""
        log_mu = torch.logsumexp(B_full, dim=2).sum(dim=1)  # (B, d)
        if self.B_glob is not None:
            log_mu = log_mu + torch.logsumexp(self.B_glob, dim=0)
        return self._logits_from_log_mu(log_mu)

    def slot_mfvi_readout(self, B_full: torch.Tensor, trace: Optional[list] = None) -> torch.Tensor:
        """Mean-field readout for one slot. If ``trace`` is given, append ``(Q_Z, Q_c)``
        after every block update, in update order, for the free-energy check."""
        qw0, sbar = self._word_prior()
        qz = torch.softmax(sbar / self.cfg.lambda_Z, dim=-1).expand(B_full.shape[0], self.cfg.d)
        for _ in range(self.cfg.tau):
            ctx, alpha = self._slot_message(qz, B_full)
            g_ctx, qg = self._global_message(qz)
            if trace is not None:
                trace.append((qz, alpha, qg))
            qz = torch.softmax((sbar + ctx + g_ctx) / self.cfg.lambda_Z, dim=-1)
            if trace is not None:
                trace.append((qz, alpha, qg))
        logits = qz @ self.S.T
        if self.b is not None:
            logits = logits + self.b
        return logits / self.cfg.lambda_W

    # ---------------------------------------------------------------------- public --

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """``logits[b, t, w] = log p̂(W_t = w | w_{<t})`` (unnormalised), ``(B, n, V)``.

        Slot ``t`` predicts the token *at* position ``t`` from the strict prefix, so the
        training target is ``idx`` itself, not a shifted copy (§18 Check 5). Slot 0
        predicts from ROOT alone, so the first word has a proper distribution and no
        BOS token is needed.
        """
        qbar = self.content_stream(idx)
        keys_q = qbar.detach() if self.cfg.detach_prefix else qbar
        Bk = self.contract(keys_q)
        if self.cfg.readout == "exact":
            return self._logits_from_log_mu(self.exact_log_mu(Bk))
        return self.mfvi_readout(Bk)

    def loss(self, idx: torch.Tensor, ignore_first: int = 0) -> torch.Tensor:
        """``L = -Σ_t log Q_{W_t}^readout(w_t)`` — the model's own NLL, nothing else.

        ``ignore_first`` drops the first slots from the average. It exists because PT and
        a GPT baseline do **not** score the same tokens by default: given a block
        ``w_0..w_{n-1}``, this model predicts every ``w_t`` from ``w_{<t}`` — including
        ``w_0`` from ROOT alone, ``n`` scored tokens — while a GPT trained the usual way
        consumes ``w_0..w_{n-2}`` and predicts ``w_1..w_{n-1}``, ``n-1`` scored tokens.
        Comparing the two averages would compare different token sets, and PT's extra
        slot is a first-word unigram prediction that GPT never has to make.

        Experiment 1 must therefore use ``ignore_first=1`` on the PT side (or give the
        baseline a BOS token; that is the less honest fix, since §18 Check 5 notes PT has
        a proper first-word distribution precisely so that no BOS hack is needed).
        """
        logits = self(idx)[:, ignore_first:]
        target = idx[:, ignore_first:]
        return torch.nn.functional.cross_entropy(
            logits.reshape(-1, self.cfg.vocab_size), target.reshape(-1)
        )

    @torch.no_grad()
    def next_token_logits(self, idx: torch.Tensor) -> torch.Tensor:
        """Readout of the slot *after* the given prefix, ``(B, V)``.

        This is step 2 of the generation loop of §18 Check 6, and the call the worked
        example of ``causal_pt_output_note.pdf`` §5 makes for slot 4.
        """
        qbar = self.content_stream(idx)
        n = idx.shape[1]
        B_full = self._slot_keys(self.contract(qbar), n)
        if self.cfg.readout == "exact":
            return self.slot_exact_readout(B_full)
        return self.slot_mfvi_readout(B_full)

    # ------------------------------------------------------------------ accounting --

    def num_parameters(self) -> dict:
        """Embedding and non-embedding parameters, reported separately.

        With tied embeddings almost all of PT's budget sits in ``S``; a single total
        hides that, and the research plan requires the split in every table.
        """
        emb = self.S.numel() + (self.b.numel() if self.b is not None else 0)
        total = sum(p.numel() for p in self.parameters())
        return {"embedding": emb, "non_embedding": total - emb, "total": total}
