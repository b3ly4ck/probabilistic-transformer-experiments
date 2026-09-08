"""Factored labels: ``K`` label variables per position instead of one.

Specification: ``developer files/causalprobabilistictransformer_1.pdf`` Part IV §22.2,
"Widening the model without leaving the graph", subsection *Factored labels — the
flagship option*, read together with §17.2 (the exact readout), §16 (the generative
chain over words) and §26 rung 4 (``K ∈ {1, 2, 4}`` at fixed total width, which is the
experiment this file exists to make runnable).

The construction
----------------
Take ``K`` label variables ``Z_t^(1..K)`` per position, each over ``d'`` values, with
``K`` binary word–label factors sharing ``W_t``::

    φ(W_t, Z_t^(k)) = exp( S^(k)_{W_t, Z_t^(k)} ),   k = 1..K

and each head channel ``c`` assigned a *child* component ``k(c)`` and a *head*
component ``k'(c)``, so that the contracted arc score reads the prefix belief of its
head component and is consumed by the belief of its child component::

    B^(c)_{j,a} = Σ_b q̄_j^{(k'(c))}(b) T^(c)_{a,b}

The predictive per-step energy is the obvious sum::

    E_t = − Σ_w Q_W(w) b_w
          − Σ_k Σ_{w,a} Q_W(w) Q_{Z^(k)}(a) S^(k)_{w,a}
          − Σ_c Σ_{j∈D_t} Σ_a Q_c(j) Q_{Z^(k(c))}(a) B^(c)_{j,a}

multilinear, with honest per-variable gradients, giving the updates::

    Q_c(j)       ∝ exp( (1/λ_H) Σ_a Q_{Z^(k(c))}(a) B^(c)_{j,a} )
    Q_{Z^(k)}(a) ∝ exp( (1/λ_Z) [ Σ_w Q_W(w) S^(k)_{w,a}
                                  + Σ_{c: k(c)=k} Σ_j Q_c(j) B^(c)_{j,a} ] )
    Q_W(w)       ∝ exp( (1/λ_W) [ b_w + Σ_k Σ_a Q_{Z^(k)}(a) S^(k)_{w,a} ] )

The slot graph is still a tree — ``W_t`` joined to ``K`` stars, one per component — so
sum-product is still exact, and the exact readout becomes a **product of K mixtures**::

    p̂(w) ∝ exp(b_w) · Π_{k=1..K} Σ_{a=1..d'} exp(S^(k)_{w,a}) μ_t^(k)(a)
    log μ_t^(k)(a) = Σ_{c: k(c)=k} LSE_{j ∈ D_t} B^(c)_{j,a}

seeded at ROOT with ``B^(c)_{ROOT,a} = r^(c)_a``, exactly as in the flat model.

Why this is worth running, and the prediction it makes
------------------------------------------------------
At total width ``D = K d'`` the parameter count of the word–label factor is
``|V| K d' = |V| D`` — **identical** to a flat label of width ``D``. §22.2 states the
consequence in its own words, and it is a sharp falsifiable prediction rather than a
hope:

    "under the mean-field readout, factoring buys nothing over flat width D
    (logits = b + Σ_k S^(k) q^(k), affine dimension ≤ D); the gain exists only
    through the exact readout."

So the experiment that follows this file is a 2×2: ``K ∈ {1, 2, 4}`` crossed with
``readout ∈ {exact, mfvi}``, at one fixed ``d``. Under ``mfvi`` the ``K`` curves must
land on top of each other (up to optimisation noise); under ``exact`` they must not, or
the mixture family bought nothing. A ``K`` effect appearing under ``mfvi`` is evidence of
a bug in this file, not of capacity.

Layout: ``K = 1`` is the flat model, tensor for tensor
------------------------------------------------------
``cfg.d`` keeps meaning the **total** width ``D`` throughout, and ``cfg.n_components``
is ``K``, required to divide it; ``d' = d // K``. Consequences, all deliberate:

* ``S`` stays a single ``(V, d)`` tensor, read as ``K`` contiguous blocks of width
  ``d'``: ``S^(k) = S[:, k·d' : (k+1)·d']``. There is no list of ``K`` parameters.
  A list would break constraint 2 of ``CLAUDE.md`` — the tying of the input and output
  embedding is the *same tensor* in both roles — and it would multiply one factor
  (one matrix on the pair ``(word, label-slot)``) into ``K`` unrelated ones.
* the arc factors become per-component: ``T^(c)`` acts ``d' × d'``, so ``U`` and ``V``
  are ``(n_dist, h, d', rank)`` and ``r_root`` is ``(h, d')``.
* the label belief is ``(B, n, K, d')`` with a softmax over the **last** axis only —
  one distribution per component, which is what "K label variables" means.

At ``K = 1`` every tensor therefore has exactly the shape it has in
:class:`~src.pt_decoder.CausalPTDecoder`, and ``tests/test_18_factored.py`` asserts the
two models produce the same logits from the same state dict to 1e-12 in float64, under
both readouts. That test is the specification of this layout: if it fails, the layout is
wrong.

The parameter budget, stated honestly. The *embedding* block ``|V|d + |V|`` is exactly
independent of ``K``, which is §22.2's claim and the one that makes the comparison fair
(it is ~99% of the budget at any realistic ``|V|``). The *non-embedding* block is not
constant: the arc factors shrink as ``h·n_dist·d'²`` (or ``2·h·n_dist·d'·r``) and
``r_root`` as ``h·d'``, so a larger ``K`` is strictly *cheaper*. Factoring is therefore
never buying its effect with parameters; :func:`num_parameters` reports the split and
the test asserts both halves.

Channel assignment
------------------
The document is explicit that it does not fix ``k(c)`` and ``k'(c)``: "Per-channel
labels are the special case ``K = h``, ``k(c) = c``; there is no reason to hard-couple
``K`` to ``h``." The choice below is therefore this project's, not the paper's:

``"roundrobin"`` (the default, and what the experiments use)
    ``k(c) = c mod K`` and ``k'(c) = (c // K) mod K``. Chosen because at ``h = K²``
    every ordered ``(child, head)`` component pair occurs exactly once, so the model can
    route information *between* components and no pair is privileged. At ``K = 1`` it is
    the flat model.

``"diagonal"``
    ``k'(c) = k(c) = c mod K``. Components never read each other, so the model is ``K``
    independent width-``d'`` PTs sharing only ``W_t`` and ``b``. Kept as the ablation
    that separates "more mixture components" from "components that communicate".

Both are buffers, not parameters — they name no factor, they *are* the factor graph's
edge list — and they are non-persistent, so a factored state dict is the flat one.

Not supported
-------------
``cfg.n_global > 0``. The B.3.3 global head is one more leaf off ``Z_t``; with ``K``
label variables §22.2 does not say which component it hangs off (nor whether there is
one ``G_t`` per component), and inventing that is a modelling decision, not an
implementation detail. It is refused rather than guessed.
"""

from typing import List, Optional, Tuple

import torch

from .config import PTConfig
from .pt_decoder import CausalPTDecoder


def component_assignment(h: int, K: int, mode: str = "roundrobin") -> Tuple[List[int], List[int]]:
    """``(k(c), k'(c))`` for every channel ``c``, as plain Python lists.

    See the module docstring for the two modes. Written on lists so that a test can
    state the expected assignment without borrowing the model's tensors.
    """
    if mode == "roundrobin":
        return [c % K for c in range(h)], [(c // K) % K for c in range(h)]
    if mode == "diagonal":
        child = [c % K for c in range(h)]
        return child, list(child)
    raise ValueError(f"unknown channel_assignment {mode!r}")


class FactoredPTDecoder(CausalPTDecoder):
    """The causal PT decoder with ``K`` label variables per position (§22.2).

    Interface identical to :class:`~src.pt_decoder.CausalPTDecoder` — ``forward``,
    ``loss``, ``arc_regulariser``, ``num_parameters``, ``content_stream``,
    ``set_word_unary``, ``next_token_logits``, both readouts, both schedules — so the
    shared training loop of ``src/train.py`` takes it without a change.

    What differs internally, and only this:

    * every label-carrying tensor gains a component axis of size ``K`` before its
      ``d'`` axis, and every softmax over labels runs on the last axis alone;
    * a head channel queries with component ``k(c)`` and keys off component ``k'(c)``;
    * the message a channel returns is scattered back into component ``k(c)``;
    * the exact readout multiplies ``K`` mixtures instead of taking one.

    Everything else — causal masks, distance bucketing, the three attention temperature
    modes, the vocabulary chunking of the readout, the L2 arc regulariser, the parameter
    accounting, ``detach_prefix`` — is inherited unchanged from the flat decoder.
    """

    supports_factored_labels = True

    def __init__(self, cfg: PTConfig):
        if cfg.n_global > 0:
            raise NotImplementedError(
                "n_global > 0 with factored labels: §22.2 introduces the B.3.3 global head "
                "and the K label variables in the same section but never says which "
                "component G_t attaches to. Choosing one is a modelling decision; make it "
                "in the paper first. Use n_global=0."
            )
        super().__init__(cfg)
        child, head = component_assignment(cfg.h, cfg.n_components, cfg.channel_assignment)
        # Non-persistent: they are derived from the config, name no factor, and must not
        # appear in the state dict — a K=1 factored checkpoint is a flat checkpoint.
        self.register_buffer("k_child", torch.tensor(child, dtype=torch.long), persistent=False)
        self.register_buffer("k_head", torch.tensor(head, dtype=torch.long), persistent=False)
        # ``child_onehot[k, c] = 1[k(c) = k]``. Gathering a channel's message back into its
        # component is a segment sum over channels, and this contracts it as a matmul rather
        # than an ``index_add``: repeated indices make ``index_add`` an atomic scatter on
        # CUDA, which is run-to-run non-deterministic and would make a training run
        # irreproducible for no gain. It is a 0/1 incidence matrix of the factor graph, not
        # a learned map, so it is a buffer and follows the module's dtype.
        onehot = torch.zeros(cfg.n_components, cfg.h)
        onehot[self.k_child, torch.arange(cfg.h)] = 1.0
        self.register_buffer("child_onehot", onehot, persistent=False)

    # -------------------------------------------------------------------- shapes --

    @property
    def K(self) -> int:
        """Number of label components per position."""
        return self.cfg.n_components

    @property
    def d_label(self) -> int:
        """Width ``d'`` of one label variable."""
        return self.cfg.d_label

    def _S_blocks(self) -> torch.Tensor:
        """``S`` viewed as ``(V, K, d')`` — the same tensor, no copy, no second matrix."""
        return self.S.view(self.cfg.vocab_size, self.K, self.d_label)

    def _components(self, x: torch.Tensor) -> torch.Tensor:
        """Split a trailing total-width ``d`` axis into ``(K, d')``."""
        return x.reshape(x.shape[:-1] + (self.K, self.d_label))

    # ------------------------------------------------------------------ factors --

    def contract(self, q: torch.Tensor, T: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``B^(c)_{j,a} = Σ_b q̄_j^{(k'(c))}(b) T^(c)_{a,b}`` — ``(n_dist, B, h, n, d')``.

        ``q`` is ``(B, n, K, d')``. The head component is selected on the *query* side
        before the contraction, which costs one ``(B, n, h, d')`` gather and leaves the
        einsum the same size as the flat model's.
        """
        qc = q.index_select(-2, self.k_head)  # (B, n, h, d')
        arcs = self.arc_scores() if T is None else T
        return torch.einsum("bjce,kcae->kbcja", qc, arcs)

    # ------------------------------------------------------- messages, vectorised --

    def _arc_message(
        self, query: torch.Tensor, Bk: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One H-update and the H → Z message, every slot in parallel.

        ``query`` is ``(B, n, K, d')`` and ``Bk`` the output of :meth:`contract`::

            F_c(i, j) = Σ_a query_i^{(k(c))}(a) B^(c)_{j,a},  Q_c = softmax_{D_i}(F_c/λ_H)
            G_i^{(k)}(a) = Σ_{c: k(c)=k} Σ_{j∈D_i} Q_c(j) B^(c)_{j,a}

        Returns ``(G, alpha)`` with ``G`` of shape ``(B, n, K, d')`` and ``alpha`` of
        shape ``(B, h, n, 1 + n)``, column 0 being ROOT — the same attention tensor the
        flat model produces, so every attention diagnostic keeps working.
        """
        n = query.shape[1]
        nd = self.cfg.n_dist
        far_B = Bk[-1]  # (B, h, n, d')

        # per-channel query: channel c asks with the belief of its child component.
        qc = query.index_select(2, self.k_child).permute(0, 2, 1, 3).contiguous()  # (B,h,n,d')

        # --- attention logits ---
        logit = torch.einsum("bcia,bcja->bcij", qc, far_B)
        if nd > 1:
            logit = logit.clone()
            for k in range(nd - 1):
                delta = k + 1
                if delta >= n:
                    break
                ii = torch.arange(delta, n, device=query.device)
                near = (qc[:, :, delta:, :] * Bk[k][:, :, : n - delta, :]).sum(-1)
                logit[:, :, ii, ii - delta] = near
        root_logit = torch.einsum("bcia,ca->bci", qc, self.r_root)  # (B, h, n)
        full = torch.cat([root_logit.unsqueeze(-1), logit], dim=-1)  # (B, h, n, 1+n)

        allowed, far_mask = self._causal_masks(n, query.device)
        # `qc` carries the channel axis, so "qnorm" reads a per-channel query norm; the
        # unsqueeze loop in _scaled_logits lines it up without a change there.
        alpha = torch.softmax(self._scaled_logits(full, qc, allowed), dim=-1)

        # --- message back to Z, per channel, then summed into its child component ---
        a_root, a_pos = alpha[..., 0], alpha[..., 1:]
        Gc = torch.einsum("bcij,bcja->bcia", a_pos * far_mask, far_B)  # (B, h, n, d')
        for k in range(nd - 1):
            delta = k + 1
            if delta >= n:
                break
            ii = torch.arange(delta, n, device=query.device)
            a_k = a_pos[:, :, ii, ii - delta]  # (B, h, n-delta)
            contrib = a_k.unsqueeze(-1) * Bk[k][:, :, : n - delta, :]
            Gc = Gc.index_add(2, ii, contrib)
        Gc = Gc + torch.einsum("bci,ca->bcia", a_root, self.r_root)

        G = torch.einsum("kc,bcna->bnka", self.child_onehot, Gc)  # (B, n, K, d')
        return G, alpha

    def _global_message(self, query: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """No global head under factoring — see the module docstring. Always zero."""
        return query.new_zeros(query.shape), None

    # -------------------------------------------------------------- content stream --

    def _content_parallel(self, idx: torch.Tensor, trace: Optional[list] = None) -> torch.Tensor:
        """Layer-parallel schedule, ``K`` beliefs per position. Returns ``(B, n, K, d')``."""
        Sw = self._components(self.S[idx])  # (B, n, K, d')
        T = self.arc_scores()
        q = torch.softmax(Sw / self.cfg.lambda_Z, dim=-1)
        for _ in range(self.cfg.n_iters):
            G, alpha = self._arc_message(q, self.contract(q, T))
            if trace is not None:
                trace.append(self._iteration_stats(q, Sw, G, alpha))
            q_new = torch.softmax((Sw + G) / self.cfg.lambda_Z, dim=-1)
            a = self.cfg.alpha_Z
            q = q_new if a == 1.0 else a * q_new + (1.0 - a) * q
        return q

    def _content_serial(self, idx: torch.Tensor, trace: Optional[list] = None) -> torch.Tensor:
        """Serial left-to-right filtering, ``K`` beliefs per position."""
        Sw = self._components(self.S[idx])
        n = idx.shape[1]
        qs: List[torch.Tensor] = []
        keys: List[torch.Tensor] = []
        for t in range(n):
            B_full = self._slot_keys_from(keys, t, idx.shape[0])
            q_t = torch.softmax(Sw[:, t] / self.cfg.lambda_Z, dim=-1)  # (B, K, d')
            for _ in range(self.cfg.tau_obs):
                ctx, alpha = self._slot_message(q_t, B_full)
                if trace is not None:
                    trace.append(
                        self._iteration_stats(
                            q_t.unsqueeze(1), Sw[:, t : t + 1], ctx.unsqueeze(1),
                            alpha.unsqueeze(2),
                        )
                    )
                q_t = torch.softmax((Sw[:, t] + ctx) / self.cfg.lambda_Z, dim=-1)
            qs.append(q_t)
            keys.append(self.contract(q_t.unsqueeze(1)))
        return torch.stack(qs, dim=1)

    # ----------------------------------------------------------- messages, one slot --

    def _slot_message(
        self, query: torch.Tensor, B_full: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The single-slot H-update and H → Z message.

        ``query`` is ``(B, K, d')``, ``B_full`` is ``(B, h, 1+t, d')``; returns
        ``(ctx, Q_c)`` with ``ctx`` of shape ``(B, K, d')``.
        """
        qc = query.index_select(1, self.k_child)  # (B, h, d')
        logit = torch.einsum("bca,bcja->bcj", qc, B_full)
        alpha = torch.softmax(self._scaled_logits(logit, qc, None), dim=-1)
        ctx_c = torch.einsum("bcj,bcja->bca", alpha, B_full)  # (B, h, d')
        ctx = torch.einsum("kc,bca->bka", self.child_onehot, ctx_c)
        return ctx, alpha

    # ------------------------------------------------------------------- readouts --

    def exact_log_mu(self, Bk: torch.Tensor) -> torch.Tensor:
        """``log μ_t^(k)(a) = Σ_{c: k(c)=k} LSE_{j∈D_t} B^(c)_{j,a}`` — ``(B, n, K, d')``.

        The per-channel prefix log-sum-exp is inherited verbatim (``_channel_log_mu``);
        factoring changes only *which* channels are summed together. A component with no
        channel assigned to it keeps ``log μ^(k) ≡ 0``, i.e. the empty product ``μ ≡ 1``,
        which is the correct reading of the formula and not a special case.
        """
        per_c = self._channel_log_mu(Bk)  # (B, h, n, d')
        return torch.einsum("kc,bcna->bnka", self.child_onehot, per_c)

    def _chunk_logits(self, log_mu: torch.Tensor, S_c: torch.Tensor) -> torch.Tensor:
        """``Σ_k LSE_a ( S^(k)_{w,a} + log μ^(k)(a) )`` for one slice of the vocabulary.

        This is the log of §22.2's product of ``K`` mixtures,
        ``Π_k Σ_a exp(S^(k)_{w,a}) μ^(k)(a)``. ``log_mu`` is ``(..., K, d')`` and ``S_c``
        is ``(chunk, d)``; the result is ``(..., chunk)``, so the chunking loop, the
        ``+ b`` and the whole of :meth:`_logits_from_log_mu` are inherited unchanged.
        """
        lead = log_mu.shape[:-2]
        S_k = S_c.view((1,) * len(lead) + (S_c.shape[0], self.K, self.d_label))
        return torch.logsumexp(log_mu.unsqueeze(-3) + S_k, dim=-1).sum(-1)

    def _word_message(self) -> torch.Tensor:
        """``s̄^(k)_a = Σ_w Q_W^(0)(w) S^(k)_{w,a}`` — §17.1's derived [MASK], ``(K, d')``."""
        return self._components(self._word_prior()[1])

    def _word_logits(self, qz: torch.Tensor) -> torch.Tensor:
        """``Σ_k Σ_a Q_{Z^(k)}(a) S^(k)_{w,a}``, the mean-field word message, ``(..., V)``.

        Affine in ``(q^(1), ..., q^(K))`` of total dimension ``D`` — which is §22.2's
        point that the mean-field readout gains nothing from factoring.
        """
        return torch.einsum("...ke,wke->...w", qz, self._S_blocks())

    def mfvi_readout(self, Bk: torch.Tensor) -> torch.Tensor:
        """Mean-field readout of §17.1 with ``K`` label variables. ``(B, n, V)``."""
        Bt, n = Bk.shape[1], Bk.shape[3]
        sbar = self._word_message()
        qz = torch.softmax(sbar / self.cfg.lambda_Z, dim=-1).expand(Bt, n, self.K, self.d_label)
        for _ in range(self.cfg.tau):
            G, _ = self._arc_message(qz, Bk)
            qz = torch.softmax((sbar + G) / self.cfg.lambda_Z, dim=-1)
        logits = self._word_logits(qz)
        if self.b is not None:
            logits = logits + self.b
        return logits / self.cfg.lambda_W

    # --------------------------------------------------------------- slot readouts --

    def slot_exact_readout(self, B_full: torch.Tensor) -> torch.Tensor:
        """Exact readout for one slot from its assembled keys ``(B, h, 1+t, d')``."""
        per_c = torch.logsumexp(B_full, dim=2)  # (B, h, d')
        log_mu = torch.einsum("kc,bca->bka", self.child_onehot, per_c)
        return self._logits_from_log_mu(log_mu)

    def slot_mfvi_readout(self, B_full: torch.Tensor, trace: Optional[list] = None) -> torch.Tensor:
        """Mean-field readout for one slot. ``trace`` receives ``(Q_Z, Q_c, None)`` after
        every block update, in update order, matching the flat model's contract."""
        sbar = self._word_message()
        qz = torch.softmax(sbar / self.cfg.lambda_Z, dim=-1).expand(
            B_full.shape[0], self.K, self.d_label
        )
        for _ in range(self.cfg.tau):
            ctx, alpha = self._slot_message(qz, B_full)
            if trace is not None:
                trace.append((qz, alpha, None))
            qz = torch.softmax((sbar + ctx) / self.cfg.lambda_Z, dim=-1)
            if trace is not None:
                trace.append((qz, alpha, None))
        logits = self._word_logits(qz)
        if self.b is not None:
            logits = logits + self.b
        return logits / self.cfg.lambda_W
