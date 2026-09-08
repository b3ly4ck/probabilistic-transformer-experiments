"""Experiment 6 — the switch ladder: which difference from a transformer costs what?

The ask
-------
Penghao Kuang, 2026-08-14:

    "if the performance of PT turns out to be significantly inferior, you might consider
     designing analytical experiments to investigate why the PT Decoder underperforms
     compared to Transformers. Haoyi Wu, the original author of the PT paper, previously
     conducted an experiment summarizing ten key differences between PT and Transformer
     models, treating them as ten 'switches.' When all switches are turned on, the model
     behaves identically to a Transformer; when all are turned off, it behaves identically to
     the original PT. This framework was used to systematically analyze the distinctions
     between the two architectures."

`src/switches.py` is that framework for the *decoder*. Ten switches, each independently
settable, with all-transformer numerically equal to `src/gpt.py` (asserted in
`tests/test_15_switches.py` by loading the GPT state dict into the switch model and comparing
logits) and all-PT structurally equal to the causal PT decoder.

What this is and is not
-----------------------
This is **analysis**, deliberately outside the modelling discipline of `src/pt_decoder.py`.
Intermediate rungs carry learned matrices that name no factor of the graph — a value
projection distinct from the key, an output projection over channels, a feed-forward block.
That is precisely why they are here: the point is to *price* the factor-graph constraint, and
a price is only visible against something that does not pay it. No rung of this ladder is a
proposed model.

Design
------
Three families, all at `n_embd = d_label = 64`, `n_layer = 4`, `n_head = 4`, PTB,
15,000 steps, blocks 16 x 64, validation checkpoint selection, seed 0.

* **S1, single switch on** — the transformer with exactly one property replaced by PT's. The
  marginal cost of each PT property in an otherwise ordinary decoder. Ten cells.
* **S2, leave one out** — the all-PT configuration with exactly one property restored to the
  transformer's. The marginal *benefit* of each transformer property inside PT. Ten cells.
  S1 and S2 answer different questions, and a property whose S1 cost is small while its S2
  benefit is large is one that only matters in combination — which is the interesting case
  and the one a single ladder hides.
**Why `n_embd = d_label`, and not a smaller label set.** The word--label factor `S` is the
transformer's `wte` read in the other direction, so the mixture rung can only *tie* the two --
which `CLAUDE.md` constraint 2 and \Cref{sec:output} say is forced rather than chosen -- when
the two widths agree. At `n_embd = 64, d_label = 32` the ladder would silently run its
PT anchor untied, i.e. as a different model from the one the paper is about; that was caught
by the verification pass on `src/switches.py` before any of these cells ran. Matching the
widths also makes the readout switch measure the thing it is supposed to measure: by
\Cref{prop:readout} the mean-field bottleneck equals a linear head's *at matched width*, so
with `n_embd = d_label` the rung isolates the mixture-versus-linear *family* difference, with
no width reduction confounded into it. Width is exp2's axis, not a switch.

* **L, cumulative ladder** — flip the switches one at a time in a fixed, declared order, from
  transformer to PT. Eleven rungs. The order puts the two properties suspected of carrying
  the gap last, so the curve shows where the cliff is rather than smearing it:

      weight_sharing, position, norm, attn_out_proj, attn_query_proj,
      attn_value, ffn, residual, state, readout

  `state` is the simplex constraint on the hidden vector and `readout` is the rank-`d_label`
  mixture-of-softmaxes head. Together they are the "all predictive information passes through
  the label variables" clause that the paper's Discussion attributes the gap to, and this
  ladder is the first measurement of that attribution.

**Every rung runs at four learning rates (1e-3, 3e-3, 1e-2, 3e-2) and is scored at its
best.** A
ladder run at one rate measures the rate's mismatch with the rung, not the rung: the PT end
of this ladder wants ~2e-2 and the transformer end wants ~1e-3, and the 2026-08 report's
whole diagnosis turned on that being a *joint* region in `(d, lr)`.

Parameter counts differ along the ladder by construction (dropping `W_V`, `W_O` and the
feed-forward block removes parameters). Each row carries its own count and the analysis
reports perplexity against it, so a rung that wins by being larger is visible as such.

Falsification
-------------
The paper's stated explanation of the gap -- the rank-`d_label` simplex bottleneck -- predicts
that `state` and `readout` account for most of the cumulative drop and that every other single
switch is cheap. If instead the drop is spread evenly across the ten, the bottleneck story is
wrong and the honest conclusion is that the construction costs a little everywhere. If some
*other* single switch dominates (the tied key/value, say, or the absence of a feed-forward
block), then that is the finding and it points at a specific repair.
"""

from experiments.sweep import Cell
from src.switches import PT_SWITCHES, TRANSFORMER_SWITCHES

LADDER_ORDER = [
    "weight_sharing",
    "position",
    "norm",
    "attn_out_proj",
    "attn_query_proj",
    "attn_value",
    "ffn",
    "residual",
    "state",
    "readout",
]

# The PT end of this ladder wants a rate the transformer end would diverge at: the all-PT
# anchor sits on the unigram (687.2) at 1e-3, and exp4c puts the causal PT's argmin at 2e-2 to
# 4e-2. 3e-2 is appended rather than inserted so that the cell ordering -- and therefore the
# shard assignment `k % n` -- of the 96 cells already run is unchanged and they resume-skip.
LRS = [1e-3, 3e-3, 1e-2, 3e-2]
STEPS = 15000


def _cell(name, switches, lr, tags):
    return Cell(
        name=name,
        model="switch",
        n_embd=64,
        n_layer=4,
        n_head=4,
        d=64,  # d_label; equal to n_embd so the word-label factor can be tied (see above)
        switches=dict(switches),
        lr=lr,
        steps=STEPS,
        eval_every=500,
        l2_arc=0.0,
        batch_size=16,
        block_size=64,
        seed=0,
        tags=tags,
    )


def build_cells(*args):
    assert set(LADDER_ORDER) == set(TRANSFORMER_SWITCHES), (
        "LADDER_ORDER must name every switch exactly once; "
        f"missing {set(TRANSFORMER_SWITCHES) - set(LADDER_ORDER)}, "
        f"unknown {set(LADDER_ORDER) - set(TRANSFORMER_SWITCHES)}"
    )
    cells = []

    for lr in LRS:
        # --- the two anchors ---------------------------------------------------------
        cells.append(_cell(f"anchor_transformer_lr{lr:g}", TRANSFORMER_SWITCHES, lr,
                           {"family": "anchor", "which": "transformer", "n_flipped": 0}))
        cells.append(_cell(f"anchor_pt_lr{lr:g}", PT_SWITCHES, lr,
                           {"family": "anchor", "which": "pt", "n_flipped": 10}))

        # --- S1: the transformer with one PT property --------------------------------
        for name in LADDER_ORDER:
            sw = dict(TRANSFORMER_SWITCHES)
            sw[name] = PT_SWITCHES[name]
            cells.append(_cell(f"S1_{name}_lr{lr:g}", sw, lr,
                               {"family": "S1", "switch": name, "n_flipped": 1}))

        # --- S2: PT with one transformer property restored ---------------------------
        for name in LADDER_ORDER:
            sw = dict(PT_SWITCHES)
            sw[name] = TRANSFORMER_SWITCHES[name]
            cells.append(_cell(f"S2_{name}_lr{lr:g}", sw, lr,
                               {"family": "S2", "switch": name, "n_flipped": 9}))

        # --- L: the cumulative ladder ------------------------------------------------
        sw = dict(TRANSFORMER_SWITCHES)
        for k, name in enumerate(LADDER_ORDER, start=1):
            sw[name] = PT_SWITCHES[name]
            cells.append(_cell(f"L{k:02d}_{name}_lr{lr:g}", sw, lr,
                               {"family": "L", "rung": k, "switch": name, "n_flipped": k}))
    return cells
