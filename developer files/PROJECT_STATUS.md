# Project status

What has been built, which runs were executed with which configs, what the numbers were,
and which decisions were made and why. Organised by component, most recent changes first.

## Recent changes

**2026-08-09 — restart.** The previous implementation (`src/`, `tests/`, `experiments/`,
old status and report files, checkpoints, slurm logs) was removed at the user's request;
the reference PDFs, `PROJECT.md` and `RESEARCH_PLAN.md` were kept, as was the PTB corpus
under `data/`. History is intact — the last commit of the old implementation is `9c77f94`
(`v0.9.0`), so anything can be read back with `git show 9c77f94:<path>`. The version was
reset to `0.1.0`.

**2026-08-09 — the causal PT decoder, written from scratch.** Experiment 0 complete:
checks 1–9 of the research plan pass, and the worked example of
`causal_pt_output_note.pdf` §5 is reproduced number for number. Details in
[`experiments/exp0_decoder_validation/EXPERIMENT_STATUS.md`](../experiments/exp0_decoder_validation/EXPERIMENT_STATUS.md).

## Model code

```
src/config.py      PTConfig — graph shape, inference schedule, message weights
src/pt_decoder.py  CausalPTDecoder — content stream, exact and mean-field readouts
src/energy.py      slot_free_energy — the functional the MFVI updates descend
```

### `CausalPTDecoder`

Parameters, which are exactly the factor list and nothing else:

| Tensor | Shape | Factor |
|---|---|---|
| `S` | `|V| × d` | word–label factor; unary when `W_t` is observed, emission when free |
| `b` | `|V|` | word unary — the LM head bias, as a factor (§16(c), optional) |
| `r_root` | `h × d` | root/sink column `r^(c)`, the ROOT entry of the contracted arc score |
| `T` | `n_dist × h × d × d` | arc score per channel per distance bucket |
| `U`, `V` | `n_dist × h × d × r` | Kruskal form `T^(c) = U^(c) V^(c)ᵀ` (Wu & Tu Eqs. 14/21), when `rank` is set |
| `B_glob` | `m × d` | single-split global head (Wu & Tu App. B.3.3), when `n_global > 0` |

Forward pass:

1. **Content stream** → `q̄` `(B, n, d)`. MFVI on the directed chain of conditional CRFs
   (Part II §12.2). `schedule="parallel"` is the layer-parallel version — the computation
   graph of a depth-`T`, parameter-shared causal transformer, and the training path.
   `schedule="serial"` is left-to-right filtering with a per-slot inner loop.
2. **Contraction** → `B^(c)_{j,a} = Σ_b q̄_j(b) T^(c)_{a,b}`, seeded at ROOT with `r^(c)_a`.
   This is the KV cache; it falls out of the model rather than being an engineering trick.
3. **Readout.** `readout="exact"` (mainline, §23.3): `log μ_t(a) = Σ_c LSE_{j∈D_t} B^(c)_{j,a}`
   implemented as a `logcumsumexp` scan over the far distance bucket plus shifts for the
   near band, then `logits(w) = b_w + LSE_a (S_{w,a} + log μ_t(a))`, chunked over the
   vocabulary. `readout="mfvi"` (ablation, §17.1): τ rounds of the (2)–(3) inner loop from
   the prior word message `s̄`, then one `Q_W` update as the LM output layer.

`logits[:, t]` predicts the token *at* position `t`, so the training target is `idx`
itself and not a shifted copy (§18 Check 5). Slot 0 predicts from ROOT alone, so there is
no BOS hack.

### Decisions

* **Exact readout is the mainline.** §17.2 recommends MFVI, §23.3 inverts it explicitly.
  The later section wins. MFVI is kept as Experiment 3's comparison object.
* **Gradients flow backwards through the frozen prefix** (`detach_prefix=False`), per Part
  II §12.3 Check 2 and Part III §18 Check 5. Ruled on 2026-08-09: `CLAUDE.md` constraint 3
  was rewritten to separate the forward claim (binding) from the gradient claim (which the
  old wording got wrong). The stop-gradient reading remains as a flag with a test.
* **Defaults are Wu & Tu Table 2, PTB masked LM**: `d = 384, h = 16, rank = 64, γ = 3,
  T = 5`. Previously `d = 64, h = 4, rank = None, T = 4`, which were arbitrary. Taking the
  source's row leaves nothing here to defend that the source has not defended. Guarded by
  `tests/test_10_diagnostics.py::test_default_config_is_the_source_table_2_row`.
* **`λ_H = 1/d`** by default (Wu & Tu §2.3.3 and App. A.5); `λ_Z = 1`, `λ_W = 1`.
* **RPE is implemented** — without it the content stream is permutation-invariant over the
  prefix. Only the causal half of the clipped table exists, so `n_dist = γ + 1`; `γ = 0`
  makes the arc score distance-insensitive, which is the setting of the note's example.
* **Global head is a flag, defaulting off** — the research plan makes it a measured
  variable (arms 1.1 / 1.2), and baking it in would assume the answer.

### Measured behaviour

`src/diagnostics.py` (`python -m src.diagnostics`) reports, per content-stream iteration,
`‖G‖/‖S_w‖`, `max|G|`, attention entropy as a fraction of its maximum, label entropy, and
the contraction constant `ρ` of Lemma 23.1. Full numbers and their reading are in the exp0
status file; the four that matter:

* **Nothing normalises `G`, and nothing needs to.** `|G_i(a)| ≤ h · max(max|T|, max|r|)`
  because every message is a convex combination. Measured at 95 % of that bound on the
  overfitted toy model, so the bound is tight, not slack. Parameter growth is the only
  unbounded direction, and the source controls it with an L2 penalty on `T` (5e-4 on PTB)
  which is **not implemented** — there is no training loop yet.
* **`λ_H = 1/d` does not sharpen the attention at initialisation** — measured `H/H_max =
  0.982` at `d = 384`. It cancels the `1/d²` variance shrinkage of App. A.5. Sharpening is
  driven by `‖T‖`: after overfitting, `H/H_max` fell to 0.22.
* **`ρ ≫ 1` means the bound is vacuous, not that the model is unstable.** Lemma 23.1
  (Part IV §23.1) bounds the divergence between the predictive and observed runs of one
  slot, and with the forcing term removed the same constant is the contraction factor of
  the slot map, so `ρ < 1` guarantees a unique fixed point. Measured `ρ = 233` at the
  source's configuration — and a direct probe from 48 random initialisations finds
  **exactly one fixed point** there, and one after training at `ρ = 207`. The empirical
  onset of a second fixed point is around `ρ ≈ 130` in a `d = 24` sweep. An earlier note
  in this file treated the root column's share of `ρ = 233` as possibly central; that is
  withdrawn — shrinking `root_init_std` 1000× changes neither `ρ` nor the fixed-point
  count once the arc scores are large. `ρ` is a cheap diagnostic to log, nothing more.
* **The root column is a measured variable, not a defect.** It is initialised ≈121×
  above the contracted arc scores, but the attention sink it was suspected of causing is
  absent: ROOT mass is 1.10× uniform untrained and 0.15× uniform after training, i.e. the
  model learns to avoid it. Default unchanged; `root_mass` is logged every iteration.
* **The two schedules can disagree completely, where the map is genuinely multistable.**
  TV between the parallel and serial `q̄` is 3e-6 at `init_std=0.02`, 0.111 at 0.5 and
  0.682 (with `TV_max = 1.000`, disjoint support) at 2.0 — and at that scale the slot map
  has 15–22 fixed points, so the two schedules land in different ones. `parallel` is the
  training mainline; re-measure on real data in Experiment 1.

### Known costs, not yet paid down

Arc scores are materialised as `d × d` per channel and bucket, so attention logits cost
`O(n² d)` instead of `O(n² r)` and the contracted scores cost `O(n_dist · B · h · n · d)`
memory. This is correct but expensive; the `r`-space fast path changes no mathematics and
is the first thing to do before Experiment 1.

## Tests

`tests/test_01..10` — 64 tests, all passing, CPU, `float64` except the overfit check.
Run with `python -m pytest`. Roughly 41 s.

The two that carry the weight:

* `test_09_exact_vs_brute.py` enumerates the slot joint with plain Python loops and
  compares against the closed-form readout — agreement to `1e-12` validates the
  factorisation, not an approximation to it. The vectorised `logcumsumexp` scan is
  separately checked against slot-by-slot assembly for `γ ∈ {0, 1, 2, 4, 9}`.
* `test_08_free_energy.py` checks the free energy is non-increasing along the inner loop,
  and contains a mutation test — the same check run against a sign-flipped H-update must
  fail. Checks 1–7 survive that mutation untouched.

## Runs

See the run log in each experiment's `EXPERIMENT_STATUS.md`. Summary of what exists:

| Experiment | State |
|---|---|
| 0 — decoder validation | complete; checks 1–9 pass, single-batch loss 2.45 → 0.0022 |
| 1 — PT vs. GPT | not started |
| 2 — PT vs. Looped | not started |
| 3 — exact vs. MFVI readout | not started (both readouts implemented and tested) |

## Required before Experiment 1 — not implemented

Ruled mandatory in the review of 2026-08-09. All three come from Wu & Tu Table 2, PTB
masked LM, and all three are training-loop concerns, which is why none exists yet.

| Item | Value | Why it is not optional |
|---|---|---|
| L2 penalty on `T` | `5e-4` | The **only** mechanism restraining `‖T‖`. Message size and `ρ` are both governed by it, and there is no layer norm to absorb growth. Wu & Tu §4.2 add it for MLM and "experimentally find beneficial". |
| Dropout | `0.15` | The source's value for this dataset. Note it is a training-time regulariser, not a factor, so it does not cross the §22.2 tripwire. |
| Weight decay | `1.4e-6` | Adam, `β1 = 0.9`, `β2 = 0.999`, lr `1e-3` on PTB. |

Also mandatory: evaluate with `loss(idx, ignore_first=1)`. PT scores `w_0..w_{n-1}` and a
GPT baseline scores `w_1..w_{n-1}`; without this the two perplexities are over different
token sets and the comparison is void.

## Data

`data/ptb/` holds the Penn Treebank splits (`ptb.train.txt` 5.1 MB, `ptb.valid.txt`,
`ptb.test.txt`), gitignored. No data pipeline is written yet.

## Environment

HPC login node, Slurm available. `.venv` with Python 3.11.15, torch 2.13.0+cpu — **CPU
only on this node**; a GPU node is needed for anything beyond the toy scale. System `git`
is 1.8.3.1, so no `git switch` / `git restore`.
