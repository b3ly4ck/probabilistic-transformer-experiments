"""Regenerate every figure and every table of the preprint from the sweep JSONs.

One script, run repeatedly as shards land. Nothing in the paper is typed by hand from a
terminal: `CLAUDE.md` says a number without a commit cannot be reproduced or defended, and the
paper's own rule is that no number appears without a run-log row. The way to keep both true is
to make the figures and the tables a *function* of the committed JSONs, so that re-running this
script after any new shard updates the paper and nothing drifts.

Usage::

    python -m experiments.make_report                 # figures to the article worktree
    python -m experiments.make_report --out-dir DIR   # somewhere else
    python -m experiments.make_report --tables-only

Every section is independent and tolerant of missing input: a grid whose shards have not
finished prints what it has and says how many cells are still missing, rather than failing or
--- worse --- silently drawing a curve through half a grid.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.analysis import (
    aggregate_seeds,
    best_by,
    fit_power_law,
    get_path,
    load_rows,
    markdown_table,
    plot_heatmap,
    plot_scaling,
    plot_traces,
    scaling_frontier,
)

REPO = Path(__file__).resolve().parent.parent
ARTICLE = Path("/public/home/belyack/work/pt-article/paper/figures")
UNIGRAM_PTB = 688.82  # val ppl of the ML unigram on the identical token set, ignore_first=1
BIGRAM_PTB = 445.24
# The unigram says whether a model learned anything; the bigram says whether it learned anything
# beyond the previous word. Once the interpretability probe found the trained model putting
# 0.976 of its positional attention mass at distance 1, the second stopped being a formality and
# became the line a reader should actually look at.

GRIDS = {
    "baselines": "experiments/exp2_scaling/spec_baselines*.json",
    "pt_scaling": "experiments/exp2_scaling/spec_pt.shard*.json",
    "pt_lowrank": "experiments/exp2_scaling/spec_pt_lowrank*.json",
    "width": "experiments/exp4_width_transfer/spec_width*.json",
    "gain": "experiments/exp4_width_transfer/spec_gain*.json",
    "lr_transfer": "experiments/exp4_width_transfer/spec_lr_transfer*.json",
    "rank": "experiments/exp4_width_transfer/spec_rank*.json",
    "init": "experiments/exp5_init/spec_init*.json",
    "globalhead": "experiments/exp5_init/spec_globalhead*.json",
    "open": "experiments/exp3_readout/spec_open*.json",
    "repro": "experiments/exp3_readout/repro_check*.json",
    "switches": "experiments/exp6_switches/spec_switches*.json",
    "factored": "experiments/exp7_factored/spec_factored*.json",
    "wt2": "experiments/exp8_transfer/spec_wt2*.json",
}


def load(name: str) -> List[Dict[str, Any]]:
    got = load_rows(str(REPO / GRIDS[name]))
    rows = list(got)
    dropped = getattr(got, "n_dropped", 0)
    print(f"  [{name}] {len(rows)} rows" + (f", {dropped} failed cells dropped" if dropped else ""))
    return rows


def tagged(rows: Sequence[Dict[str, Any]], **want) -> List[Dict[str, Any]]:
    """Rows whose ``cell.tags`` match every keyword given."""
    out = []
    for r in rows:
        tags = get_path(r, "cell.tags") or {}
        if all(tags.get(k) == v for k, v in want.items()):
            out.append(r)
    return out


def emit(md: List[str], title: str, rows, columns, note: str = "") -> None:
    md.append(f"\n### {title}\n")
    if note:
        md.append(note + "\n")
    if not rows:
        md.append("_no rows yet_\n")
        return
    md.append(markdown_table(rows, columns))


# ---------------------------------------------------------------- scaling curves --


def section_scaling(md: List[str], out: Path) -> None:
    md.append("\n## Scaling curves (Experiment 2)\n")
    base = load("baselines")
    pt = load("pt_scaling")
    low = load("pt_lowrank")
    if not base:
        md.append("_baseline grid empty_\n")
        return

    width = [r for r in base if get_path(r, "cell.tags.family") == "width"]
    depth = [r for r in base if get_path(r, "cell.tags.family") == "depth"]
    bw = best_by(width, ["cell.model", "cell.n_embd"], metric="val_ppl")
    bd = best_by(depth, ["cell.model", "cell.n_layer"], metric="val_ppl")
    # The two PT ladders are different models, not two runs of one: the control carries the
    # 2026-08 constants across widths, the rule ladder uses the standardised temperature, and
    # the low-rank ladder additionally drops the Kruskal rank. Each is its own curve, and the
    # scaling plot draws the Pareto envelope of each rather than mixing them.
    rule = [r for r in pt if get_path(r, "cell.tags.ladder") == "rule"]
    ctrl = [r for r in pt if get_path(r, "cell.tags.ladder") == "fixed"]
    bpt = best_by(rule, ["cell.d"], metric="val_ppl") if rule else []
    bctrl = best_by(ctrl, ["cell.d"], metric="val_ppl") if ctrl else []
    blow = best_by(low, ["cell.d", "cell.rank"], metric="val_ppl") if low else []

    cols = ["cell.model", "cell.n_embd", "cell.n_layer", "cell.d", "cell.lr", "val_ppl",
            "test_ppl", "train_ppl", "params.non_embedding", "params.total", "best_step"]
    emit(md, "Width family, best learning rate per point", bw, cols,
         "Each row is the better of `lr in {1e-3, 3e-3}`; 15,000 steps; checkpoint selected "
         "on validation.")
    emit(md, "Depth family (`n_embd = 64`)", bd, cols)
    for label, rows_ in (("Causal PT -- control (source constants carried across widths)", bctrl),
                         ("Causal PT -- standardised temperature, undamped", bpt),
                         ("Causal PT -- standardised temperature at low Kruskal rank", blow)):
        if rows_:
            emit(md, label, sorted(rows_, key=lambda r: (r["cell"]["d"], r["cell"]["rank"] or 0)),
                 cols + ["cell.rank"])

    for xkey, fname, xlabel in (
        ("params.non_embedding", "fig_scaling_nonemb.png", "non-embedding parameters (count)"),
        ("params.total", "fig_scaling_total.png", "total parameters (count)"),
    ):
        curves = []
        for model, label in (("gpt", "causal transformer"), ("looped", "looped transformer")):
            pts = [r for r in bw if get_path(r, "cell.model") == model]
            if pts:
                curves.append((label, pts))
        if bctrl:
            curves.append(("causal PT, source constants", bctrl))
        if bpt:
            curves.append(("causal PT, standardised temp.", bpt))
        if blow:
            curves.append(("causal PT, + low rank", blow))
        # Report the fitted exponent twice: over each curve's own range, and over the range
        # where all three models are actually measured. They differ a lot, because the
        # baselines extend into the regime where this corpus rather than the parameter count
        # binds and their flat points drag the fitted slope down. Quoting only the first
        # invites the reading that the causal PT scales like a transformer, which the second
        # refutes.
        if curves:
            md.append("\n**Fitted power-law exponents on the "
                      + ("non-embedding" if xkey.endswith("non_embedding") else "total")
                      + " axis**\n")
            rowsx = []
            for label, data in curves:
                fr = scaling_frontier(data, x=xkey, y="val_ppl")
                # `scaling_frontier` returns rows when given rows and (x, y) pairs when given
                # pairs; accept both rather than assuming which.
                P = []
                for r in fr:
                    if isinstance(r, (list, tuple)):
                        P.append((float(r[0]), float(r[1])))
                    else:
                        xv, yv = get_path(r, xkey), r.get("val_ppl")
                        if xv is not None and yv is not None:
                            P.append((float(xv), float(yv)))
                if not P:
                    continue
                full = fit_power_law([a for a, _ in P], [b for _, b in P])
                lo, hi = 4e3, 1.5e5
                Q = [q for q in P if lo <= q[0] <= hi]
                comm = fit_power_law([a for a, _ in Q], [b for _, b in Q])
                rowsx.append({
                    "curve": label, "n": len(P),
                    "own range": f"{full.exponent:+.3f} (r2 {full.r2:.2f})" if full else "--",
                    "n (common)": len(Q),
                    "4k-150k": f"{comm.exponent:+.3f} (r2 {comm.r2:.2f})" if comm else "--"})
            md.append(markdown_table(rowsx, ["curve", "n", "own range", "n (common)",
                                             "4k-150k"]))
            p = plot_scaling(
                curves, out / fname, x=xkey, x_label=xlabel,
                hline=(BIGRAM_PTB, "bigram baseline 445.2"),
                title=None,
            )
            print(f"  wrote {p}")


# ------------------------------------------------------------------ width transfer --


def section_width(md: List[str], out: Path) -> None:
    md.append("\n## Width transfer (Experiment 4)\n")
    rows = load("width")
    if not rows:
        md.append("_grid empty_\n")
        return

    A = tagged(rows, grid="A")
    B = tagged(rows, grid="B")
    C = tagged(rows, grid="C")
    D = tagged(rows, grid="D")

    diag = ["diag_final.msg_over_unary", "diag_final.q_sharpness", "diag_final.label_entropy",
            "ablation_kl"]
    emit(md, "Grid A -- alpha*(d) at the source temperature",
         sorted(A, key=lambda r: (r["cell"]["d"], -r["cell"]["alpha_Z"])),
         ["cell.d", "cell.alpha_Z", "val_ppl", "val_ppl_final", "test_ppl", "train_ppl",
          "params.non_embedding"] + diag,
         "`temp_mode = fixed`, i.e. Wu & Tu's `lambda_H = 1/d`. 6,000 steps, seed 0.")
    if A:
        plot_heatmap(A, "cell.d", "cell.alpha_Z", "val_ppl", out / "fig_width_alpha.png",
                     x_label="label-set size $d$", y_label=r"label step size $\alpha_Z$",
                     v_label="validation perplexity")
        best_alpha = []
        for d in sorted({r["cell"]["d"] for r in A}):
            cells = [r for r in A if r["cell"]["d"] == d]
            b = min(cells, key=lambda r: r["val_ppl"])
            best_alpha.append({"d": d, "alpha*": b["cell"]["alpha_Z"], "val": b["val_ppl"],
                               "n_cells": len(cells)})
        md.append("\n**alpha\\*(d), the argmin over the grid row**\n")
        md.append(markdown_table(best_alpha, ["d", "alpha*", "val", "n_cells"]))

    emit(md, "Grid B -- the temperature rule, undamped (`alpha_Z = 1`)",
         sorted(B, key=lambda r: (r["cell"]["d"], r["cell"]["tags"]["temp"])),
         ["cell.d", "cell.tags.temp", "cell.qk_gain", "val_ppl", "test_ppl", "train_ppl"] + diag)
    if B:
        # `plot_heatmap` takes categorical axis values in first-seen order, and the rows arrive
        # in shard order, which is arbitrary. Sort so the axis reads as the ladder it is:
        # the source temperature first, then the query-normalised one, then the standardised
        # one by increasing gain.
        order = {"fixed": 0, "qnorm": 1, "qkn2": 2, "qkn4": 3, "qkn8": 4}
        both = sorted([r for r in A + B if r["cell"]["alpha_Z"] == 1.0],
                      key=lambda r: order.get(get_path(r, "cell.tags.temp"), 9))
        plot_heatmap(both, "cell.d", "cell.tags.temp", "val_ppl", out / "fig_width_temp.png",
                     x_label="label-set size $d$", y_label="attention temperature rule",
                     v_label="validation perplexity")
    emit(md, "Grid C -- temperature and damping together",
         sorted(C, key=lambda r: (r["cell"]["d"], r["cell"]["tags"]["temp"], -r["cell"]["alpha_Z"])),
         ["cell.d", "cell.tags.temp", "cell.alpha_Z", "val_ppl", "test_ppl"] + diag)
    emit(md, "Grid D -- channels", sorted(D, key=lambda r: (r["cell"]["h"], r["cell"]["tags"]["temp"])),
         ["cell.h", "cell.tags.temp", "val_ppl", "test_ppl"] + diag,
         "`d = 32`, `alpha_Z = 1`. The 2026-08 deconfound found channels degrade the model at "
         "every width; the message bound `|G| <= h max|T|` is linear in `h`.")


# ---------------------------------------------------------------------- the rest --


def section_mechanism(md: List[str], out: Path) -> None:
    """The mechanism figure: what a collapse looks like while it happens.

    Section 3.2 of the paper claims the head temperature is calibrated for the uniform-belief
    limit and mis-calibrated by up to a factor d away from it, and predicts a feedback loop
    whose gain rises with width. The evidence for that is not a final perplexity, it is the
    *joint trajectory* of the message scale and the belief sharpness within a run -- so the
    figure has to put both on one time axis, and pair a run that learns with one that does not
    at the same width.
    """
    md.append("\n## The collapse mechanism, within a run\n")
    rows = load("width") + load("gain")
    pick = []
    for want in (dict(d=32, temp_mode="fixed", alpha_Z=1.0),
                 dict(d=32, temp_mode="qknorm", qk_gain=1.0, alpha_Z=1.0),
                 dict(d=96, temp_mode="fixed", alpha_Z=1.0),
                 dict(d=96, temp_mode="qknorm", qk_gain=2.0, alpha_Z=1.0)):
        hits = [r for r in rows
                if all(get_path(r, f"cell.{k}") == v for k, v in want.items())
                and r.get("diag_trace")]
        if hits:
            pick.append(min(hits, key=lambda r: r["val_ppl"]))
    if not pick:
        md.append("_no traces yet_\n")
        return

    def lbl(r):
        c = r["cell"]
        t = "1/d" if c["temp_mode"] == "fixed" else f"standardised g={c['qk_gain']:g}"
        return f"d={c['d']}, {t}"

    plot_traces(pick, out / "fig_mechanism_msg.png", diag="msg_over_unary", label=lbl,
                diag_label="head message / word unary")
    plot_traces(pick, out / "fig_mechanism_sharp.png", diag="q_sharpness", label=lbl,
                diag_label=r"belief sharpness $\|q\|_2\sqrt{d}$")
    md.append(markdown_table(pick, ["cell.d", "cell.temp_mode", "cell.qk_gain", "val_ppl",
                                    "diag_final.msg_over_unary", "diag_final.q_sharpness",
                                    "ablation_kl"]))
    md.append("\nBelief sharpness is reported scale-free as `||q||_2 sqrt(d)`, which is 1 at the "
              "uniform belief and sqrt(d) at a one-hot -- i.e. it is exactly the factor by which "
              "the effective attention temperature drifts from the value `lambda_H = 1/d` was "
              "calibrated for.\n")


def section_gain(md: List[str], out: Path) -> None:
    md.append("\n## The standardised-attention gain, three seeds (Experiment 4b)\n")
    rows = load("gain")
    if rows:
        agg = []
        for d in sorted({get_path(r, "cell.d") for r in rows}):
            for g in sorted({get_path(r, "cell.qk_gain") for r in rows}):
                cells = [r for r in rows if r["cell"]["d"] == d and r["cell"]["qk_gain"] == g]
                if not cells:
                    continue
                v = sorted(r["val_ppl"] for r in cells)
                mean = sum(v) / len(v)
                sd = (sum((x - mean) ** 2 for x in v) / (len(v) - 1)) ** 0.5 if len(v) > 1 else 0.0
                agg.append({"d": d, "qk_gain": g, "n": len(v), "mean": round(mean, 1),
                            "sd": round(sd, 1), "min": round(min(v), 1), "max": round(max(v), 1),
                            "dead": sum(1 for x in v if x > 0.95 * UNIGRAM_PTB)})
        md.append(markdown_table(agg, ["d", "qk_gain", "n", "mean", "sd", "min", "max", "dead"]))
        md.append("\n`dead` counts seeds that finished above 95% of the unigram baseline, i.e. "
                  "that never learned; a mean over a mix of learned and dead runs is not a "
                  "measure of anything, so the count is reported beside it.\n")
        plot_heatmap(rows, "cell.d", "cell.qk_gain", "val_ppl", out / "fig_gain.png",
                     agg="mean", x_label="label-set size $d$",
                     y_label="standardised-attention gain", v_label="validation perplexity")
    else:
        md.append("_grid empty_\n")

    md.append("\n## Learning-rate transfer across width (Experiment 4c)\n")
    rows = load("lr_transfer")
    if not rows:
        md.append("_grid empty_\n")
        return
    for arm in ("fixed", "qkn2"):
        sub = [r for r in rows if get_path(r, "cell.tags.arm") == arm]
        emit(md, f"arm `{arm}`", sorted(sub, key=lambda r: (r["cell"]["d"], r["cell"]["lr"])),
             ["cell.d", "cell.lr", "val_ppl", "test_ppl", "train_ppl",
              "diag_final.msg_over_unary", "diag_final.q_sharpness", "ablation_kl"])
        if sub:
            best = []
            for d in sorted({r["cell"]["d"] for r in sub}):
                cells = [r for r in sub if r["cell"]["d"] == d]
                b = min(cells, key=lambda r: r["val_ppl"])
                best.append({"d": d, "argmin lr": b["cell"]["lr"], "val": round(b["val_ppl"], 1)})
            md.append(f"\n**argmin over lr, arm `{arm}`** -- the quantity that transfers or "
                      f"does not\n")
            md.append(markdown_table(best, ["d", "argmin lr", "val"]))
            plot_heatmap(sub, "cell.d", "cell.lr", "val_ppl", out / f"fig_lr_{arm}.png",
                         x_label="label-set size $d$", y_label="learning rate",
                         v_label="validation perplexity")


def section_rank(md: List[str], out: Path) -> None:
    md.append("\n## The Kruskal rank (Experiment 4d)\n")
    rows = load("rank")
    if not rows:
        md.append("_grid empty_\n")
        return
    agg = []
    for d in sorted({r["cell"]["d"] for r in rows}):
        for rk in sorted({r["cell"]["rank"] for r in rows if r["cell"]["d"] == d}):
            cells = [r for r in rows if r["cell"]["d"] == d and r["cell"]["rank"] == rk]
            v = [r["val_ppl"] for r in cells]
            mean = sum(v) / len(v)
            sd = (sum((x - mean) ** 2 for x in v) / (len(v) - 1)) ** 0.5 if len(v) > 1 else 0.0
            agg.append({"d": d, "rank": rk, "non-emb": cells[0]["params"]["non_embedding"],
                        "total": cells[0]["params"]["total"], "n": len(v),
                        "val mean": round(mean, 1), "sd": round(sd, 1)})
    md.append(markdown_table(agg, ["d", "rank", "non-emb", "total", "n", "val mean", "sd"]))
    md.append("\nThe arc factors cost `2 K h d r`; every experiment before this one ran at "
              "`r = d`, the most expensive setting of the decomposition. A rank that costs no "
              "perplexity moves the whole PT curve left on the parameter axis.\n")


def section_init(md: List[str], out: Path) -> None:
    md.append("\n## Initialisation and the global head (Experiment 5)\n")
    rows = load("init")
    if not rows:
        md.append("_grid empty_\n")
        return
    gd = ["diag_final.glob_over_unary", "diag_final.qg_entropy_frac",
          "diag_final.max_abs_B_glob", "diag_final.msg_over_unary"]
    A = tagged(rows, grid="A")
    emit(md, "Grid A -- global head: initialisation scale x L2",
         sorted(A, key=lambda r: (r["cell"]["d"], r["cell"]["b_glob_init_std"] or -1,
                                  r["cell"]["regularise_global_head"])),
         ["cell.d", "cell.n_global", "cell.b_glob_init_std", "cell.regularise_global_head",
          "val_ppl", "test_ppl", "train_ppl"] + gd,
         "Prof. Tu: *\"Uniform initialisation (of what? B?) does not sound like a good choice. "
         "How does it work with random initialisation?\"* Every factor here is Gaussian; what "
         "the 2026-08 report called uniform was the *treatment*, not the distribution.")
    if A:
        plot_heatmap([r for r in A if r["cell"]["n_global"] > 0],
                     "cell.b_glob_init_std", "cell.regularise_global_head", "val_ppl",
                     out / "fig_init_globalhead.png",
                     x_label="initialisation std of $B'$", y_label="$B'$ in the $L_2$ term",
                     v_label="validation perplexity")
    for g, title in (("B", "Grid B -- number of global features m"),
                     ("C", "Grid C -- the initialisation distribution, taken literally"),
                     ("D", "Grid D -- overall initialisation scale"),
                     ("E", "Grid E -- a width rule for the arc-score scale")):
        sub = tagged(rows, grid=g)
        emit(md, title, sub,
             ["cell.name", "cell.d", "cell.init_dist", "cell.init_std", "cell.arc_init_std",
              "cell.n_global", "val_ppl", "test_ppl", "train_ppl"] + gd)


def section_globalhead(md: List[str], out: Path) -> None:
    md.append("\n## The B.3.3 global head, three seeds in the stable configuration (Exp. 5b)\n")
    rows = load("globalhead")
    if not rows:
        md.append("_grid empty_\n")
        return
    agg = []
    for r in rows:
        c = r["cell"]
        key = (c["d"], c["n_global"], c["b_glob_init_std"], c["regularise_global_head"],
               c["init_dist"])
        agg.append((key, r))
    seen = {}
    for key, r in agg:
        seen.setdefault(key, []).append(r)
    table = []
    for key in sorted(seen, key=lambda k: (k[0], k[1], k[2] or -1, str(k[3]), k[4])):
        cells = seen[key]
        v = [r["val_ppl"] for r in cells]
        m = sum(v) / len(v)
        sd = (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5 if len(v) > 1 else 0.0
        d, m_glob, std, reg, dist = key
        table.append({"d": d, "m": m_glob, "init std": std, "in L2": reg, "init dist": dist,
                      "n": len(v), "val mean": round(m, 1), "sd": round(sd, 1),
                      "H(Q_g)/max": round(sum(
                          get_path(r, "diag_final.qg_entropy_frac") or 0 for r in cells)
                          / len(cells), 3),
                      "glob/unary": round(sum(
                          get_path(r, "diag_final.glob_over_unary") or 0 for r in cells)
                          / len(cells), 3)})
    md.append(markdown_table(table, ["d", "m", "init std", "in L2", "init dist", "n",
                                     "val mean", "sd", "H(Q_g)/max", "glob/unary"]))
    md.append("\n`H(Q_g)/max` at 1.000 means the head posterior is uniform, i.e. the head is "
              "contributing a constant label bias rather than a feed-forward-like operator. "
              "That distinction is the whole question and it is not visible in perplexity.\n")


def section_open(md: List[str], out: Path) -> None:
    md.append("\n## Open questions closed inside the working region (Experiment 3)\n")
    rows = load("open")
    if not rows:
        md.append("_grid empty_\n")
        return
    common = ["cell.name", "val_ppl", "val_ppl_final", "test_ppl", "train_ppl",
              "diag_final.msg_over_unary", "ablation_kl", "seconds"]
    R = tagged(rows, grid="R")
    if R:
        md.append("\n### Grid R -- replication over three seeds\n")
        agg = []
        for what in sorted({get_path(r, "cell.tags.what") for r in R}):
            cells = [r for r in R if get_path(r, "cell.tags.what") == what]
            st = aggregate_seeds(cells, [], metric="val_ppl")
            st = st[0] if isinstance(st, list) else st
            agg.append({"what": what, "val (mean +- sd)": getattr(st, "format", lambda **k: str(st))(),
                        "n": getattr(st, "n", len(cells)),
                        "seeds": ",".join(str(get_path(c, "cell.seed")) for c in cells)})
        md.append(markdown_table(agg, ["what", "val (mean +- sd)", "n", "seeds"]))
    for g, title in (("S", "Grid S -- budget vs. schedule"),
                     ("T", "Grid T -- inference depth T and tau"),
                     ("G", "Grid G -- the relative positional encoding"),
                     ("X", "Grid X -- the exact readout, trained"),
                     ("L", "Grid L -- the L2 coefficient on the arc scores")):
        emit(md, title, tagged(rows, grid=g), common)

    swaps = [r for r in rows if "swap_val_ppl" in r]
    emit(md, "Readout swap: the same trained weights scored under the other readout",
         swaps, ["cell.name", "cell.readout", "val_ppl", "swap_readout", "swap_val_ppl",
                 "test_ppl", "swap_test_ppl"],
         "Only meaningful at `lambda_W = 1`; measured for every PT cell at the cost of one "
         "extra pass.")


def section_repro(md: List[str], out: Path) -> None:
    md.append("\n## Reproduction check\n")
    rows = load("repro")
    if not rows:
        md.append("_not run yet_\n")
        return
    emit(md, "The 2026-08 record configuration over three seeds, damped and undamped", rows,
         ["cell.name", "cell.alpha_Z", "cell.seed", "val_ppl", "val_ppl_final", "test_ppl",
          "diag_final.msg_over_unary", "best_step"],
         "Job 940848 of 2026-08-10 recorded **430.04** for the undamped cell. What matters "
         "here is the *spread*, not the mean.")
    for a in (1.0, 0.25):
        cells = [r for r in rows if get_path(r, "cell.alpha_Z") == a]
        if len(cells) > 1:
            vals = [r["val_ppl"] for r in cells]
            mean = sum(vals) / len(vals)
            sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1))
            md.append(f"\n`alpha_Z = {a}`: **{mean:.2f} +- {sd:.2f}** over n={len(vals)} "
                      f"(min {min(vals):.2f}, max {max(vals):.2f})\n")
    if rows:
        plot_traces(rows, out / "fig_repro_traces.png",
                    label=lambda r: f"alpha_Z={get_path(r,'cell.alpha_Z')} seed {get_path(r,'cell.seed')}",
                    title=None)


def section_switches(md: List[str], out: Path) -> None:
    """The ladder, reported as deltas against the anchors rather than as raw perplexities.

    "The transformer with the feed-forward block removed reaches 144.4" is not the finding; "the
    feed-forward block is worth 15.1 perplexity in an otherwise ordinary decoder, and 0 once the
    other nine differences are in place" is. So every row carries its distance from the anchor
    of its own family, and the two families are reported side by side: a switch whose S1 cost is
    small while its S2 benefit is large only matters in combination, which is the interesting
    case and the one a single ladder hides.
    """
    md.append("\n## The switch ladder (Experiment 6)\n")
    rows = load("switches")
    if not rows:
        md.append("_grid empty_\n")
        return
    best = best_by(rows, ["cell.tags.family", "cell.tags.switch", "cell.tags.rung",
                          "cell.tags.which"], metric="val_ppl")

    def anchor(which):
        hit = [r for r in best if get_path(r, "cell.tags.which") == which]
        return min(hit, key=lambda r: r["val_ppl"]) if hit else None

    a_tr, a_pt = anchor("transformer"), anchor("pt")
    md.append(
        "\n| anchor | val | test | non-emb | best lr |\n|---|---|---|---|---|\n"
        + "".join(
            f"| {n} | {r['val_ppl']:.1f} | {r['test_ppl']:.1f} | "
            f"{r['params']['non_embedding']:,} | {r['cell']['lr']:g} |\n"
            for n, r in (("all-transformer", a_tr), ("all-PT", a_pt)) if r
        )
    )
    if a_tr and a_pt:
        md.append(f"\nThe ladder spans **{a_pt['val_ppl'] - a_tr['val_ppl']:.1f} perplexity**; "
                  "every delta below is a share of that.\n")

    for fam, ref, sign, title in (
        ("S1", a_tr, +1, "S1 -- cost of one PT property in an otherwise ordinary decoder"),
        ("S2", a_pt, -1, "S2 -- benefit of restoring one transformer property inside PT"),
    ):
        sub = [r for r in best if get_path(r, "cell.tags.family") == fam]
        if not sub or ref is None:
            continue
        table = []
        for r in sorted(sub, key=lambda r: r["val_ppl"]):
            table.append({
                "switch": get_path(r, "cell.tags.switch"),
                "val": round(r["val_ppl"], 1),
                "delta vs anchor": round(sign * (r["val_ppl"] - ref["val_ppl"]), 1),
                "best lr": r["cell"]["lr"],
                "non-emb": r["params"]["non_embedding"],
            })
        emit(md, title, table, ["switch", "val", "delta vs anchor", "best lr", "non-emb"])

    L = [r for r in best if get_path(r, "cell.tags.family") == "L"]
    if L:
        L.sort(key=lambda r: get_path(r, "cell.tags.rung"))
        table = [{"rung": get_path(r, "cell.tags.rung"),
                  "switch flipped": get_path(r, "cell.tags.switch"),
                  "val": round(r["val_ppl"], 1),
                  "step": round(r["val_ppl"] - (L[i - 1]["val_ppl"] if i else
                                                (a_tr["val_ppl"] if a_tr else r["val_ppl"])), 1),
                  "best lr": r["cell"]["lr"],
                  "non-emb": r["params"]["non_embedding"]}
                 for i, r in enumerate(L)]
        emit(md, "L -- the cumulative ladder, transformer to PT",
             table, ["rung", "switch flipped", "val", "step", "best lr", "non-emb"],
             "`step` is the change from the previous rung, so the column locates the cliff.")
        plot_scaling([("cumulative ladder",
                       [(get_path(r, "cell.tags.rung"), r["val_ppl"]) for r in L])],
                     out / "fig_switch_ladder.png",
                     x_label="switches flipped towards the causal PT (count)",
                     frontier=False, fit=False)

    if a_pt:
        md.append("\n**Endpoint residue.** The all-PT rung is an approximation of the real "
                  "decoder (see this experiment's status file for the four named gaps). The "
                  "difference between it and the causal PT at the same width, corpus, loop and "
                  "budget is the size of everything the ladder cannot express, and it bounds "
                  "how much weight the attributions above can carry.\n")


def section_factored(md: List[str], out: Path) -> None:
    md.append("\n## Factored labels (Experiment 7)\n")
    rows = load("factored")
    if not rows:
        md.append("_grid empty_\n")
        return
    emit(md, "K label components at fixed total width", rows,
         ["cell.d", "cell.tags.K", "cell.readout", "val_ppl", "test_ppl", "train_ppl",
          "params.non_embedding", "params.total", "swap_readout", "swap_val_ppl"],
         "The parameter count is independent of K at fixed total width; the prediction is that "
         "factoring buys nothing under the mean-field readout and something under the exact one.")


def section_transfer(md: List[str], out: Path) -> None:
    md.append("\n## WikiText-2 transfer (Experiment 8)\n")
    rows = load("wt2")
    if not rows:
        md.append("_not run yet_\n")
        return
    agg = []
    for lad in sorted({get_path(r, "cell.tags.ladder") for r in rows}):
        sub = [r for r in rows if get_path(r, "cell.tags.ladder") == lad]
        for key in sorted({(r["cell"]["d"], r["cell"]["n_embd"]) for r in sub}):
            cells = [r for r in sub if (r["cell"]["d"], r["cell"]["n_embd"]) == key]
            v = [r["val_ppl"] for r in cells]
            m = sum(v) / len(v)
            sd = (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5 if len(v) > 1 else 0.0
            t = [r["test_ppl"] for r in cells]
            agg.append({"ladder": lad,
                        "config": (f"d={key[0]}" if cells[0]["cell"]["model"] == "pt"
                                   else f"n_embd={key[1]}"),
                        "n": len(v), "val": round(m, 1), "sd": round(sd, 1),
                        "test": round(sum(t) / len(t), 1),
                        "non-emb": cells[0]["params"]["non_embedding"],
                        "total": cells[0]["params"]["total"]})
    md.append(markdown_table(agg, ["ladder", "config", "n", "val", "sd", "test",
                                   "non-emb", "total"]))
    md.append("\nWikiText-2 unigram baseline: **964.8**. Every hyperparameter is the one "
              "selected on PTB; nothing was retuned, which is what makes this a transfer test "
              "rather than a second experiment.\n")


def _fmt_pm(vals: Sequence[float]) -> str:
    if not vals:
        return "---"
    m = sum(vals) / len(vals)
    if len(vals) == 1:
        return f"{m:.1f}"
    sd = (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
    return f"{m:.1f} $\\pm$ {sd:.1f}"


def emit_main_table(out: Path) -> None:
    """Write the paper's main results table as LaTeX, so no number is typed by hand.

    The table is the one place where the paper commits to a matched-budget comparison, so how
    the match is made has to be mechanical and stated: for each causal-PT configuration we
    report, the baselines shown are the ones whose *total* parameter count is nearest to it
    from below and from above. That is the bracket Penghao Kuang proposed as the fallback when
    a full PT curve is unaffordable -- "represent the PT as a single point and the Transformer
    results as a line segment" -- and it is honest in a way a single nearest-neighbour is not,
    because it shows the reader the interval the comparison lives in rather than a point that
    could have been chosen either side.
    """
    base = load("baselines")
    pt = load("pt_scaling")
    low = load("pt_lowrank")
    if not base:
        return
    width = [r for r in base if get_path(r, "cell.tags.family") == "width"]
    bw = best_by(width, ["cell.model", "cell.n_embd"], metric="val_ppl")

    lines = [
        "% Generated by `python -m experiments.make_report`. Do not edit by hand.",
        "\\begin{tabular}{llrrrr}", "\\toprule",
        "Model & Config & Emb. & Non-emb. & Val ppl & Test ppl \\\\", "\\midrule",
        "Unigram & --- & --- & --- & 688.8 & --- \\\\",
    ]

    def row(label, cfg, rows):
        if not rows:
            return
        p0 = rows[0]["params"]
        lines.append(
            f"{label} & {cfg} & {p0['embedding']:,} & {p0['non_embedding']:,} & "
            f"{_fmt_pm([r['val_ppl'] for r in rows])} & "
            f"{_fmt_pm([r['test_ppl'] for r in rows])} \\\\".replace(",", "{,}")
        )

    if pt:
        by_d: Dict[int, List[Dict[str, Any]]] = {}
        for r in pt:
            if get_path(r, "cell.tags.ladder") == "rule":
                by_d.setdefault(r["cell"]["d"], []).append(r)
        for d in sorted(by_d):
            cells = by_d[d]
            row(f"Causal PT", f"$d={d}$", cells)
            tot = cells[0]["params"]["total"]
            below = [r for r in bw if r["params"]["total"] <= tot]
            above = [r for r in bw if r["params"]["total"] > tot]
            for model in ("gpt", "looped"):
                name = "Transformer" if model == "gpt" else "Looped"
                lo = max([r for r in below if r["cell"]["model"] == model],
                         key=lambda r: r["params"]["total"], default=None)
                hi = min([r for r in above if r["cell"]["model"] == model],
                         key=lambda r: r["params"]["total"], default=None)
                for r in (lo, hi):
                    if r is not None:
                        row(f"\\quad {name}", f"$d_{{\\mathrm{{model}}}}={r['cell']['n_embd']}$", [r])
    else:
        for model, name in (("gpt", "Transformer"), ("looped", "Looped")):
            for r in sorted([x for x in bw if x["cell"]["model"] == model],
                            key=lambda x: x["params"]["total"]):
                row(name, f"$d_{{\\mathrm{{model}}}}={r['cell']['n_embd']}$", [r])

    lines += ["\\bottomrule", "\\end{tabular}"]
    (out / "table_main.tex").write_text("\n".join(lines) + "\n")
    print(f"  wrote {out / 'table_main.tex'}")


def _latex_table(path: Path, header: Sequence[str], rows: Sequence[Sequence[Any]],
                 align: str) -> None:
    """Write a booktabs tabular. Generated, so the appendix cannot drift from the JSONs."""
    def cell(v):
        if isinstance(v, bool):
            return "yes" if v else "no"
        if isinstance(v, float):
            return f"{v:.1f}"
        if isinstance(v, int):
            return f"{v:,}".replace(",", "{,}")
        return str(v)
    lines = ["% Generated by `python -m experiments.make_report`. Do not edit by hand.",
             "\\begin{tabular}{%s}" % align, "\\toprule",
             " & ".join(header) + r" \\", "\\midrule"]
    lines += [" & ".join(cell(v) for v in r) + r" \\" for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")
    print(f"  wrote {path}")


def _placeholder(path: Path, note: str) -> None:
    """A grid that has not finished still has to produce a compilable table.

    A missing ``\\input`` is a fatal LaTeX error, so an unfinished grid would break the build of
    a document whose other twenty pages are ready. A visible placeholder keeps the document
    compiling and makes the gap obvious in the PDF, which is the right failure mode: the reader
    sees "not yet run" rather than a silently absent table.
    """
    body = "\\textit{" + note + "} \\\\"
    path.write_text(
        "% Generated placeholder: this grid has not finished.\n"
        "\\begin{tabular}{l}\n\\toprule\n"
        + body + "\n\\bottomrule\n\\end{tabular}\n"
    )
    print(f"  placeholder {path.name}: {note}")


def _mean_sd(vals):
    if not vals:
        return 0.0, 0.0
    m = sum(vals) / len(vals)
    sd = (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5 if len(vals) > 1 else 0.0
    return m, sd


def _pm(vals) -> str:
    m, sd = _mean_sd(vals)
    return f"{m:.1f} $\\pm$ {sd:.1f}"


def _loginterp(table, x):
    """Log-log interpolation of a baseline curve at a parameter count.

    The baselines were run at their own widths, so almost never at exactly a causal PT's
    parameter count. Interpolating the baseline curve in log-log space -- where these curves are
    close to straight -- is what lets the comparison be made at the PT's budget rather than at
    whichever baseline point happens to be nearest, which would silently favour one side.
    Returns None outside the measured range: extrapolating a baseline into a region where it was
    not run would be inventing the comparison.
    """
    import bisect, math
    xs = [a for a, _ in table]
    i = bisect.bisect_left(xs, x)
    if i == 0 or i >= len(xs):
        return None
    (x0, y0), (x1, y1) = table[i - 1], table[i]
    t = (math.log(x) - math.log(x0)) / (math.log(x1) - math.log(x0))
    return math.exp(math.log(y0) + t * (math.log(y1) - math.log(y0)))


def emit_matched_budget(out: Path, md: List[str]) -> None:
    """The matched-budget comparison, at each causal PT point, against both baselines."""
    base = load("baselines")
    low = load("pt_lowrank")
    if not base or not low:
        _placeholder(out / "table_matched.tex", "matched-budget comparison not yet complete")
        return
    bw = best_by([r for r in base if get_path(r, "cell.tags.family") == "width"],
                 ["cell.model", "cell.n_embd"], metric="val_ppl")
    curves = {m: sorted((r["params"]["non_embedding"], r["val_ppl"])
                        for r in bw if r["cell"]["model"] == m)
              for m in ("gpt", "looped")}
    pts = best_by(low, ["cell.d", "cell.rank"], metric="val_ppl")
    table, mdrows = [], []
    for r in sorted(pts, key=lambda r: r["params"]["non_embedding"]):
        n, v = r["params"]["non_embedding"], r["val_ppl"]
        g, l = _loginterp(curves["gpt"], n), _loginterp(curves["looped"], n)
        table.append([f"$\\nlab={r['cell']['d']}$, $\\krank={r['cell']['rank']}$", n,
                      f"{v:.1f}",
                      f"{g:.1f}" if g else "---", f"{v / g:.2f}" if g else "---",
                      f"{l:.1f}" if l else "---", f"{v / l:.2f}" if l else "---"])
        mdrows.append({"config": f"d={r['cell']['d']}, r={r['cell']['rank']}", "non-emb": n,
                       "PT": round(v, 1),
                       "transformer": round(g, 1) if g else None,
                       "ratio": round(v / g, 2) if g else None,
                       "looped": round(l, 1) if l else None,
                       "ratio ": round(v / l, 2) if l else None})
    _latex_table(out / "table_matched.tex",
                 ["causal PT", "non-emb.", "val ppl", "transformer", "ratio", "looped", "ratio"],
                 table, "lrrrrrr")
    md.append("\n## Matched-budget comparison\n")
    md.append(markdown_table(mdrows, ["config", "non-emb", "PT", "transformer", "ratio",
                                      "looped", "ratio "]))
    md.append("\nBaselines are interpolated in log-log space at the causal PT's own parameter "
              "count, and left blank outside the range where they were measured -- "
              "extrapolating a baseline into a region it was not run in would be inventing the "
              "comparison.\n")


def emit_grid_tables(out: Path) -> None:
    """Emit the per-grid detail tables that Appendix G inputs.

    Section 4 carries one paragraph and one headline number per finding; the grid behind each
    lives in the appendix. Both come from here, so trimming the main text for length can never
    separate a claim from the numbers that support it.
    """
    rows = load("gain")
    if not rows:
        _placeholder(out / "table_gain.tex", "gain sweep not yet complete")
    else:
        table = []
        for d in sorted({r["cell"]["d"] for r in rows}):
            for g in sorted({r["cell"]["qk_gain"] for r in rows}):
                c = [r for r in rows if r["cell"]["d"] == d and r["cell"]["qk_gain"] == g]
                if not c:
                    continue
                dead = sum(1 for r in c if r["val_ppl"] > 0.95 * UNIGRAM_PTB)
                table.append([d, g, len(c), _pm([r["val_ppl"] for r in c]), dead])
        _latex_table(out / "table_gain.tex",
                     ["$d$", "gain", "$n$", "val ppl", "collapsed"], table, "rrrlr")

    rows = load("lr_transfer")
    if not rows:
        _placeholder(out / "table_lr.tex", "learning-rate transfer grid not yet complete")
    else:
        lrs = sorted({r["cell"]["lr"] for r in rows})
        table = []
        for d in sorted({r["cell"]["d"] for r in rows}):
            for arm, label in (("fixed", "source"), ("qkn2", "standardised")):
                sub = [r for r in rows if r["cell"]["d"] == d
                       and get_path(r, "cell.tags.arm") == arm]
                if not sub:
                    continue
                vals = []
                for lr in lrs:
                    c = [r for r in sub if r["cell"]["lr"] == lr]
                    vals.append(f"{c[0]['val_ppl']:.0f}" if c else "---")
                best = min(sub, key=lambda r: r["val_ppl"])
                table.append([d, label] + vals + [f"{best['cell']['lr']:g}"])
        _latex_table(out / "table_lr.tex",
                     ["$d$", "temperature"] + [f"{lr:g}" for lr in lrs] + ["argmin"],
                     table, "rl" + "r" * len(lrs) + "r")

    rows = load("rank")
    if not rows:
        _placeholder(out / "table_rank.tex", "rank grid not yet complete")
    else:
        table = []
        for d in sorted({r["cell"]["d"] for r in rows}):
            for rk in sorted({r["cell"]["rank"] for r in rows if r["cell"]["d"] == d}):
                c = [r for r in rows if r["cell"]["d"] == d and r["cell"]["rank"] == rk]
                table.append([d, rk, c[0]["params"]["non_embedding"], len(c),
                              _pm([r["val_ppl"] for r in c])])
        _latex_table(out / "table_rank.tex",
                     ["$d$", "$r$", "non-emb.", "$n$", "val ppl"], table, "rrrrl")

    rows = load("switches")
    if not rows:
        _placeholder(out / "table_switches.tex", "switch ladder not yet complete")
    else:
        best = best_by(rows, ["cell.tags.family", "cell.tags.switch", "cell.tags.rung",
                              "cell.tags.which"], metric="val_ppl")
        anc = {}
        for w in ("transformer", "pt"):
            hit = [r for r in best if get_path(r, "cell.tags.which") == w]
            anc[w] = min(hit, key=lambda r: r["val_ppl"]) if hit else None
        names = sorted(n for n in {get_path(r, "cell.tags.switch") for r in best} if n)
        table = []
        for name in names:
            row = [name.replace("_", r"\_")]
            for fam, ref, sign in (("S1", anc["transformer"], +1), ("S2", anc["pt"], -1)):
                hit = [r for r in best if get_path(r, "cell.tags.family") == fam
                       and get_path(r, "cell.tags.switch") == name]
                if hit and ref is not None:
                    row += [f"{hit[0]['val_ppl']:.1f}",
                            f"{sign * (hit[0]['val_ppl'] - ref['val_ppl']):+.1f}"]
                else:
                    row += ["---", "---"]
            table.append(row)
        _latex_table(out / "table_switches.tex",
                     ["switch", "S1 val", r"$\Delta$", "S2 val", r"$\Delta$"], table, "lrrrr")

    rows = load("globalhead")
    if not rows:
        _placeholder(out / "table_globalhead.tex", "global-head grid not yet complete")
    else:
        seen = {}
        for r in rows:
            c = r["cell"]
            seen.setdefault((c["d"], c["n_global"], c["b_glob_init_std"],
                             c["regularise_global_head"], c["init_dist"]), []).append(r)
        table = []
        for k in sorted(seen, key=lambda k: (k[0], k[1], k[2] if k[2] is not None else -1,
                                             str(k[3]), k[4])):
            c = seen[k]
            hq, _ = _mean_sd([get_path(r, "diag_final.qg_entropy_frac") or 0.0 for r in c])
            table.append([k[0], k[1], "---" if k[2] is None else f"{k[2]:g}",
                          "yes" if k[3] else "no", k[4], len(c),
                          _pm([r["val_ppl"] for r in c]), f"{hq:.3f}"])
        _latex_table(out / "table_globalhead.tex",
                     ["$d$", "$m$", "init sd", "in $L_2$", "dist.", "$n$", "val ppl",
                      r"$H(Q_g)/\max$"], table, "rrllrrlr")

    rows = load("wt2")
    if not rows:
        _placeholder(out / "table_wt2.tex", "WikiText-2 transfer not yet run")
    else:
        seen = {}
        for r in rows:
            c = r["cell"]
            seen.setdefault((get_path(r, "cell.tags.ladder"), c["d"], c["n_embd"],
                             c["model"]), []).append(r)
        table = []
        for k in sorted(seen, key=lambda k: (str(k[0]), k[1], k[2])):
            c = seen[k]
            cfg = f"$d={k[1]}$" if k[3] == "pt" else "$d_{\\mathrm{model}}=%d$" % k[2]
            table.append([str(k[0]), cfg, c[0]["params"]["non_embedding"], len(c),
                          _pm([r["val_ppl"] for r in c]),
                          _pm([r["test_ppl"] for r in c])])
        _latex_table(out / "table_wt2.tex",
                     ["arm", "config", "non-emb.", "$n$", "val ppl", "test ppl"],
                     table, "llrrll")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=str(ARTICLE))
    p.add_argument("--tables", default=str(REPO / "RESULTS.md"))
    p.add_argument("--tables-only", action="store_true")
    a = p.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    md: List[str] = [
        "# Results\n",
        "Generated by `python -m experiments.make_report`. Every number here comes from a "
        "committed run JSON under `experiments/`; nothing is typed by hand. Re-run after any "
        "new shard lands.\n",
    ]

    for fn in (section_repro, section_scaling, section_width, section_mechanism, section_gain, section_rank, section_init, section_globalhead, section_open,
               section_switches, section_factored, section_transfer):
        try:
            fn(md, out)
        except Exception as exc:  # a missing grid must not stop the rest of the report
            import traceback
            traceback.print_exc()
            md.append(f"\n_section {fn.__name__} failed: {type(exc).__name__}: {exc}_\n")

    try:
        emit_main_table(out)
        emit_grid_tables(out)
        emit_matched_budget(out, md)
    except Exception:
        import traceback
        traceback.print_exc()
    Path(a.tables).write_text("\n".join(md))
    print(f"\nwrote {a.tables}")


if __name__ == "__main__":
    main()
