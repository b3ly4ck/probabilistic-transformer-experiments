"""What do the label variables and the head posteriors actually represent?

The whole claim of this model class is that its latent variables mean something --- labels are
syntactic categories, head posteriors are dependency attachments --- and a paper that reports
only perplexity has offered no evidence for the one property that distinguishes it from a
transformer. Wu & Tu devote their Appendix F to exactly this, and report honestly that the
structures their encoder induces are only partially consistent with human intuition. This is the
decoder's version.

Three probes, all read off a trained checkpoint with no further training:

1. **What each label is.** The word--label factor `S` is the only place a word touches the
   graph, so the words with the largest `S[w, a]` are the vocabulary items label `a` explains.
   Ranking them raw is dominated by whichever words simply have large rows, so we report the
   *contrastive* score `S[w,a] - max_{a' != a} S[w,a']`: words this label claims against the
   competition. Frequency is reported alongside, because a label whose top words are all
   hapaxes has found nothing.

2. **Where the attention goes.** The head posterior over `D_t = {ROOT, 1..t-1}` is the model's
   dependency attachment. We report its mass on ROOT, its mean attachment distance, and the
   whole distance profile. A model that has learned nothing about word order attends uniformly
   over its prefix; one that has learned locality does not.

3. **Prior against posterior on the same slot.** Part IV's contraction bound says the divergence
   between the predictive and observed beliefs at a slot should be small on predictable tokens
   and grow with surprise --- "the embedding-space surprisal of the observed word". That is a
   falsifiable prediction about a correlation, and it is cheap: score both beliefs at every slot
   of a few batches and correlate their KL with the token's own negative log-likelihood.

Usage::

    python -m experiments.interpretability checkpoints/RECseed1_d32_pt.pt --batches 8
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List

import torch

from src import CausalPTDecoder, PTConfig
from src.data import load_ptb, sequential_batches


def load_checkpoint(path: str, device: str = "cpu"):
    """Rebuild the model from a checkpoint, tolerating configs older than the current one.

    A checkpoint pickles the `PTConfig` *instance* that produced it, so one written before a
    field existed unpickles without that field and every later access raises. Rebuilding a
    fresh config from the fields that are present, and letting the dataclass supply defaults
    for the rest, is what makes August's checkpoints readable today.
    """
    ck = torch.load(path, map_location=device, weights_only=False)
    old = ck["cfg"]
    known = {f.name for f in fields(PTConfig)}
    kw = {k: v for k, v in vars(old).items() if k in known}
    cfg = PTConfig(**kw)
    model = CausalPTDecoder(cfg)
    model.load_state_dict(ck["state_dict"])
    model.eval().to(device)
    return model, cfg


@torch.no_grad()
def label_lexicon(model, corpus, top: int = 12) -> List[Dict[str, Any]]:
    """The words each label claims, by contrastive score."""
    S = model.S.detach()
    counts = torch.bincount(corpus.train, minlength=corpus.vocab_size).double()
    freq = counts / counts.sum()
    out = []
    for a in range(S.shape[1]):
        others = torch.cat([S[:, :a], S[:, a + 1:]], dim=1)
        contrast = S[:, a] - (others.max(dim=1).values if others.numel() else 0.0)
        idx = contrast.argsort(descending=True)[:top]
        out.append({
            "label": a,
            "words": [corpus.itos[int(i)] for i in idx],
            "contrast": [round(float(contrast[i]), 2) for i in idx],
            "median_log10_freq": round(float(
                torch.log10(freq[idx].clamp_min(1e-12)).median()), 2),
        })
    return out


@torch.no_grad()
def attention_profile(model, corpus, block: int, batch: int, n_batches: int,
                      device: str) -> Dict[str, Any]:
    """Where the head posterior puts its mass, against what uniform would give."""
    root, dist_hist, tot, uni_gap = 0.0, torch.zeros(block), 0, 0.0
    n = 0
    for k, blk in enumerate(sequential_batches(corpus.valid, batch, block, device=device)):
        if k >= n_batches:
            break
        qbar = model.content_stream(blk)
        _, alpha = model._arc_message(qbar, model.contract(qbar))
        B, h, T, D = alpha.shape
        root += float(alpha[..., 0].mean())
        # attachment distance i - j for the positional columns
        for i in range(1, T):
            a = alpha[:, :, i, 1:i + 1]                      # (B, h, i)
            d = torch.arange(i, 0, -1, device=a.device).float()
            dist_hist[: i] += (a.sum((0, 1)).flip(0) * 1.0).cpu()[:i]
            uni_gap += float((a.max(-1).values - 1.0 / (i + 1)).mean())
            tot += 1
        n += 1
    prof = dist_hist / dist_hist.sum().clamp(min=1e-9)
    mean_dist = float((prof * torch.arange(1, block + 1).float()).sum())
    return {
        "root_mass": round(root / max(n, 1), 4),
        "mean_attachment_distance": round(mean_dist, 2),
        "peakedness_over_uniform": round(uni_gap / max(tot, 1), 4),
        "distance_profile_first10": [round(float(x), 4) for x in prof[:10]],
    }


@torch.no_grad()
def prior_posterior(model, corpus, block: int, batch: int, n_batches: int,
                    device: str) -> Dict[str, Any]:
    """Does the prior-to-posterior divergence at a slot grow with the token's surprisal?

    Part IV's Lemma bounds the divergence by the embedding-space surprisal of the observed
    word. We do not test the bound (it is an inequality with an unknown constant); we test the
    qualitative prediction it makes, that the two are positively related.
    """
    kls, nlls = [], []
    for k, blk in enumerate(sequential_batches(corpus.valid, batch, block, device=device)):
        if k >= n_batches:
            break
        logits = model(blk)
        logp = torch.log_softmax(logits, dim=-1)
        nll = -logp.gather(-1, blk.unsqueeze(-1)).squeeze(-1)          # (B, n)
        qbar = model.content_stream(blk)                                # observed beliefs
        Bk = model.contract(qbar)
        _, sbar = model._word_prior()
        qz = torch.softmax(sbar / model.cfg.lambda_Z, dim=-1).expand_as(qbar)
        for _ in range(model.cfg.tau):                                  # predictive beliefs
            G, _ = model._arc_message(qz, Bk)
            qz = torch.softmax((sbar + G) / model.cfg.lambda_Z, dim=-1)
        kl = (qz * (qz.clamp_min(1e-30).log() - qbar.clamp_min(1e-30).log())).sum(-1)
        kls.append(kl[:, 1:].reshape(-1))
        nlls.append(nll[:, 1:].reshape(-1))
    x = torch.cat(kls).double()
    y = torch.cat(nlls).double()
    xc, yc = x - x.mean(), y - y.mean()
    r = float((xc * yc).sum() / (xc.norm() * yc.norm()).clamp(min=1e-12))
    lo = y < y.quantile(0.25)
    hi = y > y.quantile(0.75)
    return {
        "n_slots": int(x.numel()),
        "pearson_r_kl_vs_nll": round(r, 4),
        "mean_kl_predictable_quartile": round(float(x[lo].mean()), 4),
        "mean_kl_surprising_quartile": round(float(x[hi].mean()), 4),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--batches", type=int, default=8)
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--top", type=int, default=12)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    corpus = load_ptb()
    model, cfg = load_checkpoint(a.checkpoint, a.device)
    print(f"{a.checkpoint}: d={cfg.d} h={cfg.h} rank={cfg.rank} readout={cfg.readout} "
          f"alpha_Z={cfg.alpha_Z} temp={cfg.temp_mode}")

    lex = label_lexicon(model, corpus, a.top)
    print("\n--- what each label claims (contrastive score, median log10 frequency) ---")
    for row in lex:
        print(f"  z={row['label']:<3} (freq {row['median_log10_freq']:>6}) "
              + " ".join(row["words"]))

    prof = attention_profile(model, corpus, a.block_size, a.batch_size, a.batches, a.device)
    print("\n--- where the head posterior looks ---")
    for k, v in prof.items():
        print(f"  {k}: {v}")

    pp = prior_posterior(model, corpus, a.block_size, a.batch_size, a.batches, a.device)
    print("\n--- prior against posterior ---")
    for k, v in pp.items():
        print(f"  {k}: {v}")

    if a.out:
        Path(a.out).write_text(json.dumps(
            {"checkpoint": a.checkpoint, "d": cfg.d, "lexicon": lex,
             "attention": prof, "prior_posterior": pp}, indent=1))
        print(f"\nwritten {a.out}")


if __name__ == "__main__":
    main()
