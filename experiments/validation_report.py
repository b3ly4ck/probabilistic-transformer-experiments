"""Produce the numbers of the paper's validation appendix, by running the checks.

Appendix E of the preprint lists what was checked and to what precision. Those numbers must not
be retyped from a test file's docstring: a test and an appendix that drift apart is exactly the
failure the project's own rules name, and the appendix is the only part a reader sees. So the
appendix is generated from a run of the checks themselves, and this script is that run.

    python -m experiments.validation_report            # markdown to stdout
    python -m experiments.validation_report --json     # machine-readable

Everything here executes the real code path on small real tensors in float64 where a precision
claim is being made. Nothing greps the source.
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Any, Dict

import torch

from src import CausalPTDecoder, FactoredPTDecoder, PTConfig
from src.energy import slot_free_energy


def _cfg(**kw) -> PTConfig:
    base = dict(vocab_size=6, d=4, h=2, rank=None, gamma=2, n_iters=3, tau=2,
                readout="exact", vocab_chunk=64)
    base.update(kw)
    return PTConfig(**base)


def check_exact_vs_brute() -> Dict[str, Any]:
    """Sum-product on the slot tree against explicit enumeration of the slot joint.

    The strongest correctness statement available: on one slot the graph is a star centred on
    the label variable, hence a tree, so the closed form must equal the marginal of the joint
    computed by brute force. Enumeration is written here from the raw parameters, with loops,
    touching no method of the model.
    """
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    cfg = _cfg()
    m = CausalPTDecoder(cfg)
    idx = torch.randint(0, cfg.vocab_size, (1, 4))
    with torch.no_grad():
        qbar = m.content_stream(idx)
        B_full = m._slot_keys(m.contract(qbar), idx.shape[1])  # (1, h, 1+t, d)
        model_logp = torch.log_softmax(m.slot_exact_readout(B_full)[0], dim=-1)

        V, d, h = cfg.vocab_size, cfg.d, cfg.h
        D = B_full.shape[2]
        unnorm = torch.zeros(V, dtype=torch.float64)
        for w in range(V):
            tot = 0.0
            for a in range(d):
                for js in _tuples(D, h):
                    e = float(m.b[w]) + float(m.S[w, a])
                    for c in range(h):
                        e += float(B_full[0, c, js[c], a])
                    tot += math.exp(e)
            unnorm[w] = tot
        brute_logp = torch.log(unnorm) - torch.log(unnorm.sum())
    torch.set_default_dtype(torch.float32)
    return {"check": "exact readout vs brute-force enumeration",
            "max_abs_diff": float((model_logp - brute_logp).abs().max()),
            "detail": f"V={V}, d={d}, h={h}, |D_t|={D}, float64"}


def _tuples(n: int, k: int):
    if k == 0:
        yield ()
        return
    for rest in _tuples(n, k - 1):
        for j in range(n):
            yield rest + (j,)


def check_free_energy() -> Dict[str, Any]:
    """The slot free energy must not increase along the inner loop.

    This is the one check that tests the update equations rather than their shapes: a sign
    error in the head update leaves every shape, normalisation and causality check passing and
    fails only here. The mutation is exercised alongside, so the check is shown to have teeth.
    """
    torch.set_default_dtype(torch.float64)
    out: Dict[str, Any] = {"check": "free energy non-increasing"}
    for label, flip in (("as implemented", False), ("with a sign-flipped H update", True)):
        torch.manual_seed(1)
        cfg = _cfg(readout="mfvi", tau=6, init_std=0.5)
        m = CausalPTDecoder(cfg)
        idx = torch.randint(0, cfg.vocab_size, (1, 4))
        with torch.no_grad():
            qbar = m.content_stream(idx)
            B_full = m._slot_keys(m.contract(qbar), idx.shape[1])
            if flip:
                B_full = -B_full
            trace: list = []
            m.slot_mfvi_readout(B_full, trace=trace)
            qw0, _ = m._word_prior()
            energies = []
            for qz, alpha, qg in trace:
                energies.append(float(slot_free_energy(
                    qw0, qz, alpha, B_full if not flip else -B_full, m.S, m.b,
                    cfg.lambda_W, cfg.lambda_Z, cfg.lam_H)[0]))
        rises = [b - a for a, b in zip(energies, energies[1:]) if b - a > 1e-9]
        out[label] = {"n_updates": len(energies) - 1, "n_increases": len(rises),
                      "max_increase": round(max(rises), 6) if rises else 0.0,
                      "total_decrease": round(energies[0] - energies[-1], 6)}
    torch.set_default_dtype(torch.float32)
    return out


def check_causality() -> Dict[str, Any]:
    """Changing token t+1 must leave logits at slots <= t bitwise unchanged, in every mode."""
    res = {}
    for readout in ("exact", "mfvi"):
        for temp in ("fixed", "qnorm", "qknorm"):
            torch.manual_seed(2)
            cfg = _cfg(readout=readout, temp_mode=temp, qk_gain=2.0)
            m = CausalPTDecoder(cfg)
            idx = torch.randint(0, cfg.vocab_size, (2, 6))
            alt = idx.clone()
            alt[:, -1] = (alt[:, -1] + 1) % cfg.vocab_size
            with torch.no_grad():
                a, b = m(idx)[:, :-1], m(alt)[:, :-1]
            res[f"{readout}/{temp}"] = {"max_abs_diff": float((a - b).abs().max()),
                                        "bitwise_equal": bool(torch.equal(a, b))}
    return {"check": "causality", **res}


def check_tying() -> Dict[str, Any]:
    """Input and output word factors must be the same tensor object, not merely equal."""
    torch.manual_seed(3)
    cfg = _cfg()
    m = CausalPTDecoder(cfg)
    names = [n for n, p in m.named_parameters() if p is m.S]
    return {"check": "tying", "S_is_one_object": len(names) == 1, "parameter_names": names,
            "n_parameters_named_S": len(names)}


def check_factored_equivalence() -> Dict[str, Any]:
    """K = 1 factored labels must reproduce the flat decoder, bitwise."""
    torch.set_default_dtype(torch.float64)
    worst = 0.0
    for readout in ("exact", "mfvi"):
        for rank in (None, 3):
            torch.manual_seed(4)
            cfg = _cfg(readout=readout, rank=rank, d=6, n_components=1)
            a = CausalPTDecoder(cfg)
            b = FactoredPTDecoder(cfg)
            b.load_state_dict(a.state_dict())
            idx = torch.randint(0, cfg.vocab_size, (2, 5))
            with torch.no_grad():
                worst = max(worst, float((a(idx) - b(idx)).abs().max()))
    torch.set_default_dtype(torch.float32)
    return {"check": "factored K=1 equals the flat decoder", "max_abs_diff": worst,
            "detail": "both readouts, full and Kruskal arc scores, float64"}


def check_normalisation() -> Dict[str, Any]:
    torch.manual_seed(5)
    cfg = _cfg(readout="mfvi")
    m = CausalPTDecoder(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 5))
    with torch.no_grad():
        q = m.content_stream(idx)
        logits = m(idx)
    return {"check": "normalisation",
            "max_abs_belief_sum_error": float((q.sum(-1) - 1).abs().max()),
            "logits_finite": bool(torch.isfinite(logits).all())}


CHECKS = [check_exact_vs_brute, check_free_energy, check_causality, check_tying,
          check_factored_equivalence, check_normalisation]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--json", action="store_true")
    a = p.parse_args()
    out = [fn() for fn in CHECKS]
    if a.json:
        print(json.dumps(out, indent=2))
        return
    for r in out:
        print(f"\n## {r.pop('check')}")
        for k, v in r.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
