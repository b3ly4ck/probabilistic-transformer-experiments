"""Reproduce causal_pt_output_note.pdf §5 with every intermediate tensor printed.

This is check 7 in narrative form.  The assertions live in
tests/test_07_worked_example.py; this script exists so the derivation can be
read alongside the note, and so there is something to show if it is probed.

Run:  ./.venv/bin/python experiments/exp0_decoder_validation/worked_example.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import mfvi  # noqa: E402
from src.config import PTConfig  # noqa: E402

torch.set_printoptions(precision=3, sci_mode=False)

CFG = PTConfig(vocab_size=4, d=2, n_channels=1, n_rounds=1, lambda_Z=1.0, lambda_H=1.0)
WORDS = ["the", "cat", "sat", "mat"]
LABELS = ["N", "V"]

S = torch.tensor([[1.5, -1.5], [2.0, -2.0], [-2.0, 2.0], [2.0, -2.0]])
B = torch.tensor([1.0, 0.0, 0.0, 0.0])
T = torch.tensor([[[0.0, 2.0], [1.0, 0.0]]])
R = torch.tensor([[0.0, 1.5]])
SENTENCE = [0, 1, 2]  # the cat sat []


def rule(title):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


rule("Setup")
print(f"V = {WORDS}, labels = {LABELS}, d = {CFG.d}, h = {CFG.n_channels}")
print(f"lambda_Z = {CFG.lambda_Z}, lambda_H = {CFG.lambda_H}")
print(f"S (rows {WORDS}) =\n{S}")
print(f"b = {B}\nT^(1) =\n{T[0]}\nr^(1) = {R[0]}")
print("A noun-like dependent scores 2 under a verb-like head; a verb-like")
print("dependent scores 1 under a noun-like head; verbs are attracted to ROOT.")

rule("Observed pass -- t = 1, 2, 3")
cached = [R[0]]
for t, w in enumerate(SENTENCE, start=1):
    Bkey = torch.stack(cached).unsqueeze(0)
    domain = ["ROOT"] + [str(i) for i in range(1, t)]
    Q_Z0 = mfvi.init_slot(S[w], CFG)
    Q_Z, Q_c = mfvi.run_slot_mfvi(S[w], Bkey, CFG, n_rounds=1)
    cached.append(torch.einsum("e,ae->a", Q_Z, T[0]))
    print(f"\nslot {t}: w_{t} = {WORDS[w]!r}, D_{t} = {{{', '.join(domain)}}}")
    print(f"  Q_Z^(0) = softmax(S[{WORDS[w]}]) = {Q_Z0}")
    print(f"  Q_c over D_{t}            = {Q_c[0]}")
    print(f"  qbar_{t} (frozen)          = {Q_Z}")
    print(f"  cached B_{t} = T qbar_{t}    = {cached[-1]}")

rule("Predictive slot 4 -- the word is now a variable")
Bkey = torch.stack(cached).unsqueeze(0)
Q_W0 = torch.softmax(B, dim=-1)
s_bar = mfvi.word_message(Q_W0, S)
print(f"Q_W^(0) = softmax(b)        = {Q_W0}")
print(f"s_bar = sum_w Q_W^(0)(w) S  = {s_bar}   <- the derived [MASK] embedding")
print(f"Q_Z^(0) = softmax(s_bar)    = {torch.softmax(s_bar, dim=-1)}")

Q_Z, Q_c, trace = mfvi.run_slot_mfvi(s_bar, Bkey, CFG, n_rounds=2, return_trace=True)
for i, (qz, qc) in enumerate(trace, start=1):
    F = mfvi.head_scores(trace[i - 2][0] if i > 1 else torch.softmax(s_bar, -1), Bkey)
    print(f"\nround {i}:")
    print(f"  attention logits F over (ROOT, 1, 2, 3) = {F[0]}")
    print(f"  Q_c                                     = {qc[0]}")
    print(f"  Q_Z after the Z-update                  = {qz}")

rule("Readout -- the MFVI update of W is the LM output layer")
logits = mfvi.mfvi_readout_logits(Q_Z, S, B, CFG)
probs = torch.softmax(logits, dim=-1)
print(f"{'':>10}" + "".join(f"{w:>10}" for w in WORDS))
print(f"{'logit':>10}" + "".join(f"{float(x):>10.3f}" for x in logits))
print(f"{'p(W_4)':>10}" + "".join(f"{float(x):>10.3f}" for x in probs))
print("\nExpected from the note:")
print(f"{'logit':>10}" + "".join(f"{x:>10.3f}" for x in (2.368, 1.824, -1.824, 1.824)))
print(f"{'p(W_4)':>10}" + "".join(f"{x:>10.3f}" for x in (0.460, 0.267, 0.007, 0.267)))

rule("Exact readout on the same slot (the §23.3 mainline)")
from src import exact  # noqa: E402

log_mu = exact.log_mu_slot(Bkey)
ex_logits = exact.exact_logits(log_mu, S, B)
ex_probs = torch.softmax(ex_logits, dim=-1)
brute = torch.softmax(exact.brute_force_logits(Bkey, S, B), dim=-1)
print(f"log mu_4 over labels {LABELS} = {log_mu}")
print(f"{'':>10}" + "".join(f"{w:>10}" for w in WORDS))
print(f"{'p exact':>10}" + "".join(f"{float(x):>10.3f}" for x in ex_probs))
print(f"{'p brute':>10}" + "".join(f"{float(x):>10.3f}" for x in brute))
print(f"\nmax |exact - brute force| = {float((ex_probs - brute).abs().max()):.2e}")
print("The star graph is a tree, so these must agree to numerical precision.")
