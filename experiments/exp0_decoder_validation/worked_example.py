"""Reproduce the worked example of ``causal_pt_output_note.pdf`` §5, number for number.

    V = {the, cat, sat, mat},  d = 2 labels {N, V},  h = 1 channel,
    lambda_Z = lambda_H = 1,  tau = 1 in observed mode, tau = 2 in predictive mode,
    distance-insensitive arc score (gamma = 0).

The parameters are the note's invented ones. The sentence is "the cat sat", and the
model is asked for slot 4. Every intermediate the note prints is printed here.

Run:  python -m experiments.exp0_decoder_validation.worked_example
"""

import torch

from src import CausalPTDecoder, PTConfig

WORDS = ["the", "cat", "sat", "mat"]
LABELS = ["N", "V"]


def build_model(readout: str = "mfvi") -> CausalPTDecoder:
    cfg = PTConfig(
        vocab_size=4,
        d=2,
        h=1,
        rank=None,
        gamma=0,  # one distance bucket: the note's single T
        schedule="serial",  # the note freezes each slot before advancing
        tau_obs=1,  # "one round of (2)-(3), freeze q_bar"
        tau=2,  # "tau = 2 in predictive mode"
        readout=readout,
        lambda_Z=1.0,
        lambda_H=1.0,
        lambda_W=1.0,
    )
    m = CausalPTDecoder(cfg).double()
    with torch.no_grad():
        m.S.copy_(torch.tensor([[1.5, -1.5], [2.0, -2.0], [-2.0, 2.0], [2.0, -2.0]]))
        m.b.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        m.r_root.copy_(torch.tensor([[0.0, 1.5]]))
        m.T.copy_(torch.tensor([[[[0.0, 2.0], [1.0, 0.0]]]]))
    return m


def main() -> None:
    torch.set_printoptions(precision=3, sci_mode=False)
    m = build_model("mfvi")
    idx = torch.tensor([[0, 1, 2]])  # the cat sat

    print("Parameters (the note's invented values)")
    print("  S =\n", m.S.data)
    print("  b =", m.b.data, "  T =", m.T.data.squeeze(), "  r =", m.r_root.data.squeeze())

    print("\nObserved pass — expected q_bar (.818,.182) (.957,.043) (.006,.994)")
    qbar = m.content_stream(idx)
    Sw = m.S[idx]
    for t in range(3):
        q0 = torch.softmax(Sw[0, t], dim=-1)
        print(f"  t={t + 1} w={WORDS[idx[0, t]]:4s} Q_Z^(0)={q0.data} q_bar={qbar[0, t].data}")

    Bk = m.contract(qbar)
    print("  cached B_{t,.} =", Bk[0, 0, 0].data.tolist())

    print("\nSlot 4, predictive — expected Q_Z after rounds: (.951,.049) then (.956,.044)")
    B_full = m._slot_keys(Bk, 3)
    qw0, sbar = m._word_prior()
    print("  Q_W^(0) =", qw0.data, " s_bar =", sbar.data)
    print("  Q_Z^(0) =", torch.softmax(sbar, dim=-1).data)

    trace: list = []
    logits = m.slot_mfvi_readout(B_full, trace=trace)
    for k in range(0, len(trace), 2):
        qz_in, alpha, _ = trace[k]
        qz_out = trace[k + 1][0]
        F = torch.einsum("ba,bcja->bcj", qz_in, B_full)
        print(
            f"  round {k // 2 + 1}: F over (ROOT,1,2,3) = {F[0, 0].data}"
            f"  Q_c = {alpha[0, 0].data}  Q_Z after (3) = {qz_out[0].data}"
        )

    print("\nReadout (4) — expected logits 2.368 1.824 -1.824 1.824")
    print("  logits =", logits[0].data)
    print("  p_hat  =", torch.softmax(logits[0], dim=-1).data)

    print("\nExact readout on the same slot (§17.2 mainline, not part of the note's table)")
    print("  logits =", m.slot_exact_readout(B_full)[0].data)
    print("  p_hat  =", torch.softmax(m.slot_exact_readout(B_full)[0], dim=-1).data)


if __name__ == "__main__":
    main()
