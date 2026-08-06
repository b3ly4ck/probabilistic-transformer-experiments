"""Sample text from a calibration checkpoint.

Usage:
    python experiments/exp1_language_modeling/sample.py checkpoints/calib/gpt.pt
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import PTConfig  # noqa: E402
from src.data import load_ptb  # noqa: E402
from src.generate import decode, generate  # noqa: E402
from src.gpt import GPT, GPTConfig  # noqa: E402
from src.pt_decoder import CausalPTDecoder  # noqa: E402


def load(path: Path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = CausalPTDecoder(cfg) if isinstance(cfg, PTConfig) else GPT(cfg)
    model.load_state_dict(ckpt["state_dict"])
    readout = None
    if isinstance(cfg, PTConfig):
        readout = "exact" if "exact" in ckpt["model"] else "mfvi"
    return model, readout, ckpt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--n", type=int, default=40, help="tokens per sample")
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    corpus = load_ptb(download=False)
    model, readout, ckpt = load(args.checkpoint)
    g = torch.Generator()
    g.manual_seed(args.seed)

    prompt = None
    if args.prompt:
        ids = corpus.vocab.encode(args.prompt.split())
        prompt = ids.unsqueeze(0).expand(args.samples, -1).contiguous()

    print(f"{ckpt['model']}  val ppl {ckpt['val_ppl']:.2f}  params {ckpt['params']['total']:,}")
    print(f"readout={readout} temperature={args.temperature} top_k={args.top_k}\n")

    out = generate(
        model,
        max_new_tokens=args.n,
        prompt=prompt,
        batch_size=args.samples,
        temperature=args.temperature,
        top_k=args.top_k,
        readout=readout,
        generator=g,
    )
    for i, text in enumerate(decode(out, corpus.vocab), start=1):
        print(f"--- sample {i} ---")
        print(text)
        print()


if __name__ == "__main__":
    main()
