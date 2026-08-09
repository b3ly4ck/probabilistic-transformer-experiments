"""Penn Treebank, word level — the corpus of Wu & Tu's masked-LM experiments.

The files under ``data/`` are the standard Mikolov preprocessing: lower-cased, numbers
replaced by ``N``, out-of-vocabulary words already replaced by ``<unk>``, one sentence per
line. Appending ``<eos>`` to each line gives a vocabulary of exactly 10,000 types and
929,589 training tokens, which is the setting the literature reports perplexity on.

Batching is fixed-length blocks cut from the ``<eos>``-joined stream, which is what nanoGPT
does, so that the baseline of Experiment 1 sees exactly the same blocks. A block boundary is
not a sentence boundary, so slot 0 of a block is predicted from ROOT alone with a prefix that
genuinely does not exist — that is why evaluation drops it (``ignore_first=1``), which is
also what makes the token set identical to a GPT's. See ``CausalPTDecoder.loss``.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

EOS = "<eos>"


@dataclass
class Corpus:
    train: torch.Tensor
    valid: torch.Tensor
    test: torch.Tensor
    stoi: Dict[str, int]
    itos: List[str]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def sizes(self) -> Dict[str, int]:
        return {k: int(getattr(self, k).numel()) for k in ("train", "valid", "test")}


def _tokens(path: Path) -> List[str]:
    out: List[str] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            out.extend(line.split())
            out.append(EOS)
    return out


def load_ptb(root: str = "data/ptb") -> Corpus:
    base = Path(root)
    missing = [p for p in ("ptb.train.txt", "ptb.valid.txt", "ptb.test.txt") if not (base / p).exists()]
    if missing:
        raise FileNotFoundError(f"missing {missing} under {base.resolve()}")

    splits = {name: _tokens(base / f"ptb.{name}.txt") for name in ("train", "valid", "test")}
    itos = sorted(set(splits["train"]))
    stoi = {w: i for i, w in enumerate(itos)}
    unk = stoi.get("<unk>")

    encoded = {
        name: torch.tensor([stoi.get(w, unk) for w in toks], dtype=torch.long)
        for name, toks in splits.items()
    }
    return Corpus(encoded["train"], encoded["valid"], encoded["test"], stoi, itos)


def random_batch(
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    generator: Optional[torch.Generator] = None,
    device: str = "cpu",
) -> torch.Tensor:
    """A batch of random blocks, for training."""
    hi = data.numel() - block_size
    starts = torch.randint(0, hi, (batch_size,), generator=generator)
    return torch.stack([data[s : s + block_size] for s in starts]).to(device)


def sequential_batches(
    data: torch.Tensor, batch_size: int, block_size: int, limit: Optional[int] = None,
    device: str = "cpu",
):
    """Deterministic, non-overlapping blocks — for evaluation.

    Evaluation must not be a random sample: a perplexity that moves when the seed moves
    cannot be compared across runs or against a baseline.
    """
    n_blocks = data.numel() // block_size
    blocks = data[: n_blocks * block_size].view(n_blocks, block_size)
    if limit is not None:
        blocks = blocks[: limit * batch_size]
    for i in range(0, blocks.shape[0], batch_size):
        yield blocks[i : i + batch_size].to(device)


def unigram_perplexity(
    train: torch.Tensor, evaluate: torch.Tensor, vocab_size: int,
    block_size: int, batch_size: int, ignore_first: int = 0, limit: Optional[int] = None,
) -> float:
    """Perplexity of the maximum-likelihood unigram model, on the identical token set.

    This is the reference that matters. A previous implementation of this project reached
    val ppl 664 against a unigram baseline of 687 and the gap was mistaken for learning
    until the samples were read; anything that does not clear this number by a wide margin
    has not learned to use context.
    """
    counts = torch.bincount(train, minlength=vocab_size).double()
    logp = (counts / counts.sum()).clamp_min(1e-12).log()
    total, n = 0.0, 0
    for block in sequential_batches(evaluate, batch_size, block_size, limit):
        target = block[:, ignore_first:]
        total += float(-logp[target].sum())
        n += target.numel()
    return float(torch.tensor(total / n).exp())
