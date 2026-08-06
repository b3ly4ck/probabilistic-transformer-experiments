"""Corpus loading and word-level tokenisation.

One tokenizer and one vocabulary for every model in Experiment 1 -- PT with and
without the global head, and the GPT baseline.  A difference here would
invalidate the comparison as surely as a difference in the training loop, so the
encoded tensors are produced once and shared.

Dataset: **Penn Treebank**, the Mikolov preprocessing (lowercased, numbers
mapped to N, out-of-vocabulary mapped to <unk>, no punctuation).  Chosen over
WikiText-2 because its vocabulary is ~10k rather than ~33k: with tied embeddings
PT's budget sits almost entirely in the |V| x d matrix S, so a smaller vocabulary
leaves more of a matched budget in the parts the comparison is actually about.
WikiText-103 is excluded by the research plan -- it is the regime where the
original PT is reported to fail.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

PTB_URL = "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.{split}.txt"
SPLITS = ("train", "valid", "test")
EOS = "<eos>"

DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "data" / "ptb"


def download_ptb(root: Path = DEFAULT_ROOT) -> Path:
    """Fetch the three splits if they are not already on disk."""
    root.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        target = root / f"ptb.{split}.txt"
        if not target.exists():
            urllib.request.urlretrieve(PTB_URL.format(split=split), target)
    return root


def read_tokens(path: Path) -> list[str]:
    """Whitespace tokens, with an explicit end-of-sentence symbol per line.

    <eos> is a real token the model must predict; without it nothing marks a
    sentence boundary and perplexity is not comparable to the published numbers.
    """
    tokens: list[str] = []
    for line in path.read_text().splitlines():
        tokens.extend(line.split())
        tokens.append(EOS)
    return tokens


@dataclass
class Vocab:
    itos: list[str]
    stoi: dict[str, int]

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, tokens: list[str]) -> Tensor:
        unk = self.stoi.get("<unk>")
        if unk is None:
            return torch.tensor([self.stoi[t] for t in tokens], dtype=torch.long)
        return torch.tensor([self.stoi.get(t, unk) for t in tokens], dtype=torch.long)

    @classmethod
    def from_tokens(cls, tokens: list[str]) -> "Vocab":
        """Built from the training split only -- a vocabulary that has seen the
        validation split leaks it."""
        itos = sorted(set(tokens))
        return cls(itos=itos, stoi={t: i for i, t in enumerate(itos)})


@dataclass
class Corpus:
    vocab: Vocab
    train: Tensor
    valid: Tensor
    test: Tensor

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def split(self, name: str) -> Tensor:
        return getattr(self, name)


def load_ptb(root: Path = DEFAULT_ROOT, download: bool = True) -> Corpus:
    if download:
        download_ptb(root)
    raw = {s: read_tokens(root / f"ptb.{s}.txt") for s in SPLITS}
    vocab = Vocab.from_tokens(raw["train"])
    return Corpus(vocab=vocab, **{s: vocab.encode(raw[s]) for s in SPLITS})


def batchify(data: Tensor, batch_size: int) -> Tensor:
    """Reshape a flat token stream into (batch_size, n) contiguous streams.

    The standard LM protocol: each row is a continuous slice of the corpus, so
    consecutive windows within a row follow on from one another.  The tail that
    does not divide evenly is dropped.
    """
    n = data.numel() // batch_size
    return data[: n * batch_size].view(batch_size, n)


def windows(batched: Tensor, context: int, shift: int = 0):
    """Yield (inputs, targets) windows of length ``context``.

    Targets are inputs shifted by one, so position ``t`` is scored on predicting
    the token at ``t`` from everything before it -- the same convention the PT
    decoder uses, where slot ``t`` reads ``D_t = {ROOT, 0..t-1}``.
    """
    n = batched.shape[1]
    for start in range(shift, n - context - 1, context):
        yield batched[:, start : start + context], batched[:, start + 1 : start + context + 1]
