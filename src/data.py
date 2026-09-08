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

A second corpus, WikiText-2, lives here too (``load_wikitext2``). It is not a second main
experiment: the reviewers asked us to tune on one dataset only ("To avoid repeated tuning,
start by training on a single dataset rather than multiple different ones"), so WikiText-2
serves as the transfer test of a hyperparameter rule discovered on PTB. Both corpora are
therefore read by one code path — ``_corpus_from_files`` — because a transfer test in which
the two corpora were tokenised differently would measure the tokeniser, not the rule.
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


def _corpus_from_files(base: Path, filenames: Dict[str, str]) -> Corpus:
    """Read three whitespace-tokenised split files into one ``Corpus``.

    The whole tokenisation policy of the project is this function together with ``_tokens``,
    and it is written once rather than once per corpus on purpose. Two corpora tokenised by two
    copies of the same code drift apart the first time one copy is touched, and a transfer
    test between them would then be reporting the drift. ``filenames`` maps the split name
    to the file that holds it and must be ordered train, valid, test — the order is what the
    ``missing`` message lists, and ``train`` is the split the vocabulary is built from.

    The policy itself: split on whitespace, append ``<eos>`` at the end of every line
    (blank lines included — they contribute a bare ``<eos>``), build the vocabulary from the
    training split alone, and map anything unseen to ``<unk>``. Building the vocabulary from
    train alone is what makes ``unigram_perplexity``'s unsmoothed estimate legitimate: no
    evaluation token can carry a zero count.
    """
    missing = [fn for fn in filenames.values() if not (base / fn).exists()]
    if missing:
        raise FileNotFoundError(f"missing {missing} under {base.resolve()}")

    splits = {name: _tokens(base / fn) for name, fn in filenames.items()}
    itos = sorted(set(splits["train"]))
    stoi = {w: i for i, w in enumerate(itos)}
    unk = stoi.get("<unk>")

    encoded = {
        name: torch.tensor([stoi.get(w, unk) for w in toks], dtype=torch.long)
        for name, toks in splits.items()
    }
    return Corpus(encoded["train"], encoded["valid"], encoded["test"], stoi, itos)


def load_ptb(root: str = "data/ptb") -> Corpus:
    """Mikolov's Penn Treebank: 929,589 / 73,760 / 82,430 tokens, 10,000 types.

    Those four numbers are the committed record of this project and are asserted in
    ``tests/test_11_data_and_training.py`` and ``tests/test_16_wikitext.py``; every PTB
    perplexity reported in ``REPORT_2026-08.md`` was measured on exactly this token stream.
    """
    return _corpus_from_files(
        Path(root),
        {"train": "ptb.train.txt", "valid": "ptb.valid.txt", "test": "ptb.test.txt"},
    )


def load_wikitext2(root: str = "data/wikitext2") -> Corpus:
    """WikiText-2, word level (the ``wikitext-2-v1`` release), read exactly as PTB is.

    Fetch it with ``scripts/get_wikitext2.sh``, which writes ``wiki.train.tokens``,
    ``wiki.valid.tokens`` and ``wiki.test.tokens`` under ``root``. ``data/`` is gitignored,
    so the corpus is never in the history; the script is the reproducible record of it.

    WHY IT IS HERE. Not as a second main experiment — the reviewers were explicit that
    tuning should happen on one dataset — but as the transfer test: a hyperparameter rule
    found on PTB either carries to a second corpus or it was a fit to PTB. That test is only
    worth anything if the two corpora reach the model through the same pipeline, which is
    why this function is three lines of file names over ``_corpus_from_files`` and contains
    no tokenisation logic of its own.

    THE RAW FILES ARE USED AS THEY COME. WikiText-2 keeps its article structure: blank lines
    between paragraphs and section headings written as ``= Heading =`` and ``= = Sub = =``.
    The standard word-level protocol does not strip either — it splits on whitespace and
    appends one ``<eos>`` per line, blank lines included — and that is the token stream every
    published WikiText-2 perplexity is computed over (Merity et al. 2016; the ``Corpus``
    reader in ``pytorch/examples/word_language_model`` is the same four lines). Filtering the
    headings would lower the token count, change the denominator of the perplexity and make
    our number incomparable to the literature, which is the one thing a transfer test cannot
    afford.

    MEASURED on the release fetched by the script (2026-09-08), matching the published
    figures exactly: train 2,088,628 / valid 217,646 / test 245,569 tokens, 33,278 types
    counting ``<eos>``. The vocabulary is closed — 0 types in valid or test are absent from
    train, measured — so the ``<unk>`` fallback in ``_corpus_from_files`` never fires here
    and ``unigram_perplexity`` needs no smoothing (``add_k=0``), as on PTB.

    One consequence worth stating before any run: the vocabulary is 3.3x PTB's, against the
    same ``d``. The rank-``d`` softmax bottleneck of this construction (see CLAUDE.md, "What
    counts as success") therefore bites harder here than on PTB, and a transfer result must
    be read against the GPT baseline on the same corpus rather than against the PTB number.
    """
    return _corpus_from_files(
        Path(root),
        {
            "train": "wiki.train.tokens",
            "valid": "wiki.valid.tokens",
            "test": "wiki.test.tokens",
        },
    )


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
    add_k: float = 0.0,
) -> float:
    """Perplexity of the maximum-likelihood unigram model, on the identical token set.

    This is the reference that matters. A previous implementation of this project reached
    val ppl 664 against a unigram baseline of 687 and the gap was mistaken for learning
    until the samples were read; anything that does not clear this number by a wide margin
    has not learned to use context.

    ``add_k`` is add-k smoothing, needed only when the evaluation split can contain types the
    training split never shows. It is 0 for PTB, whose vocabulary is built from train so no
    zero count can occur, and 1 for the synthetic chains of the scale bisection, where a
    state may go unvisited and an unsmoothed estimate would report an infinite baseline.
    """
    counts = torch.bincount(train, minlength=vocab_size).double() + add_k
    logp = (counts / counts.sum()).clamp_min(1e-12).log()
    total, n = 0.0, 0
    for block in sequential_batches(evaluate, batch_size, block_size, limit):
        target = block[:, ignore_first:]
        total += float(-logp[target].sum())
        n += target.numel()
    return float(torch.tensor(total / n).exp())
