"""Check 16 — WikiText-2, and the guard on the loader refactor that let it in.

WikiText-2 is not a second main experiment. The reviewers asked us to tune on a single
dataset ("To avoid repeated tuning, start by training on a single dataset rather than
multiple different ones"), so WikiText-2 is the transfer test: a hyperparameter rule found
on PTB either carries to a second corpus or it was a fit to PTB. That test only means
something if the second corpus reaches the model through the same pipeline as the first,
which is why ``load_ptb`` and ``load_wikitext2`` are now two file-name maps over one shared
``_corpus_from_files``.

Sharing that code path is exactly what puts the committed PTB numbers at risk, so the first
half of this file is the regression guard on it: the tokenisation contract is asserted on a
hand-written miniature corpus (no data needed, always runs), and the four published PTB
numbers — 929,589 / 73,760 / 82,430 tokens, 10,000 types — are asserted on the real corpus.
Every PTB perplexity in REPORT_2026-08.md was measured on that exact token stream; if these
numbers move, no PTB number in the report can be compared to a new one.

One deviation from the letter of the brief, stated plainly rather than hidden: the PTB
number check is gated on ``data/ptb`` existing, the same way ``tests/test_11`` gates it.
``data/`` is gitignored, so an unconditional assertion would turn a fresh clone's suite red
for a reason that has nothing to do with the code. It is *not* gated on WikiText-2 being
present — that was the thing to avoid, since it would let a refactor regression hide behind
a missing second corpus. On any machine that has PTB, this check runs.
"""

import math
from pathlib import Path

import pytest
import torch

from src.data import (
    EOS,
    Corpus,
    _corpus_from_files,
    load_ptb,
    load_wikitext2,
    unigram_perplexity,
)

PTB_DIR = Path("data/ptb")
WIKI_DIR = Path("data/wikitext2")

needs_ptb = pytest.mark.skipif(
    not (PTB_DIR / "ptb.train.txt").exists(),
    reason="PTB corpus not present (data/ is gitignored)",
)
needs_wiki = pytest.mark.skipif(
    not (WIKI_DIR / "wiki.train.tokens").exists(),
    reason="WikiText-2 not present; run scripts/get_wikitext2.sh (data/ is gitignored)",
)


def _lines(path: Path) -> int:
    """Lines as the loader iterates them, so <eos> count and line count are comparable."""
    with open(path, encoding="utf-8") as fh:
        return sum(1 for _ in fh)


def _types(path: Path) -> set:
    """The distinct whitespace-separated types in a file, read independently of the loader."""
    seen = set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            seen.update(line.split())
    return seen


def _count_word(path: Path, word: str) -> int:
    n = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            n += line.split().count(word)
    return n


# ------------------------------------------------- the shared tokenisation contract --
# These run everywhere: they need no corpus, only the code path both loaders go through.


def _write_toy(tmp_path: Path) -> Path:
    """A three-line corpus small enough that the expected id sequence is written by hand.

    The blank line matters: it is the case WikiText-2 has thousands of and PTB has none of,
    and the protocol says it still contributes one <eos>. The valid split carries a word the
    train split never shows, which is the only situation in which the <unk> fallback fires.

    The train split contains a literal ``<unk>`` because that is where the loader gets the
    fallback id from — both released corpora are pre-``<unk>``-ed, so the id always exists
    there; a corpus whose training split never shows the token has no id to fall back to.
    """
    (tmp_path / "toy.train.txt").write_text("a b\n\n<unk> c a\n", encoding="utf-8")
    (tmp_path / "toy.valid.txt").write_text("a z\n", encoding="utf-8")
    (tmp_path / "toy.test.txt").write_text("<unk> b\n", encoding="utf-8")
    return tmp_path


def _toy_names():
    return {"train": "toy.train.txt", "valid": "toy.valid.txt", "test": "toy.test.txt"}


def test_shared_loader_appends_one_eos_per_line_including_blank_lines(tmp_path):
    c = _corpus_from_files(_write_toy(tmp_path), _toy_names())

    # train file is "a b\n\n<unk> c a\n": the empty second line contributes a bare <eos>.
    assert c.itos == sorted({"a", "b", "c", "<unk>", EOS})
    want = ["a", "b", EOS, EOS, "<unk>", "c", "a", EOS]
    assert [c.itos[i] for i in c.train.tolist()] == want
    assert isinstance(c, Corpus) and c.train.dtype == torch.long


def test_shared_loader_builds_the_vocabulary_from_train_and_unks_the_rest(tmp_path):
    c = _corpus_from_files(_write_toy(tmp_path), _toy_names())

    assert "z" not in c.stoi, "vocabulary must come from the training split alone"
    assert [c.itos[i] for i in c.valid.tolist()] == ["a", "<unk>", EOS]
    assert [c.itos[i] for i in c.test.tolist()] == ["<unk>", "b", EOS]
    assert c.vocab_size == len(c.itos) == len(c.stoi)
    assert all(c.stoi[w] == i for i, w in enumerate(c.itos))


def test_shared_loader_reports_which_files_are_missing(tmp_path):
    (tmp_path / "toy.train.txt").write_text("a\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError) as err:
        _corpus_from_files(tmp_path, _toy_names())
    message = str(err.value)
    assert "toy.valid.txt" in message and "toy.test.txt" in message
    assert "toy.train.txt" not in message


# ------------------------------------------------------------ PTB regression guard ---


@needs_ptb
def test_ptb_token_counts_and_vocabulary_are_still_the_committed_numbers():
    """The refactor must be invisible to PTB. These are the numbers in the report."""
    c = load_ptb()
    assert c.sizes() == {"train": 929589, "valid": 73760, "test": 82430}
    assert c.vocab_size == 10000 == len(c.itos) == len(c.stoi)
    assert EOS in c.stoi and "<unk>" in c.stoi
    assert int(c.train.min()) >= 0 and int(c.train.max()) < c.vocab_size


@needs_ptb
def test_ptb_eos_count_equals_its_line_count():
    """An independent read of the file, so the check does not lean on the loader twice."""
    c = load_ptb()
    assert int((c.train == c.stoi[EOS]).sum()) == _lines(PTB_DIR / "ptb.train.txt")


# ------------------------------------------------------------------- WikiText-2 ------


@pytest.fixture(scope="module")
def wiki() -> Corpus:
    """Loaded once: 2.1M tokens through a Python list costs about a second per call."""
    return load_wikitext2()


@needs_wiki
def test_wikitext2_matches_the_published_corpus_statistics(wiki):
    """The word-level release: 2,088,628 / 217,646 / 245,569 tokens, 33,278 types.

    Measured 2026-09-08 on the archive fetched by scripts/get_wikitext2.sh, and equal to
    the figures the literature quotes. A mismatch means a different release was downloaded
    (the raw variant, most likely) and no perplexity from it is comparable to anyone's.
    """
    assert wiki.sizes() == {"train": 2088628, "valid": 217646, "test": 245569}
    assert wiki.vocab_size == 33278
    assert EOS in wiki.stoi and "<unk>" in wiki.stoi


@needs_wiki
def test_wikitext2_corpus_fields_are_internally_consistent(wiki):
    assert wiki.vocab_size == len(wiki.itos) == len(wiki.stoi)
    assert wiki.itos == sorted(wiki.itos), "itos is built by sorting the training types"
    assert all(wiki.stoi[w] == i for i, w in enumerate(wiki.itos))

    for name in ("train", "valid", "test"):
        ids = getattr(wiki, name)
        assert ids.dtype == torch.long and ids.dim() == 1
        assert int(ids.min()) >= 0
        assert int(ids.max()) < wiki.vocab_size, f"{name} holds an id outside the vocabulary"


@needs_wiki
def test_wikitext2_vocabulary_is_closed_so_the_unk_fallback_never_fires(wiki):
    """Every type in valid and test is already in the train vocabulary — measured, not assumed.

    This is the property that lets unigram_perplexity run unsmoothed (add_k=0) on this
    corpus, exactly as on PTB. It is checked by reading the raw files again rather than by
    inspecting the ids, because after encoding an unseen word is indistinguishable from a
    literal <unk>: the two would only differ in a count.
    """
    known = set(wiki.stoi)
    for split, fname in (("valid", "wiki.valid.tokens"), ("test", "wiki.test.tokens")):
        unseen = _types(WIKI_DIR / fname) - known
        assert not unseen, f"{split} has {len(unseen)} types outside the train vocabulary"

        literal = _count_word(WIKI_DIR / fname, "<unk>")
        encoded = int((getattr(wiki, split) == wiki.stoi["<unk>"]).sum())
        assert encoded == literal, "the <unk> fallback fired on a word it should not have"


@needs_wiki
def test_wikitext2_keeps_blank_lines_and_headings_as_the_protocol_requires(wiki):
    """The article structure is not filtered: that is what the literature scores on.

    Two observable consequences, both asserted here: the number of <eos> equals the number
    of lines in the file (blank lines included, so some <eos> are adjacent), and the heading
    marker '=' survives as a token of its own.
    """
    eos = wiki.stoi[EOS]
    assert int((wiki.train == eos).sum()) == _lines(WIKI_DIR / "wiki.train.tokens") == 36718
    adjacent = int(((wiki.train[:-1] == eos) & (wiki.train[1:] == eos)).sum())
    assert adjacent > 0, "blank lines were dropped; the token count is no longer comparable"
    assert "=" in wiki.stoi, "section headings were filtered out of the stream"


@needs_wiki
def test_wikitext2_unigram_baseline_runs_and_is_a_real_number(wiki):
    """The reference every WikiText-2 result of this project has to clear by a wide margin.

    Measured 2026-09-08: valid 964.82, test 911.53 at block_size=64, batch_size=16,
    ignore_first=1 — the same evaluation geometry as the PTB baseline of test_11, whose
    unigram valid perplexity is 688.82. Both are far below the 33,278 a uniform model would
    give, which is what makes them a baseline rather than a formality.
    """
    for split, want in (("valid", 964.82), ("test", 911.53)):
        ppl = unigram_perplexity(
            wiki.train, getattr(wiki, split), wiki.vocab_size, 64, 16, ignore_first=1
        )
        assert math.isfinite(ppl), f"{split} unigram perplexity is not finite"
        assert ppl > 1.0, f"{split} unigram perplexity {ppl} is below the entropy floor"
        assert ppl < wiki.vocab_size, f"{split} unigram model is worse than uniform"
        assert ppl == pytest.approx(want, abs=0.05)


@needs_ptb
@needs_wiki
def test_both_corpora_go_through_the_same_tokenisation(wiki):
    """The transfer test compares a rule across corpora, so the pipeline must be the shared one.

    The observable form of that: on both corpora the token count equals words plus one <eos>
    per line, and the vocabulary contains the two special types. If a corpus-specific
    tokenisation ever creeps back into one loader, one of these equalities breaks.
    """
    ptb = load_ptb()
    for corpus, path in (
        (ptb, PTB_DIR / "ptb.train.txt"),
        (wiki, WIKI_DIR / "wiki.train.tokens"),
    ):
        n_lines = _lines(path)
        n_words = sum(len(line.split()) for line in open(path, encoding="utf-8"))
        assert corpus.sizes()["train"] == n_words + n_lines
        assert {EOS, "<unk>"} <= set(corpus.stoi)
