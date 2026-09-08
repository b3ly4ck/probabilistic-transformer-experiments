#!/usr/bin/env bash
#
# Fetch the standard word-level WikiText-2 corpus into data/wikitext2/.
#
# WHY THIS EXISTS, AND WHY IT IS A SCRIPT RATHER THAN A NOTE IN A README.
# The reviewers asked us to tune on a single dataset ("To avoid repeated tuning, start by
# training on a single dataset rather than multiple different ones"), so WikiText-2 is not
# a second main experiment: it is the transfer test of a hyperparameter rule discovered on
# PTB. A transfer test is only evidence if the second corpus is loaded exactly the way the
# first one was, from exactly the release everyone else reports on. Hence: pinned sources,
# recorded checksums, and a token count verified before the files are installed.
#
# WHICH RELEASE. The word-level "v1" release (wikitext-2-v1), NOT wikitext-2-raw-v1.
# The word-level files are already preprocessed: rare words replaced by <unk>, one
# paragraph per line, blank lines and "= Heading =" lines left in place. That is the file
# every reported WikiText-2 perplexity is computed on, and it is what src/data.py
# load_wikitext2 expects. The raw release is byte-pair territory and is not interchangeable.
#
# SOURCES, in the order tried. Measured on this machine on 2026-09-08 through the login
# node's HTTP proxy (HTTPS_PROXY=http://127.0.0.1:8119):
#   1. https://wikitext.smerity.com/wikitext-2-v1.zip
#      The author's own mirror of the canonical archive. HTTP 200, 4475746 bytes. WORKS.
#   2. https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip
#      The URL quoted in the WikiText paper and in salesforce/awd-lstm-lm getdata.sh.
#      DEAD as of the measurement above: path-style returns PermanentRedirect, and every
#      redirect target (virtual-hosted, s3.us-west-2, s3-us-west-2) returns AccessDenied.
#      The bucket is no longer public. It is kept in the list because it is the canonical
#      address and may come back, and because a reader of this script should see that it
#      was tried rather than silently skipped.
#   3. https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/
#      wikitext-2/{train,valid,test}.txt
#      The copy shipped inside pytorch/examples. HTTP 200. WORKS, and was verified with
#      cmp(1) to be byte-identical to the three files inside the archive from source 1.
#
# The script is idempotent: if the three files are already in place and pass the token
# count check, it does nothing. Any failure is loud and leaves data/wikitext2/ untouched --
# a half-downloaded corpus that silently trains is worse than no corpus at all.
#
# data/ is gitignored on purpose. The corpus is never committed; this script is the
# reproducible record of where it came from.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-${ROOT}/data/wikitext2}"
WORK=""  # scratch directory, set in main; global so the EXIT trap can still see it

FILES=(wiki.train.tokens wiki.valid.tokens wiki.test.tokens)

# Reference token counts under the tokenisation of src/data.py: whitespace split, plus one
# <eos> per line (blank lines included, which is why the counts are one per line above the
# bare word counts). These are the numbers quoted for WikiText-2 in the literature.
EXPECT_TRAIN=2088628
EXPECT_VALID=217646
EXPECT_TEST=245569
EXPECT_VOCAB=33278  # types in train, counting <eos>; includes <unk>

# sha256 of the three files as they come out of source 1, cross-checked against source 3.
SHA_TRAIN=9e9fa1ad55b1c2c95b08e37dd8e653f638fac2c6de904b79e813611eefbc985f
SHA_VALID=f0737ed31fc1329026e95cb8b98e19c2a182c39c240ab909dc31abf2f8af58e8
SHA_TEST=d790b833ef8cf03a90db7bf1271b7520b83c45ce07ba3c1a9699df81e239eca0

ZIP_URLS=(
  "https://wikitext.smerity.com/wikitext-2-v1.zip"
  "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip"
)
PLAIN_BASE="https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2"

log() { printf '%s\n' "$*" >&2; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

# Token count under the loader's convention: every whitespace-separated field, plus one
# <eos> per line. Deliberately implemented in awk rather than by importing src.data, so
# that the check is independent of the code it is checking.
count_tokens() { awk '{ n += NF + 1 } END { print n + 0 }' "$1"; }

# Number of distinct types in the training file, plus one for <eos>. LC_ALL=C because some
# locales collate distinct byte strings as equal and would undercount the vocabulary.
count_types() {
  local n
  n=$(LC_ALL=C tr -s '[:space:]' '\n' < "$1" | LC_ALL=C sed '/^$/d' | LC_ALL=C sort -u | wc -l)
  echo $((n + 1))
}

# Verify a directory holding the three .tokens files. Prints what it measured; returns
# non-zero on any mismatch so the caller can decide whether to try the next source.
verify_dir() {
  local dir="$1" f
  for f in "${FILES[@]}"; do
    [ -s "${dir}/${f}" ] || return 1
  done

  local got_train got_valid got_test
  got_train=$(count_tokens "${dir}/wiki.train.tokens")
  got_valid=$(count_tokens "${dir}/wiki.valid.tokens")
  got_test=$(count_tokens "${dir}/wiki.test.tokens")

  local bad=0
  [ "$got_train" -eq "$EXPECT_TRAIN" ] || { log "train tokens: got ${got_train}, expected ${EXPECT_TRAIN}"; bad=1; }
  [ "$got_valid" -eq "$EXPECT_VALID" ] || { log "valid tokens: got ${got_valid}, expected ${EXPECT_VALID}"; bad=1; }
  [ "$got_test"  -eq "$EXPECT_TEST"  ] || { log "test tokens: got ${got_test}, expected ${EXPECT_TEST}"; bad=1; }
  [ "$bad" -eq 0 ] || return 1

  local got_vocab
  got_vocab=$(count_types "${dir}/wiki.train.tokens")
  if [ "$got_vocab" -ne "$EXPECT_VOCAB" ]; then
    log "train vocabulary: got ${got_vocab} types, expected ${EXPECT_VOCAB}"
    return 1
  fi

  log "verified: train ${got_train} / valid ${got_valid} / test ${got_test} tokens, ${got_vocab} types"
  return 0
}

# Checksums are reported, not enforced. A mirror that ships the same tokenisation under
# different bytes (line endings, a trailing newline) still yields a valid corpus, and the
# token count above is the property the experiment actually depends on. A mismatch is
# printed loudly so that it is never discovered later in a log nobody reads.
check_sha() {
  local dir="$1" f want got
  command -v sha256sum >/dev/null 2>&1 || { log "note: sha256sum not available, skipping checksum report"; return 0; }
  for f in "${FILES[@]}"; do
    case "$f" in
      wiki.train.tokens) want="$SHA_TRAIN" ;;
      wiki.valid.tokens) want="$SHA_VALID" ;;
      wiki.test.tokens)  want="$SHA_TEST" ;;
    esac
    got=$(sha256sum "${dir}/${f}" | cut -d' ' -f1)
    if [ "$got" != "$want" ]; then
      log "WARNING: ${f} sha256 ${got} differs from the recorded ${want}"
    fi
  done
}

fetch() {  # fetch URL OUTPUT -- returns non-zero instead of aborting, so the caller can fall through
  local url="$1" out="$2"
  curl --silent --show-error --location --fail --retry 3 --connect-timeout 30 --max-time 900 \
       -o "$out" "$url"
}

try_zip() {  # try_zip URL WORKDIR
  local url="$1" work="$2"
  command -v unzip >/dev/null 2>&1 || { log "  unzip not available, cannot use archive sources"; return 1; }
  log "trying archive: ${url}"
  if ! fetch "$url" "${work}/wikitext-2-v1.zip"; then
    log "  download failed"
    return 1
  fi
  if ! unzip -oq "${work}/wikitext-2-v1.zip" -d "${work}/unpacked"; then
    log "  not a readable zip archive (the host may have answered with an error page)"
    return 1
  fi
  # The archive holds a wikitext-2/ directory; find the files wherever they landed.
  local f src
  for f in "${FILES[@]}"; do
    src=$(find "${work}/unpacked" -type f -name "$f" | head -1)
    [ -n "$src" ] || { log "  archive does not contain ${f}"; return 1; }
    mv "$src" "${work}/staged/${f}"
  done
  return 0
}

try_plain() {  # try_plain BASEURL WORKDIR -- the pytorch/examples copy uses {train,valid,test}.txt
  local base="$1" work="$2" f split
  log "trying plain files: ${base}"
  for f in "${FILES[@]}"; do
    split="${f#wiki.}"; split="${split%.tokens}"
    if ! fetch "${base}/${split}.txt" "${work}/staged/${f}"; then
      log "  download of ${split}.txt failed"
      return 1
    fi
  done
  return 0
}

main() {
  command -v curl >/dev/null 2>&1 || die "curl not found; this machine reaches the network through HTTPS_PROXY=${HTTPS_PROXY:-unset}"

  # Idempotency. verify_dir is silent when the files are simply absent, and noisy when they
  # are present but wrong -- which is the case worth seeing, so it is not suppressed.
  if verify_dir "$DEST"; then
    log "WikiText-2 already present and verified under ${DEST} -- nothing to do"
    exit 0
  fi

  # WORK is global on purpose. The EXIT trap fires after main has returned, so a trap that
  # dereferences a function-local would abort under "set -u" with "unbound variable" and
  # turn a completed download into a non-zero exit status.
  WORK=$(mktemp -d)
  trap 'rm -rf "$WORK"' EXIT
  mkdir -p "${WORK}/staged"

  local ok=0 url
  for url in "${ZIP_URLS[@]}"; do
    if try_zip "$url" "$WORK"; then ok=1; break; fi
    rm -f "${WORK}"/staged/* 2>/dev/null || true
    rm -rf "${WORK}/unpacked" 2>/dev/null || true
  done
  if [ "$ok" -eq 0 ]; then
    if try_plain "$PLAIN_BASE" "$WORK"; then ok=1; fi
  fi
  [ "$ok" -eq 1 ] || die "every source failed; see the messages above. Sources tried: ${ZIP_URLS[*]} ${PLAIN_BASE}"

  check_sha "${WORK}/staged"

  if ! verify_dir "${WORK}/staged"; then
    die "the downloaded corpus does not match the reference token counts; refusing to install it into ${DEST}"
  fi

  mkdir -p "$DEST"
  local f
  for f in "${FILES[@]}"; do
    mv "${WORK}/staged/${f}" "${DEST}/${f}"
  done

  log "installed into ${DEST}:"
  ls -l "$DEST" >&2
  verify_dir "$DEST" >/dev/null || die "verification failed after install into ${DEST}"
  log "done. Load it with: from src.data import load_wikitext2; load_wikitext2()"
}

main "$@"
