# Build the preprint.
#
# TeX Live is installed under $HOME/texlive (scheme-small + latexmk, cleveref, pgf, makecell).
# The cluster has no system TeX and no tex module, so that private install is the only one.
# It is added to PATH below, which means `make` works from any shell, login or not.

export PATH := $(HOME)/texlive/bin/x86_64-linux:$(PATH)

# Reproducible builds. Without these, pdflatex stamps /CreationDate, /ModDate and a trailer
# /ID into the PDF, so two builds from byte-identical sources differ in 58 bytes. That matters
# because git stores a whole new compressed blob per commit -- it does not diff binaries -- so
# an unchanged paper would still cost its full size in history on every commit.
#
# The epoch is a fixed arbitrary instant, not the current time and not the last commit time:
# the point is that identical sources always produce identical bytes, so committing a rebuilt
# but unchanged PDF is free. Nothing in the document prints a date, so the value is invisible.
export SOURCE_DATE_EPOCH := 1767225600
export FORCE_SOURCE_DATE := 1

MAIN    := main
BUILD   := build
LATEXMK := latexmk

.PHONY: all clean distclean watch check-tex pages release

all: check-tex $(BUILD)/$(MAIN).pdf

check-tex:
	@command -v $(LATEXMK) >/dev/null 2>&1 || { \
	  echo "latexmk not found."; \
	  echo "Expected it under \$$HOME/texlive/bin/x86_64-linux — see paper/PAPER_STATUS.md."; \
	  exit 1; }

$(BUILD)/$(MAIN).pdf: $(MAIN).tex preamble.tex notation.tex refs.bib acl.sty acl_natbib.bst $(wildcard sections/*.tex)
	$(LATEXMK) -pdf -outdir=$(BUILD) -interaction=nonstopmode -halt-on-error $(MAIN).tex

# Continuous rebuild while writing.
watch:
	$(LATEXMK) -pdf -pvc -outdir=$(BUILD) -interaction=nonstopmode $(MAIN).tex

# The format budget is 8 pages of main body before the references. Check it, because in a
# two-column format an overrun is discovered far too late otherwise.
pages: all
	@pdfinfo $(BUILD)/$(MAIN).pdf 2>/dev/null | grep Pages || \
	  echo "pdfinfo unavailable; open $(BUILD)/$(MAIN).pdf and count"

# Promote the current build to the tracked PDF at paper/main.pdf. Run this only at a
# milestone -- a version sent to a supervisor, a tagged draft -- not on every build. Everyday
# builds live in build/, which is gitignored.
release: all
	@cp $(BUILD)/$(MAIN).pdf $(MAIN).pdf
	@echo "paper/$(MAIN).pdf updated ($$(wc -c < $(MAIN).pdf) bytes)."
	@echo "This is a milestone artifact: 'git add paper/$(MAIN).pdf' only if you mean to keep"
	@echo "this exact PDF in history forever. Rebuilds of unchanged sources are byte-identical,"
	@echo "so re-running this without source changes leaves git with nothing to commit."

clean:
	$(LATEXMK) -c -outdir=$(BUILD) $(MAIN).tex 2>/dev/null || true

distclean:
	rm -rf $(BUILD)
