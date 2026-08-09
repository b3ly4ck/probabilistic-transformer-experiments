# Build the preprint.
#
# TeX Live is installed under $HOME/texlive (scheme-small + latexmk, cleveref, pgf, makecell).
# The cluster has no system TeX and no tex module, so that private install is the only one.
# It is added to PATH below, which means `make` works from any shell, login or not.

export PATH := $(HOME)/texlive/bin/x86_64-linux:$(PATH)

MAIN    := main
BUILD   := build
LATEXMK := latexmk

.PHONY: all clean distclean watch check-tex pages

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

clean:
	$(LATEXMK) -c -outdir=$(BUILD) $(MAIN).tex 2>/dev/null || true

distclean:
	rm -rf $(BUILD)
