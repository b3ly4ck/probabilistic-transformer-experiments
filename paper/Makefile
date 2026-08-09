# Build the preprint.
#
# There is no TeX distribution on the cluster login node (checked 2026-08-09: no pdflatex,
# no latexmk, no tex module, nothing under /public/software). This Makefile is therefore
# untested here -- it is written for a machine that has TeX Live. See PAPER_STATUS.md for
# the options.

MAIN    := main
BUILD   := build
LATEXMK := latexmk

.PHONY: all clean distclean watch check-tex

all: check-tex $(BUILD)/$(MAIN).pdf

check-tex:
	@command -v $(LATEXMK) >/dev/null 2>&1 || { \
	  echo "latexmk not found. No TeX distribution is installed on this machine."; \
	  echo "See paper/PAPER_STATUS.md for how the PDF is currently produced."; \
	  exit 1; }

$(BUILD)/$(MAIN).pdf: $(MAIN).tex preamble.tex notation.tex refs.bib $(wildcard sections/*.tex)
	$(LATEXMK) -pdf -outdir=$(BUILD) -interaction=nonstopmode -halt-on-error $(MAIN).tex

# Continuous rebuild while writing.
watch:
	$(LATEXMK) -pdf -pvc -outdir=$(BUILD) -interaction=nonstopmode $(MAIN).tex

clean:
	$(LATEXMK) -c -outdir=$(BUILD) $(MAIN).tex 2>/dev/null || true

distclean:
	rm -rf $(BUILD)
