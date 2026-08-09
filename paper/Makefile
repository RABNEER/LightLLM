# LightLLM Paper Build Script
# Requires: pdflatex + bibtex (TeX Live / MiKTeX)
#
# Usage:
#   make          -> build PDF
#   make clean    -> remove auxiliary files
#   make distclean-> remove all generated files

PAPER = lightllm_paper
LATEX = pdflatex
BIBTEX = bibtex

all: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex references.bib
	$(LATEX) -interaction=nonstopmode $(PAPER).tex
	$(BIBTEX) $(PAPER)
	$(LATEX) -interaction=nonstopmode $(PAPER).tex
	$(LATEX) -interaction=nonstopmode $(PAPER).tex

clean:
	rm -f $(PAPER).aux $(PAPER).log $(PAPER).bbl $(PAPER).blg \
	      $(PAPER).toc $(PAPER).out $(PAPER).lof $(PAPER).lot

distclean: clean
	rm -f $(PAPER).pdf

.PHONY: all clean distclean
