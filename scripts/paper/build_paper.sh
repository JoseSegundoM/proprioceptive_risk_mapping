#!/usr/bin/env bash
# Build the editorial revision; never overwrite the historical submitted PDF.
set -euo pipefail
cd "$(dirname "$0")/../../paper"
export PATH=/usr/bin:$PATH TEXINPUTS="$PWD/template:" BSTINPUTS="$PWD/template:"
build() { latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build -jobname="$1" -usepretex="$2" paper_66.tex; }
build paper_66 '\def\PDFPROFILE{interactive}'
build paper_ieee_candidate '\def\PDFPROFILE{ieee}'
cp build/paper_66.pdf paper_66.pdf
