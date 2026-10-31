#!/usr/bin/env bash
# Builds into paper/build/: paper_final (blue links) and paper_ieee_candidate (no annotations).
# paper/paper_final.pdf is the submitted manuscript and is never overwritten.
set -euo pipefail
cd "$(dirname "$0")/../../paper"
export PATH=/usr/bin:$PATH TEXINPUTS="$PWD/template:" BSTINPUTS="$PWD/template:"
build() { latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build -jobname="$1" -usepretex="$2" main.tex; }
build paper_final '\\def\\PDFPROFILE{interactive}'
build paper_ieee_candidate '\\def\\PDFPROFILE{ieee}'
