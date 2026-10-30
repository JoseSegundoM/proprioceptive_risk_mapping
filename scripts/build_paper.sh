#!/usr/bin/env bash
# Three PDFs from one source: paper_final (blue links), paper_ieee_candidate (no annotations), paper_epic_margins (top 36 mm).
set -euo pipefail
cd "$(dirname "$0")/../paper"
export PATH=/usr/bin:$PATH
build() { latexmk -pdf -interaction=nonstopmode -halt-on-error -jobname="$1" -usepretex="$2" main.tex; }
build paper_final '\def\PDFPROFILE{interactive}'
build paper_ieee_candidate '\def\PDFPROFILE{ieee}'
build paper_epic_margins '\def\PDFPROFILE{interactive}\def\EPICMARGINS{}'
