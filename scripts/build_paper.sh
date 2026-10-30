#!/usr/bin/env bash
# Two PDFs from one source, both with the IEEE-EPIC margins: paper_final (blue links), paper_ieee_candidate (no annotations).
set -euo pipefail
cd "$(dirname "$0")/../paper"
export PATH=/usr/bin:$PATH
build() { latexmk -pdf -interaction=nonstopmode -halt-on-error -jobname="$1" -usepretex="$2" main.tex; }
build paper_final '\\def\\PDFPROFILE{interactive}'
build paper_ieee_candidate '\\def\\PDFPROFILE{ieee}'
