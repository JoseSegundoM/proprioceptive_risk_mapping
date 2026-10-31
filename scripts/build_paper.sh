#!/usr/bin/env bash
# paper_final is the submitted manuscript; paper_ieee_candidate has no link annotations.
set -euo pipefail
cd "$(dirname "$0")/../paper"
export PATH=/usr/bin:$PATH
build() { latexmk -pdf -interaction=nonstopmode -halt-on-error -jobname="$1" -usepretex="$2" main.tex; }
build paper_final '\\def\\PDFPROFILE{interactive}'
build paper_ieee_candidate '\\def\\PDFPROFILE{ieee}'
