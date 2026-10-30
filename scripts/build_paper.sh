#!/usr/bin/env bash
# PDFs from one source: paper_final (blue links), paper_ieee_candidate (no annotations) and, where the
# cropped photographs are present, paper_real_photos (Fig. 1 shows the physical robot).
set -euo pipefail
cd "$(dirname "$0")/../paper"
export PATH=/usr/bin:$PATH
build() { latexmk -pdf -interaction=nonstopmode -halt-on-error -jobname="$1" -usepretex="$2" main.tex; }
build paper_final '\\def\\PDFPROFILE{interactive}'
build paper_ieee_candidate '\\def\\PDFPROFILE{ieee}'
if [ -f figures/photos/robot_legged.png ]; then
  build paper_real_photos '\\def\\PDFPROFILE{interactive}\\def\\REALPHOTOS{}'
fi
