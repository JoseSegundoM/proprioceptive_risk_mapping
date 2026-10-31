#!/usr/bin/env bash
# LaTeX project in one flat folder, for a submission system: build/epic2026_latex_package.zip
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT="$PWD"; D="$ROOT/build/epic2026_latex"
mkdir -p "$D/figures" "$D/tables"
cd paper
cp main.tex references.bib template/* paper_final.pdf build/paper_final.bbl abstract_keywords.txt "$D/"
cp tables/*.tex "$D/tables/"
for f in $(grep -oE 'figures/[a-z0-9_/]+\.(pdf|tex|png)' main.tex | sort -u); do mkdir -p "$D/$(dirname "$f")"; cp "$f" "$D/$f"; done
cd "$ROOT/build"
zip -qr -FS epic2026_latex_package.zip epic2026_latex
