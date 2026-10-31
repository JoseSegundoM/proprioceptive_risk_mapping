#!/usr/bin/env bash
# LaTeX project for submission: sources, figures in use, tables, class, style, compiled PDF. No temporary or private files.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"; D="$ROOT/build/package/epic2026_latex"
mkdir -p "$D/figures" "$D/tables"
cd paper
cp main.tex profile.tex references.bib IEEEtran.cls IEEEtran.bst paper_final.pdf paper_final.bbl abstract_keywords.txt "$D/"
cp tables/*.tex "$D/tables/"
for f in $(grep -oE 'figures/[a-z0-9_]+\.(pdf|tex)' main.tex | sort -u); do cp "$f" "$D/figures/"; done
cd "$ROOT/build/package"
mkdir -p "$D/figures/photos"; cp "$ROOT"/paper/figures/photos/*.png "$D/figures/photos/"
zip -qr -FS "$ROOT/epic2026_latex_package.zip" epic2026_latex
