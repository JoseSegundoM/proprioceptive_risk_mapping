#!/usr/bin/env bash
# LaTeX project for submission: sources, figures in use, tables, class, style, compiled PDF. No temporary or private files.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"; D="$ROOT/_downloads/pkg/epic2026_latex"
mkdir -p "$D/figures" "$D/tables"
cd paper
cp main.tex profile.tex references.bib IEEEtran.cls IEEEtran.bst paper_final.pdf paper_final.bbl abstract_keywords.txt "$D/"
cp tables/*.tex "$D/tables/"
for f in $(grep -oE 'figures/[a-z0-9_]+\.(pdf|tex)' main.tex | sort -u); do cp "$f" "$D/figures/"; done
cd "$ROOT/_downloads/pkg"
zip -qr -FS "$ROOT/epic2026_latex_package.zip" epic2026_latex
# Additional version with the physical robot in Fig. 1. Local only: the photographs are not published with the repository.
if [ -f "$ROOT/paper/figures/photos/robot_legged.png" ]; then
  D2="$ROOT/_downloads/pkg_photos/epic2026_latex_real_photos"
  mkdir -p "$D2"; cp -r "$D/." "$D2/"
  mkdir -p "$D2/figures/photos"; cp "$ROOT"/paper/figures/photos/*.png "$D2/figures/photos/"
  cp "$ROOT/paper/paper_real_photos.pdf" "$D2/"; rm -f "$D2/paper_final.pdf"
  { printf '%s
' '\def\REALPHOTOS{}'; cat "$ROOT/paper/main.tex"; } > "$D2/main.tex"
  cd "$ROOT/_downloads/pkg_photos"; zip -qr -FS "$ROOT/epic2026_latex_package_real_photos.zip" epic2026_latex_real_photos
fi
