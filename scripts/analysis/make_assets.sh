#!/usr/bin/env bash
# README images: the two TikZ diagrams of the paper as PNG, and copies of paper figures.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PATH=/usr/bin:$PATH
T="$(mktemp -d)"
for f in study control; do
  cat > "$T/$f.tex" <<TEX
\documentclass[border=4pt]{standalone}
\usepackage{tikz,amsmath,amssymb,mathptmx}
\usetikzlibrary{arrows.meta,backgrounds,fit,calc}
\definecolor{amberfill}{HTML}{FBEFC9}
\definecolor{greenfill}{HTML}{E3F1EC}
\begin{document}
\input{$PWD/paper/figures/diagrams/$f.tex}
\end{document}
TEX
  (cd "$T" && pdflatex -interaction=batchmode -halt-on-error "$f.tex" >/dev/null && pdftoppm -r 220 -png -singlefile "$f.pdf" "$f")
  cp "$T/$f.png" "assets/fig_$f.png"
done
cp paper/figures/png/mujoco_simulation.png assets/mujoco_simulation.png
cp paper/figures/png/risk_maps.png assets/risk_maps.png
cp paper/figures/png/failure_models.png assets/failure_models.png
