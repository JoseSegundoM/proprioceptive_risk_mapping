#!/usr/bin/env bash
# README images: the two TikZ diagrams of the paper as PNG, and copies of paper figures.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH=/usr/bin:$PATH
T="$(mktemp -d)"
for f in fig_study fig_control; do
  cat > "$T/$f.tex" <<TEX
\documentclass[border=4pt]{standalone}
\usepackage{tikz,amsmath,amssymb,mathptmx}
\usetikzlibrary{arrows.meta,backgrounds,fit,calc}
\definecolor{amberfill}{HTML}{FBEFC9}
\definecolor{greenfill}{HTML}{E3F1EC}
\begin{document}
\input{$PWD/paper/figures/$f.tex}
\end{document}
TEX
  (cd "$T" && pdflatex -interaction=batchmode -halt-on-error "$f.tex" >/dev/null && pdftoppm -r 220 -png -singlefile "$f.pdf" "$f")
  cp "$T/$f.png" "assets/$f.png"
done
cp paper/figures/fig9_mujoco.png assets/mujoco_simulation.png
cp paper/figures/fig4_risk_maps_trajectories.png assets/risk_maps.png
cp paper/figures/fig3_measured_scenarios.png assets/failure_models.png
