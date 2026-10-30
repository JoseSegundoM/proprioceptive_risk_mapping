#!/usr/bin/env bash
# Regenerates every result, figure and the manuscript. Raw MuJoCo data are rewritten under data/raw.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 PATH=/usr/bin:$PATH
source .venv/bin/activate
python -m pytest tests -q
python scripts/run_capture.py --n 150 --jobs 14 --tag campaign --duration 16
python scripts/extract_features.py --tag campaign
python scripts/extract_features.py --tag campaign --noise-scale 3
python scripts/run_classifier.py
python scripts/run_generalization.py
python scripts/run_mc.py maps
python scripts/run_mc.py tune
for e in E1 E2 E3 E4 E5 E6 E7 E8; do python scripts/run_mc.py "$e"; done
python scripts/analyze_mc.py
python scripts/make_tables.py
python scripts/render_mujoco.py
python scripts/generate_figures.py 12
scripts/build_paper.sh
python scripts/verify_deliverables.py
