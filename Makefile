# Results run bottom-up: capture -> classifier -> monte-carlo -> analysis -> paper.
PY ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
MODELS_COMMIT = eca13b025aee82b2bd3593918d61af0ba60281ed
export OMP_NUM_THREADS = 1

.PHONY: help setup models test capture classifier monte-carlo analysis figures paper verify package all

help:
	@echo "setup        virtual environment and dependencies"
	@echo "models       robot models for the MuJoCo capture (project repository, access needed)"
	@echo "test         unit, integration and determinism tests"
	@echo "capture      MuJoCo episodes and window features (needs models)"
	@echo "classifier   terrain classifier, confusion and fall rates"
	@echo "monte-carlo  tuning on development maps, then experiments E1-E8"
	@echo "analysis     statistics and tables from the recorded missions"
	@echo "figures      figures and README images"
	@echo "paper        manuscript, into paper/build/"
	@echo "verify       checks on the manuscript and its evidence"
	@echo "package      flat LaTeX project as a ZIP, into build/"
	@echo "all          everything above, from the capture on"

setup:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt

models:
	git clone https://github.com/JoseSegundoM/mine_sentinel_autonomy.git mine_sentinel_autonomy
	git -C mine_sentinel_autonomy checkout $(MODELS_COMMIT)
	git -C mine_sentinel_autonomy submodule update --init

test:
	$(PY) -m pytest tests -q

capture:
	$(PY) scripts/experiments/run_capture.py --n 150 --jobs 14 --tag campaign --duration 16
	$(PY) scripts/experiments/extract_features.py --tag campaign
	$(PY) scripts/experiments/extract_features.py --tag campaign --noise-scale 3

classifier:
	$(PY) scripts/experiments/run_classifier.py
	$(PY) scripts/experiments/run_generalization.py

monte-carlo:
	$(PY) scripts/experiments/run_mc.py maps
	$(PY) scripts/experiments/run_mc.py tune
	for e in E1 E2 E3 E4 E5 E6 E7 E8; do $(PY) scripts/experiments/run_mc.py $$e || exit 1; done

analysis:
	$(PY) scripts/analysis/analyze_mc.py > /dev/null
	$(PY) scripts/analysis/make_tables.py

figures:
	$(PY) scripts/experiments/render_mujoco.py
	$(PY) scripts/analysis/generate_figures.py 12
	scripts/analysis/make_assets.sh

paper:
	scripts/paper/build_paper.sh

verify:
	$(PY) scripts/paper/verify_deliverables.py

package: paper
	scripts/paper/package.sh

all: test capture classifier monte-carlo analysis figures paper verify
