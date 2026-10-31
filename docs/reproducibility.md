# Reproducibility

Three levels, from least to most dependent.

| Level | Regenerates | Needs | Command |
| --- | --- | --- | --- |
| 1. Analysis | Statistics, tables, figures, manuscript, from the recorded mission rows | This repository | `python scripts/analyze_mc.py && python scripts/make_tables.py && python scripts/generate_figures.py 12 && scripts/build_paper.sh` |
| 2. Monte Carlo | Every mission row, E1-E8 (E2, E6 and part of E7 read the classifier outputs in `results/classifier`) | This repository, 16 threads, about 15 min | `python scripts/run_mc.py tune`, then `python scripts/run_mc.py E1` ... `E8` |
| 3. Dynamics | Raw MuJoCo signals, features, classifier | The robot models in `JoseSegundoM/mine_sentinel_autonomy` at commit `eca13b025aee82b2bd3593918d61af0ba60281ed`, cloned into this folder; about 6 min for capture | `scripts/reproduce_all.sh` |

`mine_sentinel_autonomy` is a separate project repository. If it is not accessible to you, level 3 is out of reach, and levels 1 and 2 still work on the recorded data. Raw signals (about 500 MB) are not stored here because they regenerate from their seeds.

## What has been checked

- `tests/test_determinism.py` regenerates mission rows and MuJoCo episodes on the same machine and compares them with the recorded values.
- `scripts/verify_deliverables.py` checks the page count, links, citations, mission counts, and that the text's numbers come from generated macros.
- Results were produced on Ubuntu 24.04 with Python 3.12 and the versions in `requirements.txt`. Exact floating-point agreement on other machines or library versions has not been tested.

## Departures from the robot repository

MuJoCo physics and the wheeled stance PD run at 1 kHz in `src/physics/capture.py`, instead of 200 Hz, because at 200 Hz the stance PD produced a numerical limit cycle of about 70 m/s^2 RMS at the IMU. The robot repository is not modified.

## Added after the protocol

Experiments E5 (lambda sweep) and E8 (spatial baseline) were added after seeing E1-E4, to answer objections from an internal review. E5 is descriptive. Parameters were tuned on development maps only; a 30-map smoke check on evaluation maps used default parameters and tuned nothing.
