<div align="center">

<img src="assets/banner.png" alt="Proprioceptive Risk Mapping" width="100%"/>

<br/><br/>

**Proprioception-Augmented Risk Mapping for a Commercial Quadruped:<br/>Terrain-Aware Navigation Across Legged and Wheeled Configurations**

<sub>José Luis Segundo Manayay · Javier Yanpier Garay Yovera · Francisco Jesús Rodríguez Huiman</sub><br/>
<sub>Faculty of Mechanical Engineering · Universidad Nacional de Ingeniería (UNI) · Lima, Peru</sub>

<br/>

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![MuJoCo](https://img.shields.io/badge/MuJoCo-3.13-0072B2?style=for-the-badge)
![Episodes](https://img.shields.io/badge/MuJoCo%20episodes-1400-1B7837?style=for-the-badge)
![Missions](https://img.shields.io/badge/Monte%20Carlo%20missions-177000-D55E00?style=for-the-badge)
![Evidence](https://img.shields.io/badge/Evidence-simulation%20only-6A4C93?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=for-the-badge)
[![CI](https://github.com/JoseSegundoM/proprioceptive_risk_mapping/actions/workflows/ci.yml/badge.svg)](https://github.com/JoseSegundoM/proprioceptive_risk_mapping/actions/workflows/ci.yml)

**English** · [Español](README.es.md)

</div>

---

> Geometry says little about friction or softness: a wet floor and dry concrete return the same
> LiDAR profile. A robot can feel the difference through its body, but only once it is standing on
> it. This repository measures **how that evidence should enter a planner's risk map**, and whether
> the answer depends on the feet.

The DOBOT Rover X1 uses point-foot and wheel modules exchanged by hand, so each mission uses one
fixed configuration; no switching during a mission is modelled. The study has two parts: dynamic
simulation in MuJoCo, which gives a terrain classifier and simulated fall rates, and an algorithmic
Monte Carlo, which compares four map-update rules on the same maps with paired statistics.

> [!IMPORTANT]
> **All results are simulation.** Nothing was measured on the physical robot and nothing validates a
> real mine. Appearance features in the Monte Carlo are synthetic, and the wheeled controller is this
> project's own, not the manufacturer's.

---

## ✨ Main results

Mission survival `P_s`, 200 evaluation maps, classifier error ε = 0.1, assumed hazard table
(mean and 95 % bootstrap interval of the paired difference over maps):

| Method | Survival `P_s` | Exposure (m) | Difference | 200 unseen maps |
| --- | --- | --- | --- | --- |
| M0 geometry only | 0.654 | 9.10 | | |
| M1 local Bayesian update | 0.657 | 9.03 | M1 − M0: +0.004 [+0.000, +0.007] | +0.001, interval includes 0 |
| M2 appearance propagation | 0.712 | 7.35 | M2 − M1: +0.055 [+0.043, +0.067] | +0.049 |
| M3 uncertainty-aware fusion | 0.727 | 6.92 | M3 − M2: +0.015 [+0.008, +0.023] | +0.014 |
| Oracle (diagnostic bound) | 0.804 | 4.73 | | |

- **Spreading evidence by appearance is what raises survival.** Updating only the crossed ground
  arrives too late to matter. The gain disappears when half the patches look alike or appearance noise is high.
- **The failure model, not perception, decides which configuration gains.** With measured confusion
  and equal hazards the difference between configurations is +0.002. With simulated falls per metre the gain is
  +0.106 for the legged model and +0.000 for the wheeled one, which never fell.
- **The classifier interpolates and does not extrapolate.** Macro-F1 on held-out episodes is 0.935 (legged)
  and 0.999 (wheeled), 0.820 and 0.842 on mixed courses, and as low as 0.450 for the legged model at unseen speeds.

<div align="center">
<img src="assets/risk_maps.png" alt="True terrain, routes and believed hazard per method on one map" width="100%"/>
<br/><sub>One evaluation map: true terrain with the four routes, and the hazard each method believes at the end.</sub>
<br/><br/>
<img src="assets/failure_models.png" alt="Survival under three failure models" width="100%"/>
<br/><sub>Survival with measured classifier confusion under three failure models.</sub>
</div>

---

## 🏗️ How the study is built

<div align="center">
<img src="assets/fig_study.png" alt="Study structure" width="100%"/>
<br/><br/>
<img src="assets/fig_control.png" alt="Control structure: inner loop in MuJoCo, outer loop in the Monte Carlo" width="62%"/>
<br/><sub>The inner loop runs in MuJoCo and the outer loop in the Monte Carlo. They are joined offline, not closed online in one simulator.</sub>
<br/><br/>
<img src="assets/mujoco_simulation.png" alt="MuJoCo renders and logged signals" width="100%"/>
<br/><sub>MuJoCo renders, with floor colour for display only, and logged signals on the four terrains.</sub>
</div>

---

## 🚀 Quick start

Ubuntu 24.04, Python 3.12, and a LaTeX distribution for the manuscript.

```bash
git clone https://github.com/JoseSegundoM/proprioceptive_risk_mapping.git
cd proprioceptive_risk_mapping
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m pytest tests -q                 # physics tests skip without the robot models
python scripts/analyze_mc.py              # statistics from the recorded missions
python scripts/make_tables.py             # tables and every number in the text
python scripts/generate_figures.py 12     # figures
scripts/build_paper.sh                    # manuscript
python scripts/verify_deliverables.py     # page count, links, citations, numbers
```

Rerunning the Monte Carlo or the MuJoCo capture is described in
[`docs/reproducibility.md`](docs/reproducibility.md). The capture needs the robot models from a
separate project repository, which is private; the recorded data here do not.

---

## 🗂️ Repository layout

```
assets/    images of this README, and the banner
configs/   Monte Carlo experiment definitions
data/      evaluation maps, episode summaries, window features, plotted raw episodes
docs/      reproducibility guide and research protocol
paper/     LaTeX source, figures, generated tables, compiled PDFs
results/   mission records (E1-E8), statistics, classifier metrics, checksums
scripts/   campaigns, analysis, tables, figures, build and checks
src/       riskmc (Monte Carlo) and physics (MuJoCo capture, classifier)
tests/     unit, integration and determinism tests
```

---

## 👥 Authors

| Author | E-mail | ORCID | GitHub |
| --- | --- | --- | --- |
| José Luis Segundo Manayay | [jose.segundo.m@uni.pe](mailto:jose.segundo.m@uni.pe) | [0009-0002-9203-8710](https://orcid.org/0009-0002-9203-8710) | [@JoseSegundoM](https://github.com/JoseSegundoM) |
| Javier Yanpier Garay Yovera | [javier.garay.y@uni.pe](mailto:javier.garay.y@uni.pe) | [0009-0001-1645-8724](https://orcid.org/0009-0001-1645-8724) | [@garx17](https://github.com/garx17) |
| Francisco Jesús Rodríguez Huiman | [francisco.rodriguez.h@uni.pe](mailto:francisco.rodriguez.h@uni.pe) | [0009-0006-6972-2484](https://orcid.org/0009-0006-6972-2484) | [@frodriguezhuiman-lgtm](https://github.com/frodriguezhuiman-lgtm) |

Faculty of Mechanical Engineering, Universidad Nacional de Ingeniería (UNI), Lima, Peru.

---

## 📄 License

MIT, see [`LICENSE`](LICENSE). Third-party components are listed in [`NOTICE`](NOTICE).

## 📚 Citation

See [`CITATION.cff`](CITATION.cff). The manuscript is in [`paper/paper_final.pdf`](paper/paper_final.pdf).
