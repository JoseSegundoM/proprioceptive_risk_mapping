# Proprioception-augmented risk mapping for a commercial quadruped

Code and data of the paper *Proprioception-Augmented Risk Mapping for a Commercial Quadruped: Terrain-Aware Navigation Across Legged and Wheeled Configurations*.

**All results are simulation.** Nothing here was measured on the physical DOBOT Rover X1, and nothing validates a real mine. Legged and wheeled are two fixed configurations; no switching during a mission is modelled.

| Path | Contents |
| --- | --- |
| `src/riskmc` | Monte Carlo: map generator, belief updates M0-M3, risk-aware A*, statistics |
| `src/physics` | MuJoCo terrains, proprioceptive capture, window features, classifier |
| `scripts` | Campaigns, analysis, tables, figures, manuscript build, verification |
| `results/mc` | One row per mission, experiments E1-E8 |
| `results/tables`, `results/classifier` | Statistics, classifier metrics, fall rates |
| `data/processed` | Evaluation maps, episode summaries, window features |
| `data/examples/mujoco` | The raw episodes plotted in Fig. 2 |
| `paper` | LaTeX source, figures, generated tables and compiled PDFs |
| `reports/03_research_decision.md` | Protocol and hypotheses (in Spanish) |
| `manifest.json` | Versions, seeds, SHA-256 of results |

## Reproduce

Ubuntu 24.04, Python 3.12. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python -m pytest tests -q
python scripts/analyze_mc.py && python scripts/make_tables.py && python scripts/generate_figures.py 12
scripts/build_paper.sh && python scripts/verify_deliverables.py
```

## License

MIT, see [LICENSE](LICENSE). `paper/IEEEtran.cls` and `paper/IEEEtran_doi.bst` keep their own licences.
