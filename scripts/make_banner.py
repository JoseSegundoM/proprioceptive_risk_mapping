#!/usr/bin/env python3
"""Generate assets/banner.png from the study's own data: evaluation map 12, its routes, and the MuJoCo renders.

Usage: python scripts/make_banner.py
"""
import json, sys
from dataclasses import replace
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.riskmc import maps, methods, observation, sim  # noqa: E402

INK, BONE, SLATE, AMBER, TEAL = "#0E1621", "#E8EDF2", "#8FA3B8", "#F2A33C", "#35C2A0"
H = np.array([0.001, 0.06, 0.04, 0.03])


def routes(s):
    tuned = json.load(open(ROOT / "results" / "mc" / "tuned_params.json"))
    out = {}
    for m in ("M0", "M3"):
        _, traj, _ = sim.run_mission(s, m, observation.parametric(0.1), H, H, 0, methods.MethodParams(**tuned))
        out[m] = np.array(traj)
    return out


fig = plt.figure(figsize=(12.8, 3.6), dpi=200, facecolor=INK)
s = maps.generate(12)
ax = fig.add_axes([0.385, 0.09, 0.235, 0.82], facecolor=INK)
t = np.where(s.free, s.terrain, 4)
ax.imshow(t, origin="lower", interpolation="nearest", aspect="auto",
          cmap=ListedColormap(["#33485F", "#2E8FC9", "#D9B13B", "#8C6D46", INK]), vmin=0, vmax=4)
r = routes(s)
ax.plot(r["M0"][:, 1], r["M0"][:, 0], color=BONE, lw=1.4, ls=(0, (2, 1.5)))
ax.plot(r["M3"][:, 1], r["M3"][:, 0], color=TEAL, lw=2.2)
ax.set_axis_off()
for name, x, label in (("portrait_legged", 0.635, "Legged"), ("portrait_wheeled", 0.815, "Wheeled")):
    a = fig.add_axes([x, 0.20, 0.17, 0.70])
    im = np.asarray(Image.open(ROOT / "paper" / "figures" / "mujoco" / f"{name}.png").convert("RGB"))
    h, w = im.shape[:2]
    top, bottom = int(0.17 * h), int(0.97 * h)
    half = int((bottom - top) * (0.17 * 12.8) / (0.70 * 3.6) / 2)      # keep the render's proportions
    a.imshow(im[top:bottom, w // 2 - half:w // 2 + half], aspect="auto")
    a.set_axis_off()
    fig.text(x + 0.085, 0.11, label, color=SLATE, fontsize=11, ha="center", family="DejaVu Sans")
fig.text(0.03, 0.70, "Proprioceptive", color=BONE, fontsize=29, weight="bold", family="DejaVu Sans")
fig.text(0.03, 0.54, "Risk Mapping", color=AMBER, fontsize=29, weight="bold", family="DejaVu Sans")
fig.text(0.03, 0.36, "How body sensing should enter a risk map, on a\ncommercial quadruped with legs or with wheels", color=SLATE, fontsize=11.5, family="DejaVu Sans", linespacing=1.4)
fig.text(0.03, 0.16, "MuJoCo  ·  Monte Carlo  ·  simulation study", color=BONE, fontsize=11, family="DejaVu Sans")
(ROOT / "assets").mkdir(exist_ok=True)
fig.savefig(ROOT / "assets" / "banner.png", facecolor=INK)
