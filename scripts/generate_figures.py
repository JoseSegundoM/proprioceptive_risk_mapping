#!/usr/bin/env python3
"""All paper figures, from saved results only: python scripts/generate_figures.py. Writes PDF, SVG and 300 dpi PNG."""
import json, sys
from dataclasses import replace
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyBboxPatch, Patch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.riskmc import maps, methods, observation, sim  # noqa: E402

F, T = ROOT / "paper" / "figures", ROOT / "results" / "tables"
F.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"], "mathtext.fontset": "stix",
                     "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "axes.grid": True, "grid.color": "#DDDDDD", "grid.linewidth": 0.4, "axes.axisbelow": True, "pdf.fonttype": 42, "svg.fonttype": "none",
                     "legend.frameon": False, "lines.linewidth": 1.2})
ONE, TWO = 3.45, 7.1
# Okabe-Ito hues, with marker and dash as a second channel for grayscale printing. Colour follows the method everywhere.
STYLE = {"M0": ("#6B6B6B", "s", (0, (1, 1))), "M1": ("#0072B2", "o", (0, (4, 1.5))), "M2": ("#D55E00", "^", "-"),
         "M3": ("#009E73", "D", (0, (6, 1, 1, 1))), "ORACLE": ("#000000", "*", (0, (1, 2)))}
LABEL = {"M0": "M0 geometry", "M1": "M1 local", "M2": "M2 appearance", "M3": "M3 fusion", "ORACLE": "Oracle (bound)"}
CFG = {"point_foot": "Legged", "wheel_legged": "Wheeled"}
CFG_COLOR = {"point_foot": "#0072B2", "wheel_legged": "#D55E00"}
TERRAIN = ["nominal", "low friction", "granular-like", "rough"]
# Terrain has its own colours so that it is not mistaken for a method.
TCOLORS = ["#F2F2F2", "#56B4E9", "#E8C547", "#8C6D46"]


def save(fig, name):
    for ext, kw in (("pdf", {}), ("svg", {}), ("png", {"dpi": 300})):
        fig.savefig(F / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02, **kw)
    plt.close(fig)


def fig_architecture():
    fig, ax = plt.subplots(figsize=(TWO, 1.6))
    ax.set_axis_off(); ax.set_xlim(0, 100); ax.set_ylim(0, 27)

    def box(x, y, w, h, text, fc="#FFFFFF", ec="#333333", ls="-"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15,rounding_size=0.6", fc=fc, ec=ec, lw=0.7, ls=ls))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.6)

    def arrow(a, b, ls="-"):
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", lw=0.7, color="#333333", ls=ls, shrinkA=0, shrinkB=0, mutation_scale=6))
    new = "#E3F1EC"
    box(0.5, 19, 21, 7.5, "Dynamic simulation (MuJoCo)\nfixed legged or wheeled robot,\nparametrised terrain contact")
    box(0.5, 10, 21, 7, "Proprioception: IMU, joint\nposition and velocity,\ncommanded torque")
    box(0.5, 1, 21, 7, "Exteroception (assumed input):\ngeometry, roughness,\nappearance features")
    box(26, 10, 17, 7, "Terrain classifier\nwindow features,\nrandom forest", fc=new)
    box(26, 19, 17, 7.5, "Held-out confusion\nobservation model\n$p(o\\mid t,b)$", fc=new)
    box(48, 5.5, 20, 16, "Risk map\nbelief $b_i(t)$ per cell\nM0 geometry prior\nM1 local Bayes\nM2 appearance vote\nM3 tempered fusion", fc=new)
    box(73, 15, 26, 7, "Risk-aware A*, replanning\ncost rate: $1+\\lambda\\,\\bar h_i$", fc=new)
    box(73, 5, 26, 7, "Monte Carlo mission:\ntrue hazard accumulates, never\nvisible to the methods", fc="#F5F5F5", ls="--")
    arrow((11, 19), (11, 17)); arrow((21.5, 13.5), (26, 13.5)); arrow((34.5, 17), (34.5, 19)); arrow((43, 22.5), (48, 18))
    arrow((43, 13.5), (48, 13.5)); arrow((21.5, 4.5), (48, 8)); arrow((68, 16), (73, 18)); arrow((86, 15), (86, 12))
    ax.annotate("", xy=(58, 5.5), xytext=(73, 7), arrowprops=dict(arrowstyle="-|>", lw=0.7, color="#333333", ls=(0, (3, 2)), mutation_scale=6, connectionstyle="arc3,rad=-0.25"))
    ax.text(65, 1.8, "new windows, replan", fontsize=6, ha="center", color="#333333")
    ax.legend(handles=[Patch(fc=new, ec="#333333", lw=0.6, label="Implemented in this work")], loc="upper center", bbox_to_anchor=(0.73, 1.03), fontsize=6.5)
    save(fig, "fig1_architecture")


def fig_configurations():
    d = pd.read_csv(ROOT / "results" / "classifier" / "failure_rates_mujoco.csv")
    d = d[d.cls.isin(maps.CLASSES)]
    fig, axs = plt.subplots(1, 4, figsize=(TWO, 1.12), constrained_layout=True)
    panels = [("mean_slip", "Contact slip (m/s)"), ("speed_tracking", "Speed / command (-)"), ("h_mean", "Falls per metre (1/m)"), ("acc_std", "Accel. std (m/s$^2$)")]
    x = np.arange(4)
    for ax, (col, lab) in zip(axs, panels):
        for i, v in enumerate(CFG):
            g = d[d.variant == v].set_index("cls").loc[maps.CLASSES]
            y = g[col].values
            kw = dict(yerr=[y - g.h_lo.values, g.h_hi.values - y], capsize=1.5, error_kw=dict(lw=0.6)) if col == "h_mean" else {}
            ax.bar(x + (i - 0.5) * 0.38, y, 0.34, color=CFG_COLOR[v], label=CFG[v], hatch=None if i == 0 else "////", edgecolor="white", linewidth=0.3, **kw)
        ax.set_ylabel(lab, labelpad=1); ax.set_xticks(x); ax.set_xticklabels(["nom.", "low $\\mu$", "gran.", "rough"], rotation=30); ax.grid(axis="x", visible=False)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.98), ncol=2)
    save(fig, "fig2_configurations")


def fig_quadrupeds():
    """Real offscreen renders of the two simulation models. No photographs."""
    from PIL import Image
    fig, axs = plt.subplots(1, 2, figsize=(ONE, 1.28))
    for ax, (n, title) in zip(axs, (("portrait_legged", "(a) Legged"), ("portrait_wheeled", "(b) Wheeled"))):
        im = np.asarray(Image.open(F / "mujoco" / f"{n}.png").convert("RGB"))
        ax.imshow(im[170:], interpolation="none", aspect="auto")
        ax.set_title(title, fontsize=7.5, pad=2); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.subplots_adjust(left=0.003, right=0.997, top=0.91, bottom=0.005, wspace=0.03)
    save(fig, "fig0_quadrupeds")


def fig_mujoco():
    """Top: real offscreen renders of the simulation. Bottom: signals from campaign episodes, as logged."""
    from PIL import Image
    M = F / "mujoco"
    d = pd.read_csv(ROOT / "data" / "processed" / "episodes_campaign.csv")
    raw = ROOT / "data" / "raw" / "mujoco" / "campaign"
    if not raw.exists():
        raw = ROOT / "data" / "examples" / "mujoco"
    colors = {"nominal": ("#333333", "-"), "low_friction": ("#1B8FD0", (0, (4, 1.5))), "granular": ("#C9A227", (0, (6, 1, 1, 1))), "rough": ("#8C6D46", (0, (1, 1)))}
    names = {"nominal": "nominal", "low_friction": "low friction", "granular": "granular-like", "rough": "rough"}

    def episode(variant, cls):
        g = d[(d.variant == variant) & (d.cls == cls) & (d.kind == "homogeneous") & (d.duration_s >= 5.5)].sort_values("instance").iloc[0]
        return np.load(raw / f"{g.episode}.npz")

    fig = plt.figure(figsize=(TWO, 2.05))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.82], left=0.05, right=0.995, top=0.945, bottom=0.2, wspace=0.26, hspace=0.27)
    shots = [("legged_rough", "(a) Legged, rough"), ("legged_low_friction_fall", "(b) Legged, low friction: fall"), ("wheeled_rough", "(c) Wheeled, rough"), ("wheeled_granular", "(d) Wheeled, granular-like")]
    for i, (n, title) in enumerate(shots):
        ax = fig.add_subplot(gs[0, i])
        im = np.asarray(Image.open(M / f"{n}.png").convert("RGB"))
        ax.imshow(im[110:], interpolation="none", aspect="auto")
        ax.set_title(title, fontsize=7, pad=2); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
    panels = [("point_foot", lambda z: z["tau"][:, 2], "Calf torque (N m)", "(e) Legged"), ("point_foot", lambda z: z["acc"][:, 2], "Accel. z (m/s$^2$)", "(f) Legged"),
              ("wheel_legged", lambda z: z["tau_w"].mean(axis=1), "Wheel torque (N m)", "(g) Wheeled"), ("wheel_legged", lambda z: z["acc"][:, 2], "Accel. z (m/s$^2$)", "(h) Wheeled")]
    for i, (v, fn, lab, title) in enumerate(panels):
        ax = fig.add_subplot(gs[1, i])
        for c in maps.CLASSES:
            z = episode(v, c)
            t = z["t"] - z["t"][0]
            k = (t >= 2.0) & (t <= 4.0)
            ax.plot(t[k], fn(z)[k], color=colors[c][0], ls=colors[c][1], lw=0.7, label=names[c])
        ax.set_ylabel(lab, labelpad=1, fontsize=6.5); ax.set_xlabel("Time (s)", labelpad=0.5, fontsize=6.5); ax.tick_params(labelsize=6, length=2, pad=1)
        ax.set_title(title, fontsize=6.5, pad=1.5)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.03), fontsize=6.5, columnspacing=1.2, handlelength=2.4)
    save(fig, "fig9_mujoco")


def run_example(map_id, eps=0.1):
    H = np.array([0.001, 0.06, 0.04, 0.03])
    s = maps.generate(map_id)
    tuned = json.load(open(ROOT / "results" / "mc" / "tuned_params.json"))
    summ = json.load(open(ROOT / "results" / "mc" / "tuning_summary.json"))
    res = {}
    for m in ("M0", "M1", "M2", "M3"):
        mp = methods.MethodParams(**tuned)
        if m == "M2":
            mp = replace(mp, kernel_sigma=summ["m2_kernel_for_eval"])
        r, traj, meth = sim.run_mission(s, m, observation.parametric(eps), H, H, 0, mp)
        res[m] = (r, np.array(traj), meth.hazard_rate().copy())
    return s, res


def fig_maps(map_id):
    s, res = run_example(map_id)
    fig = plt.figure(figsize=(TWO, 1.62))
    gs = fig.add_gridspec(2, 5, height_ratios=[1, 0.05], left=0.003, right=0.997, top=0.93, bottom=0.2, wspace=0.03, hspace=0.25)
    axs = [fig.add_subplot(gs[0, i]) for i in range(5)]
    t = np.where(s.free, s.terrain, 4)
    from matplotlib.colors import PowerNorm
    X, Y = np.arange(s.free.shape[1] + 1) - 0.5, np.arange(s.free.shape[0] + 1) - 0.5
    axs[0].pcolormesh(X, Y, t, cmap=ListedColormap(TCOLORS + ["#2B2B2B"]), vmin=0, vmax=4, rasterized=False, edgecolor="face", linewidth=0.05)
    axs[0].set_title("True terrain, all routes")
    for m in res:
        tr = res[m][1]
        axs[0].plot(tr[:, 1], tr[:, 0], color=STYLE[m][0], ls=STYLE[m][2], lw=0.9)
    cmap = plt.get_cmap("Greys").copy(); cmap.set_bad("#2B2B2B")
    norm = PowerNorm(0.6, vmin=0, vmax=0.06)
    im = None
    for ax, m in zip(axs[1:], res):
        r, tr, h = res[m]
        im = ax.pcolormesh(X, Y, np.ma.masked_where(~s.free, h), cmap=cmap, norm=norm, edgecolor="face", linewidth=0.05)
        ax.set_facecolor("#2B2B2B")
        ax.plot(tr[:, 1], tr[:, 0], color=STYLE[m][0], lw=1.0, ls=STYLE[m][2])
        ax.set_title(f"{m}: $P_s$ = {r['p_survive']:.2f}")
    for ax in axs:
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.plot(s.start[1], s.start[0], "o", color="white", mec="black", ms=3, mew=0.5)
        ax.plot(s.goal[1], s.goal[0], "*", color="white", mec="black", ms=5, mew=0.5)
    cax = fig.add_subplot(gs[1, 2:4])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("Believed hazard $\\bar h_i$ after the mission (1/m)", labelpad=1); cb.ax.tick_params(length=2, labelsize=6)
    h1 = [Patch(fc=c, ec="#888888", lw=0.3, label=n) for c, n in zip(TCOLORS, TERRAIN)]
    fig.legend(handles=h1, loc="lower left", bbox_to_anchor=(0.0, 0.0), ncol=2, fontsize=6, columnspacing=0.6, handlelength=1.0, handletextpad=0.3)
    save(fig, "fig4_risk_maps_trajectories")


def dots(ax, g, col, order, xlabel):
    for i, m in enumerate(order):
        r = g[g.method == m].iloc[0]
        c, mk, _ = STYLE[m]
        ax.errorbar(r[col], i, xerr=[[r[col] - r[col + "_lo"]], [r[col + "_hi"] - r[col]]], fmt=mk, color=c, ms=4, capsize=1.5, lw=0.8)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order); ax.invert_yaxis(); ax.set_xlabel(xlabel, labelpad=1); ax.grid(axis="y", visible=False)


def fig_performance():
    e1 = pd.read_csv(T / "E1_summary.csv")
    g = e1[e1.eps == 0.1]
    order = ["M0", "M1", "M2", "M3", "ORACLE"]
    fig, axs = plt.subplots(1, 3, figsize=(ONE, 1.3), sharey=True)
    dots(axs[0], g, "p_survive", order, "Survival $P_s$ (-)")
    dots(axs[1], g, "exposure_m", order, "Exposure (m)")
    dots(axs[2], g, "length_m", order, "Path length (m)")
    fig.tight_layout(pad=0.3, w_pad=0.5)
    save(fig, "fig5_performance")


def fig_robustness():
    e1 = pd.read_csv(T / "E1_summary.csv")
    fig, axs = plt.subplots(1, 2, figsize=(ONE, 1.45))
    for ax, col, lab in ((axs[0], "p_survive", "Survival $P_s$ (-)"), (axs[1], "exposure_m", "Hazard exposure (m)")):
        for m in STYLE:
            g = e1[e1.method == m].sort_values("eps")
            c, mk, ls = STYLE[m]
            ax.fill_between(g.eps, g[col + "_lo"], g[col + "_hi"], color=c, alpha=0.15, lw=0)
            ax.plot(g.eps, g[col], color=c, ls=ls, marker=mk, ms=3, label=LABEL[m])
        ax.set_xlabel("Classifier error rate $\\varepsilon$ (-)", labelpad=1); ax.set_ylabel(lab, labelpad=1)
    h, l = axs[0].get_legend_handles_labels()
    fig.tight_layout(pad=0.3)
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.97), columnspacing=0.8, handlelength=2.2)
    save(fig, "fig6_robustness")


def fig_confusion():
    rep = json.load(open(ROOT / "results" / "classifier" / "classifier_report.json"))
    fig, axs = plt.subplots(1, 2, figsize=(ONE, 1.45), constrained_layout=True)
    names = ["nom.", "low $\\mu$", "gran.", "rough"]
    for ax, v in zip(axs, CFG):
        c = np.array(rep[v]["heldout_test"]["confusion"], float)
        n = c / c.sum(axis=1, keepdims=True)
        ax.pcolormesh(np.arange(5) - 0.5, np.arange(5) - 0.5, n, cmap="Greys", vmin=0, vmax=1, edgecolors="#BBBBBB", linewidth=0.3)
        ax.invert_yaxis(); ax.set_aspect("equal")
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{n[i, j]:.2f}", ha="center", va="center", fontsize=6.5, color="white" if n[i, j] > 0.55 else "black")
        ax.set_xticks(range(4)); ax.set_xticklabels(names); ax.set_yticks(range(4)); ax.set_yticklabels(names)
        ax.set_xlabel("Predicted", labelpad=1); ax.set_title(f"{CFG[v]} (macro-F1 = {rep[v]['heldout_test']['f1_macro']:.3f})"); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
    axs[0].set_ylabel("True", labelpad=1)
    save(fig, "fig7_confusion")


def fig_sensitivity():
    c = pd.read_csv(T / "E3_compare.csv")
    ab = pd.read_csv(T / "E4_compare.csv"); ab = ab[ab.metric == "p_survive"]
    fig, axs = plt.subplots(1, 2, figsize=(TWO, 2.68), gridspec_kw=dict(width_ratios=[2.0, 1]))
    ax = axs[0]
    names = {"aliasing": "aliased share", "appearance_noise": "appearance noise", "hazard_scale": "true hazard $\\times$", "believed_hazard": "believed hazard $\\times$",
             "lambda": "$\\lambda$", "hazard_fraction": "hazard coverage", "believed_eps": "believed $\\varepsilon$", "window_m": "window (m)"}
    # Every factor's reference level is the same run: it is drawn once, as the first row.
    reference = {"aliasing": 0.15, "appearance_noise": 0.15, "hazard_scale": 1.0, "believed_hazard": 1.0, "lambda": 25.0, "hazard_fraction": 0.3, "believed_eps": 0.1, "window_m": 1.0}
    y, ticks, labs = 0, [], []
    for (a, b), dy, col in ((("M2", "M1"), -0.2, "M2"), (("M3", "M2"), 0.2, "M3")):
        r = c[(c.factor == "aliasing") & (c.level == 0.15) & (c.a == a) & (c.b == b)].iloc[0]
        ax.errorbar(r.mean_diff, y + dy, xerr=[[r.mean_diff - r.ci_lo], [r.ci_hi - r.mean_diff]], fmt=STYLE[col][1], color=STYLE[col][0], ms=2.2, lw=0.6, capsize=0.8)
    ticks.append(y); labs.append("reference setting"); y += 1.6
    for fac in names:
        g = c[(c.factor == fac) & (c.level != reference[fac])]
        for lv in sorted(g.level.unique()):
            for (a, b), dy, col in ((("M2", "M1"), -0.2, "M2"), (("M3", "M2"), 0.2, "M3")):
                r = g[(g.level == lv) & (g.a == a) & (g.b == b)].iloc[0]
                ax.errorbar(r.mean_diff, y + dy, xerr=[[r.mean_diff - r.ci_lo], [r.ci_hi - r.mean_diff]], fmt=STYLE[col][1], color=STYLE[col][0], ms=2.2, lw=0.6, capsize=0.8)
            ticks.append(y); labs.append(f"{names[fac]} {lv:g}"); y += 1
        y += 0.6
    ax.axvline(0, color="#333333", lw=0.6); ax.set_yticks(ticks); ax.set_yticklabels(labs, fontsize=6); ax.set_ylim(y - 0.4, -0.8); ax.grid(axis="y", visible=False); ax.tick_params(axis="y", length=1, pad=1)
    ax.set_xlabel("Paired difference in survival $P_s$ (95% CI, 100 maps)", labelpad=1)
    ax.errorbar([], [], fmt=STYLE["M2"][1], color=STYLE["M2"][0], ms=3, label="M2 $-$ M1"); ax.errorbar([], [], fmt=STYLE["M3"][1], color=STYLE["M3"][0], ms=3, label="M3 $-$ M2")
    ax.legend(loc="lower right")
    ax = axs[1]
    lab = {"M3-noconf": "no confusion model", "M3-notemper": "no tempering", "M3-nounc": "no uncertainty penalty", "M3-noprop": "no propagation"}
    y = 0; ticks, labs = [], []
    for v, name in lab.items():
        for e, dy, mk, fc in ((0.1, -0.16, "o", "#333333"), (0.3, 0.16, "s", "white")):
            r = ab[(ab.a == v) & (ab.eps == e)].iloc[0]
            ax.errorbar(r.mean_diff, y + dy, xerr=[[r.mean_diff - r.ci_lo], [r.ci_hi - r.mean_diff]], fmt=mk, mfc=fc, mec="#333333", color="#333333", ms=3, lw=0.7, capsize=1,
                        label=f"$\\varepsilon$={e}" if y == 0 else None)
        ticks.append(y); labs.append(name); y += 1
    ax.axvline(0, color="#333333", lw=0.6); ax.set_yticks(ticks); ax.set_yticklabels(labs, fontsize=6); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Ablated $-$ full M3, $\\Delta P_s$", labelpad=1); ax.legend(loc="center left", fontsize=6, bbox_to_anchor=(0.0, 0.62))
    fig.tight_layout(pad=0.3, w_pad=0.6)
    save(fig, "fig8_sensitivity_ablation")


def fig_scenarios():
    e2 = pd.read_csv(T / "E2_summary.csv")
    sc = [("equal", "Equal hazard (assumed)"), ("wheel_sensitive", "Wheel-sensitive (assumed)"), ("mujoco_falls", "Falls per metre (MuJoCo)")]
    fig, axs = plt.subplots(1, 3, figsize=(TWO, 1.2), sharey=True)
    order = ["M0", "M1", "M2", "M3", "ORACLE"]
    for ax, (k, title) in zip(axs, sc):
        for i, v in enumerate(CFG):
            g = e2[(e2.scenario == k) & (e2.config == v)]
            for j, m in enumerate(order):
                r = g[g.method == m].iloc[0]
                c, mk, _ = STYLE[m]
                ax.errorbar(j + (i - 0.5) * 0.36, r.p_survive, yerr=[[r.p_survive - r.p_survive_lo], [r.p_survive_hi - r.p_survive]], fmt=mk, color=c, mfc=c if i == 0 else "white", ms=3.5, lw=0.8, capsize=1.2)
        ax.set_xticks(range(5)); ax.set_xticklabels(["M0", "M1", "M2", "M3", "Orac."]); ax.set_title(title); ax.grid(axis="x", visible=False)
    axs[0].set_ylabel("Survival $P_s$ (-)", labelpad=1)
    axs[0].errorbar([], [], fmt="o", color="#333333", ms=3.5, label="Legged"); axs[0].errorbar([], [], fmt="o", color="#333333", mfc="white", ms=3.5, label="Wheeled")
    axs[0].legend(loc="lower right", handletextpad=0.1)
    fig.tight_layout(pad=0.3)
    save(fig, "fig3_measured_scenarios")


if __name__ == "__main__":
    fig_architecture(); fig_quadrupeds(); fig_mujoco(); fig_configurations(); fig_scenarios(); fig_performance(); fig_robustness(); fig_confusion(); fig_sensitivity()
    fig_maps(int(sys.argv[1]) if len(sys.argv) > 1 else 12)
    print(sorted(p.name for p in F.glob("*.pdf")))
