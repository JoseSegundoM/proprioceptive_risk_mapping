#!/usr/bin/env python3
"""LaTeX tables and every number quoted in the text, as macros, from saved results. Nothing is typed by hand."""
import json, re
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
T, OUT, C = ROOT / "results" / "tables", ROOT / "paper" / "tables", ROOT / "results" / "classifier"
OUT.mkdir(parents=True, exist_ok=True)
mac = {}


def _signs(text):
    """A true, unbreakable minus sign, in tables and in running text."""
    return re.sub(r"(?<![\d.])-(?=\d)", r"\\mbox{$-$}", text)


def m(name, val, fmt="{:.3f}"):
    assert re.fullmatch(r"[a-zA-Z]+", name), name
    mac[name] = _signs(fmt.format(val)) if not isinstance(val, str) else val


def ci(r, f="{:+.3f}"):
    return _signs(f"{f.format(r.mean_diff)} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]")


def get(df, **kw):
    for k, v in kw.items():
        df = df[df[k] == v]
    assert len(df) == 1, (kw, len(df))
    return df.iloc[0]


# ---- Table I: MuJoCo per terrain and configuration
d = pd.read_csv(C / "failure_rates_mujoco.csv")
names = {"nominal": "Nominal", "low_friction": "Low friction", "granular": "Granular-like", "rough": "Rough", "slope": "Slope (5--12$^\\circ$)"}
L = [r"\begin{tabular}{@{}lrcrr@{}}", r"\toprule", r"Terrain & Falls/ep. & $h$ (1/m), 95\,\% CrI & Slip (m/s) & Speed ratio \\ \midrule"]
L.append(r"\multicolumn{5}{@{}l}{\emph{Legged}} \\")
for c in names:
    r = get(d, variant="point_foot", cls=c)
    L.append(f"{names[c]} & {int(r.falls)}/{int(r.episodes)} & {r.h_mean:.4f} [{r.h_lo:.4f}, {r.h_hi:.4f}] & {r.mean_slip:.3f} & {r.speed_tracking:.2f} \\\\")
w = d[d.variant == "wheel_legged"]
L.append(r"\multicolumn{5}{@{}l}{\emph{Wheeled (all five terrains; ranges)}} \\")
L.append(f"All & {int(w.falls.sum())}/{int(w.episodes.sum())} & {w.h_mean.min():.4f}--{w.h_mean.max():.4f} [upper {w.h_hi.max():.4f}] & {w.mean_slip.min():.3f}--{w.mean_slip.max():.3f} & {w.speed_tracking.min():.2f}--{w.speed_tracking.max():.2f} \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
(OUT / "tab_physics.tex").write_text("\n".join(L))
for v, p in (("point_foot", "Leg"), ("wheel_legged", "Whe")):
    for c, q in (("low_friction", "Low"), ("granular", "Gran"), ("nominal", "Nom"), ("rough", "Rough")):
        r = get(d, variant=v, cls=c)
        m(f"falls{p}{q}", f"{int(r.falls)}/{int(r.episodes)}")
        m(f"h{p}{q}", r.h_mean, "{:.3f}")
        m(f"slip{p}{q}", r.mean_slip, "{:.3f}")
        m(f"track{p}{q}", r.speed_tracking, "{:.2f}")

# ---- Terrain parameters: the ranges of src/physics/terrains.py, assumed scenario values
L = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule", r"Terrain & Friction $\mu$ & Relief (mm) & Contact time constant (s) \\ \midrule",
     r"Nominal & 0.70--1.20 & 0--6 & 0.02 \\", r"Low friction & 0.12--0.30 & 0--6 & 0.02 \\",
     r"Granular-like & 0.45--0.80 & 5--15 & 0.05--0.15 \\", r"Rough & 0.70--1.20 & 25--60 & 0.02 \\",
     r"Ramp (5--12$^\circ$, test only) & 0.70--1.20 & none & 0.02 \\", r"\bottomrule", r"\end{tabular}"]
(OUT / "tab_terrain.tex").write_text("\n".join(L))

# ---- Table II: classifier
k = pd.read_csv(C / "classifier_metrics.csv")
rep = json.load(open(C / "classifier_report.json"))
gen = pd.read_csv(C / "generalization.csv")
L = [r"\begin{tabular}{@{}lcc@{}}", r"\toprule", r"Macro-F1 (Brier score) & Legged & Wheeled \\ \midrule"]
fname = {"imu": "IMU", "joints": "Joints", "fusion": "IMU + joints"}
for fs, mod, ml in (("imu", "forest", "forest"), ("joints", "forest", "forest"), ("fusion", "forest", "forest"), ("fusion", "logistic", "logistic")):
    if True:
        row = [f"{fname[fs]}, {ml}"]
        for v in ("point_foot", "wheel_legged"):
            r = get(k, variant=v, features=fs, model=mod, split="by_episode")
            row.append(f"{r.f1_macro:.3f} [{r.f1_lo:.3f}, {r.f1_hi:.3f}] ({r.brier:.3f})")
        L.append(" & ".join(row) + r" \\")
L.append(r"\midrule \multicolumn{3}{@{}l}{\emph{Stress tests, IMU + joints, forest}} \\")
def _shift(v, key):
    g_ = gen[(gen.variant == v) & (gen.model == "forest") & (gen.with_cmd) & gen["shift"].str.startswith(key)]
    return g_.f1_macro.min()


stress = [("Random window split (leaky)", lambda v: get(k, variant=v, features="fusion", model="forest", split="random_windows_LEAKY").f1_macro),
          ("Held-out 20\\,\\% of episodes", lambda v: rep[v]["heldout_test"]["f1_macro"]),
          ("Sensor noise $\\times 3$", lambda v: rep[v]["noise_x3"]["f1_macro"]),
          ("Mixed terrain course", lambda v: rep[v]["mixed_course"]["f1_macro"]),
          ("Unseen speed range (worse of two)", lambda v: _shift(v, "speed")),
          ("Unseen terrain parameters (worse of two)", lambda v: _shift(v, "terrain"))]
for lab, fn in stress:
    L.append(f"{lab} & {fn('point_foot'):.3f} & {fn('wheel_legged'):.3f} \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
(OUT / "tab_classifier.tex").write_text("\n".join(L))
for v, p in (("point_foot", "Leg"), ("wheel_legged", "Whe")):
    r = rep[v]
    m(f"nEp{p}", r["n_episodes"], "{:d}"); m(f"nWin{p}", r["n_windows"], "{:d}")
    m(f"fusImuLo{p}", r["fusion_minus_imu_f1_ci"][0], "{:+.4f}"); m(f"fusImuHi{p}", r["fusion_minus_imu_f1_ci"][1], "{:+.4f}")
    m(f"fusJointLo{p}", r["fusion_minus_joints_f1_ci"][0], "{:+.3f}"); m(f"fusJointHi{p}", r["fusion_minus_joints_f1_ci"][1], "{:+.3f}")
    m(f"fOne{p}", get(k, variant=v, features="fusion", model="forest", split="by_episode").f1_macro)
    m(f"fHeld{p}", r["heldout_test"]["f1_macro"])
    m(f"fCourse{p}", r["mixed_course"]["f1_macro"]); m(f"nCourse{p}", r["mixed_course"]["n_episodes"], "{:d}")
    m(f"fLeak{p}", get(k, variant=v, features="fusion", model="forest", split="random_windows_LEAKY").f1_macro)
    m(f"inferMs{p}", r["inference_ms_single_thread_host"], "{:.0f}")
    sl = r["slope_prediction_share"]
    m(f"slopeHazard{p}", 100 * (1 - sl[0]), "{:.0f}")
    m(f"slopeNom{p}", 100 * sl[0], "{:.0f}")
for v, p_ in (("point_foot", "Leg"), ("wheel_legged", "Whe")):
    for i, q in enumerate(("Nom", "Low", "Gran", "Rough")):
        m(f"rec{p_}{q}", rep[v]["heldout_test"]["recall"][i], "{:.2f}")
    z_ = np.load(ROOT / "data" / "processed" / f"features_campaign_{v}_n1.npz")
    m(f"dimImu{p_}", z_["imu"].shape[1], "{:d}"); m(f"dimJoint{p_}", z_["joints"].shape[1], "{:d}")
    for key, q in (("speed_shift", "Sh"), ("noise_x3", "Nx")):
        m(f"f{q}{p_}", rep[v][key]["f1_macro"])
    m(f"eceFor{p_}", get(k, variant=v, features="fusion", model="forest", split="by_episode").ece)
    m(f"eceLog{p_}", get(k, variant=v, features="fusion", model="logistic", split="by_episode").ece)
    m(f"fLog{p_}", get(k, variant=v, features="fusion", model="logistic", split="by_episode").f1_macro)
g = gen[(gen.model == "forest") & (gen.with_cmd)]
gl = gen[(gen.model == "logistic") & (gen.with_cmd)]
m("speedLegMin", g[(g.variant == "point_foot") & g["shift"].str.startswith("speed")].f1_macro.min())
m("speedLegLogMin", gl[(gl.variant == "point_foot") & gl["shift"].str.startswith("speed")].f1_macro.min())
m("paramLegMin", g[(g.variant == "point_foot") & g["shift"].str.startswith("terrain")].f1_macro.min())
m("wheelShiftMin", g[(g.variant == "wheel_legged")].f1_macro.min())
m("hazFrac", 100 * np.mean([(lambda q: (q.terrain[q.free] > 0).mean())(__import__("src.riskmc.maps", fromlist=["generate"]).generate(i)) for i in range(200)]), "{:.0f}")

# ---- Table III: main result
e1 = pd.read_csv(T / "E1_summary.csv")
ec = pd.read_csv(T / "E1_compare.csv")
s = e1[e1.eps == 0.1]
L = [r"\begin{tabular}{@{}lcccrr@{}}", r"\toprule", r"Method & $P_s$ & Success & Exposure (m) & Path (m) & Replans \\ \midrule"]
for me, lab in (("M0", "M0 geometry"), ("M1", "M1 local"), ("M2", "M2 appearance"), ("M3", "M3 fusion"), ("ORACLE", "Oracle (bound)")):
    r = get(s, method=me)
    L.append(f"{lab} & {r.p_survive:.3f} [{r.p_survive_lo:.3f}, {r.p_survive_hi:.3f}] & {r.success:.3f} & {r.exposure_m:.2f} [{r.exposure_m_lo:.2f}, {r.exposure_m_hi:.2f}] & {r.length_m:.1f} & {r.replans:.1f} \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
(OUT / "tab_main.tex").write_text("\n".join(L))
for me, p in (("M0", "Mz"), ("M1", "Mo"), ("M2", "Mt"), ("M3", "Mh"), ("ORACLE", "Orc")):
    r = get(s, method=me)
    m(f"ps{p}", r.p_survive); m(f"exp{p}", r.exposure_m, "{:.2f}"); m(f"len{p}", r.length_m, "{:.1f}"); m(f"rep{p}", r.replans, "{:.1f}")
    m(f"planMs{p}", r.plan_ms_mean, "{:.1f}")
    m(f"wilsonLo{p}", r.success_wilson_lo); m(f"wilsonHi{p}", r.success_wilson_hi); m(f"succZero{p}", r.success_rep0)
    r5 = get(e1[e1.eps == 0.5], method=me)
    m(f"psFive{p}", r5.p_survive)

# ---- Table IV: contrasts across conditions
e2 = pd.read_csv(T / "E2_compare.csv"); e6 = pd.read_csv(T / "E6_compare.csv"); e7 = pd.read_csv(T / "E7_compare.csv")
conds = [("Synthetic $\\varepsilon$=0.1", ec[ec.eps == 0.1]), ("Synthetic $\\varepsilon$=0.3", ec[ec.eps == 0.3]),
         ("Repeat, $\\varepsilon$=0.1", e7[(e7.scenario == "equal") & (e7.eps == 0.1)]),
         ("Held-out conf., legged", e2[(e2.scenario == "equal") & (e2.config == "point_foot")]),
         ("Held-out conf., wheeled", e2[(e2.scenario == "equal") & (e2.config == "wheel_legged")]),
         ("Course conf., legged", e6[(e6.scenario == "equal") & (e6.config == "point_foot")]),
         ("MuJoCo falls, legged", e2[(e2.scenario == "mujoco_falls") & (e2.config == "point_foot")]),
         ("MuJoCo falls, wheeled", e2[(e2.scenario == "mujoco_falls") & (e2.config == "wheel_legged")])]
L = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule", r"Condition & M1 $-$ M0 & M2 $-$ M1 & M3 $-$ M2 \\ \midrule"]
for lab, g in conds:
    g = g[g.metric == "p_survive"]
    L.append(lab + " & " + " & ".join(ci(get(g, a=a, b=b)) for a, b in (("M1", "M0"), ("M2", "M1"), ("M3", "M2"))) + r" \\")
L += [r"\bottomrule", r"\end{tabular}"]
(OUT / "tab_contrasts.tex").write_text("\n".join(L))


# ---- Scenarios with measured confusion
su = pd.read_csv(T / "E2_summary.csv")
L = [r"\begin{tabular}{@{}llcccc@{}}", r"\toprule", r"Failure model & Config. & M0 & M1 & M2 & M3 \\ \midrule"]
for sc, lab in (("equal", "Equal (assumed)"), ("wheel_sensitive", "Wheel-sensitive (assumed)"), ("mujoco_falls", "Falls per metre (MuJoCo)")):
    for cfg, cl in (("point_foot", "Legged"), ("wheel_legged", "Wheeled")):
        row = [lab if cfg == "point_foot" else "", cl]
        for me in ("M0", "M1", "M2", "M3"):
            r = get(su, scenario=sc, config=cfg, method=me)
            row.append(f"{r.p_survive:.3f} ({r.exposure_m:.1f})")
        L.append(" & ".join(row) + r" \\")
L += [r"\bottomrule", r"\end{tabular}"]
(OUT / "tab_scenarios.tex").write_text("\n".join(L))


def pair(prefix, g, a, b):
    r = get(g[g.metric == "p_survive"], a=a, b=b)
    m(prefix, r.mean_diff, "{:+.3f}"); m(prefix + "Lo", r.ci_lo, "{:+.3f}"); m(prefix + "Hi", r.ci_hi, "{:+.3f}")
    m(prefix + "Dz", r.dz, "{:.2f}")
    if "p_holm" in r and r.p_holm == r.p_holm:
        m(prefix + "Holm", r.p_holm, "{:.3f}")


e = ec[ec.eps == 0.1]
pair("dOneZero", e, "M1", "M0"); pair("dTwoOne", e, "M2", "M1"); pair("dThreeTwo", e, "M3", "M2"); pair("dThreeOne", e, "M3", "M1"); pair("dTwoZero", e, "M2", "M0"); pair("dThreeZero", e, "M3", "M0")
pair("dThreeTwoFive", ec[ec.eps == 0.5], "M3", "M2"); pair("dTwoOneFive", ec[ec.eps == 0.5], "M2", "M1")
r7 = e7[(e7.scenario == "equal") & (e7.eps == 0.1)]
pair("repOneZero", r7, "M1", "M0"); pair("repTwoOne", r7, "M2", "M1"); pair("repThreeTwo", r7, "M3", "M2")
f = e2[(e2.scenario == "mujoco_falls") & (e2.config == "point_foot")]
pair("fallTwoOne", f, "M2", "M1"); pair("fallThreeTwo", f, "M3", "M2"); pair("fallOneZero", f, "M1", "M0")
sm = e7[(e7.scenario == "mujoco_falls_legged")]
pair("repFallTwoOne", sm, "M2", "M1")
exr = get(e[e.metric == "cum_hazard"], a="M2", b="M0"); m("riskRedTwo", exr.risk_reduction_pct, "{:.0f}")
exr = get(e[e.metric == "cum_hazard"], a="M3", b="M0"); m("riskRedThree", exr.risk_reduction_pct, "{:.0f}")
exr = get(e[e.metric == "cum_hazard"], a="M1", b="M0"); m("riskRedOne", exr.risk_reduction_pct, "{:.1f}")
x = get(e[e.metric == "exposure_m"], a="M2", b="M0"); m("expRedTwoAbs", -x.mean_diff, "{:.2f}")
x = get(e[e.metric == "length_m"], a="M3", b="M0"); m("lenAddThree", x.mean_diff, "{:.2f}")
x = get(e[e.metric == "length_m"], a="M2", b="M0"); m("lenAddTwo", x.mean_diff, "{:.2f}")

su2 = pd.read_csv(T / "E2_summary.csv")
for cfg, p in (("point_foot", "Leg"), ("wheel_legged", "Whe")):
    for me, q in (("M0", "Mz"), ("M3", "Mh"), ("ORACLE", "Orc")):
        m(f"fallPs{p}{q}", get(su2, scenario="mujoco_falls", config=cfg, method=me).p_survive)
ii = pd.read_csv(T / "E2_interaction.csv")
for sc, p in (("equal", "Equal"), ("mujoco_falls", "Falls"), ("wheel_sensitive", "Sens")):
    r = get(ii, scenario=sc, contrast="M2-M1")
    m(f"intWhe{p}", r.wheels_gain, "{:+.3f}"); m(f"intLeg{p}", r.legs_gain, "{:+.3f}")
    m(f"intDiff{p}", r.mean_diff, "{:+.3f}"); m(f"intLo{p}", r.ci_lo, "{:+.3f}"); m(f"intHi{p}", r.ci_hi, "{:+.3f}")

for cfg, p_ in (("point_foot", "Leg"), ("wheel_legged", "Whe")):
    for (sc_tab, name_), q in (((e6, "equal"), "Cc"),):
        m(f"courseGain{p_}", get(sc_tab[(sc_tab.scenario == name_) & (sc_tab.config == cfg) & (sc_tab.metric == "p_survive")], a="M2", b="M1").mean_diff, "{:+.3f}")
        m(f"courseThree{p_}", get(sc_tab[(sc_tab.scenario == name_) & (sc_tab.config == cfg) & (sc_tab.metric == "p_survive")], a="M3", b="M2").mean_diff, "{:+.3f}")
s3 = pd.read_csv(T / "E3_summary.csv")
for lv, q in ((0.25, "Low"), (4.0, "High")):
    m(f"psScale{q}", get(s3, factor="hazard_scale", level=lv, method="M0").p_survive)
e3 = pd.read_csv(T / "E3_compare.csv"); e3 = e3[e3.metric == "p_survive"]
r = get(e3, factor="believed_eps", level=0.02, a="M3", b="M2"); m("believedLow", r.mean_diff, "{:+.3f}")
r = get(e3, factor="believed_eps", level=0.3, a="M3", b="M2"); m("believedHigh", r.mean_diff, "{:+.3f}")
r = get(e3, factor="window_m", level=2.0, a="M2", b="M1"); m("windowLong", r.mean_diff, "{:+.3f}")
r = get(e3, factor="window_m", level=0.5, a="M2", b="M1"); m("windowShort", r.mean_diff, "{:+.3f}")
r = get(e3, factor="aliasing", level=0.5, a="M2", b="M1"); m("aliasHigh", r.mean_diff, "{:+.3f}"); m("aliasHighLo", r.ci_lo, "{:+.3f}"); m("aliasHighHi", r.ci_hi, "{:+.3f}")
r = get(e3, factor="aliasing", level=0.0, a="M2", b="M1"); m("aliasZero", r.mean_diff, "{:+.3f}")
r = get(e3, factor="appearance_noise", level=0.5, a="M2", b="M1"); m("noiseHigh", r.mean_diff, "{:+.3f}"); m("noiseHighLo", r.ci_lo, "{:+.3f}"); m("noiseHighHi", r.ci_hi, "{:+.3f}")
r = get(e3, factor="hazard_scale", level=0.25, a="M2", b="M1"); m("hzLow", r.mean_diff, "{:+.3f}")
r = get(e3, factor="hazard_scale", level=4.0, a="M2", b="M1"); m("hzHigh", r.mean_diff, "{:+.3f}")

e4 = pd.read_csv(T / "E4_compare.csv"); e4 = e4[(e4.metric == "p_survive") & (e4.eps == 0.1)]
for v, p in (("M3-noconf", "Conf"), ("M3-notemper", "Temper"), ("M3-nounc", "Unc"), ("M3-noprop", "Prop")):
    r = get(e4, a=v); m(f"ab{p}", r.mean_diff, "{:+.3f}"); m(f"ab{p}Lo", r.ci_lo, "{:+.3f}"); m(f"ab{p}Hi", r.ci_hi, "{:+.3f}")
b = pd.read_csv(T / "E5_best_vs_best.csv").set_index("eps")
m("bestBest", b.loc[0.1].mean_diff, "{:+.3f}"); m("bestBestLo", b.loc[0.1].ci_lo, "{:+.3f}"); m("bestBestHi", b.loc[0.1].ci_hi, "{:+.3f}")
s5 = pd.read_csv(T / "E5_summary.csv")
m("lamHighNouncThree", get(s5, eps=0.3, lam=400.0, method="M3-nounc").p_survive); m("lamHighThree", get(s5, eps=0.3, lam=400.0, method="M3").p_survive); m("lamHighTwo", get(s5, eps=0.3, lam=400.0, method="M2").p_survive)
t = json.load(open(ROOT / "results" / "mc" / "tuning_summary.json"))
m("tunedKernelThree", t["m3_best"]["kernel_sigma"], "{:g}"); m("tunedKernelTwo", t["m2_best"]["kernel_sigma"], "{:g}"); m("tunedTau", t["m2_best"]["vote_threshold"], "{:g}")
m("tunedRho", t["m3_best"]["tempering"], "{:g}"); m("tunedKappa", t["m3_best"]["uncertainty_kappa"], "{:g}")
e8 = pd.read_csv(T / "E8_compare.csv"); e8 = e8[e8.metric == "p_survive"]
x = get(e8[(e8.scenario == "equal") & (e8.eps == 0.1)], a="M2", b="M1s"); m("spatialGain", x.mean_diff, "{:+.3f}"); m("spatialGainLo", x.ci_lo, "{:+.3f}"); m("spatialGainHi", x.ci_hi, "{:+.3f}")
x = get(e8[(e8.scenario == "equal") & (e8.eps == 0.1)], a="M1s", b="M1"); m("spatialVsLocal", x.mean_diff, "{:+.3f}"); m("spatialVsLocalLo", x.ci_lo, "{:+.3f}"); m("spatialVsLocalHi", x.ci_hi, "{:+.3f}")
ep = pd.read_csv(ROOT / "data" / "processed" / "episodes_campaign.csv")
m("nEpisodesTotal", len(ep), "{:d}")
total = sum(len(pd.read_csv(ROOT / "results" / "mc" / f"{n}.csv.gz", usecols=["map_id"])) for n in ("E1_perception_sweep", "E2_measured_perception", "E3_sensitivity", "E4_ablation", "E5_lambda", "E6_course_perception", "E7_independent_repeat", "E8_spatial_baseline"))
m("nMissionsTotal", f"{total:,}".replace(",", "\\,"))
(OUT / "values.tex").write_text("".join(f"\\newcommand{{\\v{k}}}{{{v}}}\n" for k, v in sorted(mac.items())))
print(len(mac), "macros")
