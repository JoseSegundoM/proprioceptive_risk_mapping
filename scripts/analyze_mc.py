#!/usr/bin/env python3
"""Map-level statistics from the mission rows: python scripts/analyze_mc.py. Writes results/tables/*.csv."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.riskmc.stats import bootstrap_mean_ci, holm, paired_compare, wilson  # noqa: E402

MC, T = ROOT / "results" / "mc", ROOT / "results" / "tables"
T.mkdir(parents=True, exist_ok=True)
METRICS = ["p_survive", "success", "reached", "exposure_m", "cum_hazard", "length_m", "replans", "plan_ms_mean"]


def by_map(df, keys):
    return df.groupby(keys + ["method", "map_id"], as_index=False)[METRICS].mean()


def summary(df, keys):
    rows = []
    m = by_map(df, keys)
    for k, g in m.groupby(keys + ["method"]):
        k = (k,) if not isinstance(k, tuple) else k
        r = dict(zip(keys + ["method"], k))
        r["n_maps"] = len(g)
        for c in METRICS:
            lo, hi = bootstrap_mean_ci(g[c].values)
            r[c], r[c + "_lo"], r[c + "_hi"] = g[c].mean(), lo, hi
        first = df[(df.replicate == 0)]
        for kk, vv in zip(keys + ["method"], k):
            first = first[first[kk] == vv]
        s = int(first.success.sum())
        r["success_wilson_lo"], r["success_wilson_hi"] = wilson(s, len(first))
        r["success_rep0"] = s / max(len(first), 1)
        rows.append(r)
    return pd.DataFrame(rows)


def comparisons(df, keys, pairs, metrics=("p_survive", "exposure_m", "length_m", "cum_hazard")):
    rows = []
    m = by_map(df, keys)
    for k, g in m.groupby(keys):
        k = (k,) if not isinstance(k, tuple) else k
        piv = {c: g.pivot(index="map_id", columns="method", values=c) for c in metrics}
        fam = []
        for a, b in pairs:
            if a not in piv["p_survive"] or b not in piv["p_survive"]:
                continue
            for c in metrics:
                r = dict(zip(keys, k))
                r.update(a=a, b=b, metric=c, **paired_compare(piv[c][a].values, piv[c][b].values))
                if c == "cum_hazard":
                    base = piv[c][b].mean()
                    r["risk_reduction_pct"] = 100 * (base - piv[c][a].mean()) / base if base > 0 else float("nan")
                fam.append(r)
        ps = [x for x in fam if x["metric"] == "p_survive"]
        for x, adj in zip(ps, holm([x["p_wilcoxon"] for x in ps])):
            x["p_holm"] = adj
        rows += fam
    return pd.DataFrame(rows)


PAIRS = [("M1", "M0"), ("M2", "M0"), ("M2", "M1"), ("M3", "M0"), ("M3", "M1"), ("M3", "M2"), ("ORACLE", "M3")]
e1 = pd.read_csv(MC / "E1_perception_sweep.csv.gz")
summary(e1, ["eps"]).to_csv(T / "E1_summary.csv", index=False)
comparisons(e1, ["eps"], PAIRS).to_csv(T / "E1_compare.csv", index=False)

e2 = pd.read_csv(MC / "E2_measured_perception.csv.gz")
summary(e2, ["scenario", "config"]).to_csv(T / "E2_summary.csv", index=False)
comparisons(e2, ["scenario", "config"], PAIRS).to_csv(T / "E2_compare.csv", index=False)

# H5: does the gain from propagation differ between configurations? Interaction per map.
rows = []
m = by_map(e2, ["scenario", "config"])
for sc, g in m.groupby("scenario"):
    for a, b in (("M2", "M1"), ("M3", "M1"), ("M3", "M2"), ("M2", "M0"), ("M1", "M0")):
        p = g.pivot_table(index="map_id", columns=["config", "method"], values="p_survive")
        wheels = p["wheel_legged"][a] - p["wheel_legged"][b]
        legs = p["point_foot"][a] - p["point_foot"][b]
        rows.append(dict(scenario=sc, contrast=f"{a}-{b}", wheels_gain=wheels.mean(), legs_gain=legs.mean(), **paired_compare(wheels.values, legs.values)))
pd.DataFrame(rows).to_csv(T / "E2_interaction.csv", index=False)

e3 = pd.read_csv(MC / "E3_sensitivity.csv.gz")
summary(e3, ["factor", "level"]).to_csv(T / "E3_summary.csv", index=False)
comparisons(e3, ["factor", "level"], [("M1", "M0"), ("M2", "M1"), ("M3", "M2"), ("M3", "M1")], metrics=("p_survive",)).to_csv(T / "E3_compare.csv", index=False)

e4 = pd.read_csv(MC / "E4_ablation.csv.gz")
summary(e4, ["eps"]).to_csv(T / "E4_summary.csv", index=False)
comparisons(e4, ["eps"], [(v, "M3") for v in ("M3-noconf", "M3-notemper", "M3-nounc", "M3-noprop")]).to_csv(T / "E4_compare.csv", index=False)

pd.set_option("display.width", 230); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 200)
cols = ["method", "p_survive", "p_survive_lo", "p_survive_hi", "success", "exposure_m", "length_m", "replans", "plan_ms_mean", "reached"]
print(summary(e1, ["eps"])[["eps"] + cols].round(3).to_string())
c = comparisons(e1, ["eps"], PAIRS, metrics=("p_survive",))
print(c[["eps", "a", "b", "mean_diff", "ci_lo", "ci_hi", "p_holm", "dz", "rank_biserial"]].round(4).to_string())
print(summary(e2, ["scenario", "config"])[["scenario", "config"] + cols].round(3).to_string())
c = comparisons(e2, ["scenario", "config"], PAIRS, metrics=("p_survive",))
print(c[["scenario", "config", "a", "b", "mean_diff", "ci_lo", "ci_hi", "p_holm", "dz"]].round(4).to_string())
print(pd.DataFrame(rows).round(4).to_string())
print(summary(e4, ["eps"])[["eps"] + cols].round(3).to_string())
print(comparisons(e4, ["eps"], [(v, "M3") for v in ("M3-noconf", "M3-notemper", "M3-nounc", "M3-noprop")], metrics=("p_survive",))[["eps", "a", "mean_diff", "ci_lo", "ci_hi", "p_holm"]].round(4).to_string())
print(comparisons(e3, ["factor", "level"], [("M2", "M1"), ("M3", "M2")], metrics=("p_survive",))[["factor", "level", "a", "b", "mean_diff", "ci_lo", "ci_hi"]].round(4).to_string())

# Added experiments: lambda sweep, mixed-course perception, independent repetition.
e5 = pd.read_csv(MC / "E5_lambda.csv.gz")
s5 = summary(e5, ["eps", "lam"]); s5.to_csv(T / "E5_summary.csv", index=False)
m5 = by_map(e5, ["eps", "lam"])
rows5 = []
for eps, g in m5.groupby("eps"):
    best = g.groupby(["method", "lam"]).p_survive.mean().groupby("method").idxmax()
    a = g[(g.method == "M3") & (g.lam == best["M3"][1])].set_index("map_id")
    b = g[(g.method == "M2") & (g.lam == best["M2"][1])].set_index("map_id").loc[a.index]
    rows5.append(dict(eps=eps, lam_m3=best["M3"][1], lam_m2=best["M2"][1], len_m3=a.length_m.mean(), len_m2=b.length_m.mean(), **paired_compare(a.p_survive.values, b.p_survive.values)))
pd.DataFrame(rows5).to_csv(T / "E5_best_vs_best.csv", index=False)
e6 = pd.read_csv(MC / "E6_course_perception.csv.gz")
summary(e6, ["scenario", "config"]).to_csv(T / "E6_summary.csv", index=False)
comparisons(e6, ["scenario", "config"], PAIRS).to_csv(T / "E6_compare.csv", index=False)
e7 = pd.read_csv(MC / "E7_independent_repeat.csv.gz")
summary(e7, ["scenario", "eps"]).to_csv(T / "E7_summary.csv", index=False)
comparisons(e7, ["scenario", "eps"], PAIRS).to_csv(T / "E7_compare.csv", index=False)

e8 = pd.read_csv(MC / "E8_spatial_baseline.csv.gz")
summary(e8, ["scenario", "eps"]).to_csv(T / "E8_summary.csv", index=False)
comparisons(e8, ["scenario", "eps"], [("M1s", "M1"), ("M2", "M1s"), ("M3", "M1s"), ("M3", "M2")]).to_csv(T / "E8_compare.csv", index=False)
