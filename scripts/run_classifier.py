#!/usr/bin/env python3
"""Compare IMU, joints and fusion by episode; write metrics and the observation models for the Monte Carlo."""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats as st
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.physics import classify as c  # noqa: E402
from src.physics.features import episode_windows  # noqa: E402
import joblib  # noqa: E402

P = ROOT / "data" / "processed"
OUT = ROOT / "results" / "classifier"
OUT.mkdir(parents=True, exist_ok=True)
rows, report = [], {}
for variant in ("point_foot", "wheel_legged"):
    z = dict(np.load(P / f"features_campaign_{variant}_n1.npz", allow_pickle=False))
    home = z["kind"] == "homogeneous"
    y, g = c.label_index(z["label"][home]), z["episode"][home]
    report[variant] = dict(n_windows=int(home.sum()), n_episodes=int(len(np.unique(g))),
                           windows_per_class=np.bincount(y, minlength=4).tolist())
    probs = {}
    for fs in ("imu", "joints", "fusion"):
        x = c.design(z, fs, home)
        for name in ("logistic", "forest"):
            p = c.oof(x, y, g, name)
            s = c.scores(p, y)
            lo, hi = c.episode_bootstrap_f1(p, y, g)
            probs[(fs, name)] = p
            rows.append(dict(variant=variant, features=fs, model=name, split="by_episode", **{k: s[k] for k in ("accuracy", "balanced_accuracy", "f1_macro", "brier", "ece")},
                             f1_lo=lo, f1_hi=hi, f1_per_class=json.dumps(s["f1"]), recall_per_class=json.dumps(s["recall"]), confusion=json.dumps(s["confusion"])))
    # Measure what the leak would have been: the same model with a random split of windows.
    x = c.design(z, "fusion", home)
    p_leak = c.oof(x, y, g, "forest", grouped=False)
    s = c.scores(p_leak, y)
    rows.append(dict(variant=variant, features="fusion", model="forest", split="random_windows_LEAKY", **{k: s[k] for k in ("accuracy", "balanced_accuracy", "f1_macro", "brier", "ece")}))
    # H4: fusion against each single channel, paired over episodes
    fus = probs[("fusion", "forest")]
    report[variant]["fusion_minus_imu_f1_ci"] = c.episode_bootstrap_f1(fus, y, g, other=probs[("imu", "forest")])
    report[variant]["fusion_minus_joints_f1_ci"] = c.episode_bootstrap_f1(fus, y, g, other=probs[("joints", "forest")])
    # Condition shift: train on slower commands, test on faster ones, no shared episode.
    vx = z["cmd"][home, 0]
    tr, te = vx < 0.65, vx >= 0.65
    m = c.model("forest").fit(x[tr], y[tr])
    report[variant]["speed_shift"] = c.scores(m.predict_proba(x[te]), y[te])
    # Noise: three times the assumed sensor noise
    z3 = dict(np.load(P / f"features_campaign_{variant}_n3.npz", allow_pickle=False))
    home3 = z3["kind"] == "homogeneous"
    p3 = c.oof(c.design(z3, "fusion", home3), c.label_index(z3["label"][home3]), z3["episode"][home3], "forest")
    report[variant]["noise_x3"] = c.scores(p3, c.label_index(z3["label"][home3]))
    # Final model, three-way split; observation models for the Monte Carlo
    tr, ca, te = c.split_episodes(y, g)
    final = c.model("forest").fit(x[tr], y[tr])
    calib, test = c.binned_confusion(final.predict_proba(x[ca]), y[ca]), c.binned_confusion(final.predict_proba(x[te]), y[te])
    report[variant]["heldout_test"] = c.scores(final.predict_proba(x[te]), y[te])
    report[variant]["inference_ms_single_thread_host"] = None
    final1 = c.model("forest"); final1.set_params(n_jobs=1); final1.fit(x[tr], y[tr])
    report[variant]["inference_ms_single_thread_host"] = c.inference_ms(final1, x[te])
    json.dump(dict(calibration=calib, test=test, threshold=c.CONF_THRESHOLD), open(OUT / f"obs_model_{variant}.json", "w"), indent=1)
    joblib.dump(final, OUT / f"final_forest_{variant}.joblib", compress=3)
    # Conditions never trained on
    slope = z["kind"] == "slope"
    if slope.any():
        pr = final.predict_proba(c.design(z, "fusion", slope))
        report[variant]["slope_prediction_share"] = (np.bincount(pr.argmax(axis=1), minlength=4) / len(pr)).tolist()
        report[variant]["slope_n_windows"] = int(slope.sum())
    course = z["kind"] == "course"
    if course.any():
        pr = final.predict_proba(c.design(z, "fusion", course))
        yc = c.label_index(z["label"][course])
        report[variant]["mixed_course"] = {**c.scores(pr, yc), "n_windows": int(course.sum()), "n_episodes": int(len(np.unique(z["episode"][course])))}
pd.DataFrame(rows).to_csv(OUT / "classifier_metrics.csv", index=False)
json.dump(report, open(OUT / "classifier_report.json", "w"), indent=1)

# Failure model from the same campaign, kept apart from the classifier: falls per metre, Jeffreys posterior.
d = pd.read_csv(P / "episodes_campaign.csv")
d = d[d.kind.isin(["homogeneous", "slope"]) & d.ok]
hz = []
for (v, cls), g in d.groupby(["variant", "cls"]):
    k, D = int(g.fell.sum()), float(g.distance_m.sum())
    hz.append(dict(variant=v, cls=cls, episodes=len(g), falls=k, distance_m=D, h_mean=(k + 0.5) / D,
                   h_lo=float(st.gamma.ppf(0.025, k + 0.5) / D), h_hi=float(st.gamma.ppf(0.975, k + 0.5) / D),
                   mean_slip=g.slip_in_contact.mean(), speed_tracking=g.speed_tracking.mean(), tau_rms=g.tau_rms.mean(), acc_std=g.acc_std.mean(),
                   vertical_force=g.contact_force_mean.mean()))
pd.DataFrame(hz).to_csv(OUT / "failure_rates_mujoco.csv", index=False)
print(pd.DataFrame(rows)[["variant", "features", "model", "split", "accuracy", "balanced_accuracy", "f1_macro", "f1_lo", "f1_hi", "brier", "ece"]].round(3).to_string())
print(pd.DataFrame(hz).round(4).to_string())
