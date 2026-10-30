#!/usr/bin/env python3
"""How far the classifier reaches beyond its training conditions; and the mixed-course observation model."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.physics import classify as c  # noqa: E402
import joblib  # noqa: E402
P, OUT = ROOT / "data" / "processed", ROOT / "results" / "classifier"
ep = pd.read_csv(P / "episodes_campaign.csv").set_index("episode")
rows = []
for v in ("point_foot", "wheel_legged"):
    z = dict(np.load(P / f"features_campaign_{v}_n1.npz"))
    home = z["kind"] == "homogeneous"
    y, g = c.label_index(z["label"][home]), z["episode"][home]
    vx = z["cmd"][home, 0]
    # Per class, the parameter that defines the class: friction, softness or relief.
    prim = {"nominal": "mu", "low_friction": "mu", "granular": "solref_time", "rough": "relief_m"}
    value = np.array([ep.loc[e, prim[l]] for e, l in zip(g, z["label"][home])])
    upper = np.zeros(len(y), bool)
    for k, name in enumerate(c.CLASSES):
        m = y == k
        upper[m] = value[m] >= np.median(ep[(ep.variant == v) & (ep.cls == name)][prim[name]])
    for fs in ("fusion",):
        for with_cmd in (True, False):
            x = c.design(z, fs, home)
            if not with_cmd:
                x = x[:, :-2]
            for name in ("forest", "logistic"):
                for label, tr, te in (("speed: train <0.65, test >=0.65", vx < 0.65, vx >= 0.65), ("speed: train >=0.65, test <0.65", vx >= 0.65, vx < 0.65),
                                      ("terrain parameter: train lower half, test upper", ~upper, upper), ("terrain parameter: train upper half, test lower", upper, ~upper)):
                    m = c.model(name).fit(x[tr], y[tr])
                    s = c.scores(m.predict_proba(x[te]), y[te])
                    rows.append(dict(variant=v, model=name, with_cmd=with_cmd, shift=label, f1_macro=s["f1_macro"], accuracy=s["accuracy"], brier=s["brier"], n_test=int(te.sum())))
    # Observation model from the mixed course, never used in training
    final = joblib.load(OUT / f"final_forest_{v}.joblib")
    course = z["kind"] == "course"
    pr = final.predict_proba(c.design(z, "fusion", course))
    yc = c.label_index(z["label"][course])
    conf = json.load(open(OUT / f"obs_model_{v}.json"))
    conf["course"] = c.binned_confusion(pr, yc)
    conf["course_counts"] = c.scores(pr, yc)["confusion"]
    json.dump(conf, open(OUT / f"obs_model_{v}.json", "w"), indent=1)
    print(v, "course confusion\n", np.array(conf["course_counts"]))
d = pd.DataFrame(rows); d.to_csv(OUT / "generalization.csv", index=False)
pd.set_option("display.width", 200)
print(d.round(3).to_string())
