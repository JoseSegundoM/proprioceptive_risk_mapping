#!/usr/bin/env python3
"""Monte Carlo campaigns: python scripts/run_mc.py {tune|maps|E1|E2|E3|E4}. Writes one row per mission."""
import argparse, json, os, sys, time, itertools
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "configs")]
from mc_experiments import *  # noqa
from src.riskmc import maps, methods, observation, sim  # noqa: E402

OUT = ROOT / "results" / "mc"
OUT.mkdir(parents=True, exist_ok=True)
TUNED = OUT / "tuned_params.json"


def method_params():
    return methods.MethodParams(**json.load(open(TUNED))) if TUNED.exists() else methods.MethodParams()


def obs_from_file(variant):
    d = json.load(open(ROOT / "results" / "classifier" / f"obs_model_{variant}.json"))
    # Observations are sampled from the held-out test windows; the method reasons with calibration.
    return observation.ObservationModel(d["test"]["table"], d["calibration"]["table"], d["test"]["p_bin"])


def obs_from_course(variant):
    """Sampled from the mixed course, whose windows cross terrain boundaries; beliefs stay homogeneous calibration."""
    d = json.load(open(ROOT / "results" / "classifier" / f"obs_model_{variant}.json"))
    return observation.ObservationModel(d["course"]["table"], d["calibration"]["table"], d["course"]["p_bin"])


def make_obs(spec):
    if spec["kind"] == "parametric":
        return observation.parametric(spec["eps"], spec.get("believed_eps"))
    return obs_from_course(spec["variant"]) if spec["kind"] == "course" else obs_from_file(spec["variant"])


_cache = {}


def run(job):
    key = (job["map_id"], json.dumps(job.get("map_params", {}), sort_keys=True))
    if key not in _cache:
        if len(_cache) > 8:
            _cache.clear()
        _cache[key] = maps.generate(job["map_id"], maps.MapParams(**job.get("map_params", {})))
    s = _cache[key]
    mp = replace(method_params(), **job.get("method_params", {}))
    sp = sim.SimParams(**job.get("sim_params", {}))
    h = np.asarray(job["true_h"], float)
    hb = np.asarray(job.get("believed_h", job["true_h"]), float)
    rec, _, _ = sim.run_mission(s, job["method"], make_obs(job["obs"]), h, hb, job["replicate"], mp, sp)
    rec.update({k: job[k] for k in job["tags"]})
    return rec


def expand(exp, map_ids, method_list, reps, settings):
    """settings: list of dicts with true_h, obs and tag fields."""
    jobs = []
    for st in settings:
        tags = [k for k in st if k not in ("true_h", "believed_h", "obs", "map_params", "method_params", "sim_params")]
        for m, method, r in itertools.product(map_ids, method_list, range(reps)):
            jobs.append(dict(st, map_id=m, method=method, replicate=r, tags=tags + ["exp"], exp=exp))
    return jobs


def execute(name, jobs, procs=15):
    t = time.time()
    with Pool(procs) as p:
        rows = p.map(run, jobs, chunksize=40)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"{name}.csv.gz", index=False)
    print(f"{name}: {len(df)} missions in {time.time() - t:.0f} s")
    return df


def tune():
    """Development maps only. Chooses the appearance kernel, M2's threshold and M3's tempering and penalty."""
    base = [dict(eps=e, true_h=H_EQUAL, obs=dict(kind="parametric", eps=e)) for e in (0.1, 0.3)]
    grid_m2 = [dict(kernel_sigma=s, vote_threshold=t) for s in (0.2, 0.35, 0.5, 0.7) for t in (0.5, 1.0, 2.0, 4.0)]
    grid_m3 = [dict(kernel_sigma=s, tempering=r, uncertainty_kappa=k) for s in (0.2, 0.35, 0.5, 0.7) for r in (0.1, 0.3, 0.6) for k in (0.0, 0.5, 1.0)]
    jobs = []
    for b in base:
        for i, g in enumerate(grid_m2):
            jobs += expand("tune", DEV_MAPS, ["M2"], 2, [dict(b, method_params=g, grid=i)])
        for i, g in enumerate(grid_m3):
            jobs += expand("tune", DEV_MAPS, ["M3"], 2, [dict(b, method_params=g, grid=i)])
    df = execute("tuning_dev", jobs)
    m2 = df[df.method == "M2"].groupby("grid").p_survive.mean()
    m3 = df[df.method == "M3"].groupby("grid").p_survive.mean()
    best2, best3 = grid_m2[int(m2.idxmax())], grid_m3[int(m3.idxmax())]
    # One appearance kernel for each method, as found; both recorded.
    json.dump(dict(kernel_sigma=best3["kernel_sigma"], tempering=best3["tempering"], uncertainty_kappa=best3["uncertainty_kappa"],
                   vote_threshold=best2["vote_threshold"]), open(TUNED, "w"), indent=1)
    json.dump(dict(m2_best=best2, m2_score=float(m2.max()), m3_best=best3, m3_score=float(m3.max()),
                   m2_kernel_for_eval=best2["kernel_sigma"]), open(OUT / "tuning_summary.json", "w"), indent=1)
    print(open(OUT / "tuning_summary.json").read())


def m2_kernel():
    return json.load(open(OUT / "tuning_summary.json"))["m2_kernel_for_eval"]


def split_by_method(jobs):
    """M2 uses the kernel tuned for it; M3 its own."""
    for j in jobs:
        if j["method"] == "M2":
            j["method_params"] = dict(j.get("method_params", {}), kernel_sigma=j.get("method_params", {}).get("kernel_sigma", m2_kernel()))
    return jobs


def E1():
    st = [dict(eps=e, scenario="equal", true_h=H_EQUAL, obs=dict(kind="parametric", eps=e)) for e in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)]
    execute("E1_perception_sweep", split_by_method(expand("E1", EVAL_MAPS, METHODS, REPLICATES, st)))


def measured_h():
    d = pd.read_csv(ROOT / "results" / "classifier" / "failure_rates_mujoco.csv")
    out = {}
    for v in ("point_foot", "wheel_legged"):
        g = d[d.variant == v].set_index("cls")
        out[v] = [float(g.loc[c, "h_mean"]) for c in maps.CLASSES]
    return out


def E2():
    st = []
    mh = measured_h()
    for v in ("point_foot", "wheel_legged"):
        o = dict(kind="file", variant=v)
        st.append(dict(config=v, scenario="equal", true_h=H_EQUAL, obs=o))
        st.append(dict(config=v, scenario="wheel_sensitive", true_h=H_WHEEL_SENSITIVE[v], obs=o))
        st.append(dict(config=v, scenario="mujoco_falls", true_h=mh[v], obs=o))
    execute("E2_measured_perception", split_by_method(expand("E2", EVAL_MAPS, METHODS, REPLICATES, st)))


def E3():
    base = dict(scenario="equal", true_h=H_EQUAL, obs=dict(kind="parametric", eps=0.1), eps=0.1)
    st = []
    for v in (0.0, 0.15, 0.3, 0.5):
        st.append(dict(base, factor="aliasing", level=v, map_params=dict(p_alias=v, p_decoy=v)))
    for v in (0.05, 0.15, 0.3, 0.5):
        st.append(dict(base, factor="appearance_noise", level=v, map_params=dict(appearance_noise=v)))
    for v in (0.25, 0.5, 1.0, 2.0, 4.0):
        st.append(dict(base, factor="hazard_scale", level=v, true_h=[0.001] + [v * x for x in H_EQUAL[1:]]))
    for v in (0.25, 0.5, 1.0, 2.0, 4.0):
        st.append(dict(base, factor="believed_hazard", level=v, believed_h=[0.001] + [v * x for x in H_EQUAL[1:]]))
    for v in (5.0, 10.0, 25.0, 50.0, 100.0):
        st.append(dict(base, factor="lambda", level=v, sim_params=dict(lam=v)))
    for v in (0.15, 0.3, 0.45):
        st.append(dict(base, factor="hazard_fraction", level=v, map_params=dict(hazard_fraction=v)))
    for v in (0.02, 0.1, 0.3):
        st.append(dict(base, factor="believed_eps", level=v, obs=dict(kind="parametric", eps=0.1, believed_eps=v)))
    for v in (0.5, 1.0, 2.0):
        st.append(dict(base, factor="window_m", level=v, sim_params=dict(window_m=v)))
    execute("E3_sensitivity", split_by_method(expand("E3", EVAL_MAPS[:100], ["M0", "M1", "M2", "M3"], 3, st)))


def E4():
    st = [dict(eps=e, scenario="equal", true_h=H_EQUAL, obs=dict(kind="parametric", eps=e)) for e in (0.1, 0.3)]
    execute("E4_ablation", expand("E4", EVAL_MAPS, ["M3", "M3-noconf", "M3-notemper", "M3-nounc", "M3-noprop"], REPLICATES, st))


def E5():
    """Is M3's edge over M2 only risk aversion? Sweep lambda for both, with the planner's own weight."""
    st = [dict(eps=e, scenario="equal", true_h=H_EQUAL, obs=dict(kind="parametric", eps=e), lam=v, sim_params=dict(lam=v))
          for e in (0.1, 0.3) for v in (10.0, 25.0, 50.0, 100.0, 200.0, 400.0)]
    execute("E5_lambda", split_by_method(expand("E5", EVAL_MAPS, ["M2", "M3", "M3-nounc"], 3, st)))


def E6():
    """Observations from the mixed course (hardest held-out classifier data)."""
    st = []
    mh = measured_h()
    for v in ("point_foot", "wheel_legged"):
        o = dict(kind="course", variant=v)
        st.append(dict(config=v, scenario="equal", true_h=H_EQUAL, obs=o))
        st.append(dict(config=v, scenario="mujoco_falls", true_h=mh[v], obs=o))
    execute("E6_course_perception", split_by_method(expand("E6", EVAL_MAPS, METHODS, REPLICATES, st)))


def E7():
    """Independent repetition: 200 maps never touched before, a fresh replicate stream."""
    st = [dict(eps=0.1, scenario="equal", true_h=H_EQUAL, obs=dict(kind="parametric", eps=0.1)),
          dict(eps=0.3, scenario="equal", true_h=H_EQUAL, obs=dict(kind="parametric", eps=0.3))]
    for v in ("point_foot",):
        st.append(dict(eps=-1.0, scenario="mujoco_falls_legged", true_h=measured_h()[v], obs=dict(kind="file", variant=v)))
    jobs = split_by_method(expand("E7", list(range(200000, 200200)), METHODS, REPLICATES, st))
    execute("E7_independent_repeat", jobs)


def E8():
    """Added after internal review: does spatial smoothing without appearance explain the propagation gain?
    The reach is chosen on development maps; evaluation then runs on the original 200 maps."""
    grid = (0.5, 1.0, 2.0, 4.0)
    jobs = []
    for sg in grid:
        for e in (0.1, 0.3):
            jobs += expand("tuneE8", DEV_MAPS, ["M1s"], 2, [dict(eps=e, sg=sg, true_h=H_EQUAL, obs=dict(kind="parametric", eps=e), method_params=dict(spatial_sigma_m=sg))])
    d = execute("tuning_dev_E8", jobs)
    best = float(d.groupby("sg").p_survive.mean().idxmax())
    print("best spatial reach", best, d.groupby("sg").p_survive.mean().round(4).to_dict())
    json.dump(dict(spatial_sigma_m=best), open(OUT / "tuning_E8.json", "w"))
    st = [dict(eps=e, scenario="equal", true_h=H_EQUAL, obs=dict(kind="parametric", eps=e), method_params=dict(spatial_sigma_m=best)) for e in (0.1, 0.3)]
    st.append(dict(eps=-1.0, scenario="mujoco_falls_legged", true_h=measured_h()["point_foot"], obs=dict(kind="file", variant="point_foot"), method_params=dict(spatial_sigma_m=best)))
    execute("E8_spatial_baseline", split_by_method(expand("E8", EVAL_MAPS, ["M1", "M1s", "M2", "M3"], REPLICATES, st)))


def save_maps():
    d = ROOT / "data" / "processed" / "maps"
    d.mkdir(parents=True, exist_ok=True)
    for m in EVAL_MAPS:
        s = maps.generate(m)
        np.savez_compressed(d / f"map_{m:04d}.npz", free=s.free, terrain=s.terrain.astype(np.int8), appearance=s.appearance.astype(np.float32),
                            patch_mult=s.patch_mult.astype(np.float32), roughness_obs=s.roughness_obs.astype(np.float32), start=s.start, goal=s.goal)
    json.dump(dict(generator_seed_root=20261030, map_ids=EVAL_MAPS, params=vars(maps.MapParams()) if False else maps.MapParams().__dict__), open(d / "params.json", "w"), indent=1, default=list)
    print("saved", len(EVAL_MAPS), "maps")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["tune", "maps", "E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8"])
    {"tune": tune, "maps": save_maps, "E1": E1, "E2": E2, "E3": E3, "E4": E4, "E5": E5, "E6": E6, "E7": E7, "E8": E8}[ap.parse_args().what]()
