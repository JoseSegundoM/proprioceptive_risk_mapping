#!/usr/bin/env python3
"""Capture proprioception in MuJoCo over parametrised terrain: python scripts/run_capture.py --n 60 --jobs 12"""
import argparse, json, os, sys, time, csv, platform
from multiprocessing import Pool
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.physics.capture import plan_episodes, plan_extra, worker  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60, help="episodes per class and configuration")
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--tag", default="campaign")
    ap.add_argument("--duration", type=float, default=16.0)
    a = ap.parse_args()
    out = ROOT / "data" / "raw" / "mujoco" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    eps = plan_episodes(a.n) + plan_extra("slope", a.n // 3, 21) + plan_extra("course", a.n // 3, 22)
    jobs = [dict(variant=v, episode=e, out_dir=str(out), duration_s=a.duration) for v in ("point_foot", "wheel_legged") for e in eps]
    t0 = time.time()
    with Pool(a.jobs, maxtasksperchild=20) as pool:
        rows = pool.map(worker, jobs, chunksize=1)
    keys = sorted({k for r in rows for k in r})
    proc = ROOT / "data" / "processed"
    proc.mkdir(parents=True, exist_ok=True)
    with open(proc / f"episodes_{a.tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} episodes in {time.time() - t0:.0f} s; python {platform.python_version()}")
