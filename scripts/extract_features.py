#!/usr/bin/env python3
"""Raw episodes -> window features, written with their episode id: python scripts/extract_features.py --tag campaign"""
import argparse, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.physics.features import episode_windows, FS  # noqa: E402
from src.physics.terrains import X_START  # noqa: E402

COURSE_START = X_START + 9.0


def one(job):
    row, raw_dir, scale = job
    path = Path(raw_dir) / f"{row['episode']}.npz"
    if not path.exists():
        return None
    z = np.load(path)
    ep = {k: z[k] for k in z.files}
    ep["fell_flag"] = bool(row["fell"])
    imu, joints, c = episode_windows(ep, int(row["seed"]) % (2 ** 31) + 5, scale)
    if len(c) == 0:
        return None
    x = ep["pos_true"][c, 0]
    if row["kind"] == "course":              # label by where the robot truly was: for scoring only
        tiles = ["nominal"] + row["cls"].split("+")
        idx = np.where(x < COURSE_START, 0, np.minimum(((x - COURSE_START) // 5.0).astype(int) + 1, 4))
        labels = [tiles[i] for i in idx]
    else:
        labels = [row["cls"]] * len(c)
    cmd = np.tile([row["vx_cmd"], row["wz_cmd"]], (len(c), 1))
    return dict(episode=row["episode"], variant=row["variant"], kind=row["kind"], labels=labels, imu=imu, joints=joints, cmd=cmd)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="campaign")
    ap.add_argument("--noise-scale", type=float, default=1.0)
    a = ap.parse_args()
    d = pd.read_csv(ROOT / "data" / "processed" / f"episodes_{a.tag}.csv")
    raw = ROOT / "data" / "raw" / "mujoco" / a.tag
    with Pool(14) as p:
        res = [r for r in p.map(one, [(row, str(raw), a.noise_scale) for row in d.to_dict("records")]) if r]
    for v in ("point_foot", "wheel_legged"):
        rs = [r for r in res if r["variant"] == v]
        out = ROOT / "data" / "processed" / f"features_{a.tag}_{v}_n{a.noise_scale:g}.npz"
        np.savez_compressed(out, imu=np.vstack([r["imu"] for r in rs]), joints=np.vstack([r["joints"] for r in rs]),
                            cmd=np.vstack([r["cmd"] for r in rs]),
                            label=np.concatenate([r["labels"] for r in rs]),
                            episode=np.concatenate([[r["episode"]] * len(r["labels"]) for r in rs]),
                            kind=np.concatenate([[r["kind"]] * len(r["labels"]) for r in rs]))
        print(v, "windows", sum(len(r["labels"]) for r in rs), "episodes", len(rs))
