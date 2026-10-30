#!/usr/bin/env python3
"""Real offscreen renders of the MuJoCo simulation, from campaign terrain instances. Floor tint is for display only."""
import os, sys, json
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.physics.terrains import build, sample_params  # noqa: E402
from src.physics.capture import _Substeps, _single_thread_ort, plan_episodes  # noqa: E402
import mujoco  # noqa: E402
from PIL import Image  # noqa: E402

OUT = ROOT / "paper" / "figures" / "mujoco"
OUT.mkdir(parents=True, exist_ok=True)
TINT = dict(nominal=[0.72, 0.72, 0.72, 1], low_friction=[0.45, 0.72, 0.90, 1], granular=[0.88, 0.76, 0.35, 1], rough=[0.58, 0.47, 0.34, 1])
PLAN = {(e["cls"], e["instance"]): e for e in plan_episodes(150)}


def shoot(variant, cls, instance, name, t_end=None, until_fall=False):
    _single_thread_ort()
    from locomotion import Locomotion, Mode, PHYSICS_DT
    from wheels import WheelLocomotion
    e = PLAN[(cls, instance)]
    p = sample_params(cls, np.random.default_rng(e["seed"]), e["seed"])
    model = build([(p, 24.0)], variant)
    model.geom("floor0").rgba = TINT[cls]
    model.vis.global_.offwidth, model.vis.global_.offheight = 1200, 800
    model.light_castshadow[:] = 1
    data = mujoco.MjData(model)
    loco = Locomotion(model, data) if variant == "point_foot" else WheelLocomotion(model, data)
    if variant == "point_foot":
        loco.assist = False
    loco._mj = _Substeps(mujoco, 5, loco if variant == "wheel_legged" else None)
    model.opt.timestep = PHYSICS_DT / 5
    while loco.mode == Mode.STANDUP:
        loco.step()
    loco.set_command(e["vx"], 0.0, e["wz"])
    fell = False
    for k in range(int(16.0 / PHYSICS_DT)):
        loco.step()
        if loco.mode == Mode.FALLEN:
            fell = True
            for _ in range(60):
                loco.step()
            break
        if not until_fall and data.time - 2.5 >= t_end:
            break
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [data.qpos[0], data.qpos[1], max(0.2, data.qpos[2] - 0.1)]
    cam.distance, cam.azimuth, cam.elevation = 1.75, 125, -18
    with mujoco.Renderer(model, 800, 1200) as r:
        r.update_scene(data, cam)
        Image.fromarray(r.render()).save(OUT / f"{name}.png")
    return dict(name=name, variant=variant, cls=cls, instance=instance, seed=e["seed"], mu=p.mu, relief_m=p.relief_m, solref_time=p.solref_time, sim_time_s=float(data.time), fell=fell)


def portrait(variant, name):
    """Both configurations standing on a plain floor, for the opening figure."""
    _single_thread_ort()
    from locomotion import Locomotion, Mode, PHYSICS_DT
    from wheels import WheelLocomotion
    from src.physics.terrains import TerrainParams
    p = TerrainParams("nominal", 0.9, 0.02, 1.0, 0.8, 0.0, 0.3, 1)
    model = build([(p, 24.0)], variant)
    model.geom("floor0").rgba = [0.62, 0.62, 0.62, 1]
    model.vis.global_.offwidth, model.vis.global_.offheight = 1400, 1000
    model.vis.quality.shadowsize = 8192
    model.vis.map.shadowclip = 0.6
    model.light_castshadow[:] = 1
    data = mujoco.MjData(model)
    loco = Locomotion(model, data) if variant == "point_foot" else WheelLocomotion(model, data)
    loco._mj = _Substeps(mujoco, 5, loco if variant == "wheel_legged" else None)
    model.opt.timestep = PHYSICS_DT / 5
    while loco.mode == Mode.STANDUP:
        loco.step()
    for _ in range(200):
        loco.step()
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [data.qpos[0], data.qpos[1], 0.27]
    cam.distance, cam.azimuth, cam.elevation = 1.45, 140, -14
    with mujoco.Renderer(model, 1000, 1400) as r:
        r.update_scene(data, cam)
        Image.fromarray(r.render()).save(OUT / f"{name}.png")


if __name__ == "__main__":
    import pandas as pd
    d = pd.read_csv(ROOT / "data" / "processed" / "episodes_campaign.csv")
    fell = d[(d.variant == "point_foot") & (d.cls == "low_friction") & (d.kind == "homogeneous") & d.fell].sort_values("instance").iloc[0]
    rows = [shoot("point_foot", "rough", 0, "legged_rough", t_end=6.0),
            shoot("point_foot", "low_friction", int(fell.instance), "legged_low_friction_fall", until_fall=True),
            shoot("wheel_legged", "rough", 0, "wheeled_rough", t_end=6.0),
            shoot("wheel_legged", "granular", 0, "wheeled_granular", t_end=6.0)]
    json.dump(rows, open(OUT / "renders.json", "w"), indent=1)
    print(rows)
    portrait("point_foot", "portrait_legged")
    portrait("wheel_legged", "portrait_wheeled")
