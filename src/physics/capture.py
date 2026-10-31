"""Drive a robot over a terrain and record what its body senses. Truth is logged for validation only."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from .terrains import CLASSES, HALF_Y, TerrainParams, X_START, build, sample_params, slope_params, ROOT


def _single_thread_ort():
    import onnxruntime as ort
    original = ort.InferenceSession

    def session(path, providers=None, **kw):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        return original(path, sess_options=so, providers=providers)
    ort.InferenceSession = session


class _Substeps:
    """Physics at PHYSICS_DT/n instead of PHYSICS_DT. For the wheeled robot the repo's stance PD (kp 400, kd 7)
    is also closed at this rate: held for 5 ms it sits near its discrete stability limit and produces a
    numerical limit cycle (70 m/s^2 RMS lateral at the IMU on a plane). A servo closes that loop much faster."""

    def __init__(self, mj, n, loco=None):
        self._mj_mod, self.n, self.loco = mj, n, loco

    def __getattr__(self, k):
        return getattr(self._mj_mod, k)

    def mj_step(self, m, d):
        for i in range(self.n):
            if self.loco is not None and i and self.loco.mode.name in ("ROLL", "STANDUP"):
                self.loco._hold_legs(self.loco.target, self.loco.kp, self.loco.kd)
            self._mj_mod.mj_step(m, d)


def run_episode(variant: str, tiles: list[tuple[TerrainParams, float]], vx: float, wz: float,
                duration_s: float = 16.0, log_every: int = 2, substeps: int = 5) -> dict:
    """The legged robot walks on the manufacturer's policy; the wheeled one rolls on the repo's own controller."""
    import mujoco
    _single_thread_ort()
    from locomotion import Locomotion, Mode, PHYSICS_DT
    from wheels import WheelLocomotion
    model = build(tiles, variant)
    data = mujoco.MjData(model)
    if variant == "point_foot":
        loco = Locomotion(model, data)
        loco.assist = False                   # the assist uses the true pose
        loco._mj = _Substeps(mujoco, substeps)
    else:
        loco = WheelLocomotion(model, data)
        loco._mj = _Substeps(mujoco, substeps, loco)
    model.opt.timestep = PHYSICS_DT / substeps
    accel = model.sensor("lidar-acceleration").adr[0]
    gyro = model.sensor("lidar-angular-velocity").adr[0]
    vel = model.sensor("linear-velocity").adr[0]
    wheeled = variant == "wheel_legged"

    while loco.mode == Mode.STANDUP:
        loco.step()
    if loco.mode == Mode.FALLEN:
        return dict(ok=False)
    loco.set_command(vx, 0.0, wz)
    steps = int(duration_s / PHYSICS_DT)
    rows = {k: [] for k in ("t", "acc", "gyro", "grav", "q", "qd", "tau", "q_w", "qd_w", "tau_w",
                            "vel_true", "pos_true", "foot_force_true", "foot_slip_true", "contact_true")}
    x_end = X_START + sum(l for _, l in tiles) - 1.0
    fell_at, end_reason = None, "time"
    feet = loco._feet if not wheeled else loco.tyres
    v6, force = np.zeros(6), np.zeros(6)
    acc_sum, gyro_sum = np.zeros(3), np.zeros(3)
    for k in range(steps):
        loco.step()
        acc_sum += data.sensordata[accel:accel + 3]    # an IMU's anti-alias filter: mean over the log interval
        gyro_sum += data.sensordata[gyro:gyro + 3]
        if loco.mode == Mode.FALLEN:
            fell_at, end_reason = data.time, "fall"
            break
        if not (X_START + 1.0 < data.qpos[0] < x_end and abs(data.qpos[1]) < HALF_Y - 1.0):
            end_reason = "edge"
            break
        if (k + 1) % log_every:
            continue
        rows["t"].append(data.time)
        rows["acc"].append(acc_sum / log_every)
        rows["gyro"].append(gyro_sum / log_every)
        acc_sum, gyro_sum = np.zeros(3), np.zeros(3)
        rows["grav"].append(np.asarray(loco.gravity_body(), float))
        rows["q"].append(loco.leg_positions())
        rows["qd"].append(loco.leg_velocities())
        rows["tau"].append(loco.leg_efforts())      # commanded actuator torque, not a contact force
        if wheeled:
            rows["q_w"].append(loco.wheel_positions())
            rows["qd_w"].append(loco.wheel_velocities())
            rows["tau_w"].append(loco.wheel_efforts())
        rows["vel_true"].append(data.sensordata[vel:vel + 3].copy())
        rows["pos_true"].append(data.qpos[:3].copy())
        ff, sl, cn = np.zeros(4), np.zeros(4), np.zeros(4)
        for i in range(data.ncon):
            c = data.contact[i]
            for j, g in enumerate(feet):
                if g in (c.geom1, c.geom2):
                    mujoco.mj_contactForce(model, data, i, force)
                    ff[j] += force[0] * c.frame[2]      # vertical component of the normal force
                    if not cn[j]:
                        # Velocity of the material point at the contact, which is what slips
                        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_GEOM, g, v6, 0)
                        v = v6[3:6] + np.cross(v6[0:3], c.pos - data.geom_xpos[g])
                        sl[j] = float(np.hypot(v[0], v[1]))
                    cn[j] = 1.0
                    break
        rows["foot_force_true"].append(ff)
        rows["foot_slip_true"].append(sl)
        rows["contact_true"].append(cn)
    out = {k: np.asarray(v, np.float32) for k, v in rows.items() if len(v)}
    out.update(ok=True, fell=fell_at is not None, end_reason=end_reason, dt=PHYSICS_DT * log_every,
               n_steps_logged=len(rows["t"]))
    return out


def plan_extra(kind: str, n: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    return [dict(kind=kind, cls=kind, instance=i, seed=int(rng.integers(1 << 31)), vx=float(rng.uniform(0.4, 0.9)), wz=0.0) for i in range(n)]


def plan_episodes(n_per_class: int, seed: int = 11) -> list[dict]:
    """A fresh terrain instance and command for every episode; the same plan for both configurations."""
    eps = []
    rng = np.random.default_rng(seed)
    for cls in CLASSES:
        for i in range(n_per_class):
            s = int(rng.integers(1 << 31))
            r = np.random.default_rng(s)
            vx = float(r.uniform(0.4, 0.9))
            wz = float(r.choice([0.0, 0.0, 0.35, -0.35]))
            eps.append(dict(cls=cls, instance=i, seed=s, vx=vx, wz=wz))
    return eps


def _summary(row: dict, r: dict, vx: float) -> dict:
    pos = r["pos_true"]
    dist = float(np.sum(np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1))) if len(pos) > 1 else 0.0
    dur = float(r["t"][-1] - r["t"][0]) if len(r["t"]) > 1 else 0.0
    in_c = r["contact_true"] > 0
    row.update(fell=bool(r["fell"]), end_reason=r["end_reason"], duration_s=dur, distance_m=dist,
               mean_speed=dist / max(dur, 1e-6), speed_tracking=(dist / max(dur, 1e-6)) / vx,
               slip_in_contact=float(r["foot_slip_true"][in_c].mean()) if in_c.any() else float("nan"),
               contact_force_mean=float(r["foot_force_true"].sum(axis=1).mean()),
               tau_rms=float(np.sqrt((r["tau"] ** 2).mean())), qd_rms=float(np.sqrt((r["qd"] ** 2).mean())),
               acc_std=float(r["acc"].std(axis=0).mean()))
    return row


def worker(job: dict) -> dict:
    """Runs one episode and writes its raw signals. Returns a summary row."""
    variant, e, out_dir = job["variant"], job["episode"], Path(job["out_dir"])
    rng = np.random.default_rng(e["seed"])
    kind = e.get("kind", "homogeneous")
    if kind == "course":
        order = rng.permutation(CLASSES)
        tiles = [(sample_params("nominal", rng, e["seed"]), 9.0)] + \
                [(sample_params(str(c), rng, e["seed"] + i + 1), 5.0) for i, c in enumerate(order)]
        cls, p = "+".join(str(c) for c in order), tiles[0][0]
        duration = 40.0
    else:
        p = slope_params(rng, e["seed"]) if kind == "slope" else sample_params(e["cls"], rng, e["seed"])
        tiles, cls, duration = [(p, 24.0)], p.cls, job.get("duration_s", 16.0)
    r = run_episode(variant, tiles, e["vx"], e["wz"], duration_s=duration)
    name = f"{variant}_{kind}_{e['cls']}_{e['instance']:03d}"
    row = dict(episode=name, kind=kind, variant=variant, cls=cls, instance=e["instance"], seed=e["seed"], vx_cmd=e["vx"], wz_cmd=e["wz"],
               mu=p.mu, solref_time=p.solref_time, solimp_dmin=p.solimp_dmin, relief_m=p.relief_m, slope_deg=p.slope_deg, ok=bool(r.get("ok")))
    if not r.get("ok"):
        return row
    np.savez_compressed(out_dir / f"{name}.npz", **{k: v for k, v in r.items() if isinstance(v, np.ndarray)})
    return _summary(row, r, e["vx"])
