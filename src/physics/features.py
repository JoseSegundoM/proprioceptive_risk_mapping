"""Window features from signals a robot could read. No truth channel (vel_true, pos_true, foot_*_true) enters."""
from __future__ import annotations
import numpy as np

FS = 100.0
WINDOW_S, STRIDE_S = 2.0, 1.0
SKIP_S = 2.0              # transient after the command starts
PRE_FALL_S = 1.0          # the fall itself must not be what identifies a terrain
BANDS = [(0.5, 5.0), (5.0, 15.0), (15.0, 45.0)]

# Assumed sensor imperfections, applied here so the raw data stay clean. IMU white noise from the
# manufacturer's model is lower; these are deliberately larger. Bias is constant over an episode.
NOISE = dict(acc=0.05, gyro=0.01, acc_bias=0.05, gyro_bias=0.005, q=0.002, qd=0.05, tau=0.1)


def _stats(x: np.ndarray) -> list[float]:
    """Per channel of [n, c]: std, first-difference rms, three band powers (log)."""
    x = x - x.mean(axis=0, keepdims=True)
    out = [x.std(axis=0), np.sqrt((np.diff(x, axis=0) ** 2).mean(axis=0))]
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))[:, None], axis=0)) ** 2
    freq = np.fft.rfftfreq(len(x), 1.0 / FS)
    for lo, hi in BANDS:
        out.append(np.log(spec[(freq >= lo) & (freq < hi)].sum(axis=0) + 1e-9))
    return np.concatenate(out).tolist()


def _groups(a: np.ndarray, kinds: int = 3) -> np.ndarray:
    """Reduce [n, 12] leg channels to per-joint-type rms-combined channels, invariant to which leg it is."""
    return np.stack([np.sqrt((a[:, k::kinds] ** 2).mean(axis=1)) for k in range(kinds)], axis=1)


def imu_features(ep, i, j, rng_noise) -> list[float]:
    acc = ep["acc"][i:j] + rng_noise["acc"][i:j]
    gyro = ep["gyro"][i:j] + rng_noise["gyro"][i:j]
    grav = ep["grav"][i:j]
    tilt = [float(np.abs(grav[:, 0]).mean()), float(np.abs(grav[:, 1]).mean()), float(grav[:, 0].std()), float(grav[:, 1].std())]
    return _stats(acc) + _stats(gyro) + tilt


def joint_features(ep, i, j, rng_noise) -> list[float]:
    q = ep["q"][i:j] + rng_noise["q"][i:j]
    qd = ep["qd"][i:j] + rng_noise["qd"][i:j]
    tau = ep["tau"][i:j] + rng_noise["tau"][i:j]
    f = _stats(_groups(q - q.mean(axis=0))) + _stats(_groups(qd)) + _stats(_groups(tau))
    f += np.abs(tau).mean(axis=0).reshape(4, 3).mean(axis=0).tolist()       # mean load per joint type
    if "tau_w" in ep:
        qdw, tauw = ep["qd_w"][i:j] + rng_noise["qdw"][i:j], ep["tau_w"][i:j] + rng_noise["tauw"][i:j]
        f += _stats(np.abs(qdw).mean(axis=1, keepdims=True)) + _stats(tauw.mean(axis=1, keepdims=True))
        f += [float(np.abs(tauw).mean()), float(np.abs(qdw).mean())]
    return f


def episode_windows(ep: dict, noise_seed: int, noise_scale: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (imu [m, a], joints [m, b], window centre index [m])."""
    n = len(ep["t"])
    rng = np.random.default_rng(noise_seed)
    s = noise_scale
    noise = dict(acc=rng.normal(0, s * NOISE["acc"], (n, 3)) + rng.normal(0, s * NOISE["acc_bias"], 3),
                 gyro=rng.normal(0, s * NOISE["gyro"], (n, 3)) + rng.normal(0, s * NOISE["gyro_bias"], 3),
                 q=rng.normal(0, s * NOISE["q"], (n, 12)), qd=rng.normal(0, s * NOISE["qd"], (n, 12)),
                 tau=rng.normal(0, s * NOISE["tau"], (n, 12)), qdw=rng.normal(0, s * NOISE["qd"], (n, 4)),
                 tauw=rng.normal(0, s * NOISE["tau"], (n, 4)))
    w, st, skip = int(WINDOW_S * FS), int(STRIDE_S * FS), int(SKIP_S * FS)
    end = n - int(PRE_FALL_S * FS) if ep.get("fell_flag", False) else n
    imu, joints, centres = [], [], []
    for i in range(skip, end - w + 1, st):
        imu.append(imu_features(ep, i, i + w, noise))
        joints.append(joint_features(ep, i, i + w, noise))
        centres.append(i + w // 2)
    if not imu:
        return np.zeros((0, 0)), np.zeros((0, 0)), np.zeros(0, int)
    return np.asarray(imu), np.asarray(joints), np.asarray(centres)
