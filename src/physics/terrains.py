"""Terrain classes as MuJoCo contact parameters and height fields, on the repo's unmodified robot models."""
from __future__ import annotations
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "mine_sentinel_autonomy"
sys.path[:0] = [str(REPO / "sim" / "mujoco"), str(REPO)]

CLASSES = ["nominal", "low_friction", "granular", "rough"]
RES = 0.05            # m per height-field sample
HALF_Y = 8.0          # m
SPAWN_FLAT_M = 1.2    # flat around the start
X_START = -6.0        # the course begins behind the origin, so turning episodes stay on the tile


@dataclass
class TerrainParams:
    cls: str
    mu: float             # tangential friction coefficient of the floor (floor has contact priority)
    solref_time: float    # s, contact time constant; larger is softer
    solref_damp: float
    solimp_dmin: float
    relief_m: float       # peak-to-peak scale of the height field
    relief_wavelength_m: float
    seed: int
    slope_deg: float = 0.0  # >0: a ramp rising along +x beyond the flat start


def sample_params(cls: str, rng: np.random.Generator, seed: int) -> TerrainParams:
    """Ranges are assumed scenario parameters. A small relief on every class keeps "no vibration at all"
    from being an artefact that identifies a class."""
    base = dict(solref_time=0.02, solref_damp=1.0, solimp_dmin=0.8, relief_wavelength_m=float(rng.uniform(0.2, 0.5)))
    if cls == "nominal":
        return TerrainParams(cls, float(rng.uniform(0.7, 1.2)), relief_m=float(rng.uniform(0.0, 0.006)), seed=seed, **base)
    if cls == "low_friction":
        return TerrainParams(cls, float(rng.uniform(0.12, 0.30)), relief_m=float(rng.uniform(0.0, 0.006)), seed=seed, **base)
    if cls == "granular":
        # Soft, lossy contact with some irregularity. NOT a particle model of granular media.
        base.update(solref_time=float(rng.uniform(0.05, 0.15)), solimp_dmin=float(rng.uniform(0.3, 0.6)))
        return TerrainParams(cls, float(rng.uniform(0.45, 0.8)), relief_m=float(rng.uniform(0.005, 0.015)), seed=seed, **base)
    if cls == "rough":
        return TerrainParams(cls, float(rng.uniform(0.7, 1.2)), relief_m=float(rng.uniform(0.025, 0.06)), seed=seed, **base)
    raise ValueError(cls)


def relief(p: TerrainParams, nrow: int, ncol: int, x0: float = 0.0) -> np.ndarray:
    """Smooth random relief in metres with minimum 0, tapered to 0 within SPAWN_FLAT_M of the origin."""
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(p.seed + 77)
    field = gaussian_filter(rng.normal(size=(nrow, ncol)), max(1.0, p.relief_wavelength_m / RES / 2.0), mode="wrap")
    field = (field - field.min()) / max(1e-9, field.max() - field.min()) * p.relief_m
    y = (np.arange(nrow) - (nrow - 1) / 2) * RES
    x = x0 + np.arange(ncol) * RES
    r = np.hypot(*np.meshgrid(x, y))
    if p.slope_deg > 0:
        return np.clip(x - SPAWN_FLAT_M, 0.0, None)[None, :].repeat(nrow, axis=0) * np.tan(np.radians(p.slope_deg))
    return field * np.clip((r - SPAWN_FLAT_M) / 1.0, 0.0, 1.0)


def slope_params(rng: np.random.Generator, seed: int) -> TerrainParams:
    """A rigid, high-friction ramp. Evaluated only as a condition the classifier never trained on."""
    deg = float(rng.uniform(5.0, 12.0))
    return TerrainParams("slope", float(rng.uniform(0.7, 1.2)), 0.02, 1.0, 0.8, 18.0 * np.tan(np.radians(deg)), 0.3, seed, deg)


def build(tiles: list[tuple[TerrainParams, float]], variant: str):
    """Sequence of (terrain, length m) along +x starting at x = X_START. Returns the compiled model."""
    import mujoco
    from scene import _add_lidar_imu, MESHES, VENDOR
    spec = mujoco.MjSpec.from_file(str(VENDOR / "dobot_quad.xml"))
    spec.meshdir = str(MESHES) + "/"
    _add_lidar_imu(spec)
    if variant == "wheel_legged":
        from wheels import add_wheels
        add_wheels(spec)
    spec.delete(spec.geom("geom_floor"))
    world = spec.worldbody
    x = X_START
    for i, (p, length) in enumerate(tiles):
        ncol, nrow = int(round(length / RES)) + 1, int(round(2 * HALF_Y / RES)) + 1
        name = f"tile{i}"
        top = max(p.relief_m, 1e-3)
        field = relief(p, nrow, ncol, x) / top
        spec.add_hfield(name=name, size=[length / 2, HALF_Y, top, 0.5], nrow=nrow, ncol=ncol,
                        userdata=field.ravel().astype(np.float32).tolist())
        g = world.add_geom(name=f"floor{i}", type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname=name,
                           pos=[x + length / 2, 0.0, 0.0], rgba=[0.5, 0.5, 0.5, 1])
        g.friction = [p.mu, 0.1, 0.01]
        g.priority = 1                 # otherwise MuJoCo uses the maximum of floor and foot friction
        g.solref = [p.solref_time, p.solref_damp]
        g.solimp = [p.solimp_dmin, 0.99, 0.001, 0.5, 2]
        x += length
    return spec.compile()
