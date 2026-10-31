"""Physics capture checks on the simulation itself. Short episodes; these run MuJoCo."""
from pathlib import Path
import numpy as np
import pytest
if not (Path(__file__).resolve().parents[1] / "mine_sentinel_autonomy" / "sim" / "mujoco" / "vendor").exists():
    pytest.skip("robot models from mine_sentinel_autonomy not present", allow_module_level=True)
from src.physics.terrains import TerrainParams, sample_params, build, relief, slope_params
from src.physics.capture import run_episode
from src.physics import features, classify

FLAT = TerrainParams("nominal", 0.9, 0.02, 1.0, 0.8, 0.0, 0.3, 1)
LOW = TerrainParams("low_friction", 0.15, 0.02, 1.0, 0.8, 0.0, 0.3, 1)


def test_floor_friction_really_governs_contact():
    """Without priority MuJoCo takes the max of the pair, and lowering the floor would change nothing."""
    import mujoco
    m = build([(LOW, 24.0)], "point_foot")
    g = m.geom("floor0").id
    assert m.geom_priority[g] == 1 and m.geom_friction[g, 0] == pytest.approx(0.15)


def test_masses_are_the_repo_models_unmodified():
    a, b = build([(FLAT, 24.0)], "point_foot"), build([(FLAT, 24.0)], "wheel_legged")
    assert sum(a.body_mass) == pytest.approx(18.65, abs=0.02) and sum(b.body_mass) == pytest.approx(22.08, abs=0.02)
    assert a.nu == 12 and b.nu == 16


@pytest.mark.parametrize("variant,weight", [("point_foot", 18.65 * 9.81), ("wheel_legged", 22.08 * 9.81)])
def test_vertical_force_balances_weight_and_robot_moves(variant, weight):
    r = run_episode(variant, [(FLAT, 24.0)], 0.6, 0.0, duration_s=8)
    assert not r["fell"]
    assert r["foot_force_true"][200:].sum(axis=1).mean() == pytest.approx(weight, rel=0.03)
    assert r["vel_true"][200:, 0].mean() > 0.3


def test_wheeled_imu_has_no_numerical_limit_cycle():
    """At 200 Hz the repo's stance PD oscillates at about 70 m/s^2; the 1 kHz wrapper must not."""
    r = run_episode("wheel_legged", [(FLAT, 24.0)], 0.6, 0.0, duration_s=8)
    assert r["acc"][200:].std(axis=0).max() < 2.0


def test_low_friction_changes_what_the_foot_does():
    a = run_episode("point_foot", [(FLAT, 24.0)], 0.6, 0.0, duration_s=6)
    b = run_episode("point_foot", [(LOW, 24.0)], 0.6, 0.0, duration_s=6)
    sa = a["foot_slip_true"][a["contact_true"] > 0].mean()
    sb = b["foot_slip_true"][b["contact_true"] > 0].mean()
    assert sb > 2 * sa


def test_relief_scale_and_flat_start():
    p = sample_params("rough", np.random.default_rng(0), 3)
    f = relief(p, 321, 481, -6.0)
    assert f.max() <= p.relief_m + 1e-9 and f.min() >= 0.0
    assert f[160, 120] == 0.0                  # the origin
    s = slope_params(np.random.default_rng(0), 3)
    assert 5 <= s.slope_deg <= 12


def test_feature_windows_exclude_truth_and_pre_fall():
    n = 1000
    ep = dict(t=np.arange(n) / 100.0, acc=np.zeros((n, 3)), gyro=np.zeros((n, 3)), grav=np.zeros((n, 3)), q=np.zeros((n, 12)),
              qd=np.zeros((n, 12)), tau=np.zeros((n, 12)), pos_true=np.full((n, 3), np.nan), vel_true=np.full((n, 3), np.nan),
              foot_force_true=np.full((n, 4), np.nan), foot_slip_true=np.full((n, 4), np.nan), contact_true=np.full((n, 4), np.nan))
    i0, j0, c0 = features.episode_windows(dict(ep), 0)
    assert np.isfinite(i0).all() and np.isfinite(j0).all()     # NaN in truth channels cannot reach features
    i1, _, c1 = features.episode_windows(dict(ep, fell_flag=True), 0)
    assert c1.max() + 100 <= n - 100 and len(c1) < len(c0)


def test_split_is_by_episode():
    groups = np.repeat([f"e{i}" for i in range(40)], 5)
    y = np.repeat(np.arange(4), 50)
    tr, ca, te = classify.split_episodes(y, groups)
    assert not set(groups[tr]) & set(groups[te]) and not set(groups[tr]) & set(groups[ca]) and not set(groups[ca]) & set(groups[te])
    assert (tr | ca | te).all()


def test_binned_confusion_is_a_distribution():
    p = np.random.default_rng(0).dirichlet(np.ones(4), 300)
    d = classify.binned_confusion(p, np.random.default_rng(1).integers(0, 4, 300))
    assert np.allclose(np.sum(d["table"], axis=2), 1.0) and np.allclose(np.sum(d["p_bin"], axis=1), 1.0)
