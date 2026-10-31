import dataclasses
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from src.riskmc import maps, planner, methods, observation, sim, stats
from src.riskmc.maps import K, generate, MapParams

H = np.array([0.001, 0.06, 0.04, 0.03])


def test_maps_are_reproducible_and_differ():
    a, b, c = generate(3), generate(3), generate(4)
    assert np.array_equal(a.terrain, b.terrain) and np.array_equal(a.appearance, b.appearance)
    assert a.start == b.start and not np.array_equal(a.terrain, c.terrain)


@pytest.mark.parametrize("m", range(25))
def test_goal_reachable_and_hazards_present(m):
    s = generate(m)
    assert s.free[s.start] and s.free[s.goal]
    assert maps._reachable(s.free, s.start)[s.goal]
    assert len(np.unique(s.terrain[s.free])) >= 3
    assert (s.terrain == -1).sum() == (~s.free).sum()


def test_astar_matches_dijkstra_on_random_costs():
    s = generate(1)
    rng = np.random.default_rng(0)
    rate = 1 + 5 * rng.random(s.free.shape)
    _, cost = planner.astar(s.free, rate, s.start, s.goal, 0.5)
    h, w = s.free.shape
    row, col, val = [], [], []
    for r in range(h):
        for c in range(w):
            if not s.free[r, c]:
                continue
            for dr, dc, m in planner.MOVES:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w) or not s.free[nr, nc]:
                    continue
                if dr and dc and not (s.free[r + dr, c] and s.free[r, c + dc]):
                    continue
                row.append(r * w + c); col.append(nr * w + nc)
                val.append(m * 0.5 * 0.5 * (rate[r, c] + rate[nr, nc]))
    g = csr_matrix((val, (row, col)), shape=(h * w, h * w))
    d = dijkstra(g, indices=s.start[0] * w + s.start[1])[s.goal[0] * w + s.goal[1]]
    assert cost == pytest.approx(d, rel=1e-9)


def test_bayes_update_matches_hand_calculation():
    s = generate(0)
    obs = observation.parametric(0.2)
    m = methods.M1Local(s, obs, H, methods.MethodParams())
    cell = (s.start[0], s.start[1])
    prior = m.belief[cell].copy()
    m.update([cell], 1, 0)
    c = observation.symmetric_confusion(0.2)[:, 1]
    expect = prior * c / (prior * c).sum()
    assert m.belief[cell] == pytest.approx(expect)


def test_m1_touches_only_traversed_cells_and_m2_reaches_others():
    s = generate(2)
    obs = observation.parametric(0.1)
    m1 = methods.M1Local(s, obs, H, methods.MethodParams())
    before = m1.belief.copy()
    cell = s.start
    m1.update([cell], 1, 0)
    changed = np.any(before != m1.belief, axis=-1)
    assert changed.sum() == 1 and changed[cell]
    m2 = methods.M2Appearance(s, obs, H, methods.MethodParams(vote_threshold=0.5))
    b2 = m2.belief.copy()
    m2.update([cell], 1, 0)
    assert np.any(b2 != m2.belief, axis=-1).sum() > 1


def test_methods_do_not_read_true_terrain():
    """Replacing the truth must leave beliefs unchanged given the same observations."""
    s = generate(5)
    fake = dataclasses.replace(s, terrain=np.where(s.free, 0, -1))
    obs = observation.parametric(0.1)
    for name in ("M0", "M1", "M2", "M3"):
        a, b = methods.build(name, s, obs, H, methods.MethodParams()), methods.build(name, fake, obs, H, methods.MethodParams())
        for cell, o in ((s.start, 1), (s.goal, 2)):
            a.update([cell], o, 0); b.update([cell], o, 0)
        assert np.array_equal(a.belief, b.belief), name


def test_m0_ignores_observations_and_oracle_beats_m0_on_average():
    obs = observation.parametric(0.1)
    ps = {"M0": [], "ORACLE": []}
    for m in range(15):
        s = generate(m)
        for name in ps:
            r, _, _ = sim.run_mission(s, name, obs, H, H, 0)
            ps[name].append(r["p_survive"])
    assert np.mean(ps["ORACLE"]) > np.mean(ps["M0"])


def test_replicates_share_random_numbers_across_methods():
    s = generate(0)
    obs = observation.parametric(0.2)
    r1, _, _ = sim.run_mission(s, "M0", obs, H, H, 3)
    r2, _, _ = sim.run_mission(s, "M0", obs, H, H, 3)
    assert r1["cum_hazard"] == r2["cum_hazard"]
    a = sim.common_random(0, 3, s.free.shape); b = sim.common_random(0, 3, s.free.shape)
    assert a["threshold"] == b["threshold"] and np.array_equal(a["u"], b["u"])


def test_survival_is_exp_of_hazard_and_exposure_bounded_by_length():
    s = generate(7)
    r, traj, _ = sim.run_mission(s, "M1", observation.parametric(0.1), H, H, 0)
    assert r["p_survive"] == pytest.approx(np.exp(-r["cum_hazard"])) or not r["reached"]
    assert 0 <= r["exposure_m"] <= r["length_m"] + 1e-9 and traj[0] == s.start


def test_sampler_matches_confusion_frequencies():
    o = observation.parametric(0.3)
    rng = np.random.default_rng(1)
    draws = [o.sample(2, u)[0] for u in rng.random(20000)]
    freq = np.bincount(draws, minlength=K) / 20000
    assert freq == pytest.approx(observation.symmetric_confusion(0.3)[2], abs=0.015)


def test_stats():
    assert stats.wilson(0, 10)[0] == pytest.approx(0.0, abs=1e-12)
    lo, hi = stats.wilson(81, 100)
    assert lo == pytest.approx(0.722, abs=0.002) and hi == pytest.approx(0.875, abs=0.002)
    assert stats.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    a = np.arange(30.0)
    r = stats.paired_compare(a + 1, a)
    assert r["mean_diff"] == pytest.approx(1.0) and r["ci_lo"] == pytest.approx(1.0)


def test_m1s_reaches_neighbours_by_distance_only():
    s = generate(2)
    obs = observation.parametric(0.1)
    m = methods.M1Spatial(s, obs, H, methods.MethodParams(spatial_sigma_m=1.0))
    before = m.belief.copy()
    m.update([s.start], 1, 0)
    delta = np.abs(m.belief - before).sum(axis=-1)
    r, c = s.start
    near = [(r, c + 1), (r + 1, c + 1)]
    far = (s.goal[0], s.goal[1])
    assert any(delta[n] > 1e-6 for n in near if s.free[n])
    assert delta[far] < 1e-6
    a = dataclasses.replace(s, appearance=np.zeros_like(s.appearance))
    m2 = methods.M1Spatial(a, obs, H, methods.MethodParams(spatial_sigma_m=1.0))
    m2.update([s.start], 1, 0)
    assert np.array_equal(m.belief, m2.belief)      # appearance plays no part
