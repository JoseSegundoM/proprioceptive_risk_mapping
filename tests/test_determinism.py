"""The paper says results regenerate from recorded seeds. These tests hold it to that."""
from pathlib import Path
import numpy as np, pandas as pd, pytest
from src.physics.capture import plan_episodes, worker
from src.riskmc import maps, observation, sim
ROOT = Path(__file__).resolve().parents[1]
H = [0.001, 0.06, 0.04, 0.03]


def test_mission_rows_regenerate_exactly():
    rows = pd.read_csv(ROOT / "results" / "mc" / "E4_ablation.csv.gz")
    rows = rows[(rows.method == "M3-noprop") & (rows.eps == 0.1) & (rows.map_id.isin([0, 7])) & (rows.replicate == 0)]
    assert len(rows) == 2
    import json
    from src.riskmc.methods import MethodParams
    mp = MethodParams(**json.load(open(ROOT / "results" / "mc" / "tuned_params.json")))
    for _, r in rows.iterrows():
        rec, _, _ = sim.run_mission(maps.generate(int(r.map_id)), "M3-noprop", observation.parametric(0.1), np.array(H), np.array(H), 0, mp)
        assert rec["cum_hazard"] == pytest.approx(r.cum_hazard, rel=1e-9)
        assert rec["replans"] == r.replans


def test_saved_maps_match_generator():
    z = np.load(ROOT / "data" / "processed" / "maps" / "map_0003.npz")
    s = maps.generate(3)
    assert np.array_equal(z["terrain"], s.terrain) and np.array_equal(z["free"], s.free)


@pytest.mark.parametrize("variant", ["point_foot", "wheel_legged"])
def test_mujoco_episode_regenerates_exactly(tmp_path, variant):
    """Needs the mine_sentinel_autonomy clone (robot models)."""
    if not (ROOT / "mine_sentinel_autonomy" / "sim" / "mujoco" / "vendor").exists():
        pytest.skip("robot models not present")
    e = next(x for x in plan_episodes(150) if x["cls"] == "rough" and x["instance"] == 0)
    row = worker(dict(variant=variant, episode=dict(e, kind="homogeneous"), out_dir=str(tmp_path), duration_s=16.0))
    ref = pd.read_csv(ROOT / "data" / "processed" / "episodes_campaign.csv").set_index("episode").loc[row["episode"]]
    assert row["distance_m"] == pytest.approx(ref.distance_m, rel=1e-5)
    assert row["tau_rms"] == pytest.approx(ref.tau_rms, rel=1e-5)
