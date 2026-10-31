"""One mission: plan, move, sense, update, replan. True hazard accumulates; beliefs never see it."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import time
import numpy as np
from .maps import CELL_M, NOMINAL, Scenario
from .methods import MethodParams, build
from .observation import ObservationModel
from .planner import astar


@dataclass(frozen=True)
class SimParams:
    lam: float = 25.0            # metres per unit of expected -log survival
    window_m: float = 1.0        # distance a classified proprioceptive window spans
    budget_factor: float = 3.0   # abort beyond this times the geometric shortest path


def common_random(map_id: int, replicate: int, shape: tuple):
    """The same draws for every method on a (map, replicate): CRN, the basis of the paired design."""
    rng = np.random.default_rng(np.random.SeedSequence([7301, map_id, replicate]))
    return dict(u=rng.random((*shape, 16)), u_bin=rng.random((*shape, 16)), threshold=float(rng.exponential(1.0)))


def run_mission(s: Scenario, method_name: str, obs: ObservationModel, true_h: np.ndarray, believed_h: np.ndarray,
                replicate: int, mp: MethodParams = MethodParams(), sp: SimParams = SimParams()) -> dict:
    crn = common_random(s.map_id, replicate, s.free.shape)
    method = build(method_name, s, obs, believed_h, mp)
    ones = np.ones(s.free.shape)
    _, shortest = astar(s.free, ones, s.start, s.goal, CELL_M)
    budget = sp.budget_factor * shortest

    t0 = time.perf_counter()
    path, _ = astar(s.free, method.cost_rate(sp.lam), s.start, s.goal, CELL_M)
    plan_ms = [1e3 * (time.perf_counter() - t0)]
    pos, idx = s.start, 0
    length = exposure = hazard = 0.0
    replans = n_obs = 0
    visits = np.zeros(s.free.shape, int)
    window, window_len = [], 0.0
    reached = False
    trajectory = [pos]
    while True:
        if pos == s.goal:
            reached = True
            break
        if length > budget or path is None or idx + 1 >= len(path):
            break
        nxt = path[idx + 1]
        step = CELL_M * np.hypot(nxt[0] - pos[0], nxt[1] - pos[1])
        for cell, ds in ((pos, 0.5 * step), (nxt, 0.5 * step)):
            t = s.terrain[cell]
            hazard += true_h[t] * s.patch_mult[cell] * ds
            if t != NOMINAL:
                exposure += ds
        length += step
        pos, idx = nxt, idx + 1
        trajectory.append(pos)
        window.append(pos)
        window_len += step
        if window_len >= sp.window_m:
            terrains = [s.terrain[c] for c in window]
            t_win = int(np.bincount(terrains).argmax())    # a mixed window is labelled by its majority
            anchor = window[-1]
            v = min(visits[anchor], 15)
            visits[anchor] += 1
            o, b = obs.sample(t_win, crn["u"][anchor][v], crn["u_bin"][anchor][v])
            method.update(list(window), o, b)
            n_obs += 1
            window, window_len = [], 0.0
            if method_name not in ("M0", "ORACLE"):
                t0 = time.perf_counter()
                new_path, _ = astar(s.free, method.cost_rate(sp.lam), pos, s.goal, CELL_M)
                plan_ms.append(1e3 * (time.perf_counter() - t0))
                if new_path is not None and new_path[1:] != path[idx + 1:]:
                    replans += 1
                if new_path is not None:
                    path, idx = new_path, 0
    p_survive = float(np.exp(-hazard)) if reached else 0.0
    return dict(map_id=s.map_id, replicate=replicate, method=method_name, reached=int(reached),
                success=int(reached and hazard <= crn["threshold"]), p_survive=p_survive,
                cum_hazard=hazard, exposure_m=exposure, length_m=length, shortest_m=shortest,
                plan_ms_mean=float(np.mean(plan_ms)), plan_ms_max=float(np.max(plan_ms)),
                replans=replans, n_obs=n_obs), trajectory, method
