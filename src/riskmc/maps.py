"""Reproducible indoor maps inspired by mine inspection: drifts, crossings, rooms, pillars, terrain."""
from __future__ import annotations
from dataclasses import dataclass
from collections import deque
import numpy as np

NOMINAL, LOW_FRICTION, GRANULAR, ROUGH = 0, 1, 2, 3
CLASSES = ["nominal", "low_friction", "granular", "rough"]
K = len(CLASSES)
CELL_M = 0.5


@dataclass(frozen=True)
class MapParams:
    width: int = 48              # cells
    height: int = 32
    n_vertical: int = 4          # drifts
    n_horizontal: int = 3
    drift_cells: tuple = (3, 5)  # corridor width range, cells
    n_rooms: int = 3
    n_pillars: int = 10
    p_remove_segment: float = 0.15
    hazard_fraction: float = 0.30   # target share of free cells that are not nominal
    p_alias: float = 0.15        # a hazard patch that looks like the nominal floor
    p_decoy: float = 0.15        # a nominal patch that looks like a hazard
    appearance_dim: int = 8
    appearance_noise: float = 0.15   # per-cell, per-dimension
    patch_noise: float = 0.10        # per-patch offset, per-dimension
    n_decoy_looks: int = 2
    hazard_patch_sigma: float = 0.3  # lognormal spread of the true hazard among patches


@dataclass
class Scenario:
    map_id: int
    params: MapParams
    free: np.ndarray            # bool [h, w]
    terrain: np.ndarray         # int [h, w]; -1 in rock
    appearance: np.ndarray      # float [h, w, d]
    patch_mult: np.ndarray      # float [h, w]: true hazard multiplier
    patch_id: np.ndarray        # int [h, w]; -1 for the base floor
    roughness_obs: np.ndarray   # float [h, w]: noisy geometric roughness from exteroception
    start: tuple
    goal: tuple


def _reachable(free: np.ndarray, start: tuple) -> np.ndarray:
    seen = np.zeros_like(free, bool)
    seen[start] = True
    q = deque([start])
    h, w = free.shape
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and free[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return seen


def _carve(rng: np.random.Generator, p: MapParams) -> np.ndarray:
    free = np.zeros((p.height, p.width), bool)
    xs = np.linspace(3, p.width - 4, p.n_vertical).astype(int) + rng.integers(-1, 2, p.n_vertical)
    ys = np.linspace(3, p.height - 4, p.n_horizontal).astype(int) + rng.integers(-1, 2, p.n_horizontal)
    xs, ys = np.clip(xs, 2, p.width - 3), np.clip(ys, 2, p.height - 3)
    for x in xs:
        half = int(rng.integers(*p.drift_cells)) // 2 + 0
        free[1:p.height - 1, max(1, x - half):min(p.width - 1, x + half + 1)] = True
    for y in ys:
        half = int(rng.integers(*p.drift_cells)) // 2 + 0
        free[max(1, y - half):min(p.height - 1, y + half + 1), 1:p.width - 1] = True
    # Close some segments between crossings so routes differ from map to map.
    for i in range(len(xs)):
        for a, b in zip(ys[:-1], ys[1:]):
            if rng.random() < p.p_remove_segment:
                m = (a + b) // 2
                free[m - 1:m + 2, max(1, xs[i] - 3):min(p.width - 1, xs[i] + 4)] = False
    for j in range(len(ys)):
        for a, b in zip(xs[:-1], xs[1:]):
            if rng.random() < p.p_remove_segment:
                m = (a + b) // 2
                free[max(1, ys[j] - 3):min(p.height - 1, ys[j] + 4), m - 1:m + 2] = False
    for _ in range(p.n_rooms):             # rooms at crossings
        cx, cy = int(rng.choice(xs)), int(rng.choice(ys))
        rw, rh = int(rng.integers(3, 6)), int(rng.integers(2, 5))
        free[max(1, cy - rh):min(p.height - 1, cy + rh + 1), max(1, cx - rw):min(p.width - 1, cx + rw + 1)] = True
    return free


def _endpoints(free: np.ndarray) -> tuple[tuple, tuple]:
    rows, cols = np.nonzero(free)
    a = int(np.argmin(rows + cols))
    b = int(np.argmax(rows + cols))
    return (int(rows[a]), int(cols[a])), (int(rows[b]), int(cols[b]))


def generate(map_id: int, params: MapParams = MapParams(), roughness_sigma: float = 0.5) -> Scenario:
    """Same map_id and parameters give the same map. Rejects and redraws unreachable layouts."""
    ss = np.random.SeedSequence([20261030, map_id])
    rng = np.random.default_rng(ss)
    for _ in range(200):
        free = _carve(rng, params)
        start, goal = _endpoints(free)
        pillars = free.copy()
        for _ in range(params.n_pillars):
            rr, cc = np.nonzero(pillars)
            i = int(rng.integers(len(rr)))
            r, c = int(rr[i]), int(cc[i])
            if abs(r - start[0]) + abs(c - start[1]) < 4 or abs(r - goal[0]) + abs(c - goal[1]) < 4:
                continue
            trial = pillars.copy()
            trial[r, c] = False
            if _reachable(trial, start)[goal]:
                pillars = trial
        free = pillars
        seen = _reachable(free, start)
        if seen[goal] and abs(start[0] - goal[0]) + abs(start[1] - goal[1]) > 30:
            free = free & seen
            break
    else:
        raise RuntimeError(f"map {map_id}: no reachable layout")

    terrain = np.full(free.shape, -1, int)
    terrain[free] = NOMINAL
    patch_id = np.full(free.shape, -1, int)
    patch_mult = np.ones(free.shape)
    target = params.hazard_fraction * free.sum()
    rr, cc = np.nonzero(free)
    d = params.appearance_dim
    protos = rng.normal(size=(3 + 1 + params.n_decoy_looks, d))   # floor, 3 hazards, decoys
    protos /= np.linalg.norm(protos, axis=1, keepdims=True)
    # Per-map prototypes: there is no terrain <-> look relation to carry between maps.
    looks = np.zeros((*free.shape, d))
    looks[free] = protos[0]

    n_patch, covered = 0, 0
    while covered < target and n_patch < 60:
        i = int(rng.integers(len(rr)))
        cr, cc0 = int(rr[i]), int(cc[i])
        ra, rb = int(rng.integers(2, 6)), int(rng.integers(2, 6))
        y, x = np.ogrid[:free.shape[0], :free.shape[1]]
        mask = ((y - cr) / ra) ** 2 + ((x - cc0) / rb) ** 2 <= 1.0
        mask &= free & (terrain == NOMINAL)
        mask[start] = False
        mask[goal] = False
        if mask.sum() < 3:
            n_patch += 1
            continue
        cls = int(rng.integers(1, K))
        terrain[mask] = cls
        patch_id[mask] = n_patch
        patch_mult[mask] = rng.lognormal(0.0, params.hazard_patch_sigma)
        look = protos[0] if rng.random() < params.p_alias else protos[cls]
        looks[mask] = look + rng.normal(0, params.patch_noise, d)
        covered += int(mask.sum())
        n_patch += 1

    # Decoys: nominal patches that look like hazards (or like harmless other textures).
    for _ in range(int(n_patch * params.p_decoy / max(1e-9, 1 - params.p_decoy)) + 1):
        if rng.random() > 0.5 + params.p_decoy:
            continue
        i = int(rng.integers(len(rr)))
        cr, cc0 = int(rr[i]), int(cc[i])
        ra, rb = int(rng.integers(2, 5)), int(rng.integers(2, 5))
        y, x = np.ogrid[:free.shape[0], :free.shape[1]]
        mask = ((y - cr) / ra) ** 2 + ((x - cc0) / rb) ** 2 <= 1.0
        mask &= free & (terrain == NOMINAL)
        if mask.sum() < 3:
            continue
        look = protos[int(rng.integers(1, K))] if rng.random() < params.p_decoy / max(params.p_decoy, 0.5) \
            else protos[K + int(rng.integers(params.n_decoy_looks))]
        looks[mask] = look + rng.normal(0, params.patch_noise, d)

    appearance = looks + rng.normal(0, params.appearance_noise, looks.shape) * free[..., None]
    # Geometry sees relief, not friction or softness: an observation centred on 1 for rough cells.
    roughness = (terrain == ROUGH).astype(float) + rng.normal(0, roughness_sigma, free.shape)
    roughness[~free] = 0.0
    return Scenario(map_id, params, free, terrain, appearance, patch_mult, patch_id, roughness, start, goal)
