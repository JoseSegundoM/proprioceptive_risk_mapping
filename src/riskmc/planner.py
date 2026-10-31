"""A* on an 8-connected grid; no corner cutting. Edge cost = length * mean of the two cells' cost rate."""
from __future__ import annotations
import heapq
import math
import numpy as np

SQRT2 = math.sqrt(2.0)
MOVES = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
         (1, 1, SQRT2), (1, -1, SQRT2), (-1, 1, SQRT2), (-1, -1, SQRT2)]


def astar(free: np.ndarray, rate: np.ndarray, start: tuple, goal: tuple, cell_m: float = 0.5):
    """Minimum of sum(length * rate). `rate` must be >= 1 on free cells, which makes the
    octile distance in metres an admissible heuristic. Returns (path, cost); path is None if none."""
    h, w = free.shape
    gr, gc = goal

    def heur(r, c):
        dr, dc = abs(r - gr), abs(c - gc)
        return cell_m * (max(dr, dc) + (SQRT2 - 1.0) * min(dr, dc))

    best = {start: 0.0}
    parent = {}
    pq = [(heur(*start), 0.0, start)]
    while pq:
        _, g, node = heapq.heappop(pq)
        if g > best.get(node, math.inf):
            continue
        if node == goal:
            path = [node]
            while node in parent:
                node = parent[node]
                path.append(node)
            return path[::-1], g
        r, c = node
        for dr, dc, m in MOVES:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w) or not free[nr, nc]:
                continue
            if dr and dc and not (free[r + dr, c] and free[r, c + dc]):
                continue
            ng = g + m * cell_m * 0.5 * (rate[r, c] + rate[nr, nc])
            if ng < best.get((nr, nc), math.inf):
                best[(nr, nc)] = ng
                parent[(nr, nc)] = node
                heapq.heappush(pq, (ng + heur(nr, nc), ng, (nr, nc)))
    return None, math.inf
