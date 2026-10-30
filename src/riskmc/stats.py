"""Paired, map-level statistics. The map is the unit: replicates are averaged before any comparison."""
from __future__ import annotations
import numpy as np
from scipy import stats as st


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(c - h), float(c + h))


def bootstrap_mean_ci(x: np.ndarray, n_boot: int = 10000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(axis=1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def paired_compare(a: np.ndarray, b: np.ndarray, n_boot: int = 10000, seed: int = 0) -> dict:
    """a - b per map: mean, bootstrap CI, Wilcoxon p, dz, rank-biserial r."""
    d = np.asarray(a, float) - np.asarray(b, float)
    lo, hi = bootstrap_mean_ci(d, n_boot, seed)
    nz = d[d != 0]
    if len(nz) == 0:
        p, rb = 1.0, 0.0
    else:
        p = float(st.wilcoxon(nz, alternative="two-sided").pvalue)
        ranks = st.rankdata(np.abs(nz))
        rb = float((ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum())
    sd = d.std(ddof=1) if len(d) > 1 else 0.0
    return dict(mean_diff=float(d.mean()), ci_lo=lo, ci_hi=hi, p_wilcoxon=p, dz=float(d.mean() / sd) if sd > 0 else 0.0,
                rank_biserial=rb, n=len(d))


def holm(pvalues: list[float]) -> list[float]:
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    adj = np.empty(len(p))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj.tolist()
