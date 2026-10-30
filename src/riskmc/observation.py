"""Observation models of the proprioceptive classifier. These are perception, not failure, models."""
from __future__ import annotations
import numpy as np
from .maps import K


class ObservationModel:
    """Truth side `sample_table[b][t, o]` generates the observations; belief side `like[b][t, o]`
    is what a method believes. They differ when the method's confusion is estimated or misspecified.
    b is a confidence bin; p_bin[t, b] the probability of each bin given the true class."""

    def __init__(self, sample_table, like, p_bin=None):
        self.sample_table = np.asarray(sample_table, float)
        self.like = np.asarray(like, float)
        self.n_bins = self.sample_table.shape[0]
        self.p_bin = np.ones((K, 1)) if p_bin is None else np.asarray(p_bin, float)

    def sample(self, t: int, u: float, u_bin: float = 0.5) -> tuple[int, int]:
        """Inverse-CDF draw, so that the same u gives methods a common random number."""
        b = int(np.searchsorted(np.cumsum(self.p_bin[t]), u_bin, side="right"))
        b = min(b, self.n_bins - 1)
        o = int(np.searchsorted(np.cumsum(self.sample_table[b, t]), u, side="right"))
        return min(o, K - 1), b

    def log_like(self, o: int, b: int) -> np.ndarray:
        return np.log(np.clip(self.like[b, :, o], 1e-3, 1.0))


def symmetric_confusion(eps: float) -> np.ndarray:
    c = np.full((K, K), eps / (K - 1))
    np.fill_diagonal(c, 1.0 - eps)
    return c


def parametric(eps: float, believed_eps: float | None = None) -> ObservationModel:
    believed = eps if believed_eps is None else believed_eps
    return ObservationModel(symmetric_confusion(eps)[None], symmetric_confusion(believed)[None])
