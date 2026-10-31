"""M0-M3 and the oracle. A method sees geometry, appearance, its own observations - never `terrain`."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .maps import CELL_M, K, ROUGH, Scenario
from .observation import ObservationModel


@dataclass(frozen=True)
class MethodParams:
    prior: tuple = (0.70, 0.10, 0.10, 0.10)
    roughness_sigma: float = 0.5
    kernel_sigma: float = 0.35        # appearance kernel width; set on development maps
    vote_threshold: float = 1.0       # M2: kernel mass required to relabel a cell
    tempering: float = 0.3            # M3: rho, weight of propagated evidence
    uncertainty_kappa: float = 0.5    # M3: planner penalty on the hazard's posterior std
    evidence_cap: float = 6.0         # M3: bound on accumulated propagated log-evidence
    spatial_sigma_m: float = 1.0      # M1s: reach of evidence by distance alone


def geometric_prior(s: Scenario, mp: MethodParams) -> np.ndarray:
    """Eq. (prior): pi_i(t) proportional to pi(t) N(r_i; mu_t, sigma_g^2). Geometry only separates ROUGH."""
    prior = np.asarray(mp.prior, float)
    mu = np.zeros(K)
    mu[ROUGH] = 1.0
    r = s.roughness_obs[..., None]
    log_b = np.log(prior) - 0.5 * ((r - mu) / mp.roughness_sigma) ** 2
    b = np.exp(log_b - log_b.max(axis=-1, keepdims=True))
    return b / b.sum(axis=-1, keepdims=True)


class Method:
    name = "base"

    def __init__(self, s: Scenario, obs: ObservationModel, believed_h: np.ndarray, mp: MethodParams):
        self.s, self.obs, self.h, self.mp = s, obs, np.asarray(believed_h, float), mp
        self.belief = geometric_prior(s, mp)

    def update(self, cells: list[tuple], o: int, b: int) -> None:
        """A window of proprioception, classified as o, covering these traversed cells."""

    def hazard_rate(self) -> np.ndarray:
        return self.belief @ self.h

    def cost_rate(self, lam: float) -> np.ndarray:
        """Eq. (cost): metres plus lam * expected -log survival; >= 1 on every cell."""
        return 1.0 + lam * self.hazard_rate()


class M0GeometryOnly(Method):
    name = "M0"


class M1Local(Method):
    """Bayesian update of the cells actually traversed; nothing else."""
    name = "M1"

    def update(self, cells, o, b):
        ll = self.obs.log_like(o, b)
        for c in cells:
            post = np.log(np.clip(self.belief[c], 1e-12, 1.0)) + ll
            post = np.exp(post - post.max())
            self.belief[c] = post / post.sum()


def _kernel(s: Scenario, cells, sigma: float) -> np.ndarray:
    a = np.mean([s.appearance[c] for c in cells], axis=0)
    d2 = ((s.appearance - a) ** 2).sum(axis=-1)
    w = np.exp(-0.5 * d2 / sigma ** 2)
    w[~s.free] = 0.0
    return w


class M1Spatial(Method):
    """A stronger baseline than M1, added after internal review: evidence also reaches nearby cells, by
    distance alone. Tempered like M3, no appearance and no planner penalty."""
    name = "M1s"

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.log_prior = np.log(np.clip(self.belief, 1e-12, 1.0))
        self.direct = np.zeros((*self.s.free.shape, K))
        self.spread = np.zeros((*self.s.free.shape, K))
        self.yy, self.xx = np.indices(self.s.free.shape)

    def update(self, cells, o, b):
        ll = self.obs.log_like(o, b)
        r, c = np.mean(cells, axis=0)
        d2 = ((self.yy - r) ** 2 + (self.xx - c) ** 2) * CELL_M ** 2
        w = np.exp(-0.5 * d2 / self.mp.spatial_sigma_m ** 2)
        self.spread += self.mp.tempering * w[..., None] * ll
        for cell in cells:
            self.direct[cell] += ll
        prop = np.maximum(self.spread - self.spread.max(axis=-1, keepdims=True), -self.mp.evidence_cap)
        log_b = self.log_prior + self.direct + prop
        log_b -= log_b.max(axis=-1, keepdims=True)
        b_ = np.exp(log_b)
        self.belief = b_ / b_.sum(axis=-1, keepdims=True)


class M2Appearance(Method):
    """Online appearance vote: labels are taken as correct, the winning label replaces the belief
    where enough kernel mass agrees. The strong, simple form of the propagation baseline."""
    name = "M2"

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.votes = np.zeros((*self.s.free.shape, K))
        self.prior_belief = self.belief.copy()

    def update(self, cells, o, b):
        self.votes[..., o] += _kernel(self.s, cells, self.mp.kernel_sigma)
        mass = self.votes.sum(axis=-1)
        relabel = mass >= self.mp.vote_threshold
        self.belief = self.prior_belief.copy()
        idx = self.votes.argmax(axis=-1)
        self.belief[relabel] = np.eye(K)[idx[relabel]]


class M3Fusion(Method):
    """Tempered (power-likelihood) Bayes: every observation adds rho * kernel * log p(o|t) to every cell,
    full weight on traversed cells; the planner also pays for the hazard's posterior std.
    Differs from M2 in three removable ways, which the ablation removes one by one."""
    name = "M3"

    def __init__(self, *a, use_confusion=True, tempering=None, kappa=None, **k):
        super().__init__(*a, **k)
        self.log_prior = np.log(np.clip(self.belief, 1e-12, 1.0))
        self.direct = np.zeros((*self.s.free.shape, K))
        self.propagated = np.zeros((*self.s.free.shape, K))
        self.use_confusion = use_confusion
        self.rho = self.mp.tempering if tempering is None else tempering
        self.kappa = self.mp.uncertainty_kappa if kappa is None else kappa

    def update(self, cells, o, b):
        if self.use_confusion:
            ll = self.obs.log_like(o, b)
        else:                                 # ablation: the label is believed as if nearly perfect
            ll = np.log(np.where(np.arange(K) == o, 0.97, 0.01))
        self.propagated += self.rho * _kernel(self.s, cells, self.mp.kernel_sigma)[..., None] * ll
        for c in cells:
            self.direct[c] += ll
        cap = self.mp.evidence_cap
        prop = self.propagated - self.propagated.max(axis=-1, keepdims=True)
        prop = np.maximum(prop, -cap)         # correlated evidence must not reach certainty
        log_b = self.log_prior + self.direct + prop
        log_b -= log_b.max(axis=-1, keepdims=True)
        b_ = np.exp(log_b)
        self.belief = b_ / b_.sum(axis=-1, keepdims=True)

    def hazard_rate(self):
        mean = self.belief @ self.h
        var = np.clip(self.belief @ (self.h ** 2) - mean ** 2, 0.0, None)
        return mean + self.kappa * np.sqrt(var)


class Oracle(Method):
    """Diagnostic bound on simulator truth. Never a deployable baseline."""
    name = "ORACLE"

    def __init__(self, s, obs, believed_h, mp):
        super().__init__(s, obs, believed_h, mp)
        t = np.clip(s.terrain, 0, K - 1)
        self.belief = np.eye(K)[t]


REGISTRY = {"M0": M0GeometryOnly, "M1": M1Local, "M1s": M1Spatial, "M2": M2Appearance, "M3": M3Fusion, "ORACLE": Oracle}


def build(name: str, s, obs, h, mp) -> Method:
    if name == "M3-noconf":
        return M3Fusion(s, obs, h, mp, use_confusion=False)
    if name == "M3-notemper":
        return M3Fusion(s, obs, h, mp, tempering=1.0)
    if name == "M3-nounc":
        return M3Fusion(s, obs, h, mp, kappa=0.0)
    if name == "M3-noprop":
        return M3Fusion(s, obs, h, mp, tempering=0.0)
    return REGISTRY[name](s, obs, h, mp)
