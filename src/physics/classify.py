"""Terrain classifier. Splits are by episode: windows of one episode are never on both sides."""
from __future__ import annotations
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CLASSES = ["nominal", "low_friction", "granular", "rough"]
CONF_THRESHOLD = 0.8


def design(z, feature_set: str, mask=None) -> np.ndarray:
    parts = {"imu": [z["imu"]], "joints": [z["joints"]], "fusion": [z["imu"], z["joints"]]}[feature_set] + [z["cmd"]]
    x = np.hstack(parts)
    return x if mask is None else x[mask]


def model(name: str, seed: int = 0):
    if name == "forest":
        return RandomForestClassifier(200, max_depth=14, min_samples_leaf=2, n_jobs=-1, random_state=seed, class_weight="balanced")
    if name == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"))
    raise ValueError(name)


def label_index(y) -> np.ndarray:
    return np.array([CLASSES.index(v) for v in y])


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(((p - np.eye(len(CLASSES))[y]) ** 2).sum(axis=1).mean())


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    conf, pred = p.max(axis=1), p.argmax(axis=1)
    edges = np.linspace(0, 1, bins + 1)
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            out += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(out)


def oof(x, y, groups, name: str, seed: int = 0, grouped: bool = True) -> np.ndarray:
    """Out-of-fold probabilities. grouped=False is the leaky random-window split, kept to measure the leakage."""
    p = np.zeros((len(y), len(CLASSES)))
    split = StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(x, y, groups) if grouped else \
        StratifiedKFold(5, shuffle=True, random_state=seed).split(x, y)
    for tr, te in split:
        m = model(name, seed).fit(x[tr], y[tr])
        p[te] = m.predict_proba(x[te])
    return p


def scores(p, y) -> dict:
    pred = p.argmax(axis=1)
    pr, rc, f1, _ = precision_recall_fscore_support(y, pred, labels=range(len(CLASSES)), zero_division=0)
    return dict(accuracy=float((pred == y).mean()), balanced_accuracy=float(balanced_accuracy_score(y, pred)),
                f1_macro=float(f1_score(y, pred, average="macro")), brier=brier(p, y), ece=ece(p, y),
                precision=pr.tolist(), recall=rc.tolist(), f1=f1.tolist(),
                confusion=confusion_matrix(y, pred, labels=range(len(CLASSES))).tolist())


def episode_bootstrap_f1(p, y, groups, n: int = 2000, seed: int = 0, other=None) -> tuple:
    """CI of macro-F1, or of its paired difference with `other`, resampling episodes."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    index = [np.nonzero(inv == i)[0] for i in range(len(uniq))]
    pred = p.argmax(axis=1)
    pred2 = None if other is None else other.argmax(axis=1)
    vals = []
    for _ in range(n):
        take = np.concatenate([index[i] for i in rng.integers(0, len(uniq), len(uniq))])
        v = f1_score(y[take], pred[take], average="macro")
        if pred2 is not None:
            v -= f1_score(y[take], pred2[take], average="macro")
        vals.append(v)
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def split_episodes(y, groups, seed: int = 3):
    """60/20/20 train/calibration/test by episode, stratified by class."""
    rng = np.random.default_rng(seed)
    train, calib, test = [], [], []
    for c in np.unique(y):
        eps = np.unique(groups[y == c])
        rng.shuffle(eps)
        a, b = int(0.6 * len(eps)), int(0.8 * len(eps))
        train += eps[:a].tolist(); calib += eps[a:b].tolist(); test += eps[b:].tolist()
    return np.isin(groups, train), np.isin(groups, calib), np.isin(groups, test)


def binned_confusion(p, y, smooth: float = 0.5) -> dict:
    """p(o | t, bin) and p(bin | t) with add-half smoothing, for confidence bins low / high."""
    K = len(CLASSES)
    conf, pred = p.max(axis=1), p.argmax(axis=1)
    bins = (conf >= CONF_THRESHOLD).astype(int)
    table = np.full((2, K, K), smooth, float)
    for b, t, o in zip(bins, y, pred):
        table[b, t, o] += 1
    pb = np.full((K, 2), smooth, float)
    for b, t in zip(bins, y):
        pb[t, b] += 1
    return dict(table=(table / table.sum(axis=2, keepdims=True)).tolist(), p_bin=(pb / pb.sum(axis=1, keepdims=True)).tolist())


def inference_ms(m, x) -> float:
    t = time.perf_counter()
    for row in x[:200]:
        m.predict_proba(row[None])
    return 1e3 * (time.perf_counter() - t) / min(200, len(x))
