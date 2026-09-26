import numpy as np
import pandas as pd
import pvlib
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture


def filter_dbscan(day, full, eps=0.055, min_samples=90):
    X = StandardScaler().fit_transform(day[["g", "Potencia"]].to_numpy())
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X) == -1


def filter_isolation_forest(day, full, contamination=0.25):
    X = StandardScaler().fit_transform(day[["g", "Potencia"]].to_numpy())
    return IsolationForest(contamination=contamination, random_state=42).fit_predict(X) == -1


def filter_lof(day, full, n_neighbors=350, contamination=0.5):
    X = StandardScaler().fit_transform(day[["g", "Potencia"]].to_numpy())
    return LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination).fit_predict(X) == -1


def filter_gmm(day, full, n_components=10, contamination=0.33, random_state=42):
    X = StandardScaler().fit_transform(day[["g", "Potencia"]].to_numpy())
    gmm = GaussianMixture(n_components=n_components, random_state=random_state).fit(X)
    scores = gmm.score_samples(X)
    return scores < np.quantile(scores, contamination)


def make_clearsky_filter(location):
    def filter_clearsky_pvlib(day, full, window_length=30):
        cs = location.get_clearsky(full.index)
        clear = pvlib.clearsky.detect_clearsky(full["ghi"], cs["ghi"], window_length=window_length)
        return (~clear.loc[day.index]).to_numpy()
    return filter_clearsky_pvlib


def _irls_residuals(day, weight_fn, iters=20):
    p = day["Potencia"].to_numpy()
    g = day["g"].to_numpy()
    X = np.column_stack([g, g * (day["panT"].to_numpy() - 25.0)])
    w = np.ones(len(p))
    u = np.zeros(len(p))
    for _ in range(iters):
        sw = np.sqrt(w)
        beta, *_ = np.linalg.lstsq(X * sw[:, None], p * sw, rcond=None)
        r = p - X @ beta
        sigma = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
        u = r / sigma
        w = np.clip(weight_fn(u), 1e-8, None)
    return u


def _w_l2(u):
    return np.ones_like(u)


def _w_l1(u):
    return 1.0 / np.maximum(np.abs(u), 1e-8)


def _w_cauchy(u, k=2.385):
    return 1.0 / (1.0 + (u / k) ** 2)


def _w_geman_mcclure(u, k=1.0):
    return 1.0 / (1.0 + (u / k) ** 2) ** 2


def _w_welsch(u, k=2.985):
    return np.exp(-((u / k) ** 2))


def _w_tukey(u, k=4.685):
    return np.where(np.abs(u) < k, (1.0 - (u / k) ** 2) ** 2, 0.0)


def _w_andrew(u, k=1.339):
    a = np.abs(u) / k
    return np.where(a <= np.pi, np.sinc(a / np.pi), 0.0)


def _w_fair(u, k=1.4):
    return 1.0 / (1.0 + np.abs(u) / k)


def _w_logistic(u, k=1.205):
    a = np.abs(u / k) + 1e-8
    return np.tanh(a) / a


def _w_student(u, nu=4.0):
    return (nu + 1.0) / (nu + u ** 2)


def _make_irls_filter(weight_fn, z_crit_default=0.9):
    def f(day, full, z_crit=z_crit_default):
        return np.abs(_irls_residuals(day, weight_fn)) > z_crit
    return f


def _trimmed_fit(day, keep_frac, iters=8):
    p = day["Potencia"].to_numpy()
    g = day["g"].to_numpy()
    X = np.column_stack([g, g * (day["panT"].to_numpy() - 25.0)])
    n_keep = max(10, int(keep_frac * len(p)))
    keep = np.ones(len(p), dtype=bool)
    r = np.zeros(len(p))
    for _ in range(iters):
        beta, *_ = np.linalg.lstsq(X[keep], p[keep], rcond=None)
        r = np.abs(p - X @ beta)
        keep = np.zeros(len(p), dtype=bool)
        keep[np.argsort(r)[:n_keep]] = True
    return r, keep


def filter_vartrimmed_ficp(day, full, lam=0.5):
    best_frmsd, best_mask = np.inf, None
    for f_keep in np.arange(0.40, 0.96, 0.05):
        r, keep = _trimmed_fit(day, f_keep, iters=6)
        frmsd = np.sqrt(np.mean(r[keep] ** 2)) / f_keep ** (1.0 + lam)
        if frmsd < best_frmsd:
            best_frmsd, best_mask = frmsd, ~keep
    return best_mask


def build_filters(location):
    return {
        "DBSCAN": filter_dbscan,
        "IsolationForest": filter_isolation_forest,
        "LOF": filter_lof,
        "GMM": filter_gmm,
        "ClearSky_pvlib": make_clearsky_filter(location),
        "L2_ZScore": _make_irls_filter(_w_l2),
        "VarTrimmed_FICP": filter_vartrimmed_ficp,
        "Cauchy": _make_irls_filter(_w_cauchy),
        "GemanMcClure": _make_irls_filter(_w_geman_mcclure),
        "Welsch": _make_irls_filter(_w_welsch),
        "Tukey": _make_irls_filter(_w_tukey),
        "Andrew": _make_irls_filter(_w_andrew),
        "L1_IRLS": _make_irls_filter(_w_l1),
        "Fair": _make_irls_filter(_w_fair),
        "Logistic": _make_irls_filter(_w_logistic),
        "StudentT": _make_irls_filter(_w_student),
    }


FILTER_NAMES = [
    "DBSCAN", "IsolationForest", "LOF", "GMM", "ClearSky_pvlib",
    "L2_ZScore", "VarTrimmed_FICP", "Cauchy", "GemanMcClure", "Welsch",
    "Tukey", "Andrew", "L1_IRLS", "Fair", "Logistic", "StudentT",
]
