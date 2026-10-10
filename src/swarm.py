"""Swarm velocity estimator of Chen, Huang and Tang (2025)."""
import numpy as np
import pandas as pd

import config


def load_returns():
    px = pd.read_csv(config.DATA, index_col=0, parse_dates=True)[config.TICKERS]
    return np.log(px / px.shift(1)).dropna()


def split(ret):
    return ret.loc[: config.SPLIT], ret.loc[config.SPLIT:]


def positions(R, L=config.LAGS):
    """x[t, i, j] = r_{i, t-j-1}, the L lagged returns of asset i at date t."""
    T, n = R.shape
    X = np.full((T, n, L), np.nan)
    for j in range(L):
        X[j + 1:, :, j] = R[: -(j + 1)]
    return X


def leader(R, X, t, rule):
    """Highest return on day t ('day') or over the position window ('window', paper 5.2)."""
    if rule == "day":
        return int(np.argmax(R[t]))
    return int(np.argmax(X[t].sum(axis=1)))


def decompose(R, L=config.LAGS, rule="window"):
    """Pooled numerator as leader pull plus own increment, median squared norm ratio."""
    X = positions(R, L)
    pull, own, ratio = [], [], []
    for t in range(L + 1, len(R)):
        Xt, Xp = X[t], X[t - 1]
        if np.isnan(Xt).any() or np.isnan(Xp).any():
            continue
        ld = leader(R, X, t, rule)
        nu, d = Xt - Xp, Xt[ld][None, :] - Xt
        den = (d * d).sum()
        if den <= 0:
            continue
        pull.append((nu * Xt[ld][None, :]).sum() / den)
        own.append(-(nu * Xt).sum() / den)
        ratio.append((nu * nu).sum() / den)
    return {"pull": float(np.mean(pull)), "increment": float(np.mean(own)),
            "ratio": float(np.median(ratio))}


def alpha(R, L=config.LAGS, rule="window", pooled=False):
    """Per asset alpha of equation (14), or the pooled common speed when pooled is True."""
    X = positions(R, L)
    T, n = R.shape
    A = np.full((T, n), np.nan)
    pool = np.full(T, np.nan)
    for t in range(L + 1, T):
        Xt, Xp = X[t], X[t - 1]
        if np.isnan(Xt).any() or np.isnan(Xp).any():
            continue
        ld = leader(R, X, t, rule)
        nu = Xt - Xp
        d = Xt[ld][None, :] - Xt
        den = (d * d).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            A[t] = (nu * d).sum(axis=1) / den
        A[t, ld] = np.nan                      # the leader has a zero gap
        total = den.sum()
        if total > 0:
            pool[t] = (nu * d).sum() / total
    return pool if pooled else A
