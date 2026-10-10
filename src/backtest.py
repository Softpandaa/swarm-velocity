"""Trading rules and the cost accounting they share."""
import numpy as np

import config


def _step(cap, w, prev_dollar, fwd):
    """One holding period, charging the one way cost on the change in dollar position."""
    dollar = w * cap
    pnl = float(dollar @ fwd) - config.TCOST * float(np.abs(dollar - prev_dollar).sum())
    return cap + pnl, dollar, pnl


def _pair(n, long_i, short_i, sign):
    """Half the capital long and half short, so gross exposure is one times capital."""
    w = np.zeros(n)
    w[long_i] = sign * config.GROSS / 2.0
    w[short_i] = -sign * config.GROSS / 2.0
    return w


def _weights(R, pool, R_lead, R_lose, t, thresholds, gated, sign):
    """Leader versus loser weights on day t, gated on the pooled speed when gated is True."""
    n, lead, lose, r_lead = R.shape[1], R_lead[t], R_lose[t], R[t, R_lead[t]]
    if r_lead > 0 and (not gated or pool[t] > thresholds[1]):
        return _pair(n, lead, lose, sign)
    if r_lead < 0 and (not gated or pool[t] < thresholds[0]):
        return _pair(n, lose, lead, sign)
    return np.zeros(n)


def threshold(R, pool, R_lead, R_lose, thresholds=None, flip=False, gated=True,
              L=config.LAGS):
    """Leader versus loser pair trade, gated on the pooled speed when gated is True."""
    T, n = R.shape
    cap, prev, path, per = 1.0, np.zeros(n), [], []
    sign = -1.0 if flip else 1.0
    for t in range(L + 1, T - 1):
        w = _weights(R, pool, R_lead, R_lose, t, thresholds, gated, sign)
        prev_cap = cap
        cap, prev, pnl = _step(cap, w, prev, R[t + 1])
        path.append(cap)
        per.append(pnl / prev_cap)
    return np.array(path), np.array(per)


def cross_section(R, A, k=config.K, h=config.H, flip=False, equal_weight=False,
                  L=config.LAGS):
    """Sort on the per asset speed, buy the k slowest, sell the k fastest, hold h days."""
    T, n = R.shape
    cap, prev, dates, path, per = 1.0, np.zeros(n), [], [], []
    t = L + 1
    while t + h < T:
        w = np.zeros(n)
        if equal_weight:
            w[:] = 1.0 / n
        else:
            valid = np.where(~np.isnan(A[t]))[0]
            if len(valid) >= 2 * k:
                order = valid[np.argsort(A[t][valid])]
                w[order[:k]] = config.GROSS / (2.0 * k)
                w[order[-k:]] = -config.GROSS / (2.0 * k)
                if flip:
                    w = -w
        prev_cap = cap
        cap, prev, pnl = _step(cap, w, prev, R[t + 1:t + 1 + h].sum(axis=0))
        per.append(pnl / prev_cap)
        path.append(cap)
        dates.append(t + h)
        t += h
    return np.array(path), np.array(dates), np.array(per)


def threshold_activity(R, pool, R_lead, R_lose, thresholds=None, gated=True, L=config.LAGS):
    """Share of days the rule holds a position and mean one way turnover per day."""
    T, n = R.shape
    prev, total, active = np.zeros(n), 0.0, 0
    for t in range(L + 1, T - 1):
        w = _weights(R, pool, R_lead, R_lose, t, thresholds, gated, 1.0)
        total += float(np.abs(w - prev).sum())
        active += int(np.abs(w).sum() > 0)
        prev = w
    days = T - L - 2
    return {"active": active / days, "turnover": total / days,
            "cost": config.TCOST * total / days * config.TRADING_DAYS}
