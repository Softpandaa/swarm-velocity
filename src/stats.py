"""Summary statistics reported in the appendix."""
import numpy as np
import pandas as pd
from scipy import stats as sps

import config


def summary(ret):
    out = pd.DataFrame(index=ret.columns,
                       columns=["ann_return", "ann_vol", "skew", "ex_kurtosis"], dtype=float)
    for c in ret.columns:
        r = ret[c].to_numpy(float)
        out.loc[c] = [r.mean() * config.TRADING_DAYS,
                      r.std(ddof=1) * np.sqrt(config.TRADING_DAYS),
                      sps.skew(r), sps.kurtosis(r)]
    return out


def latex_summary(ret):
    s = summary(ret)
    rows = []
    for name, r in s.iterrows():
        rows.append("%s & %.1f & %.1f & %+.2f & %.1f \\\\"
                    % (name, 100 * r["ann_return"], 100 * r["ann_vol"],
                       r["skew"], r["ex_kurtosis"]))
    return "\n".join(rows)


def tstat(per):
    return per.mean() / (per.std(ddof=1) / np.sqrt(len(per)))


def latex_corr(ret):
    """Lower triangle of the correlation matrix, as LaTeX rows."""
    c = ret.corr()
    names = list(ret.columns)
    rows = []
    for i, name in enumerate(names):
        cells = ["%.2f" % c.iloc[i, j] for j in range(i + 1)] + [""] * (len(names) - i - 1)
        rows.append("%s & %s \\\\" % (name, " & ".join(cells)))
    return "\n".join(rows)


def mean_correlation(ret):
    """Mean pairwise correlation of the return panel."""
    c = ret.corr().to_numpy(float)
    off = ~np.eye(len(c), dtype=bool)
    return float(c[off].mean())


def performance(per, periods_per_year):
    """Total return, annualized volatility and Sharpe ratio at a zero risk free rate."""
    per = np.asarray(per, dtype=float)
    vol = per.std(ddof=1) * np.sqrt(periods_per_year)
    return {"total": float(np.prod(1.0 + per) - 1.0),
            "vol": float(vol),
            "sharpe": float(per.mean() * periods_per_year / vol) if vol > 0 else float("nan"),
            "t": float(tstat(per)), "n": len(per)}
