"""All numbers quoted in the report. Run plots.py, which calls this module."""
import numpy as np
import pandas as pd

import backtest
import config
import stats
import swarm


def legs(R):
    return np.argmax(R, axis=1), np.argmin(R, axis=1)


def threshold_grid(R, pool, lead, lose):
    """Terminal capital at every pair of thresholds."""
    g = pd.DataFrame(index=config.TAU_LOW, columns=config.TAU_HIGH, dtype=float)
    for hi in config.TAU_HIGH:
        for lo in config.TAU_LOW:
            if lo > hi:
                continue
            g.loc[lo, hi] = backtest.threshold(R, pool, lead, lose, [lo, hi])[0][-1]
    return g


def argmax(g):
    a = g.to_numpy(float)
    i, j = np.unravel_index(np.nanargmax(a), a.shape)
    return float(g.index[i]), float(g.columns[j]), float(a[i, j])


def results():
    ret = swarm.load_returns()
    tr, te = swarm.split(ret)
    out = {"ret": ret, "train": tr, "test": te}

    R = {"train": tr.to_numpy(float), "test": te.to_numpy(float)}
    pool = {k: swarm.alpha(v, rule="day", pooled=True) for k, v in R.items()}
    legs_ = {k: legs(v) for k, v in R.items()}
    A = {k: swarm.alpha(v, rule="window") for k, v in R.items()}

    g = threshold_grid(R["train"], pool["train"], *legs_["train"])
    lo, hi, best = argmax(g)
    out["grid"] = {"train": g, "test": threshold_grid(R["test"], pool["test"], *legs_["test"])}
    out["tau"] = (lo, hi)
    out["threshold"], out["activity"] = {}, {}
    for s in ("train", "test"):
        ld, ls = legs_[s]
        runs = {"rule": backtest.threshold(R[s], pool[s], ld, ls, [lo, hi]),
                "benchmark": backtest.threshold(R[s], pool[s], ld, ls, gated=False),
                "opposite": backtest.threshold(R[s], pool[s], ld, ls, [lo, hi], flip=True),
                "benchmark_opposite": backtest.threshold(R[s], pool[s], ld, ls,
                                                         gated=False, flip=True)}
        out["threshold"][s] = {k: stats.performance(v[1], config.TRADING_DAYS)
                               for k, v in runs.items()}
        out["activity"][s] = {
            "rule": backtest.threshold_activity(R[s], pool[s], ld, ls, [lo, hi]),
            "benchmark": backtest.threshold_activity(R[s], pool[s], ld, ls, gated=False)}

    R["full"] = ret.to_numpy(float)
    A["full"] = swarm.alpha(R["full"], rule="window")
    out["hold"] = {}
    for h in config.H_GRID:
        row = {}
        for flip in (False, True):
            _, _, per = backtest.cross_section(R["full"], A["full"], h=h, flip=flip)
            row["reverse" if flip else "direct"] = stats.performance(per, config.TRADING_DAYS / h)
        out["hold"][h] = row

    out["baseline"] = {}
    for flip in (False, True):
        _, _, per = backtest.cross_section(R["full"], A["full"], flip=flip)
        out["baseline"]["reverse" if flip else "direct"] = stats.performance(
            per, config.TRADING_DAYS / config.H)
    _, _, per = backtest.cross_section(R["full"], A["full"], equal_weight=True)
    out["baseline"]["equal"] = stats.performance(per, config.TRADING_DAYS / config.H)
    out["alpha"] = {
        "per_asset": describe(swarm.alpha(R["train"], rule="window")),
        "pooled": describe(swarm.alpha(R["train"], rule="window", pooled=True)),
        "decomposition": swarm.decompose(R["train"]),
        "mean_corr": stats.mean_correlation(ret)}
    out["rank_corr"] = {h: rank_corr(ret.to_numpy(float), swarm.alpha(ret.to_numpy(float), rule="window"), h)
                        for h in (5, 10, config.H, 63)}
    return out


def describe(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return {"mean": float(x.mean()), "sd": float(x.std(ddof=1))}


def rank_corr(R, A, h, L=config.LAGS):
    """Mean cross sectional rank correlation between the speed and the next h day return."""
    rows, t = [], L + 1
    while t + h < len(R):
        v = np.where(~np.isnan(A[t]))[0]
        if len(v) >= 4:
            f = R[t + 1:t + 1 + h].sum(axis=0)[v]
            rows.append(pd.Series(A[t][v]).corr(pd.Series(f), method="spearman"))
        t += h
    rows = np.array(rows)
    return {"mean": rows.mean(), "t": stats.tstat(rows), "n": len(rows)}


def report(out):
    lo, hi = out["tau"]
    print(f"sample {len(out['ret'])} returns, train {len(out['train'])}, test {len(out['test'])}")
    print(f"threshold optimum ({lo}, {hi})")
    for s in ("train", "test"):
        for k, d in out["threshold"][s].items():
            print(f"  {s:5s} {k:9s} return {d['total']:+8.1%}  vol {d['vol']:6.1%}  sharpe {d['sharpe']:+6.2f}")
    print("holding period, net of cost")
    for h, row in out["hold"].items():
        d, e = row["direct"], row["reverse"]
        print(f"  h={h:3d} | slow-fast {d['total']:+7.1%} vol {d['vol']:5.1%} SR {d['sharpe']:+5.2f}"
              f" | fast-slow {e['total']:+7.1%} vol {e['vol']:5.1%} SR {e['sharpe']:+5.2f}")
    for key, b in out["baseline"].items():
        print(f"  baseline h={config.H} full {key:7s}: return {b['total']:+7.1%}, "
              f"vol {b['vol']:.1%}, sharpe {b['sharpe']:+.2f}, t={b['t']:+.2f}, n={b['n']}")
    a = out["alpha"]
    print(f"  alpha per asset mean {a['per_asset']['mean']:+.3f} sd {a['per_asset']['sd']:.3f}; "
          f"pooled mean {a['pooled']['mean']:+.3f}")
    print(f"  decomposition: pull {a['decomposition']['pull']:+.2f}, "
          f"increment {a['decomposition']['increment']:+.2f}, "
          f"ratio median {a['decomposition']['ratio']:.2f}; "
          f"mean pairwise correlation {a['mean_corr']:.2f}")
    for s in ("train", "test"):
        for k, d in out["activity"][s].items():
            print(f"  {s:5s} {k:9s} active {d['active']:.1%} of days, turnover {d['turnover']:.2f}/day, "
                  f"cost {d['cost']:.1%}/yr")
    for h, d in out["rank_corr"].items():
        print(f"  rank corr with next {h:2d} day return: {d['mean']:+.3f}, t={d['t']:+.2f}, n={d['n']}")
    for s in ("train", "test"):
        a = out["grid"][s].to_numpy(float)
        ok = np.isfinite(a)
        print(f"  grid {s}: {int((a[ok] > 1).sum())} of {int(ok.sum())} profitable, "
              f"best {np.nanmax(a[ok]) - 1:+.1%}")
    print("\nsummary statistics, LaTeX rows")
    print(stats.latex_summary(out["ret"]))
    print("\ncorrelation matrix, lower triangle, LaTeX rows")
    print(stats.latex_corr(out["ret"]))
