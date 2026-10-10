"""Single entry point. Prints every number quoted in the report and writes the figures."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import backtest
import config
import main as m
import swarm

OKABE = {"blue": "#0072B2", "vermillion": "#D55E00", "grey": "#666666", "sky": "#56B4E9"}
DASH = {"blue": (0, ()), "vermillion": (0, (5, 2)), "grey": (0, (1, 2)), "sky": (0, (3, 1, 1, 1))}

plt.rcParams.update({"font.family": "serif", "font.size": 10,
                     "axes.grid": True, "grid.alpha": 0.3})


def _capital(ret, flip=False, equal_weight=False):
    """The cross sectional rule estimates nothing, so it runs on the full sample."""
    R = ret.to_numpy(float)
    A = swarm.alpha(R, rule="window")
    p, idx, _ = backtest.cross_section(R, A, flip=flip, equal_weight=equal_weight)
    return ret.index[idx], p


def _save_capital(fig, ax, name):
    ax.axhline(1.0, color="black", linewidth=0.6, alpha=0.5)
    ax.set_ylabel("Capital")
    ax.legend(frameon=False)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / name, dpi=200)
    plt.close(fig)


def figure_baseline(ret):
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, kw, col in [("Slow minus fast", dict(), "blue"),
                           ("Fast minus slow", dict(flip=True), "vermillion"),
                           ("Equally weighted sectors", dict(equal_weight=True), "grey")]:
        d, p = _capital(ret, **kw)
        ax.plot(d, p, label=label, color=OKABE[col], linestyle=DASH[col], linewidth=1.4)
    _save_capital(fig, ax, "cross_section_h21.png")


def figure_grid(out):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    a = [(out["grid"][s].to_numpy(float) - 1) * 100 for s in ("train", "test")]
    vmax = max(np.nanmax(np.abs(x)) for x in a)
    g = out["grid"]["train"]
    for ax, arr, title in zip(axes, a, ("Training", "Testing")):
        im = ax.imshow(arr, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(title)
        ax.set_xticks(range(0, len(g.columns), 3), [f"{c:+.1f}" for c in g.columns[::3]])
        ax.set_yticks(range(len(g.index)), [f"{r:+.1f}" for r in g.index])
        ax.set_xlabel(r"$\tau_u$")
        ax.grid(False)
    axes[0].set_ylabel(r"$\tau_l$")
    fig.colorbar(im, ax=axes, label="Total return in per cent", fraction=0.03)
    fig.savefig(config.FIG_DIR / "threshold_grid.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def figure_threshold(out):
    """Testing period only, each rule starting from one unit of capital."""
    lo, hi = out["tau"]
    fig, ax = plt.subplots(figsize=(8, 4))
    part = out["test"]
    R = part.to_numpy(float)
    pool = swarm.alpha(R, rule="day", pooled=True)
    ld, ls = m.legs(R)
    paths = m.runs(R, pool, ld, ls, [lo, hi])
    idx = part.index[config.LAGS + 1:len(R) - 1]
    labels = {"rule": "Threshold rule", "opposite": "Opposite rule",
              "benchmark": "Benchmark", "benchmark_opposite": "Opposite benchmark"}
    for k, col in [("rule", "blue"), ("opposite", "vermillion"),
                   ("benchmark", "grey"), ("benchmark_opposite", "sky")]:
        ax.plot(idx, paths[k][0], label=labels[k], color=OKABE[col],
                linestyle=DASH[col], linewidth=1.4)
    _save_capital(fig, ax, "threshold_performance.png")


def figure_holding(out):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax2 = ax.twinx()
    h = list(out["hold"].keys())
    x = np.arange(len(h))
    for label, key, col in [("Slow minus fast", "direct", "blue"),
                            ("Fast minus slow", "reverse", "vermillion")]:
        ax.plot(x, [100 * out["hold"][k][key]["total"] for k in h], marker="o", markersize=4,
                label=f"{label}, return", color=OKABE[col], linestyle=DASH[col], linewidth=1.4)
        ax2.plot(x, [out["hold"][k][key]["sharpe"] for k in h], marker="s", markersize=4,
                 markerfacecolor="white", label=f"{label}, Sharpe",
                 color=OKABE[col], linestyle=DASH["grey"], linewidth=1.1)
    ax.axhline(0.0, color="black", linewidth=0.6)
    ax.set_xticks(x, [str(v) for v in h])
    ax.set_xlabel("Holding period in trading days")
    ax.set_ylabel("Total return in per cent")
    ax2.set_ylabel("Sharpe ratio")
    ax2.grid(False)
    lines = ax.get_lines()[:2] + ax2.get_lines()[:2]
    ax.legend(handles=lines, frameon=False, ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "holding_period.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = m.results()
    m.report(out)
    figure_grid(out)
    figure_threshold(out)
    figure_baseline(out["ret"])
    figure_holding(out)
    print(f"\nfigures written to {config.FIG_DIR}")
