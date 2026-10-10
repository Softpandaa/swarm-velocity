# Swarm Velocity Dynamics in US Sector ETFs

This study utilizes the leader-following swarm velocity estimator of Chen, Huang and Tang (2025), *Classifying Factor Velocity with Swarm Intelligence* on eleven Select Sector SPDR ETFs at daily frequency and asks whether the estimated velocity carries tradable information. 

We backtest two strategies: a threshold rule that applies the cross-sectional average velocity to a leader-versus-loser pair trade, and a cross-sectional rule that ranks the ETFs by their own velocities. The swarm velocity proves to be informative but does not offer a standalone tradable edge. The cross-sectional rule loses when trading from slow to fast ETFs but gains when doing the opposite, as the correlated ETFs cause the velocity estimator to favor an asset's own movement over its distance to the leader. Backtesting results reveal that all strategies are outperformed by an equally weighted portfolio of the same ETFs.

## Layout

```
data/     committed input prices
src/      analysis modules, every parameter declared once in config.py
report.pdf
slides.pdf
```

The report and slides are distributed as compiled PDFs. Their typesetting source is not included.

## Data

`data/sector_etf_prices.csv` is the eleven Select Sector SPDR ETFs daily close from Yahoo Finance from October 2018 to December 2025.

## Reproducing

Python 3.13.

```
pip install -r requirements.txt
python src/plots.py
```

This prints the table of the report and rewrites the four figures it uses to `latex/figures/`, which is created on the first run and is not tracked.