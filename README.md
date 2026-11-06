# Swarm Velocity Dynamics in US Sector ETFs

This study examines whether the leader-following swarm velocity of Chen, Huang and Tang (2025), *Classifying Factor Velocity with Swarm Intelligence*, carries tradable information for eleven Select Sector SPDR ETFs at daily frequency. A threshold rule applies the pooled velocity to a leader-versus-loser pair trade, and a cross-sectional rule ranks the ETFs by their own speeds.

## Findings

The velocity threshold cuts transaction costs but does not pick better trades. Out of sample, its trades earn −0.06% per active day before costs, no better than the −0.03% of the ungated trade. Neither rule earns a significantly positive return, and an equally weighted portfolio of the same ETFs has a higher Sharpe ratio.

The estimated speeds are negative, whereas Chen, Huang and Tang (2025) find positive ones. Because the gap to the leader is dated on the same day as the velocity, the estimate sits near −1/2, its value for returns without leader-following.

## Layout

```
data/     committed input prices
src/      analysis modules, every parameter declared once in config.py
report.pdf
```

The report is distributed as a compiled PDF. Its typesetting source is not included.

## Data

`data/sector_etf_prices.csv` holds the daily adjusted closes of the eleven Select Sector SPDR ETFs from Yahoo Finance for October 2018 to December 2025.

## Reproducing

Python 3.13.

```
pip install -r requirements.txt
python src/plots.py
```

This prints every number quoted in the report and draws its four figures.