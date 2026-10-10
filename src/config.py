"""Parameters for the swarm velocity study. Every constant is declared here once."""
from pathlib import Path

# Paths -----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "sector_etf_prices.csv"
FIG_DIR = ROOT / "latex" / "figures"

# Sample ----------------------------------------------------------------------
TICKERS = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]
SPLIT = "2024-01-01"                  # testing starts here
TRADING_DAYS = 252

# Strategies ------------------------------------------------------------------
LAGS = 5                              # dimension of the swarm space
TAU_LOW = [round(-1.3 + 0.1 * i, 2) for i in range(10)]    # -1.3 .. -0.4
TAU_HIGH = [round(-0.4 + 0.1 * i, 2) for i in range(20)]   # -0.4 ..  1.5
K = 2                                 # ETFs bought and sold on each side
H = 21                                # baseline holding period, trading days
H_GRID = [1, 2, 3, 5, 10, 21, 63]

# Trading ---------------------------------------------------------------------
GROSS = 1.0                           # half long, half short, no leverage
TCOST = 0.0005                        # per unit of one way turnover
