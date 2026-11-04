"""VaR and ES by three methods for an equal-weight and an AGG-heavy basket.

Uses portfolio_risk at 99% confidence on the six-ticker store.
Run: uv run python notebooks/portfolio_risk_demo.py
"""

from portfolio_risk_engine.data import (
    add_returns,
    read_prices,
    returns_wide,
)
from portfolio_risk_engine.risk_stats import portfolio_risk

CONFIDENCE = 0.99

wide = returns_wide(add_returns(read_prices()))
tickers = [c for c in wide.columns if c != "date"]

baskets = {
    "equal weight": {t: 1 / len(tickers) for t in tickers},
    "AGG heavy": {"AAPL": 0.1, "AGG": 0.4, "JPM": 0.1, "SPY": 0.2, "VEA": 0.1, "XOM": 0.1},
}

for name, by_ticker in baskets.items():
    weights = [by_ticker[t] for t in tickers]
    print(f"\n{name}: {by_ticker}")
    print(portfolio_risk(weights, wide, CONFIDENCE, seed=23))
