"""Parametric 99% VaR of an equal-weight six-ticker portfolio.

Compares it with the weighted average of the single-ticker parametric VaRs;
the gap is the diversification benefit from imperfect correlation.
Run: uv run python notebooks/portfolio_var.py
"""

import numpy as np

from portfolio_risk_engine.data import (
    add_returns,
    read_prices,
    returns_wide,
)
from portfolio_risk_engine.risk_stats import (
    parametric_portfolio_var,
    parametric_var,
)

CONFIDENCE = 0.99

wide = returns_wide(add_returns(read_prices()))
tickers = [c for c in wide.columns if c != "date"]
weights = np.full(len(tickers), 1 / len(tickers))

singles = [parametric_var(wide[t], CONFIDENCE) for t in tickers]
weighted_average = float(weights @ singles)
portfolio = parametric_portfolio_var(weights, wide, CONFIDENCE)

print("tickers:", tickers)
print(f"weighted average of single-ticker parametric VaR: {weighted_average:.6f}")
print(f"equal-weight portfolio parametric VaR:            {portfolio:.6f}")
print(f"portfolio as a share of the weighted average:     {portfolio / weighted_average:.1%}")
