"""Parametric 99% VaR of an equal-weight six-ticker portfolio.

Compares it with the weighted average of the single-ticker parametric VaRs;
the gap is the diversification benefit from imperfect correlation.
Run: uv run python notebooks/portfolio_var.py
"""

import numpy as np
from scipy.stats import kurtosis, skew

from portfolio_risk_engine.data import (
    add_returns,
    read_prices,
    returns_wide,
)
from portfolio_risk_engine.risk_stats import (
    historical_es,
    historical_var,
    monte_carlo_portfolio_var,
    parametric_portfolio_var,
    parametric_var,
    portfolio_returns,
)

CONFIDENCE = 0.99

wide = returns_wide(add_returns(read_prices()))
tickers = [c for c in wide.columns if c != "date"]
weights = np.full(len(tickers), 1 / len(tickers))

singles = [parametric_var(wide[t], CONFIDENCE) for t in tickers]
weighted_average = float(weights @ singles)
portfolio = parametric_portfolio_var(weights, wide, CONFIDENCE)
simulated = monte_carlo_portfolio_var(weights, wide, CONFIDENCE, seed=23)

print("tickers:", tickers)
print(f"weighted average of single-ticker parametric VaR: {weighted_average:.6f}")
print(f"equal-weight portfolio parametric VaR:            {portfolio:.6f}")
print(f"portfolio as a share of the weighted average:     {portfolio / weighted_average:.1%}")
print(f"equal-weight portfolio Monte Carlo VaR (seed 23): {simulated:.6f}")
print(f"Monte Carlo as a share of parametric:             {simulated / portfolio:.1%}")

daily = portfolio_returns(weights, wide)
hist_var = historical_var(daily, CONFIDENCE)
hist_es = historical_es(daily, CONFIDENCE)
print(f"equal-weight portfolio historical VaR:            {hist_var:.6f}")
print(f"equal-weight portfolio historical ES:             {hist_es:.6f}")
print(f"parametric as a share of historical VaR:          {portfolio / hist_var:.1%}")
print(f"Monte Carlo as a share of historical VaR:         {simulated / hist_var:.1%}")
print(f"portfolio skew {skew(daily.to_numpy()):.2f}, excess kurtosis {kurtosis(daily.to_numpy()):.2f}")
