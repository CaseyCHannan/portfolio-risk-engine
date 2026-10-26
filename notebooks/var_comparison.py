"""Compares historical, parametric and Monte Carlo VaR at 99% on the six-ticker basket.

Prints 1-day and 10-day tables of daily log-return VaR, as positive losses.
Run: uv run python notebooks/var_comparison.py
"""

import polars as pl

from portfolio_risk_engine.data import (
    add_returns,
    read_prices,
)

from portfolio_risk_engine.risk_stats import (
    historical_var_multiday,
    monte_carlo_var,
    monte_carlo_var_multiday,
    parametric_var,
    risk_by_ticker,
)


prices = read_prices()
returns = add_returns(prices)


def parametric_var_by_ticker(returns, confidence=0.95, column="log_return"):
    rows = []
    for (ticker,), group in returns.group_by("ticker", maintain_order=True):
        series = group[column].drop_nulls()  # the `column` of group, nulls dropped
        rows.append({"ticker": ticker, "var": parametric_var(series, confidence)})
    return pl.DataFrame(rows)

def monte_carlo_var_by_ticker(returns, confidence=0.95, column="log_return", n_paths=100_000, seed=23):
    rows = []
    for (ticker,), group in returns.group_by("ticker", maintain_order=True):
        series = group[column].drop_nulls()
        rows.append({"ticker": ticker, "var": monte_carlo_var(series, confidence, n_paths, seed)})
    return pl.DataFrame(rows)

def monte_carlo_var_multiday_by_ticker(returns, confidence=0.95, horizon=10, column="log_return", n_paths=100_000, seed=23):
    rows = []
    for (ticker,), group in returns.group_by("ticker", maintain_order=True):
        series = group[column].drop_nulls()
        rows.append({"ticker": ticker, "var": monte_carlo_var_multiday(series, confidence, horizon, n_paths, seed)})
    return pl.DataFrame(rows)

result = parametric_var_by_ticker(returns, confidence=0.99)
print(result)


historical = risk_by_ticker(returns, confidence=0.99)
print(historical)

monte_carlo = monte_carlo_var_by_ticker(returns, confidence=0.99)
print(monte_carlo)

comparison = (
    historical.rename({"var": "var_historical", "es": "es_historical"})
    .join(result.rename({"var": "var_parametric"}), on="ticker")
    .join(monte_carlo.rename({"var": "var_monte_carlo"}), on="ticker")
)
print(comparison)

monte_carlo_10day = monte_carlo_var_multiday_by_ticker(returns, confidence=0.99, horizon=10)
print(monte_carlo_10day)

def historical_var_multiday_by_ticker(returns, confidence=0.95, horizon=10, column="log_return"):
    rows = []
    for (ticker,), group in returns.group_by("ticker", maintain_order=True):
        series = group[column].drop_nulls()
        rows.append({"ticker": ticker, "var": historical_var_multiday(series, confidence, horizon)})
    return pl.DataFrame(rows)

historical_10day = historical_var_multiday_by_ticker(returns, confidence=0.99, horizon=10)
print(historical_10day)

comparison_10day = (
    historical_10day.rename({"var": "var_historical_10d"})
    .join(monte_carlo_10day.rename({"var": "var_monte_carlo_10d"}), on="ticker")
    .with_columns(
        (pl.col("var_monte_carlo_10d") / pl.col("var_historical_10d")).alias("mc_as_share_of_historical")
    )
)
print(comparison_10day)
