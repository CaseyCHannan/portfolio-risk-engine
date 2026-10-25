import polars as pl

from portfolio_risk_engine.data import (
    add_returns,
    read_prices,
)

from portfolio_risk_engine.risk_stats import (
    monte_carlo_var,
    monte_carlo_var_multiday,
    parametric_var,
    risk_by_ticker,
   )


prices = read_prices()
returns = add_returns(prices)

spy_returns = (
    returns
    .filter(pl.col("ticker") == "SPY")
    .select("log_return")
    .drop_nulls()
)
log_returns = spy_returns["log_return"]

spy_var = parametric_var(log_returns, confidence=0.99)
print(spy_var)

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