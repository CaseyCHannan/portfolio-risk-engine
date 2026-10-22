import polars as pl

from portfolio_risk_engine.data import (
    add_returns,
    read_prices,
)

from portfolio_risk_engine.risk_stats import (
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

result = parametric_var_by_ticker(returns, confidence=0.99)
print(result)


historical = risk_by_ticker(returns, confidence=0.99)
print(historical)



comparison = historical.join(result, on="ticker", suffix="_parametric")
print(comparison)