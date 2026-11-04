"""Which days are each ticker's worst five?

Checks whether AGG's high excess kurtosis comes from a few days, and whether
stocks and bonds share their worst days.

Run: uv run python notebooks/worst_days.py
"""

from portfolio_risk_engine.data import add_returns, read_prices

returns = add_returns(read_prices()).drop_nulls("log_return")

for (ticker,), group in returns.group_by("ticker", maintain_order=True):
    print(ticker)
    print(group.sort("log_return").head(5).select("date", "log_return"))
