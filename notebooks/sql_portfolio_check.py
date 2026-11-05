"""Does SQL build the same portfolio return series as portfolio_returns?

Joins each day's log returns to a weights table in DuckDB and sums weight times
return per date, for the equal-weight and AGG-heavy baskets, then compares the
result with the Polars version.

Run: uv run python notebooks/sql_portfolio_check.py
"""

import duckdb
import polars as pl

from portfolio_risk_engine.data import DB_PATH, add_returns, read_prices, returns_wide
from portfolio_risk_engine.risk_stats import portfolio_returns

TICKERS = ["AAPL", "AGG", "JPM", "SPY", "VEA", "XOM"]

BASKETS = {
    "equal weight": {ticker: 1 / 6 for ticker in TICKERS},
    "AGG heavy": {"AAPL": 0.1, "AGG": 0.4, "JPM": 0.1, "SPY": 0.2, "VEA": 0.1, "XOM": 0.1},
}

PORTFOLIO_SQL = """
WITH rets AS (
    SELECT
        date,
        ticker,
        LN(close / LAG(close) OVER (PARTITION BY ticker ORDER BY date)) AS log_return
    FROM prices
),
weights(ticker, weight) AS (
    VALUES {values}
)
SELECT date, SUM(weight * log_return) AS sql_return
FROM rets
JOIN weights USING (ticker)
WHERE log_return IS NOT NULL
GROUP BY date
ORDER BY date
"""

con = duckdb.connect(DB_PATH, read_only=True)
wide = returns_wide(add_returns(read_prices()))
columns = [c for c in wide.columns if c != "date"]

for name, by_ticker in BASKETS.items():
    values = ", ".join(f"('{ticker}', {weight})" for ticker, weight in by_ticker.items())
    sql_series = con.sql(PORTFOLIO_SQL.format(values=values)).pl()

    weights = [by_ticker[ticker] for ticker in columns]
    polars_series = wide.select("date").with_columns(
        portfolio_returns(weights, wide).alias("polars_return")
    )

    joined = sql_series.join(polars_series, on="date")
    max_diff = joined.select((pl.col("sql_return") - pl.col("polars_return")).abs().max()).item()
    print(name, "rows: sql", sql_series.height, "polars", polars_series.height, "joined", joined.height)
    print(name, "max abs diff:", max_diff)
