"""Does a SQL join build the same wide returns table as returns_wide?

Builds one column of log returns per ticker in DuckDB (a CTE per ticker, joined
on date), then compares it with the Polars pivot in returns_wide.

Run: uv run python notebooks/sql_wide_check.py
"""

import duckdb
import polars as pl

from portfolio_risk_engine.data import DB_PATH, add_returns, read_prices, returns_wide

WIDE_SQL = """
WITH rets AS (
    SELECT
        date,
        ticker,
        LN(close / LAG(close) OVER (PARTITION BY ticker ORDER BY date)) AS log_return
    FROM prices
),
t_aapl AS (SELECT date, log_return AS AAPL FROM rets WHERE ticker = 'AAPL'),
t_agg AS (SELECT date, log_return AS AGG FROM rets WHERE ticker = 'AGG'),
t_jpm AS (SELECT date, log_return AS JPM FROM rets WHERE ticker = 'JPM'),
t_spy AS (SELECT date, log_return AS SPY FROM rets WHERE ticker = 'SPY'),
t_vea AS (SELECT date, log_return AS VEA FROM rets WHERE ticker = 'VEA'),
t_xom AS (SELECT date, log_return AS XOM FROM rets WHERE ticker = 'XOM')
SELECT *
FROM t_aapl
JOIN t_agg USING (date)
JOIN t_jpm USING (date)
JOIN t_spy USING (date)
JOIN t_vea USING (date)
JOIN t_xom USING (date)
WHERE t_aapl.AAPL IS NOT NULL
ORDER BY date
"""

sql_wide = duckdb.connect(DB_PATH, read_only=True).sql(WIDE_SQL).pl()
polars_wide = returns_wide(add_returns(read_prices()))

joined = sql_wide.join(polars_wide, on="date", suffix="_pl")
print("rows: sql", sql_wide.height, "polars", polars_wide.height, "joined", joined.height)
print("columns match:", sql_wide.columns == polars_wide.columns)

for ticker in [c for c in polars_wide.columns if c != "date"]:
    null_mismatch = joined.select(
        (pl.col(ticker).is_null() != pl.col(f"{ticker}_pl").is_null()).sum()
    ).item()
    max_diff = joined.select(
        (pl.col(ticker) - pl.col(f"{ticker}_pl")).abs().max()
    ).item()
    print(ticker, "null mismatches:", null_mismatch, "max abs diff:", max_diff)
