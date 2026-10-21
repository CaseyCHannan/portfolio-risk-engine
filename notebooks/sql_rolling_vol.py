import duckdb
import polars as pl

from portfolio_risk_engine.data import add_returns, read_prices
from portfolio_risk_engine.risk_stats import add_rolling_vol

"""Rolling 60-day annualized vol of log returns for SPY, computed with raw SQL.

Run  uv run python notebooks/sql_rolling_vol.py to 
cross-check against the Polars add_rolling_vol output.
"""


QUERY = """
WITH rets AS (
    SELECT
        ticker,
        date,
        LN(close / LAG(close) OVER (PARTITION BY ticker ORDER BY date)) AS log_return
    FROM prices
)
SELECT
    ticker,
    date,
    log_return,
    CASE
        WHEN COUNT(log_return) OVER w = 60
        THEN STDDEV_SAMP(log_return) OVER w * SQRT(252)
    END AS rolling_vol
FROM rets
WHERE ticker = 'SPY'
WINDOW w AS (PARTITION BY ticker ORDER BY date ROWS BETWEEN 59 PRECEDING AND CURRENT ROW)
ORDER BY date
"""

con = duckdb.connect("data/risk.duckdb", read_only=True)
df = con.execute(QUERY).pl()
print(df.tail(5))

polars_spy = (
    add_rolling_vol(add_returns(read_prices()))
    .filter(pl.col("ticker") == "SPY")
    .select("date", pl.col("rolling_vol").alias("polars_vol"))
)
both = df.join(polars_spy, on="date")
print("null mismatches:", (both["rolling_vol"].is_null() != both["polars_vol"].is_null()).sum())
print("max abs diff:", (both["rolling_vol"] - both["polars_vol"]).abs().max())