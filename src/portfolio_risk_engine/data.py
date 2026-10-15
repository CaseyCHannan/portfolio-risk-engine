import duckdb
import polars as pl

from portfolio_risk_engine.ingest import DB_PATH


RETURNS_SQL = """
    SELECT
        date,
        ticker,
        close / LAG(close) OVER (PARTITION BY ticker ORDER BY date) - 1 AS simple_return_sql,
        LN(close / LAG(close) OVER (PARTITION BY ticker ORDER BY date)) AS log_return_sql
    FROM prices
    ORDER BY ticker, date
"""

PRICES_SQL = "SELECT date, ticker, close FROM prices ORDER BY ticker, date"

def read_prices(db_path: str = DB_PATH) -> pl.DataFrame:
    return duckdb.connect(db_path, read_only=True).sql('SELECT date, ticker, close FROM prices ORDER BY ticker, date').pl()


def add_returns(prices: pl.DataFrame) -> pl.DataFrame:
    prev_close = pl.col('close').shift(1).over('ticker')
    return prices.with_columns(
        simple_return = pl.col('close') / prev_close - 1,
        log_return = (pl.col('close') / prev_close).log(),
    )

def read_returns_sql():
    con = duckdb.connect(DB_PATH, read_only=True)
    return con.sql(RETURNS_SQL).pl()