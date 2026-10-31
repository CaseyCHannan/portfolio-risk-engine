import duckdb
import polars as pl

from portfolio_risk_engine.ingest import DB_PATH


PRICES_SQL = "SELECT date, ticker, close FROM prices ORDER BY ticker, date"

RETURNS_SQL = """
    SELECT
        date,
        ticker,
        close / LAG(close) OVER (PARTITION BY ticker ORDER BY date) - 1 AS simple_return_sql,
        LN(close / LAG(close) OVER (PARTITION BY ticker ORDER BY date)) AS log_return_sql
    FROM prices
    ORDER BY ticker, date
"""


def read_prices(db_path: str = DB_PATH) -> pl.DataFrame:
    return duckdb.connect(db_path, read_only=True).sql(PRICES_SQL).pl()


def add_returns(prices: pl.DataFrame) -> pl.DataFrame:
    """Add simple and log daily returns, computed separately for each ticker.

    Expects rows sorted by ticker, then date (as read_prices returns them),
    because shift(1) goes by row position. The first row of each ticker is null.
    """
    prev_close = pl.col("close").shift(1).over("ticker")
    return prices.with_columns(
        simple_return=pl.col("close") / prev_close - 1,
        log_return=(pl.col("close") / prev_close).log(),
    )


def read_returns_sql(db_path: str = DB_PATH) -> pl.DataFrame:
    return duckdb.connect(db_path, read_only=True).sql(RETURNS_SQL).pl()


def returns_wide(returns: pl.DataFrame, column: str = "log_return") -> pl.DataFrame:
    """Pivot long returns to one column per ticker and one row per date.

    Tickers are ordered alphabetically by name, and the first date (null for
    every ticker) is dropped, so every remaining row is complete.
    """
    wide = returns.pivot(on="ticker", index="date", values=column)
    tickers = sorted(c for c in wide.columns if c != "date")
    return wide.select("date", *tickers).sort("date").drop_nulls()
