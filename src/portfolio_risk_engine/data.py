import duckdb
import polars as pl

from portfolio_risk_engine.ingest import DB_PATH


def read_prices(db_path: str = DB_PATH) -> pl.DataFrame:
    return duckdb.connect(db_path, read_only=True).sql('SELECT date, ticker, close FROM prices ORDER BY ticker, date').pl()