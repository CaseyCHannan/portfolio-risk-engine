import yfinance as yf
import pandas as pd
import duckdb
from datetime import date

CREATE_PRICES = """
CREATE TABLE IF NOT EXISTS prices (
    date   DATE    NOT NULL,
    ticker VARCHAR NOT NULL,
    close  DOUBLE  NOT NULL,
    PRIMARY KEY (date, ticker)
)
"""

def to_long(wide: pd.DataFrame) -> pd.DataFrame:
    """Reshape wide closes (date x ticker) to long: date, ticker, close."""
    return (
        wide.rename_axis(index="date", columns="ticker")
        .stack()
        .rename("close")
        .reset_index()
    )

DB_PATH = "data/risk.duckdb"

def load_prices(long: pd.DataFrame, db_path: str = DB_PATH) -> None:
    """Upsert completed trading days into the prices table."""
    long = long[long["date"] < pd.Timestamp(date.today())]
    with duckdb.connect(db_path) as con:
        con.execute(CREATE_PRICES)
        con.register("incoming", long)
        con.execute("INSERT OR REPLACE INTO prices SELECT date, ticker, close FROM incoming")

# Basket design: a small, diversified mix across asset classes and sectors.
#   SPY  - US large-cap equity (market benchmark)
#   VEA  - international developed-market equity
#   AGG  - US aggregate bonds (low correlation with equities, useful for
#          portfolio risk and correlation modeling later)
#   AAPL, JPM, XOM - big single names from different sectors (tech, financials,
#          energy) to add idiosyncratic risk
# All have  history back to at least 2015 or earlier.
TICKERS = ["SPY", "VEA", "AGG", "AAPL", "JPM", "XOM"]


def fetch_prices(tickers, start="2015-01-01"):
    """Download daily ADJUSTED prices for a list of specified tickers."""
    return yf.download(
        tickers, start=start, auto_adjust=True, progress=False, threads=False
    )

def main():
    prices = fetch_prices(TICKERS)["Close"]
    empty = prices.columns[prices.isna().all()].tolist()
    if empty:
        raise RuntimeError(f"No data returned for: {empty}")
    load_prices(to_long(prices))


if __name__ == "__main__":
    main()