# Portfolio Risk Engine

A Python project implementing common market risk methods: historical,
parametric,
and Monte Carlo Value at Risk (VaR), Expected Shortfall (ES),
and (eventually) Monte Carlo option pricing, validated with Black-Scholes.

This is a learning project and a real-software project at the same time.
I'm using it to deepen my understanding of the methods themselves, to build
something tested and reproducible, and to gain fluency with the Polars and
DuckDB.

## Status

So far, the initial phase seems done: prices are ingested into DuckDB and
daily simple and log returns are computed in Polars, cross-checked against
an independent SQL `LAG()` implementation. Risk metrics come next.

## Data

Adjusted daily prices from 2015 onward via yfinance for SPY, VEA, AGG,
AAPL, JPM, and XOM, stored in a local DuckDB file (`data/risk.duckdb`,
not committed).

## Setup

```
uv sync
uv run python -m portfolio_risk_engine.ingest
```

## Notes and limitations

- Tickers are used as identifiers. 
- Rows dated today are dropped at load time, so the database holds only
  completed trading days.
- Built with AI assistance (Claude). I review each change before committing, and cross-check results independently (e.g., Polars returns vs. a SQL LAG() implementation).