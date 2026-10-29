# Portfolio Risk Engine

A Python project implementing common market risk methods: historical,
parametric, and Monte Carlo Value at Risk (VaR), Expected Shortfall (ES),
and (eventually) Monte Carlo option pricing, validated against Black-Scholes.

This is a learning project and a real software project at the same time.
I'm using it to deepen my understanding of the methods themselves, to build
something tested and reproducible, and to gain fluency with Polars and
DuckDB.

## Main finding

On six liquid tickers (2015 to Oct 2026), a normal-distribution VaR
understates the historical 99% one-day VaR by about 13% to 22%. Daily
returns have fat tails (excess kurtosis 6 to 20) and negative skew, which
the normal model cannot capture. Monte Carlo VaR agrees with parametric VaR
to within sampling noise, since both assume normal shocks. At 10 days the
normal model still understates historical VaR (by 4% to 20%), but the gap
does not grow with the horizon.

One-day 99% VaR, log returns, reported as a positive loss:

| Ticker | Historical | Parametric | Monte Carlo | Parametric as % of historical |
| --- | --- | --- | --- | --- |
| AAPL | 0.0497 | 0.0411 | 0.0413 | 83% |
| AGG | 0.0087 | 0.0076 | 0.0076 | 87% |
| JPM | 0.0455 | 0.0386 | 0.0388 | 85% |
| SPY | 0.0322 | 0.0252 | 0.0253 | 78% |
| VEA | 0.0297 | 0.0252 | 0.0253 | 85% |
| XOM | 0.0484 | 0.0399 | 0.0400 | 83% |

The full write-up is in [`docs/var_comparison.md`](docs/var_comparison.md)
(in progress). It also includes a study of Monte Carlo standard error: the
spread of the estimate falls by about 3.3x for every 10x increase in paths,
matching the theoretical formula.

## What's implemented

- Historical VaR and ES, plus a per-ticker wrapper
- Parametric (normal) VaR
- Monte Carlo VaR, single-step and multi-day, seeded for reproducibility
- Historical multi-day VaR from rolling sums of log returns
- Rolling annualized volatility and rolling Sharpe ratio
- Normality check (skew, excess kurtosis, Jarque-Bera) on daily returns
- Returns cross-checked against an independent SQL implementation
  (`LAG()` for returns, a window-frame query for rolling volatility)
- 15 pytest tests, mostly against small datasets with hand-computed answers

## Status

Phases 0 to 2 (data foundation, descriptive risk stats, VaR comparison) are
built. The comparison report is being finished. Planned next:

- Correlated multi-asset simulation and portfolio-level VaR/ES
- Monte Carlo option pricing, validated against Black-Scholes
- CI, linting, and packaging

## Data

Adjusted daily prices from 2015 onward via yfinance for SPY, VEA, AGG,
AAPL, JPM, and XOM, stored in a local DuckDB file (`data/risk.duckdb`,
not committed). Anyone running the project downloads the data themselves.

## Setup

```
uv sync
uv run python -m portfolio_risk_engine.ingest
uv run pytest -v
uv run python notebooks/var_comparison.py
```

## Conventions

- VaR and ES are reported as positive losses.
- Quantiles use the "lower" convention (the loss the worst 1% or 5% of
  observations reach), so risk is never understated by interpolation.
- Returns are log returns; standard deviations are sample (n - 1).

## Notes and limitations

- Tickers are used as identifiers. A production system would key on
  CUSIP/ISIN via a security master.
- Rows dated today are dropped at load time, so the database holds only
  completed trading days.
- Results rest on one 11-year sample. Multi-day historical windows overlap,
  so the 10-day figures are noisier than the window count suggests.
- Built with AI assistance (Claude). I review each change before committing,
  and cross-check results independently (e.g., Polars returns vs. a SQL
  LAG() implementation).

## About

I'm Casey, an instructor and former software engineer with a master's in
statistics. I've been interested in risk for a long time, and working
through actuarial exams and CFA Level II brought that interest back. I'm
currently preparing for CAS MAS-I in January 2027. This project is where
I'm turning that interest into tested code. Feedback, especially on
methodology, is welcome via issues.

[LinkedIn](https://www.linkedin.com/in/caseyhannan)

