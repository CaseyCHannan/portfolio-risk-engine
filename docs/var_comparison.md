# VaR Methods Compared: Historical, Parametric and Monte Carlo
### results from the first stage of my "risk-engine" project

## 1. Setup and normality

### Setup

- **Basket:** SPY (US large-cap), VEA (international developed), AGG (US aggregate bonds), and three single names: AAPL (tech), JPM (financials), XOM (energy).
- **Data:** adjusted daily closes from yfinance, stored in DuckDB, 2015-01-02 to 2026-10-09. That gives 2,959 daily log returns per ticker (the first row of each ticker has no prior price).
- **Convention:** VaR and ES are computed on daily log returns and reported as positive losses. A 99% VaR of 0.032 means that on the worst 1% of days the loss was at least 3.2%.
- **Methods compared:**
  1. *Historical:* the empirical 1% quantile of the observed returns (`'lower'` interpolation).
  2. *Parametric:* assume returns are normal with the sample mean and sample standard deviation; VaR = -(mean + z x std), with z = -2.326 at 99%.
  3. *Monte Carlo:* fit the same normal, draw 100,000 returns (seed 23), and take the 1% quantile of the draws.
- **Where the code lives:** the tested functions are in `src/portfolio_risk_engine/risk_stats.py`; every number in this report is produced by `notebooks/var_comparison.py` (normality statistics by `notebooks/normality_check.py`).

### Are the returns normal? No.

Parametric and Monte Carlo VaR (as built here) both assume normal returns, so the first question is whether that assumption holds. Skew and excess kurtosis are both 0 for a normal distribution.

| Ticker | n | Skew | Excess kurtosis | Jarque-Bera p |
| --- | --- | --- | --- | --- |
| AAPL | 2959 | -0.12 | 6.21 | 0 |
| AGG | 2959 | -1.11 | 19.67 | 0 |
| JPM | 2959 | -0.08 | 12.76 | 0 |
| SPY | 2959 | -0.57 | 14.53 | 0 |
| VEA | 2959 | -1.05 | 14.86 | 0 |
| XOM | 2959 | -0.20 | 6.01 | 0 |

- **Fat tails:** all six tickers have excess kurtosis between 6 and 20, so extreme days are far more frequent than a normal distribution predicts.
- **Negative skew:** all six have a longer LOSS tail than GAIN tail.
- **Read the p-values with care:** a Jarque-Bera p-value of 0 means below floating-point range. With about 3,000 observations the test rejects normality for almost any real data, so the skew and kurtosis columns carry the information, not the p-value.
- **Kurtosis is not riskiness:** AGG has the highest kurtosis but the smallest VaR in the basket. A fourth moment is dominated by a few extreme days, and AGG is calm on most days. (That its kurtosis comes from a handful of specific days has not yet been checked against the data.)
- **The i.i.d. assumption is shaky:** daily returns show volatility clustering, so the Jarque-Bera test is best treated as a screen, not a precise test.

Stage 2 will the obvious next question -> 
If the returns are not normal, how wrong is a normal-based VaR?
