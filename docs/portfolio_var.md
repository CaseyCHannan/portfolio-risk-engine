# Portfolio VaR and ES: Three methods on a weighted basket

*Phase 3 of the portfolio risk engine. Follows the single-ticker comparison in `docs/var_comparison.md`.*

## What I did.

I extended the single-ticker comparison to a weighted portfolio of the same six tickers (AAPL, AGG, JPM, SPY, VEA, XOM). `portfolio_risk(weights, wide, confidence)` returns the 1-day VaR and Expected Shortfall by three methods: historical (the portfolio's own daily return series), parametric (a normal fitted to the portfolio's mean and variance, `w' Sigma w`), and Monte Carlo (100,000 correlated draws using a Cholesky factor of the sample covariance, seed 23). Weights are fixed and rebalanced every day. Everything is at 99% confidence on daily log returns, and losses are reported as positive numbers. The demo is `notebooks/portfolio_risk_demo.py`.

## Equal-weight basket

| Method | VaR 99% | ES 99% |
| --- | --- | --- |
| Historical | 0.027661 | 0.042366 |
| Parametric | 0.022501 | 0.025847 |
| Monte Carlo | 0.022592 | 0.025923 |

- **Monte Carlo reproduces the results we see in the parametric.** The gaps are 0.4% for VaR and 0.3% for ES. Both methods assume the same fitted normal, so this checks the code; it says nothing about the market.
- **The normal model understates the tail.** Parametric VaR is 81% of historical VaR, so the normal model understates it by about 19%, inside the 13% to 22% range I found for the single tickers.
- **It understates ES much more than VaR.** Parametric ES is 61% of historical ES, an understatement of about 39%. The ES/VaR ratio is about 1.15 under the normal model and about 1.53 in the history. The normal model gets the cutoff roughly right and the average loss beyond it badly wrong.

## What diversification did

- The portfolio's parametric VaR is 76% of the weighted average of the six single-ticker VaRs. Correlations below 1 cut the scale of the risk.
- It did not thin the tails. The portfolio's excess kurtosis is 15.85 (skew −0.76), higher than SPY's 14.53 and every single name except AGG.
- A possible reason is a shared market factor: stocks and bonds may fall together on the worst days. **I have not tested this.** Checking the dates of AGG's worst days would be a first test.

## A second basket: AGG-heavy

AGG 40%, SPY 20%, and AAPL, JPM, VEA, XOM 10% each.

| Method | VaR 99% | ES 99% |
| --- | --- | --- |
| Historical | 0.019300 | 0.030847 |
| Parametric | 0.016210 | 0.018622 |
| Monte Carlo | 0.016289 | 0.018637 |

- Moving weight into bonds cuts the risk: historical VaR falls to about 70% of the equal-weight figure.
- The same pattern shows up. The normal model understates VaR by about 16% and ES by about 40%, and the ES/VaR ratio is about 1.15 under the normal model and 1.60 historically.
- The VaR gap is a little smaller here (84% against 81%). That is one pair of baskets, so I treat it as an observation and not a finding.

## Caveats

- One 11-year in-sample history: the same data fits the models and tests them.
- Weights are fixed and rebalanced daily. A buy-and-hold portfolio would drift away from the target weights.
- A weighted sum of log returns only approximates the portfolio return (it is exact for simple returns).
- Monte Carlo ES is noisier than Monte Carlo VaR, because it averages a thin slice of the simulated tail.
- The normal-based methods ignore volatility clustering.

## Takeaway

Correlation changes the scale of the risk, but not the shape of the tail. A normal model understates the 99% VaR by about a fifth and the 99% ES by about two fifths, in both baskets I tried.

## Next

- Look at the dates of AGG's worst days, and test whether stocks and bonds share their worst days.
- Add a Student-t column as a fourth method.

