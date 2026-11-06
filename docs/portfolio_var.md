# Portfolio VaR and ES: Four methods on a weighted basket

*Phase 3 of the portfolio risk engine. Follows the single-ticker comparison in `docs/var_comparison.md`.*

## What I did.

I extended the single-ticker comparison to a weighted portfolio of the same six tickers (AAPL, AGG, JPM, SPY, VEA, XOM). `portfolio_risk(weights, wide, confidence)` returns the 1-day VaR and Expected Shortfall by four methods: historical (the portfolio's own daily return series), parametric (a normal fitted to the portfolio's mean and variance, `w' Sigma w`), Monte Carlo (100,000 correlated draws using a Cholesky factor of the sample covariance, seed 23), and Student-t (a Student-t distribution fitted by maximum likelihood to the portfolio's daily return series, with VaR and ES from the fitted t's closed form). Weights are fixed and rebalanced every day. Everything is at 99% confidence on daily log returns, and losses are reported as positive numbers. The demo is `notebooks/portfolio_risk_demo.py`.

## Equal-weight basket

| Method | VaR 99% | ES 99% |
| --- | --- | --- |
| Historical | 0.027661 | 0.042366 |
| Parametric | 0.022501 | 0.025847 |
| Monte Carlo | 0.022592 | 0.025923 |
| Student-t | 0.026141 | 0.040793 |

- **Monte Carlo reproduces the results we see in the parametric.** The gaps are 0.4% for VaR and 0.3% for ES. Both methods assume the same fitted normal, so this checks the code; it says nothing about the market.
- **The normal model understates the tail.** Parametric VaR is 81% of historical VaR, so the normal model understates it by about 19%, inside the 13% to 22% range I found for the single tickers.
- **It understates ES much more than VaR.** Parametric ES is 61% of historical ES, an understatement of about 39%. The ES/VaR ratio is about 1.15 under the normal model and about 1.53 in the history. The normal model gets the cutoff roughly right and the average loss beyond it badly wrong.
- **A Student-t fitted to the portfolio gets most of the way back.** Its VaR is about 95% of historical VaR (0.0261 against 0.0277) and its ES is about 96% of historical ES (0.0408 against 0.0424). The fitted degrees of freedom are about 3.0, and its ES/VaR ratio is about 1.56, close to the 1.53 in the history.

## What diversification did

- The portfolio's parametric VaR is 76% of the weighted average of the six single-ticker VaRs. Correlations below 1 cut the scale of the risk.
- It did not thin the tails. The portfolio's excess kurtosis is 15.85 (skew −0.76), higher than SPY's 14.53 and every single name except AGG.
- A possible reason is a shared market factor: stocks and bonds may fall together on the worst days. I listed each ticker's five worst days (`notebooks/worst_days.py`). March 12, 2020 is on all six lists, AGG included, and 20 of the 30 slots are days in March 2020. Four of AGG's five worst days are in March 2020, so its high kurtosis looks like a few days and not steady tail risk. This fits a shared factor, but it is only five days per ticker. **I have not measured how much of the kurtosis those days explain.**

## A second basket: AGG-heavy

AGG 40%, SPY 20%, and AAPL, JPM, VEA, XOM 10% each.

| Method | VaR 99% | ES 99% |
| --- | --- | --- |
| Historical | 0.019300 | 0.030847 |
| Parametric | 0.016210 | 0.018622 |
| Monte Carlo | 0.016289 | 0.018637 |
| Student-t | 0.026227 | 0.053947 |

- Moving weight into bonds cuts the risk: historical VaR falls to about 70% of the equal-weight figure.
- The same pattern shows up. The normal model understates VaR by about 16% and ES by about 40%, and the ES/VaR ratio is about 1.15 under the normal model and 1.60 historically.
- The VaR gap is a little smaller here (84% against 81%). That is one pair of baskets, so I treat it as an observation and not a finding.
- **The Student-t fit does not hold up here.** Its VaR is about 136% of historical VaR and its ES about 175% of historical ES, so it overshoots instead of understating. The fitted degrees of freedom are 1.98. A t with about 2 degrees of freedom has infinite variance, and its ES grows quickly as the degrees of freedom fall toward 2 (for a standard t at 99%, ES is about 7.0 at 3 degrees of freedom and about 14.1 at 2). The fitted VaR (0.0262) is about the same as the equal-weight fit (0.0261), although the historical VaR is only 70% as large, which a calmer portfolio should not do. **I have not tested why the fit lands so low.** One guess is a tight everyday center with a few very large days (March 2020) pulling the fit toward heavy tails.

## Caveats

- One 11-year in-sample history: the same data fits the models and tests them.
- Weights are fixed and rebalanced daily. A buy-and-hold portfolio would drift away from the target weights.
- A weighted sum of log returns only approximates the portfolio return (it is exact for simple returns).
- Monte Carlo ES is noisier than Monte Carlo VaR, because it averages a thin slice of the simulated tail.
- The normal-based methods ignore volatility clustering.
- The Student-t row fits one t to the combined portfolio series. It models fat tails but not crash-time correlation between the assets (that would need a multivariate t or a copula).
- The Student-t result depends on the fitted degrees of freedom. Near 2 the tail estimate is unstable.

## Takeaway

Correlation changes the scale of the risk, but not the shape of the tail. A normal model understates the 99% VaR by about a fifth and the 99% ES by about two fifths, in both baskets I tried. A Student-t fitted to the portfolio series closes most of that gap for the equal-weight basket, but the same fit overshoots for the AGG-heavy basket, where the fitted degrees of freedom fall to about 2. The t row is only as reliable as its fitted degrees of freedom.

## Next

- Refit the Student-t, and recompute the kurtosis, with March 2020 removed, to see how much of the tail those days explain.
