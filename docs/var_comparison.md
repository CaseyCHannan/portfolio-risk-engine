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


## 2. One-day comparison at 99%

| Ticker | Historical VaR | Historical ES | Parametric VaR | Monte Carlo VaR | Parametric as % of historical |
| --- | --- | --- | --- | --- | --- |
| AAPL | 0.0497 | 0.0685 | 0.0411 | 0.0413 | 83% |
| AGG | 0.0087 | 0.0131 | 0.0076 | 0.0076 | 87% |
| JPM | 0.0455 | 0.0660 | 0.0386 | 0.0388 | 85% |
| SPY | 0.0322 | 0.0471 | 0.0252 | 0.0253 | 78% |
| VEA | 0.0297 | 0.0463 | 0.0252 | 0.0253 | 85% |
| XOM | 0.0484 | 0.0651 | 0.0399 | 0.0400 | 83% |

*Daily log-return VaR and ES at 99%, as positive losses. This Monte Carlo uses 100,000 paths and seed 23.*

- **The normal model understates the historical 99% VaR for all six tickers**, by about 13% (AGG) to 22% (SPY). This is what the fat tails in section 1 predicted: a normal curve fitted to the same mean and standard deviation puts too little probability on extreme days.
- **Monte Carlo and parametric agree.** Both assume the same normal, so Monte Carlo is the parametric answer plus some sampling noise. It sits about 0.36% above parametric in every row; section 3 shows why that is noise and not a bug.
- **ES is well above VaR in every row**, by about 35% to 56%. Under a normal distribution at 99%, ES is only about 15% above VaR (ignoring the mean), so the real tails are heavier past the cutoff as well as at it.
- **SPY has the largest gap** even though its VaR is below the single names. Diversification shrinks the size of the risk but not the shape of the tail.



## 3. How many paths? The sampling noise in Monte Carlo VaR

Section 2 showed Monte Carlo VaR sitting about 0.36% above parametric in every row. To see whether that is a bug or noise, I ran the Monte Carlo VaR for SPY at 99% fifty times (seeds 0 to 49) at three path counts and compared the spread of the estimates with the theoretical standard error of a simulated quantile, sqrt(p(1-p)/n) / f(q), where f is the normal density at the cutoff.

| n_paths | Mean estimate | Std across seeds | Theory SE | Mean minus exact |
| --- | --- | --- | --- | --- |
| 1,000 | 0.025543 | 0.001459 | 0.001305 | +0.000335 |
| 10,000 | 0.025173 | 0.000425 | 0.000413 | -0.000035 |
| 100,000 | 0.025207 | 0.000128 | 0.000131 | -0.000002 |

*Exact answer for this model (parametric VaR, SPY, 99%): 0.025208.*

- **Error shrinks as 1/sqrt(n).** Each 10x increase in paths cuts the spread by about 3.3 to 3.4x (sqrt(10) is 3.16). To halve the error you need 4x the paths.
- **The formula works.** The measured spread is within about 12% of the theoretical standard error at every size (closest at 100,000 paths, where it is within 3%).
- **What that means for VaR.** Relative to the VaR, the standard error is about 5% at 1,000 paths, 1.6% at 10,000 and 0.5% at 100,000.
- **No clear bias.** The average estimate is close to the exact answer at every size. The +0.000335 at 1,000 paths is about 1.6 standard errors of the 50-seed average, not clearly different from zero.
- **The +0.36% offset is noise.** At 100,000 paths the standard error is about 0.5% of the VaR, so a 0.36% gap is under one standard error. It looks identical in every ticker because the fixed seed gives every ticker the same standard-normal draws, so one slightly unlucky tail quantile shows up in all six rows. It is one sampling error, not six independent confirmations.

## 4. Ten-day comparison at 99%

Section 2 compared the methods over one day. Risk is often quoted over longer horizons, so I repeated the comparison at 10 days. The historical figure uses rolling 10-day sums of daily log returns; the Monte Carlo figure simulates 10 days of normal returns per path and sums them (100,000 paths, seed 23).

| Ticker | Historical 10-day VaR | Monte Carlo 10-day VaR | Monte Carlo as % of historical | 1-day parametric as % of historical |
| --- | --- | --- | --- | --- |
| AAPL | 0.1375 | 0.1238 | 90% | 83% |
| AGG | 0.0244 | 0.0235 | 96% | 87% |
| JPM | 0.1381 | 0.1175 | 85% | 85% |
| SPY | 0.0953 | 0.0761 | 80% | 78% |
| VEA | 0.0959 | 0.0774 | 81% | 85% |
| XOM | 0.1436 | 0.1235 | 86% | 83% |

*10-day log-return VaR at 99%, as positive losses. The last column repeats the 1-day result from section 2 for comparison.*

- **The normal model still understates historical VaR at 10 days**, by about 4% (AGG) to 20% (SPY).
- **The gap does not grow with the horizon.** It narrows for AAPL, AGG and XOM, is about flat for JPM and SPY, and widens slightly for VEA. That fits the idea that sums of more days look more normal, but the pattern is not uniform and I have not tested the idea.
- **Square-root-of-time roughly holds.** Historical 10-day VaR is 2.8 to 3.2 times the 1-day figure (square root of 10 is 3.16). Monte Carlo is 3.0 to 3.1 times, a bit under 3.16 because the positive average daily return grows linearly with time and pulls the loss down.
- **Caveat: the historical windows overlap.** There are about 2,950 rolling 10-day windows, but only about 295 non-overlapping ones. The 99% historical figure rests on a few clustered episodes, so it is noisier than the number of windows suggests.
