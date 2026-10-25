"""How many Monte Carlo paths are enough?
 Runs monte_carlo_var on SPY at several path counts and seeds,
and compares the spread of the estimates to the standard-error formula.
"""
import numpy as np
import polars as pl
from scipy.stats import norm

from portfolio_risk_engine.data import (
    add_returns,
    read_prices,
)
from portfolio_risk_engine.risk_stats import (
    monte_carlo_var,
    parametric_var,
)

CONFIDENCE = 0.99
SEEDS = range(50)

returns = add_returns(read_prices())
spy = returns.filter(pl.col("ticker") == "SPY")["log_return"].drop_nulls()

exact = parametric_var(spy, confidence=CONFIDENCE)
sigma = spy.std()
tail_prob = 1 - CONFIDENCE
density = norm.pdf(norm.ppf(tail_prob)) / sigma  # normal density at the VaR cutoff

print(f"parametric (exact answer for this model): {exact:.6f}")
print("n_paths | mean estimate | std across seeds | theory SE | mean - exact")
for n_paths in [1_000, 10_000, 100_000]:
    estimates = np.array(
        [monte_carlo_var(spy, CONFIDENCE, n_paths, seed) for seed in SEEDS]
    )
    theory_se = np.sqrt(tail_prob * (1 - tail_prob) / n_paths) / density
    print(
        f"{n_paths:>7} | {estimates.mean():.6f} | {estimates.std(ddof=1):.6f} "
        f"| {theory_se:.6f} | {estimates.mean() - exact:+.6f}"
    )