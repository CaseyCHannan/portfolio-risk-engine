from scipy import stats

from portfolio_risk_engine.data import add_returns, read_prices
"""check whether daily log returns look normal (skew, excess kurtosis, Jarque-Bera)."""

returns = add_returns(read_prices())
for (ticker,), group in returns.group_by("ticker", maintain_order=True):
    x = group["log_return"].drop_nulls().to_numpy()
    jb = stats.jarque_bera(x)
    print(
        f"{ticker}: n={len(x)} skew={stats.skew(x):.2f} "
        f"excess_kurtosis={stats.kurtosis(x):.2f} JB_p={jb.pvalue:.3g}"
    )