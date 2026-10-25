import numpy as np
import polars as pl
from scipy.stats import norm

TRADING_DAYS = 252

def add_rolling_vol(returns: pl.DataFrame, window: int = 60) -> pl.DataFrame:
    """Rolling annualized volatility of log returns, per ticker (sample std times sqrt(TRADING_DAYS))."""   
    return returns.with_columns(
        rolling_vol = pl.col('log_return').rolling_std(window_size=window).over('ticker') * TRADING_DAYS**.5)

def historical_var(returns: pl.Series, confidence: float = 0.95) -> float:
    """Historical VaR as a positive loss. Drops nulls before computing."""
    if not 0 < confidence < 1:
     raise ValueError("confidence must be between 0 and 1")
    clean = returns.drop_nulls()
    if clean.is_empty():
        raise ValueError("returns is empty after dropping nulls")
    returns_var = clean.quantile(1-confidence, interpolation='lower')
    return -returns_var


def historical_es(returns: pl.Series, confidence: float = 0.95) -> float:
    """Use result of historical var to determine ES cutoff """
    cutoff = -historical_var(returns, confidence)
    tail = returns.filter(returns <= cutoff)
    return -tail.mean()

def add_rolling_sharpe(df: pl.DataFrame, window: int = 60) -> pl.DataFrame:
    """sharpe ratio over 60 session window (initial RFR of zero)"""
    mean = pl.col("log_return").rolling_mean(window_size=window).over("ticker")
    std = pl.col("log_return").rolling_std(window_size=window).over("ticker")
    return df.with_columns((mean / std * TRADING_DAYS**0.5).alias("rolling_sharpe"))


def risk_by_ticker(
    returns: pl.DataFrame,
    confidence: float = 0.95,
    column: str = "log_return",
) -> pl.DataFrame:
    """One row per ticker with historicalVaR and ES (losses are reported as positive numbers)."""
    rows = []
    for (ticker,), group in returns.group_by("ticker", maintain_order=True):
        series = group[column]
        rows.append(
            {
                "ticker": ticker,
                "var": historical_var(series, confidence),
                "es": historical_es(series, confidence),
            }
        )
    return pl.DataFrame(rows)


def parametric_var(returns: pl.Series, confidence: float = 0.95) -> float:
    """Parametric (normal) VaR as a positive loss, from the sample MEAN and STDEV. Drops nulls before computing."""
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    clean = returns.drop_nulls()
    if len(clean) < 2:
        raise ValueError("need at least 2 observations to estimate a standard deviation")
    z = norm.ppf(1 - confidence)
    return float(-(clean.mean() + z * clean.std()))

def monte_carlo_var(
    returns: pl.Series,
    confidence: float = 0.95,
    n_paths: int = 100_000,
    seed: int | None = None,
) -> float:
    """Monte Carlo VaR as a positive loss: simulate one-day log returns from a normal fitted to the sample, take the lower-tail quantile."""
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    clean = returns.drop_nulls()
    if len(clean) < 2:
        raise ValueError("need at least 2 observations to estimate a standard deviation")
    rng = np.random.default_rng(seed)
    simulated = rng.normal(loc=clean.mean(), scale=clean.std(), size=n_paths)
    return float(-np.quantile(simulated, 1 - confidence, method="lower"))


def monte_carlo_var_multiday(
    returns: pl.Series,
    confidence: float = 0.95,
    horizon: int = 10,
    n_paths: int = 100_000,
    seed: int | None = None,
) -> float:
    
    """Monte Carlo VaR over `horizon` days: sum each path, take the left-tail quantile."""
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if horizon < 1:
        raise ValueError("horizon must be at least 1 day")
    clean = returns.drop_nulls()
    if len(clean) < 2:
        raise ValueError("need at least 2 observations")
    mean = clean.mean()
    std = clean.std()
    rng = np.random.default_rng(seed)
    daily = rng.normal(loc=mean, scale=std, size=(n_paths, horizon))
    total = daily.sum(axis=1)
    return -float(np.quantile(total, 1 - confidence, method="lower"))


def historical_var_multiday(
    returns: pl.Series, confidence: float = 0.95, horizon: int = 10
) -> float:
    """Historical VaR over `horizon` days, from overlapping rolling sums of daily log returns."""
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    sums = returns.drop_nulls().rolling_sum(window_size=horizon).drop_nulls()
    return historical_var(sums, confidence)
