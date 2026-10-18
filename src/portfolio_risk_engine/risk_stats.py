import polars as pl


def add_rolling_vol(returns: pl.DataFrame, window: int = 60) -> pl.DataFrame:
    return returns.with_columns(
        rolling_vol = pl.col('log_return').rolling_std(window_size=window).over('ticker') * 252**.5)

def historical_var(returns: pl.Series, confidence: float = 0.95) -> float:
    """Historical VaR as a positive loss. Drops nulls before computing."""
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
    return df.with_columns((mean / std * 252**0.5).alias("rolling_sharpe"))