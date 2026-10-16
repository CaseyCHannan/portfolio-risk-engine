import polars as pl


def add_rolling_vol(returns: pl.DataFrame, window: int = 60) -> pl.DataFrame:
    return returns.with_columns(
        rolling_vol = pl.col('log_return').rolling_std(window_size=window).over('ticker') * 252**.5)

def historical_var(returns: pl.Series, confidence: float = 0.95) -> float:
    """Historical VaR as a positive loss. Drops nulls before computing."""
    returns_var = returns.drop_nulls().quantile(1-confidence)
    return (-1)*returns_var


def historical_es(returns: pl.Series, confidence: float = 0.95) -> float:
    cutoff = -historical_var(returns, confidence)
    tail = returns.drop_nulls().filter(returns.drop_nulls() <= cutoff)
    return -tail.mean()