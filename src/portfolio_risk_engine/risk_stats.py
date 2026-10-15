import polars as pl


def add_rolling_vol(returns: pl.DataFrame, window: int = 60) -> pl.DataFrame:
    return returns.with_columns(
        rolling_vol = pl.col('log_return').rolling_std(window_size=window).over('ticker') * 252**.5)