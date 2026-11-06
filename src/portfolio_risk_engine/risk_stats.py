import numpy as np
import polars as pl
from scipy.stats import norm
from scipy.stats import t as student_t

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


def covariance_matrix(wide: pl.DataFrame) -> np.ndarray:
    """Sample covariance matrix (n - 1) of the ticker columns of a wide returns frame.

    Ignores the date column; rows and columns follow the frame's column order.
    """
    return np.cov(wide.drop("date").to_numpy(), rowvar=False)


def portfolio_variance(weights, cov) -> float:
    """Variance of a portfolio return: W' Sigma W."""
    weights = np.asarray(weights, dtype=float)
    cov = np.asarray(cov, dtype=float)
    if cov.shape != (len(weights), len(weights)):
        raise ValueError("cov must be square and match the number of weights")
    return float(weights @ cov @ weights)


def parametric_portfolio_var(weights, wide: pl.DataFrame, confidence: float = 0.95) -> float:
    """Normal VaR of a fixed-weight portfolio, reported as a positive loss.

    Weights follow the ticker column order of the wide frame and must sum to 1.
    Treats the weighted sum of log returns as the portfolio return, which is an
    approximation (it is exact for simple returns).
    """
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    weights = np.asarray(weights, dtype=float)
    returns = wide.drop("date").to_numpy()
    if returns.shape[1] != len(weights):
        raise ValueError("need one weight per ticker column")
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("weights must sum to 1")
    if returns.shape[0] < 2:
        raise ValueError("need at least 2 observations")
    mean = float(weights @ returns.mean(axis=0))
    std = portfolio_variance(weights, covariance_matrix(wide)) ** 0.5
    return -(mean + norm.ppf(1 - confidence) * std)


def _simulate_portfolio_returns(weights, wide: pl.DataFrame, n_paths: int, seed) -> np.ndarray:
    """Simulated portfolio returns from a multivariate normal fitted to the frame.

    Correlates independent normal draws with the Cholesky factor of the sample
    covariance matrix. Weights follow the ticker column order and must sum to 1.
    """
    weights = np.asarray(weights, dtype=float)
    returns = wide.drop("date").to_numpy()
    if returns.shape[1] != len(weights):
        raise ValueError("need one weight per ticker column")
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("weights must sum to 1")
    if returns.shape[0] < 2:
        raise ValueError("need at least 2 observations")
    mean = returns.mean(axis=0)
    chol = np.linalg.cholesky(covariance_matrix(wide))
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_paths, len(weights)))
    return (mean + z @ chol.T) @ weights


def monte_carlo_portfolio_var(
    weights,
    wide: pl.DataFrame,
    confidence: float = 0.95,
    n_paths: int = 100_000,
    seed=None,
) -> float:
    """Monte Carlo VaR of a fixed-weight portfolio, as a positive loss.

    Takes the lower quantile of the simulated portfolio returns (the same
    convention as monte_carlo_var).
    """
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    simulated = _simulate_portfolio_returns(weights, wide, n_paths, seed)
    return -float(np.quantile(simulated, 1 - confidence, method="lower"))


def portfolio_returns(weights, wide: pl.DataFrame) -> pl.Series:
    """Daily portfolio return series for fixed weights (the weights apply every day).

    Weights follow the ticker column order of the wide frame and must sum to 1.
    """
    weights = np.asarray(weights, dtype=float)
    returns = wide.drop("date").to_numpy()
    if returns.shape[1] != len(weights):
        raise ValueError("need one weight per ticker column")
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("weights must sum to 1")
    return pl.Series("portfolio", returns @ weights)


def portfolio_risk(
    weights,
    wide: pl.DataFrame,
    confidence: float = 0.95,
    n_paths: int = 100_000,
    seed=None,
) -> pl.DataFrame:
    """VaR and ES of a fixed-weight portfolio by four methods, as positive losses.

    Returns one row per method (historical, parametric, monte_carlo, student_t) with
    columns method, var and es. Weights follow the ticker column order of the wide frame.
    The student_t row fits a Student-t (location, scale, degrees of freedom) to the
    portfolio return series by maximum likelihood.
    """
    series = portfolio_returns(weights, wide)
    historical = (historical_var(series, confidence), historical_es(series, confidence))

    mean = float(series.mean())
    std = float(series.std())
    z = norm.ppf(1 - confidence)
    parametric = (
        parametric_portfolio_var(weights, wide, confidence),
        -mean + std * norm.pdf(z) / (1 - confidence),
    )

    simulated = _simulate_portfolio_returns(weights, wide, n_paths, seed)
    cutoff = np.quantile(simulated, 1 - confidence, method="lower")
    monte_carlo = (-float(cutoff), -float(simulated[simulated <= cutoff].mean()))

    dof, loc, scale = student_t.fit(series.to_numpy())
    student = student_t_risk(loc, scale, dof, confidence)

    return pl.DataFrame(
        {
            "method": ["historical", "parametric", "monte_carlo", "student_t"],
            "var": [historical[0], parametric[0], monte_carlo[0], student[0]],
            "es": [historical[1], parametric[1], monte_carlo[1], student[1]],
        }
    )


def student_t_risk(
    loc: float, scale: float, df: float, confidence: float = 0.95
) -> tuple[float, float]:
    """Student-t VaR and ES as positive losses, from location, scale and degrees of freedom.

    The return is loc + (scale * T), where T is a standard t variable. ES needs df > 1.
    """
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if scale <= 0:
        raise ValueError("scale must be positive")
    if df <= 1:
        raise ValueError("df must be greater than 1")
    tail = 1 - confidence
    q = student_t.ppf(tail, df)
    var = -(loc + scale * q)
    es = -loc + scale * student_t.pdf(q, df) / tail * (df + q**2) / (df - 1)
    return float(var), float(es)
