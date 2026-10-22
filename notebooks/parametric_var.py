import polars as pl
from portfolio_risk_engine.data import read_prices, add_returns
from scipy.stats import norm

# Z_99 =  -2.326 # left tail


def parametric_var(returns, confidence=0.95):
    mean = returns.mean()
    std = returns.std()
    z = norm.ppf(1 - confidence)
    return -(mean + z * std)


prices = read_prices()
returns = add_returns(prices)

spy_returns = (
    returns
    .filter(pl.col("ticker") == "SPY")
    .select("log_return")
    .drop_nulls()
)
log_returns = spy_returns["log_return"]

spy_var = parametric_var(log_returns, confidence=0.99)
print(spy_var)

# spy_returns = (returns.filter(pl.col("ticker") == "SPY").select("log_return").drop_nulls())
              
# log_returns = spy_returns["log_return"]

# mean = log_returns.mean()
# std = log_returns.std()

# print(mean)
# print(std)

# VaR = -(mean + z * std)

# print(VaR)
# print(z)
              
# def read_prices(db_path: str = DB_PATH) -> pl.DataFrame:
#     return duckdb.connect(db_path, read_only=True).sql(PRICES_SQL).pl()


# def add_returns(prices: pl.DataFrame) -> pl.DataFrame: