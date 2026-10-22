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

def parametric_var_by_ticker(returns, confidence=0.95, column="log_return"):
    rows = []
    for (ticker,), group in returns.group_by("ticker", maintain_order=True):
        series = group[column].drop_nulls()  # the `column` of group, nulls dropped
        rows.append({"ticker": ticker, "var": parametric_var(series, confidence)})
    return pl.DataFrame(rows)

result = parametric_var_by_ticker(returns, confidence=0.99)
print(result)

# def risk_by_ticker(
#     returns: pl.DataFrame,
#     confidence: float = 0.95,
#     column: str = "log_return",
# ) -> pl.DataFrame:
#     """One row per ticker with historicalVaR and ES (losses are reported as positive numbers)."""
#     rows = []
#     for (ticker,), group in returns.group_by("ticker", maintain_order=True):
#         series = group[column]
#         rows.append(
#             {
#                 "ticker": ticker,
#                 "var": historical_var(series, confidence),
#                 "es": historical_es(series, confidence),
#             }
#         )
#     return pl.DataFrame(rows)

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