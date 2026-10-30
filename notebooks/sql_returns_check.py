"""Cross-check the Polars returns against the SQL LAG() returns.

Joins the two on ticker and date, then reports row counts, null-pattern
mismatches and the largest absolute difference for each return column.
Run: uv run python notebooks/sql_returns_check.py
"""

import polars as pl

from portfolio_risk_engine.data import (
    add_returns,
    read_prices,
    read_returns_sql,
)

polars_df = add_returns(read_prices())
sql_df = read_returns_sql()
print("rows: polars", polars_df.height, "| sql", sql_df.height)

joined = polars_df.join(sql_df, on=["ticker", "date"], how="full", coalesce=True)
print("rows after full join:", joined.height)

for c in ("simple_return", "log_return"):
    sql_c = f"{c}_sql"
    null_mismatch = joined.filter(pl.col(c).is_null() != pl.col(sql_c).is_null()).height
    both = joined.filter(pl.col(c).is_not_null() & pl.col(sql_c).is_not_null())
    max_diff = both.select((pl.col(c) - pl.col(sql_c)).abs().max()).item()
    print(f"{c}: null mismatches = {null_mismatch}, rows compared = {both.height}, max abs diff = {max_diff}")

