import polars as pl
import math
import pytest

from portfolio_risk_engine.risk_stats import (
    add_rolling_vol, 
    historical_var, 
    historical_es, 
    add_rolling_sharpe, 
)


def test_rolling_vol_toy_example():
    df = pl.DataFrame({
        'ticker': ['A', 'A', 'A', 'A'],
        'log_return': [0.01, -0.01, 0.01, -0.01],
    })
    out = add_rolling_vol(df, window=2)
    expected = 0.0002 ** 0.5  * 252**.5
    assert out['rolling_vol'][0] is None
    assert out['rolling_vol'][1] == pytest.approx(expected)
    assert out['rolling_vol'][3] == pytest.approx(expected)

def test_historical_var_toy_example():
    returns = pl.Series([x / 100 for x in range(-10, 10)])  
    result = historical_var(returns, confidence=0.95) 
    assert result == pytest.approx(.10)    

def test_historical_es_toy_example():
    returns = pl.Series([x / 100 for x in range(-10, 10)])  
    result = historical_es(returns, confidence=0.90) 
    assert result == pytest.approx(.095) 
    assert result >= historical_var(returns, confidence=0.90)

def test_historical_var_empty_series_raises():
    with pytest.raises(ValueError):
        historical_var(pl.Series([], dtype=pl.Float64))

def test_historical_es_empty_series_raises():
    with pytest.raises(ValueError):
        historical_es(pl.Series([], dtype=pl.Float64))

def test_rolling_sharpe_toy_example():
    df = pl.DataFrame({"ticker": ["A", "A"], "log_return": [0.01, 0.03]})
    out = add_rolling_sharpe(df, window=2)
    assert out["rolling_sharpe"][0] is None
    assert out["rolling_sharpe"][1] == pytest.approx(math.sqrt(504))


def test_rolling_sharpe_does_not_cross_tickers():
    df = pl.DataFrame({
        "ticker": ["A", "A", "B", "B"],
        "log_return": [0.01, 0.03, 0.10, 0.30],
    })
    out = add_rolling_sharpe(df, window=2)
    assert out["rolling_sharpe"][0] is None
    assert out["rolling_sharpe"][1] == pytest.approx(math.sqrt(504))
    assert out["rolling_sharpe"][2] is None
    assert out["rolling_sharpe"][3] == pytest.approx(math.sqrt(504))