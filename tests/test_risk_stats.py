import polars as pl
import math
import pytest

from portfolio_risk_engine.risk_stats import (
   add_rolling_sharpe,
    add_rolling_vol,
    historical_es,
    historical_var,
    monte_carlo_var,
    parametric_var,
    risk_by_ticker,
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

def test_historical_var_es_99():
    returns = pl.Series([i / 100 for i in range(-50, 51)])  # 101   points to show that ES is greater than VaR (would be idential with only 100)
    var = historical_var(returns, 0.99)
    es = historical_es(returns, 0.99)
    assert var == pytest.approx(0.49)
    assert es == pytest.approx(0.495)
    assert es >= var

def test_risk_by_ticker_99():
    # B is A scaled by 2, so VaR and ES double: 0.98 and 0.99.
    a = [i / 100 for i in range(-50, 51)]
    b = [2 * x for x in a]
    df = pl.DataFrame({"ticker": ["A"] * 101 + ["B"] * 101, "log_return": a + b})
    out = risk_by_ticker(df, 0.99)
    assert out["ticker"].to_list() == ["A", "B"]
    assert out["var"].to_list() == pytest.approx([0.49, 0.98])
    assert out["es"].to_list() == pytest.approx([0.495, 0.99])

def test_historical_var_bad_confidence_raises():
    with pytest.raises(ValueError):
        historical_var(pl.Series([0.01, -0.02, 0.03]), 95)   # test if mistakenly percentage is passed to function 

def test_rolling_vol_does_not_cross_tickers():
    df = pl.DataFrame({
        'ticker': ['A', 'A', 'B', 'B'],
        'log_return': [0.01, 0.03, 0.10, 0.30],
    })
    out = add_rolling_vol(df, window=2)
    # A: mean 0.02, deviations ±0.01, sum of squares 0.0002, divided by n-1 = 1
    expected_a = 0.0002 ** 0.5 * 252 ** 0.5
    # B is A times 10, so its std is 10x: deviations ±0.10, sum of squares 0.02
    expected_b = 0.02 ** 0.5 * 252 ** 0.5
    assert out['rolling_vol'][0] is None
    assert out['rolling_vol'][1] == pytest.approx(expected_a)
    assert out['rolling_vol'][2] is None
    assert out['rolling_vol'][3] == pytest.approx(expected_b)

def test_parametric_var_95():
    # returns 0.00 and 0.02: mean = 0.01, sample std = sqrt(0.0002) = 0.014142
    # z at 5% = -1.644854, so cutoff return = 0.01 - 1.644854 * 0.014142 = -0.013262
    # reported as a positive loss: 0.013262
    returns = pl.Series([0.00, 0.02])
    assert parametric_var(returns, confidence=0.95) == pytest.approx(0.013262, abs=1e-6)

def test_monte_carlo_var_close_to_parametric():
    # same toy series as the parametric test: mean 0.01, sample std 0.014142
    # the simulation draws from that same normal, so it should land near the
    # parametric answer 0.013262; the standard error of a 5% quantile from
    # 100,000 draws is about 0.0001, so a tolerance of 0.0005 is roughly 5 standard errors
    returns = pl.Series([0.00, 0.02])
    result = monte_carlo_var(returns, confidence=0.95, n_paths=100_000, seed=23) #23 is the GOAT
    assert result == pytest.approx(0.013262, abs=5e-4)