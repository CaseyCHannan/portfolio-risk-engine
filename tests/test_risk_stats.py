from datetime import date
import numpy as np
import polars as pl
import math
import pytest

from portfolio_risk_engine.data import returns_wide

from portfolio_risk_engine.risk_stats import (
    add_rolling_sharpe,
    add_rolling_vol,
    covariance_matrix,
    historical_es,
    historical_var,
    historical_var_multiday,
    monte_carlo_portfolio_var,
    monte_carlo_var,
    monte_carlo_var_multiday,
    parametric_portfolio_var,
    parametric_var,
    portfolio_returns, 
    portfolio_risk,
    portfolio_variance,
    risk_by_ticker,
    student_t_risk,
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

def test_monte_carlo_var_multiday_scales_with_sqrt_time():
    returns = pl.Series([-0.01, 0.03])
    result = monte_carlo_var_multiday(returns, confidence=0.95, horizon=10, seed=23)
    # daily mean 0.01, sample std sqrt(0.0008) = 0.028284
    # 10 days: mean 0.1, std sqrt(10 * 0.0008) = 0.089443 (sqrt-time, not 10x)
    # VaR = -(0.1 - 1.644854 * 0.089443) = 0.047122
    # scaling std by 10 instead of sqrt(10) would give about 0.365
    assert result == pytest.approx(0.047122, abs=3e-3)


def test_historical_var_multiday_uses_rolling_sums():
    # horizon 2 on 21 returns gives 20 two-day sums: -0.10, -0.04, then 18 of 0.02.
    # 95% VaR is the worst of the 20 (position 0.05 * 19 = 0.95, 'lower'), so 0.10.
    # Single-day VaR on the same series is 0.05, so only real summing passes.
    returns = pl.Series([-0.05, -0.05] + [0.01] * 19)
    result = historical_var_multiday(returns, confidence=0.95, horizon=2)
    assert result == pytest.approx(0.10)


def test_returns_wide_one_column_per_ticker_and_drops_null_dates():
    long = pl.DataFrame(
        {
            "date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)] * 2,
            "ticker": ["B"] * 3 + ["A"] * 3,
            "log_return": [None, 0.10, 0.20, None, 0.01, 0.02],
        }
    )
    wide = returns_wide(long)
    assert wide.columns == ["date", "A", "B"]
    assert wide["date"].to_list() == [date(2024, 1, 2), date(2024, 1, 3)]
    assert wide["A"].to_list() == [0.01, 0.02]
    assert wide["B"].to_list() == [0.10, 0.20]


def test_covariance_matrix_uses_n_minus_1():
    # A deviates -0.01, 0, 0.01: variance 0.0002 / 2 = 0.0001. B is 2x A,
    # so its variance is 0.0004 and the covariance is 0.0002.
    wide = pl.DataFrame(
        {
            "date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)],
            "A": [0.01, 0.02, 0.03],
            "B": [0.02, 0.04, 0.06],
        }
    )
    result = covariance_matrix(wide)
    assert result.flatten().tolist() == pytest.approx([0.0001, 0.0002, 0.0002, 0.0004])


def test_portfolio_variance_two_assets():
    # sigma 0.10 and 0.20, correlation 0.5, so the covariance is 0.01.
    # 0.6^2 * 0.01 + 0.4^2 * 0.04 + 2 * 0.6 * 0.4 * 0.01 = 0.0036 + 0.0064 + 0.0048
    cov = [[0.01, 0.01], [0.01, 0.04]]
    assert portfolio_variance([0.6, 0.4], cov) == pytest.approx(0.0148)


def test_portfolio_variance_wrong_size_raises():
    with pytest.raises(ValueError):
        portfolio_variance([0.5, 0.3, 0.2], [[0.01, 0.01], [0.01, 0.04]])


def test_parametric_portfolio_var_95_two_assets():
    # A = 0, 0.01, 0.02: mean 0.01, variance 0.0001. B = 2x A: mean 0.02,
    # variance 0.0004, covariance 0.0002. Weights 0.5 / 0.5:
    # mean 0.015; variance 0.25*0.0001 + 0.25*0.0004 + 2*0.25*0.0002 = 0.000225,
    # std 0.015. VaR = -(0.015 - 1.644854 * 0.015) = 0.009673.
    wide = pl.DataFrame(
        {
            "date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)],
            "A": [0.00, 0.01, 0.02],
            "B": [0.00, 0.02, 0.04],
        }
    )
    result = parametric_portfolio_var([0.5, 0.5], wide, confidence=0.95)
    assert result == pytest.approx(0.009673, abs=1e-6)


def test_parametric_portfolio_var_weights_must_sum_to_one():
    wide = pl.DataFrame(
        {
            "date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)],
            "A": [0.00, 0.01, 0.02],
            "B": [0.00, 0.02, 0.04],
        }
    )
    with pytest.raises(ValueError):
        parametric_portfolio_var([0.5, 0.6], wide)


def test_monte_carlo_portfolio_var_close_to_parametric():
    # A: mean 0.01, variance 0.0000667. B: mean 0.015, variance 0.0001667.
    # Covariance -0.0000333 (correlation -0.32), so the matrix is not singular.
    # Weights 0.5 / 0.5: mean 0.0125, variance 0.0000417, std 0.006455.
    # 99% VaR = -(0.0125 - 2.326348 * 0.006455) = 0.002516.
    # Standard error at 100,000 paths is about 7.6e-5, so abs=4e-4 is about 5 SE.
    wide = pl.DataFrame(
        {
            "date": [date(2024, 1, d) for d in (1, 2, 3, 4)],
            "A": [0.00, 0.01, 0.02, 0.01],
            "B": [0.02, 0.00, 0.01, 0.03],
        }
    )
    result = monte_carlo_portfolio_var([0.5, 0.5], wide, confidence=0.99, seed=23)
    assert result == pytest.approx(0.002516, abs=4e-4)


def test_portfolio_returns_applies_weights_per_row():
    # 0.75 * A + 0.25 * B: 0.75*0.01 + 0.25*0.03 = 0.015; 0.75*0.02 + 0.25*0.04 = 0.025.
    # Swapped weights would give 0.025 and 0.035, so uneven weights catch that mistake.
    wide = pl.DataFrame(
        {
            "date": [date(2024, 1, 1), date(2024, 1, 2)],
            "A": [0.01, 0.02],
            "B": [0.03, 0.04],
        }
    )
    result = portfolio_returns([0.75, 0.25], wide)
    assert result.to_list() == pytest.approx([0.015, 0.025])


def _toy_wide():
    # Weights 0.75/0.25 give portfolio returns -0.04, -0.02, 0.00, 0.01, 0.03.
    return pl.DataFrame(
        {
            "date": ["d1", "d2", "d3", "d4", "d5"],
            "A": [-0.04, 0.00, 0.04, -0.02, 0.02],
            "B": [-0.04, -0.08, -0.12, 0.10, 0.06],
        }
    )


def test_portfolio_risk_uneven_weights_three_methods():
    result = portfolio_risk([0.75, 0.25], _toy_wide(), confidence=0.6, seed=23)
    rows = {row["method"]: row for row in result.iter_rows(named=True)}

    # Historical: worst 2 of 5 returns are -0.04 and -0.02, so VaR 0.02, ES (0.04 + 0.02) / 2 = 0.03.
    assert rows["historical"]["var"] == pytest.approx(0.02)
    assert rows["historical"]["es"] == pytest.approx(0.03)

    # Parametric: mean -0.004, sample std sqrt(0.00073) = 0.027019, z = -0.253347.
    assert rows["parametric"]["var"] == pytest.approx(0.010845, abs=1e-6)
    assert rows["parametric"]["es"] == pytest.approx(0.030096, abs=1e-6)

    # Monte Carlo: about 5 standard errors for VaR; ES has a larger error, so a looser bound.
    assert rows["monte_carlo"]["var"] == pytest.approx(0.010845, abs=5e-4)
    assert rows["monte_carlo"]["es"] == pytest.approx(0.030096, abs=1e-3)

    for row in rows.values():
        assert row["es"] >= row["var"]


def test_portfolio_risk_monte_carlo_matches_monte_carlo_portfolio_var():
    wide = _toy_wide()
    result = portfolio_risk([0.75, 0.25], wide, confidence=0.6, seed=23)
    mc_var = result.filter(pl.col("method") == "monte_carlo")["var"][0]
    assert mc_var == pytest.approx(
        monte_carlo_portfolio_var([0.75, 0.25], wide, confidence=0.6, seed=23)
    )


def test_student_t_risk_df4_99():
    # df = 4: the 1% t quantile is -3.746947, its density there is 0.0086819 and
    # (df + q**2) / (df - 1) = 6.0132, so the standard ES is 0.0086819 / 0.01 * 6.0132 = 5.2206.
    # The non-zero loc catches a sign slip on the mean term.
    var, es = student_t_risk(0.001, 0.01, 4, 0.99)
    assert var == pytest.approx(0.036469, abs=1e-6)  # -(0.001 + 0.01 * -3.746947)
    assert es == pytest.approx(0.051206, abs=1e-6)  # -0.001 + 0.01 * 5.2206


def test_student_t_risk_large_df_matches_normal():
    # As df grows the t approaches the normal: 99% VaR 2.326, ES 2.665 (the normal figures).
    var, es = student_t_risk(0, 1, 1e7, 0.99)
    assert var == pytest.approx(2.326348, abs=1e-4)
    assert es == pytest.approx(2.665214, abs=1e-4)


def test_student_t_risk_bad_inputs_raise():
    with pytest.raises(ValueError):
        student_t_risk(0, 1, 1, 0.99)  # ES needs df > 1
    with pytest.raises(ValueError):
        student_t_risk(0, 0, 4, 0.99)  # scale must be positive
    with pytest.raises(ValueError):
        student_t_risk(0, 1, 4, 99)  # confidence, not a percentage


def test_portfolio_risk_student_t_recovers_known_t():
    # Returns drawn from a t with 4 degrees of freedom (loc 0.0005, scale 0.01). Asset B has
    # weight 0 and only keeps the covariance matrix invertible for the simulation.
    rng = np.random.default_rng(23)
    n = 5000
    a = 0.0005 + 0.01 * rng.standard_t(4, size=n)
    b = rng.normal(0, 0.01, size=n)
    wide = pl.DataFrame({"date": list(range(n)), "A": a, "B": b})

    result = portfolio_risk([1.0, 0.0], wide, confidence=0.99, n_paths=1000, seed=23)
    rows = {row["method"]: row for row in result.iter_rows(named=True)}

    # The fit lands within a few percent of the true t (sampling error with 5,000 points).
    true_var, true_es = student_t_risk(0.0005, 0.01, 4, 0.99)
    assert rows["student_t"]["var"] == pytest.approx(true_var, rel=0.06)
    assert rows["student_t"]["es"] == pytest.approx(true_es, rel=0.08)

    # The normal row understates the ES by about a quarter on the same data.
    assert rows["parametric"]["es"] < 0.8 * true_es
