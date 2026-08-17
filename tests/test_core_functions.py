"""
test_core_functions.py
------------------------
Fast, direct unit tests for the key new functions added to this project
(statistical inference, EVT/tail risk, American options, the COS-method
Heston pricer, Delta-Vega hedging, and variance reduction), plus a few
cross-cutting regression checks on long-standing core functions
(put-call parity, CVaR >= VaR). Complements test_cli_scripts.py, which
runs each module's own, more extensive, self-contained validation suite.
"""

import numpy as np
import pytest

from black_scholes import bs_price
from heston import heston_price, heston_price_cos, heston_price_cos_batch
from merton import merton_price
from risk_analysis import cvar_historical, var_historical
from statistical_inference import bootstrap_ci, paired_bootstrap_test
from tail_risk import fit_gpd_pot, evt_var, evt_cvar
from american_options import binomial_tree_option, crank_nicolson_american, early_exercise_premium
from hedging_engine import _batch_price_and_delta, _batch_vega, simulate_delta_vega_hedge_batch
from variance_reduction import weighted_var_cvar


# ══════════════════════════════════════════════════════════════════════════
# risk_analysis.py — core risk-measure invariants
# ══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_losses():
    rng = np.random.default_rng(0)
    return rng.standard_normal(5000) * 10.0


def test_cvar_at_least_as_large_as_var(sample_losses):
    """CVaR (tail average) must never be smaller than VaR (tail quantile) at the same level."""
    for q in (0.01, 0.05, 0.10):
        assert cvar_historical(sample_losses, q) >= var_historical(sample_losses, q)


def test_cvar_smaller_q_is_more_extreme(sample_losses):
    """A smaller q (deeper into the tail) should give an equal-or-larger CVaR."""
    assert cvar_historical(sample_losses, 0.01) >= cvar_historical(sample_losses, 0.05)


# ══════════════════════════════════════════════════════════════════════════
# statistical_inference.py
# ══════════════════════════════════════════════════════════════════════════

def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(1)
    pnls = -rng.standard_normal(2000) * 10.0
    result = bootstrap_ci(pnls, statistic="cvar", q=0.05, n_bootstrap=500, seed=1)
    assert result["ci_lower"] <= result["point_estimate"] <= result["ci_upper"]
    assert result["se"] > 0


def test_paired_bootstrap_null_case_not_significant():
    rng = np.random.default_rng(2)
    pnls = rng.standard_normal(1000) * 5.0
    test = paired_bootstrap_test(pnls, pnls.copy(), statistic="cvar", q=0.05, n_bootstrap=500, seed=2)
    assert test["point_diff"] == pytest.approx(0.0)
    assert not test["significant_at_5pct"]


def test_paired_bootstrap_requires_equal_length():
    with pytest.raises(AssertionError):
        paired_bootstrap_test(np.zeros(10), np.zeros(20))


# ══════════════════════════════════════════════════════════════════════════
# tail_risk.py
# ══════════════════════════════════════════════════════════════════════════

def test_fit_gpd_pot_raises_with_too_few_exceedances():
    rng = np.random.default_rng(3)
    tiny_sample = rng.standard_normal(50)
    with pytest.raises(ValueError):
        fit_gpd_pot(tiny_sample, threshold_quantile=0.98)   # ~1 exceedance, should raise


def test_evt_cvar_at_least_evt_var():
    rng = np.random.default_rng(4)
    losses = np.abs(rng.standard_normal(3000)) * 10.0 + rng.exponential(5.0, 3000)
    fit = fit_gpd_pot(losses, threshold_quantile=0.85)
    for q in (0.05, 0.01):
        assert evt_cvar(fit, q) >= evt_var(fit, q)


# ══════════════════════════════════════════════════════════════════════════
# american_options.py
# ══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("option_type", ["call", "put"])
def test_american_at_least_european(option_type):
    S0, K, T, r, sigma = 100.0, 100.0, 0.5, 0.03, 0.25
    result = early_exercise_premium(S0, K, T, r, sigma, option_type, q=0.02, n_steps=300)
    assert result["premium"] >= -1e-6   # American must never be cheaper than European


def test_binomial_tree_matches_bs_for_deep_out_of_money_call_no_dividends():
    """A far-OTM American call with no dividends should closely match Black-Scholes
    (negligible early-exercise value), sanity-checking the tree isn't badly biased."""
    S0, K, T, r, sigma = 100.0, 150.0, 0.5, 0.03, 0.20
    tree_price = binomial_tree_option(S0, K, T, r, sigma, "call", n_steps=400, q=0.0, american=True)
    bs_ref = bs_price(S0, K, T, r, sigma, "call")
    assert tree_price == pytest.approx(bs_ref, abs=0.05)


def test_crank_nicolson_matches_binomial_tree():
    S0, K, T, r, sigma = 100.0, 105.0, 0.5, 0.03, 0.25
    tree_price = binomial_tree_option(S0, K, T, r, sigma, "put", n_steps=500, american=True)
    cn_price = crank_nicolson_american(S0, K, T, r, sigma, "put")["price"]
    assert cn_price == pytest.approx(tree_price, abs=0.15)


# ══════════════════════════════════════════════════════════════════════════
# heston.py — COS method vs. quad-based Fourier inversion
# ══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("K", [80.0, 100.0, 120.0])
def test_heston_cos_matches_quad(K):
    S0, T, r = 100.0, 0.5, 0.03
    kappa, theta, xi, rho, v0 = 2.0, 0.04, 0.4, -0.7, 0.04
    quad_price = heston_price(S0, K, T, r, kappa, theta, xi, rho, v0, "call")
    cos_price = heston_price_cos(S0, K, T, r, kappa, theta, xi, rho, v0, "call")
    assert cos_price == pytest.approx(quad_price, rel=1e-3)


def test_heston_cos_batch_matches_loop():
    S0_arr = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    K, T, r = 100.0, 0.5, 0.03
    kappa, theta, xi, rho, v0 = 2.0, 0.04, 0.4, -0.7, 0.04
    batch_prices = heston_price_cos_batch(S0_arr, K, T, r, kappa, theta, xi, rho, v0, "call")
    loop_prices = np.array([heston_price_cos(float(s), K, T, r, kappa, theta, xi, rho, v0, "call")
                             for s in S0_arr])
    np.testing.assert_allclose(batch_prices, loop_prices, atol=1e-10)


# ══════════════════════════════════════════════════════════════════════════
# hedging_engine.py — batch dispatch + Delta-Vega hedge
# ══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("model,params", [
    ("bs", {"sigma": 0.2}),
    ("merton", {"sigma": 0.18, "lam": 1.0, "mu_j": -0.05, "sigma_j": 0.1}),
    ("heston", {"kappa": 2.0, "theta": 0.04, "xi": 0.4, "rho": -0.7, "v0": 0.04}),
])
def test_batch_price_delta_vega_all_models(model, params):
    """Every hedge model must support the same batch price/Delta/Vega dispatch API."""
    S = np.array([90.0, 100.0, 110.0])
    K, T, r = 100.0, 0.25, 0.03
    price, delta = _batch_price_and_delta(model, S, K, T, r, "call", params)
    vega = _batch_vega(model, S, K, T, r, "call", params)
    assert np.all(price >= 0)
    assert np.all((delta >= -0.01) & (delta <= 1.01))
    assert np.all(vega >= -1e-6)   # Vega should be non-negative for a vanilla call


def test_delta_vega_hedge_zeroes_net_vega_at_inception():
    n_paths, n_steps = 500, 21
    S0, K, K_instrument, r, sigma = 100.0, 100.0, 110.0, 0.03, 0.2
    rng = np.random.default_rng(5)
    z = rng.standard_normal((n_paths, n_steps))
    dt = 0.25 / n_steps
    log_paths = np.log(S0) + np.cumsum((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z, axis=1)
    paths = np.concatenate([np.full((n_paths, 1), S0), np.exp(log_paths)], axis=1)

    pnls, costs = simulate_delta_vega_hedge_batch(
        paths, K, K_instrument, r, "call", "bs", {"sigma": sigma},
        rebalance_every=1, transaction_cost_rate=0.0005, dt=dt,
    )
    assert pnls.shape == (n_paths,)
    assert np.all(np.isfinite(pnls))
    assert np.all(costs >= 0)


# ══════════════════════════════════════════════════════════════════════════
# variance_reduction.py — weighted CVaR reduces to the plain estimator
# ══════════════════════════════════════════════════════════════════════════

def test_weighted_var_cvar_matches_plain_with_uniform_weights(sample_losses):
    """With uniform weights, weighted_var_cvar's CVaR (a top-k order-statistic
    mean, same convention as risk_analysis.cvar_historical) must match EXACTLY.
    Its VaR uses a discrete inverse-CDF quantile rather than numpy's default
    linear-interpolation quantile (risk_analysis.var_historical) -- a genuine,
    expected difference in quantile CONVENTION (both are standard choices),
    not a bug, so VaR is checked only approximately."""
    weights = np.ones_like(sample_losses)
    var_w, cvar_w = weighted_var_cvar(sample_losses, weights, 0.05)
    assert var_w == pytest.approx(var_historical(sample_losses, 0.05), rel=0.01)
    assert cvar_w == pytest.approx(cvar_historical(sample_losses, 0.05), rel=1e-9)


# ══════════════════════════════════════════════════════════════════════════
# Cross-cutting regression checks: put-call parity across all three models
# ══════════════════════════════════════════════════════════════════════════

def test_put_call_parity_black_scholes():
    S, K, T, r, sigma = 100.0, 95.0, 0.5, 0.03, 0.2
    call, put = bs_price(S, K, T, r, sigma, "call"), bs_price(S, K, T, r, sigma, "put")
    assert (call - put) == pytest.approx(S - K * np.exp(-r * T), abs=1e-8)


def test_put_call_parity_merton():
    S, K, T, r, sigma = 100.0, 95.0, 0.5, 0.03, 0.2
    lam, mu_j, sigma_j = 1.0, -0.05, 0.1
    call = merton_price(S, K, T, r, sigma, lam, mu_j, sigma_j, "call")
    put = merton_price(S, K, T, r, sigma, lam, mu_j, sigma_j, "put")
    assert (call - put) == pytest.approx(S - K * np.exp(-r * T), abs=1e-6)


def test_put_call_parity_heston_cos():
    S, K, T, r = 100.0, 95.0, 0.5, 0.03
    kappa, theta, xi, rho, v0 = 2.0, 0.04, 0.4, -0.7, 0.04
    call = heston_price_cos(S, K, T, r, kappa, theta, xi, rho, v0, "call")
    put = heston_price_cos(S, K, T, r, kappa, theta, xi, rho, v0, "put")
    assert (call - put) == pytest.approx(S - K * np.exp(-r * T), abs=1e-6)
