"""
dashboard.py
------------
Interactive analytics product for the Options Market-Making & Multi-Model
Delta-Hedging Simulator.

Design philosophy: this is NOT a "pick one configuration and click Run"
tool. Every tab is built to surface results across MULTIPLE parameter
combinations at once (hedge model x rebalancing frequency x moneyness x
asset), rendered with interactive Plotly charts (zoom, pan, hover,
legend-toggle) so a reader can explore the model-risk story themselves
instead of trusting a single static number.

Tabs
----
1.  Overview            — headline finding, at a glance, both assets
2.  Data & Calibration   — real market data + historical-P calibration
3.  Live Market Check    — live SPY options chain vs. historical model
4.  Volatility Smile     — pricing validation across all three models
5.  Headline CVaR Result — the project's central experiment
6.  Statistical Rigor    — bootstrap confidence intervals + hypothesis tests
7.  Tail Risk (EVT)      — Extreme Value Theory vs. historical CVaR
8.  P&L Attribution      — Greeks-based P&L explain
9.  Heston Finding       — stochastic vol is not a jump hedge (now full-scale via COS)
10. Vega Hedging         — Delta-Vega multi-instrument hedge, a fair fight for Heston
11. Efficient Frontier & Bands — rebalancing cost vs. risk tradeoff
12. American Options     — binomial tree + Crank-Nicolson PDE, early-exercise boundary
13. Stress Testing       — deterministic, distribution-free adverse scenarios
14. Model Validation     — Kupiec POF test + Basel traffic light VaR backtesting
15. Interactive Risk Lab — live multi-parameter sweeps, no button required
16. Methodology & Notes  — assumptions, limitations, engineering notes

Run with:
    streamlit run src/dashboard.py
"""

import os
import sys

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(__file__))

from data_loader import load_prices
from calibration import calibrate_heston, calibrate_merton
from black_scholes import bs_price, implied_volatility
from heston import heston_price, heston_simulate_paths
from merton import merton_price, merton_simulate_paths
from hedging_engine import simulate_delta_hedge, simulate_delta_hedge_batch, simulate_delta_vega_hedge_batch
from risk_analysis import cvar_historical, var_historical, build_band_hedging_pnl_and_cost, compare_delta_vs_delta_vega_hedge
from pnl_attribution import compute_pnl_attribution, compute_pnl_attribution_batch
from stress_testing import run_stress_scenarios
from model_validation import run_var_backtest
from market_data_live import fetch_live_chain
from statistical_inference import bootstrap_ci, paired_bootstrap_test
from tail_risk import compare_historical_vs_evt, fit_gpd_pot, mean_excess_plot_data, hill_plot_data
from american_options import binomial_tree_option, crank_nicolson_american, early_exercise_premium

# ══════════════════════════════════════════════════════════════════════════
# Page, theme, and constants
# ══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Options Market-Making & Delta-Hedging Simulator",
    page_icon="\U0001F4C8",
    layout="wide",
    initial_sidebar_state="collapsed",
)
pio.templates.default = "plotly_white"

COLOR = {
    "bs": "#E15759",       # coral red   — the naive / jump-blind hedge
    "heston": "#59A14F",   # green       — stochastic-vol hedge
    "merton": "#4E79A7",   # steel blue  — the jump-aware hedge
    "spy": "#4E79A7",
    "btc": "#F28E2B",
}
HEDGE_COLOR_MAP = {"BS hedge": COLOR["bs"], "Heston hedge": COLOR["heston"], "Merton hedge": COLOR["merton"]}

ROOT = os.path.join(os.path.dirname(__file__), "..")
TABLE_DIR = os.path.join(ROOT, "results", "tables")

R_FREE = 0.04
OPTION_T = 0.25
N_STEPS = 63
TRANSACTION_COST = 0.0005

ASSET_LABELS = {"spy": "SPY (calm equity)", "btc": "BTC-USD (volatile crypto)"}

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
      div[data-testid="stMetric"] {
          background: #F7F9FC; border: 1px solid #E3E8F0; border-radius: 10px;
          padding: 0.7rem 0.9rem 0.4rem 0.9rem;
      }
      div[data-testid="stMetricValue"] {font-size: 1.55rem;}
      .stTabs [data-baseweb="tab-list"] {gap: 4px;}
      .stTabs [data-baseweb="tab"] {
          padding: 8px 16px; border-radius: 8px 8px 0 0; font-weight: 600;
      }
      .banner {
          padding: 1.5rem 1.9rem; border-radius: 14px; margin-bottom: 1.1rem;
          background: linear-gradient(100deg, #0B2545 0%, #144272 55%, #205295 100%);
          color: white;
      }
      .banner h1 {margin:0; font-size: 1.85rem;}
      .banner p {margin: 0.35rem 0 0 0; font-size: 1.0rem; opacity: 0.92; max-width: 950px;}
      .pill {
          display:inline-block; padding: 2px 10px; margin-right: 6px; border-radius: 999px;
          background: rgba(255,255,255,0.14); font-size: 0.78rem; font-weight: 600;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="banner">
      <h1>Options Market-Making &amp; Multi-Model Delta-Hedging Simulator</h1>
      <p>Quantifying <b>hidden tail risk from model misspecification</b> in derivatives hedging.
      A market maker who hedges Black-Scholes Delta in a market that actually jumps carries
      materially more tail risk than their own risk model can see &mdash; measured here with
      Conditional Value-at-Risk, P&amp;L attribution, deterministic stress scenarios, and
      regulatory-style VaR backtesting, on both simulated paths and real market history.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════
# Cached data loading & precomputed-table loading
# ══════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Loading market data and calibrating models...")
def get_asset_data(asset_key: str):
    df = load_prices(asset_key)
    log_returns = df["log_return"].values
    realized_var = df["realized_vol_21d"].values ** 2
    heston_params = calibrate_heston(log_returns, realized_var)
    merton_params = calibrate_merton(log_returns)
    S0 = float(df["close"].iloc[-1])
    return df, S0, heston_params, merton_params


@st.cache_data
def load_table(filename: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(TABLE_DIR, filename))


DATA = {k: get_asset_data(k) for k in ASSET_LABELS}
headline_df = load_table("headline_cvar_comparison.csv")
heston_secondary_df = load_table("heston_secondary_comparison.csv")
frontier_df = {"spy": load_table("efficient_frontier_spy.csv"), "btc": load_table("efficient_frontier_btc.csv")}
realdata_df = load_table("real_data_backtest.csv")
band_hedging_df = load_table("band_hedging_comparison.csv")
stress_testing_df = load_table("stress_testing.csv")
var_backtest_df = load_table("var_backtest.csv")
statistical_significance_df = load_table("statistical_significance.csv")
evt_tail_risk_df = load_table("evt_tail_risk.csv")
vega_hedging_df = load_table("vega_hedging_comparison.csv")
american_options_df = load_table("american_options.csv")
lookahead_bias_df = load_table("real_data_backtest_lookahead_bias_check.csv")
variance_reduction_pricing_df = load_table("variance_reduction_pricing.csv")
variance_reduction_is_df = load_table("variance_reduction_importance_sampling.csv")


# ══════════════════════════════════════════════════════════════════════════
# Cached analytics (all recomputed live, but memoized per parameter set —
# this is what lets the app show MULTI-parameter sweeps interactively
# instead of a single "configure + click Run" simulation)
# ══════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Computing implied-volatility smiles...")
def compute_smile(asset_key: str, n_strikes: int = 17) -> pd.DataFrame:
    df, S0, hp, mp = DATA[asset_key]
    T = OPTION_T
    strikes = np.linspace(0.8 * S0, 1.2 * S0, n_strikes)
    flat_sigma = mp["sigma"]
    rows = []
    for K in strikes:
        heston_p = heston_price(S0, K, T, R_FREE, hp["kappa"], hp["theta"], hp["xi"], hp["rho"], hp["v0"], "call")
        merton_p = merton_price(S0, K, T, R_FREE, mp["sigma"], mp["lam"], mp["mu_j"], mp["sigma_j"], "call")
        rows.append({
            "moneyness": K / S0,
            "Black-Scholes (flat)": flat_sigma * 100,
            "Heston": implied_volatility(heston_p, S0, K, T, R_FREE, "call") * 100,
            "Merton": implied_volatility(merton_p, S0, K, T, R_FREE, "call") * 100,
        })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="Simulating hedging P&L distributions (vectorized batch)...")
def compute_pnl_distributions(asset_key: str, n_paths: int = 5000, seed: int = 2024) -> dict:
    df, S0, hp, mp = DATA[asset_key]
    K = S0
    dt = OPTION_T / N_STEPS
    true_paths = merton_simulate_paths(
        S0, OPTION_T, R_FREE, mp["sigma"], mp["lam"], mp["mu_j"], mp["sigma_j"],
        n_paths=n_paths, n_steps=N_STEPS, seed=seed,
    )
    hedge_specs = {
        "BS hedge": ("bs", {"sigma": mp["sigma"]}),
        "Merton hedge": ("merton", {"sigma": mp["sigma"], "lam": mp["lam"],
                                     "mu_j": mp["mu_j"], "sigma_j": mp["sigma_j"]}),
    }
    out = {}
    for name, (model, params) in hedge_specs.items():
        out[name] = simulate_delta_hedge_batch(
            true_paths, K, R_FREE, "call", model, params,
            rebalance_every=1, transaction_cost_rate=TRANSACTION_COST, dt=dt,
        )
    return out


@st.cache_data(show_spinner="Sweeping rebalancing frequency x hedge model...")
def compute_frequency_sweep(asset_key: str, option_type: str, n_paths: int,
                             tc_bps: float, r_free: float, seed: int) -> pd.DataFrame:
    df, S0, hp, mp = DATA[asset_key]
    K = S0
    dt = OPTION_T / N_STEPS
    true_paths = merton_simulate_paths(
        S0, OPTION_T, r_free, mp["sigma"], mp["lam"], mp["mu_j"], mp["sigma_j"],
        n_paths=n_paths, n_steps=N_STEPS, seed=seed,
    )
    hedge_specs = {
        "BS hedge": ("bs", {"sigma": mp["sigma"]}),
        "Merton hedge": ("merton", {"sigma": mp["sigma"], "lam": mp["lam"],
                                     "mu_j": mp["mu_j"], "sigma_j": mp["sigma_j"]}),
    }
    rows = []
    for freq in [1, 2, 5, 10, 21]:
        for name, (model, params) in hedge_specs.items():
            pnls = simulate_delta_hedge_batch(
                true_paths, K, r_free, option_type, model, params,
                rebalance_every=freq, transaction_cost_rate=tc_bps / 10_000, dt=dt,
            )
            losses = -pnls
            rows.append({
                "rebalance_every": freq, "hedge_model": name,
                "mean_pnl": float(pnls.mean()), "std_pnl": float(pnls.std()),
                "CVaR_5pct": cvar_historical(losses, 0.05), "VaR_5pct": var_historical(losses, 0.05),
            })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="Sweeping strike / moneyness x hedge model...")
def compute_moneyness_sweep(asset_key: str, option_type: str, n_paths: int,
                             tc_bps: float, r_free: float, seed: int) -> pd.DataFrame:
    df, S0, hp, mp = DATA[asset_key]
    dt = OPTION_T / N_STEPS
    true_paths = merton_simulate_paths(
        S0, OPTION_T, r_free, mp["sigma"], mp["lam"], mp["mu_j"], mp["sigma_j"],
        n_paths=n_paths, n_steps=N_STEPS, seed=seed,
    )
    hedge_specs = {
        "BS hedge": ("bs", {"sigma": mp["sigma"]}),
        "Merton hedge": ("merton", {"sigma": mp["sigma"], "lam": mp["lam"],
                                     "mu_j": mp["mu_j"], "sigma_j": mp["sigma_j"]}),
    }
    rows = []
    for moneyness in np.linspace(0.85, 1.15, 7):
        K = S0 * moneyness
        for name, (model, params) in hedge_specs.items():
            pnls = simulate_delta_hedge_batch(
                true_paths, K, r_free, option_type, model, params,
                rebalance_every=1, transaction_cost_rate=tc_bps / 10_000, dt=dt,
            )
            losses = -pnls
            rows.append({
                "moneyness": round(float(moneyness), 3), "hedge_model": name,
                "mean_pnl": float(pnls.mean()), "std_pnl": float(pnls.std()),
                "CVaR_5pct": cvar_historical(losses, 0.05),
            })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="Computing an illustrative hedging path...")
def compute_illustrative_path(asset_key: str, option_type: str, hedge_model: str,
                               rebalance_every: int, tc_bps: float, r_free: float,
                               seed: int, path_index: int = 0, n_paths_pool: int = 50):
    df, S0, hp, mp = DATA[asset_key]
    K = S0
    dt = OPTION_T / N_STEPS
    hedge_params = {
        "bs": {"sigma": mp["sigma"]},
        "merton": {"sigma": mp["sigma"], "lam": mp["lam"], "mu_j": mp["mu_j"], "sigma_j": mp["sigma_j"]},
        "heston": hp,
    }[hedge_model]
    true_paths = merton_simulate_paths(
        S0, OPTION_T, r_free, mp["sigma"], mp["lam"], mp["mu_j"], mp["sigma_j"],
        n_paths=n_paths_pool, n_steps=N_STEPS, seed=seed,
    )
    result = simulate_delta_hedge(
        true_paths[path_index], K, r_free, option_type, hedge_model, hedge_params,
        rebalance_every=rebalance_every, transaction_cost_rate=tc_bps / 10_000, dt=dt,
    )
    return result, K


def _hedge_params_for(model: str, mp: dict, hp: dict) -> dict:
    return {
        "bs": {"sigma": mp["sigma"]},
        "merton": {"sigma": mp["sigma"], "lam": mp["lam"], "mu_j": mp["mu_j"], "sigma_j": mp["sigma_j"]},
        "heston": hp,
    }[model]


@st.cache_data(show_spinner="Computing Greeks-based P&L attribution...")
def compute_attribution_path(asset_key: str, option_type: str, hedge_model: str,
                              seed: int, path_index: int = 0, n_paths_pool: int = 50):
    df, S0, hp, mp = DATA[asset_key]
    K = S0
    dt = OPTION_T / N_STEPS
    hedge_params = _hedge_params_for(hedge_model, mp, hp)
    true_paths = merton_simulate_paths(
        S0, OPTION_T, R_FREE, mp["sigma"], mp["lam"], mp["mu_j"], mp["sigma_j"],
        n_paths=n_paths_pool, n_steps=N_STEPS, seed=seed,
    )
    attribution = compute_pnl_attribution(
        true_paths[path_index], K, R_FREE, option_type, hedge_model, hedge_params, dt)
    return attribution, true_paths[path_index], K


@st.cache_data(show_spinner="Aggregating P&L attribution residuals across many paths...")
def compute_attribution_batch_residuals(asset_key: str, option_type: str, n_paths: int, seed: int):
    df, S0, hp, mp = DATA[asset_key]
    K = S0
    dt = OPTION_T / N_STEPS
    true_paths = merton_simulate_paths(
        S0, OPTION_T, R_FREE, mp["sigma"], mp["lam"], mp["mu_j"], mp["sigma_j"],
        n_paths=n_paths, n_steps=N_STEPS, seed=seed,
    )
    hedge_specs = {
        "BS hedge": ("bs", {"sigma": mp["sigma"]}),
        "Merton hedge": ("merton", {"sigma": mp["sigma"], "lam": mp["lam"],
                                     "mu_j": mp["mu_j"], "sigma_j": mp["sigma_j"]}),
    }
    out = {}
    for name, (model, params) in hedge_specs.items():
        out[name] = compute_pnl_attribution_batch(true_paths, K, R_FREE, option_type, model, params, dt)
    return out


@st.cache_data(show_spinner="Running deterministic stress scenarios...")
def compute_stress_scenarios(asset_key: str, option_type: str) -> pd.DataFrame:
    df, S0, hp, mp = DATA[asset_key]
    K = S0
    dt = OPTION_T / N_STEPS
    hedge_specs = {
        "BS hedge": ("bs", {"sigma": mp["sigma"]}),
        "Merton hedge": ("merton", {"sigma": mp["sigma"], "lam": mp["lam"],
                                     "mu_j": mp["mu_j"], "sigma_j": mp["sigma_j"]}),
    }
    return run_stress_scenarios(S0, K, R_FREE, option_type, N_STEPS, dt, mp["sigma"],
                                 hedge_specs, TRANSACTION_COST)


@st.cache_data(show_spinner="Backtesting the VaR model (Kupiec POF + Basel traffic light)...")
def compute_var_backtest(asset_key: str, n_paths: int, q: float, train_fraction: float):
    df, S0, hp, mp = DATA[asset_key]
    K = S0
    dt = OPTION_T / N_STEPS
    true_paths = merton_simulate_paths(
        S0, OPTION_T, R_FREE, mp["sigma"], mp["lam"], mp["mu_j"], mp["sigma_j"],
        n_paths=n_paths, n_steps=N_STEPS, seed=2024,
    )
    hedge_specs = {
        "BS hedge": ("bs", {"sigma": mp["sigma"]}),
        "Merton hedge": ("merton", {"sigma": mp["sigma"], "lam": mp["lam"],
                                     "mu_j": mp["mu_j"], "sigma_j": mp["sigma_j"]}),
    }
    rows = []
    for name, (model, params) in hedge_specs.items():
        pnls = simulate_delta_hedge_batch(
            true_paths, K, R_FREE, "call", model, params,
            rebalance_every=1, transaction_cost_rate=TRANSACTION_COST, dt=dt,
        )
        result = run_var_backtest(pnls, q=q, train_fraction=train_fraction)
        result["hedge_model"] = name
        rows.append(result)
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="Sweeping band width x hedge model...", ttl=3600)
def compute_band_sweep(asset_key: str, option_type: str, n_paths: int, tc_bps: float, seed: int):
    df, S0, hp, mp = DATA[asset_key]
    K = S0
    dt = OPTION_T / N_STEPS
    true_paths = merton_simulate_paths(
        S0, OPTION_T, R_FREE, mp["sigma"], mp["lam"], mp["mu_j"], mp["sigma_j"],
        n_paths=n_paths, n_steps=N_STEPS, seed=seed,
    )
    rows = []
    for bw in [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]:
        pnls, costs = build_band_hedging_pnl_and_cost(
            true_paths, K, R_FREE, option_type, "bs", {"sigma": mp["sigma"]},
            band_width=bw, transaction_cost_rate=tc_bps / 10_000, dt=dt,
        )
        losses = -pnls
        rows.append({
            "band_width": bw, "CVaR_5pct": cvar_historical(losses, 0.05),
            "mean_transaction_cost": float(costs.mean()), "mean_pnl": float(pnls.mean()),
        })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="Fetching live options chain from Yahoo Finance...", ttl=300)
def get_live_chain(ticker: str, target_dte_days: int):
    return fetch_live_chain(ticker, target_dte_days)


@st.cache_data(show_spinner="Bootstrapping confidence intervals (this resamples thousands of paths)...")
def compute_bootstrap_live(asset_key: str, n_paths: int, n_bootstrap: int, q: float, seed: int = 2024):
    df, S0, hp, mp = DATA[asset_key]
    K = S0
    dt = OPTION_T / N_STEPS
    true_paths = merton_simulate_paths(
        S0, OPTION_T, R_FREE, mp["sigma"], mp["lam"], mp["mu_j"], mp["sigma_j"],
        n_paths=n_paths, n_steps=N_STEPS, seed=seed,
    )
    pnls_bs = simulate_delta_hedge_batch(true_paths, K, R_FREE, "call", "bs", {"sigma": mp["sigma"]},
                                          rebalance_every=1, transaction_cost_rate=TRANSACTION_COST, dt=dt)
    pnls_merton = simulate_delta_hedge_batch(
        true_paths, K, R_FREE, "call", "merton",
        {"sigma": mp["sigma"], "lam": mp["lam"], "mu_j": mp["mu_j"], "sigma_j": mp["sigma_j"]},
        rebalance_every=1, transaction_cost_rate=TRANSACTION_COST, dt=dt,
    )
    ci_bs = bootstrap_ci(pnls_bs, statistic="cvar", q=q, n_bootstrap=n_bootstrap, seed=0)
    ci_merton = bootstrap_ci(pnls_merton, statistic="cvar", q=q, n_bootstrap=n_bootstrap, seed=0)
    test = paired_bootstrap_test(pnls_bs, pnls_merton, statistic="cvar", q=q, n_bootstrap=n_bootstrap, seed=0)
    return ci_bs, ci_merton, test


@st.cache_data(show_spinner="Fitting Extreme Value Theory (GPD) tail models...")
def compute_evt_live(asset_key: str, n_paths: int, threshold_quantile: float, seed: int = 2024):
    df, S0, hp, mp = DATA[asset_key]
    K = S0
    dt = OPTION_T / N_STEPS
    true_paths = merton_simulate_paths(
        S0, OPTION_T, R_FREE, mp["sigma"], mp["lam"], mp["mu_j"], mp["sigma_j"],
        n_paths=n_paths, n_steps=N_STEPS, seed=seed,
    )
    out = {}
    for name, (model, params) in {
        "BS hedge": ("bs", {"sigma": mp["sigma"]}),
        "Merton hedge": ("merton", {"sigma": mp["sigma"], "lam": mp["lam"],
                                     "mu_j": mp["mu_j"], "sigma_j": mp["sigma_j"]}),
    }.items():
        pnls = simulate_delta_hedge_batch(true_paths, K, R_FREE, "call", model, params,
                                           rebalance_every=1, transaction_cost_rate=TRANSACTION_COST, dt=dt)
        losses = -pnls
        comparison = compare_historical_vs_evt(pnls, q_levels=(0.05, 0.02, 0.01, 0.005, 0.001),
                                                threshold_quantile=threshold_quantile)
        mep = mean_excess_plot_data(losses)
        hp_plot = hill_plot_data(losses, k_min=20, k_max=min(int(len(losses) * 0.15), 600))
        out[name] = {"comparison": comparison, "fit": comparison.attrs["gpd_fit"],
                      "mean_excess": mep, "hill_plot": hp_plot, "losses": losses}
    return out


@st.cache_data(show_spinner="Simulating Delta-only vs. Delta-Vega hedging...")
def compute_vega_hedge_live(asset_key: str, n_paths: int, k_instrument_mult: float,
                             true_dgp: str, rebalance_every: int, seed: int = 321):
    df, S0, hp, mp = DATA[asset_key]
    K = S0
    K_instrument = S0 * k_instrument_mult
    dt = OPTION_T / N_STEPS

    if true_dgp == "heston":
        true_paths, _ = heston_simulate_paths(S0, OPTION_T, R_FREE, hp["kappa"], hp["theta"],
                                               hp["xi"], hp["rho"], hp["v0"], n_paths=n_paths,
                                               n_steps=N_STEPS, seed=seed)
    else:
        true_paths = merton_simulate_paths(S0, OPTION_T, R_FREE, mp["sigma"], mp["lam"], mp["mu_j"],
                                            mp["sigma_j"], n_paths=n_paths, n_steps=N_STEPS, seed=seed)

    summary = compare_delta_vs_delta_vega_hedge(
        true_paths, K, K_instrument, R_FREE, "call", "heston", hp,
        rebalance_every=rebalance_every, transaction_cost_rate=TRANSACTION_COST, dt=dt,
    )
    pnls_delta = simulate_delta_hedge_batch(true_paths, K, R_FREE, "call", "heston", hp,
                                             rebalance_every=rebalance_every,
                                             transaction_cost_rate=TRANSACTION_COST, dt=dt)
    pnls_vega, _ = simulate_delta_vega_hedge_batch(true_paths, K, K_instrument, R_FREE, "call", "heston", hp,
                                                    rebalance_every=rebalance_every,
                                                    transaction_cost_rate=TRANSACTION_COST, dt=dt)
    return summary, pnls_delta, pnls_vega


@st.cache_data(show_spinner="Pricing American options (binomial tree + Crank-Nicolson PDE)...")
def compute_american_options_live(asset_key: str, option_type: str, n_steps_tree: int = 500):
    df, S0, hp, mp = DATA[asset_key]
    K = S0
    sigma = mp["sigma"]
    prem = early_exercise_premium(S0, K, OPTION_T, R_FREE, sigma, option_type, q=0.0, n_steps=n_steps_tree)
    cn = crank_nicolson_american(S0, K, OPTION_T, R_FREE, sigma, option_type, q=0.0, track_boundary=True)

    moneyness_grid = np.linspace(0.8, 1.2, 15)
    premium_curve = []
    for m in moneyness_grid:
        K_m = S0 * m
        p = early_exercise_premium(S0, K_m, OPTION_T, R_FREE, sigma, option_type, q=0.0, n_steps=300)
        premium_curve.append(p["premium"])
    premium_df = pd.DataFrame({"moneyness": moneyness_grid, "premium": premium_curve})
    return prem, cn, premium_df, S0, sigma


def fmt(x, decimals=2):
    return f"{x:,.{decimals}f}"


# ══════════════════════════════════════════════════════════════════════════
# Tabs
# ══════════════════════════════════════════════════════════════════════════

tab_overview, tab_data, tab_live, tab_smile, tab_headline, tab_stats, tab_evt, tab_attribution, \
    tab_heston, tab_vega, tab_frontier, tab_american, tab_stress, tab_validation, tab_lab, tab_notes = st.tabs([
        "\U0001F3E0  Overview", "\U0001F4CA  Data & Calibration", "\U0001F4E1  Live Market Check",
        "\U0001F32A\uFE0F  Volatility Smile", "\U0001F3AF  Headline CVaR Result",
        "\U0001F4D0  Statistical Rigor", "\U0001F30B  Tail Risk (EVT)",
        "\U0001F9EE  P&L Attribution", "\U0001F30A  Heston Finding",
        "\u2696\uFE0F  Vega Hedging", "\U0001F4C9  Efficient Frontier & Bands",
        "\U0001F332  American Options", "\u26A0\uFE0F  Stress Testing",
        "\u2705  Model Validation", "\U0001F9EA  Interactive Risk Lab", "\U0001F4C4  Methodology & Notes",
    ])

# ─────────────────────────────────────────────────────────────────────────
# TAB 1 — Overview
# ─────────────────────────────────────────────────────────────────────────
with tab_overview:
    st.markdown("#### The experiment, in one picture")
    steps = [
        ("1. True market", "Calibrated Merton jump-diffusion — a crash-realistic market, fit to each asset's own historical returns"),
        ("2. Hedge model", "What the market maker BELIEVES and hedges with: Black-Scholes, Heston, or Merton"),
        ("3. Discrete hedging sim.", "Delta-hedge at realistic rebalancing frequency, with proportional transaction costs"),
        ("4. CVaR of P&L", "Tail risk of the realised hedging loss — the gap between belief and reality"),
    ]
    for col, (title, desc) in zip(st.columns(4), steps):
        col.info(f"**{title}**\n\n{desc}")


    st.markdown("#### BS vs. Merton hedge — simulated and real-data, both assets at once")
    combo_rows = []
    for _, r in headline_df.iterrows():
        model = "BS hedge" if "BS" in r["hedge_model"] else "Merton hedge"
        combo_rows.append({"asset": r["asset"], "hedge_model": model, "CVaR_5pct": r["CVaR_5pct"],
                            "source": "Simulated (calibrated Merton market, 8,000 paths)"})
    for _, r in realdata_df.iterrows():
        combo_rows.append({"asset": r["asset"], "hedge_model": "BS hedge", "CVaR_5pct": r["BS_CVaR_5pct"],
                            "source": f"Real-data backtest ({int(r['n_windows'])} windows)"})
        combo_rows.append({"asset": r["asset"], "hedge_model": "Merton hedge", "CVaR_5pct": r["Merton_CVaR_5pct"],
                            "source": f"Real-data backtest ({int(r['n_windows'])} windows)"})
    combo_df = pd.DataFrame(combo_rows)

    fig = px.bar(
        combo_df, x="source", y="CVaR_5pct", color="hedge_model", barmode="group", facet_col="asset",
        color_discrete_map=HEDGE_COLOR_MAP, text_auto=".2s",
        labels={"CVaR_5pct": "CVaR of hedging loss (95%)", "source": ""}, height=460,
    )
    fig.update_xaxes(tickangle=0, title_text="")
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_layout(legend_title_text="", margin=dict(t=50))
    st.plotly_chart(fig, width='stretch')

    st.success(
        "Across BOTH a calm equity (SPY) and a volatile crypto asset (BTC), and BOTH pure simulation "
        "and a replay of actual market history, the pattern repeats: a market maker hedging with "
        "Black-Scholes Delta carries meaningfully more tail risk than a jump-aware Merton hedge — "
        "risk that the BS-based risk model itself cannot see."
    )

    with st.expander("Methodological check: is the real-data backtest look-ahead biased? (click to see the fix)"):
        st.markdown(
            "An earlier version of this backtest calibrated Merton parameters **once**, from the "
            "asset's **entire** 2018-2025 history, then used those same parameters to hedge every "
            "historical window — including windows from 2018, hedged with parameters that were only "
            "knowable using data through 2025. That is a textbook look-ahead bias. It is now fixed: "
            "every window is hedged using parameters calibrated **only** from log-returns strictly "
            "**before** that window begins (an expanding window, refreshed quarterly), with the first "
            "~2 years of history used only to seed the initial calibration. The table below shows both "
            "versions side by side, so the effect of the fix itself is visible rather than hidden."
        )
        bias_display = lookahead_bias_df.copy()
        bias_display["BS_pct_overstated"] = (
            bias_display["BS_CVaR_5pct_fullsample_biased"] / bias_display["BS_CVaR_5pct_walkforward"] - 1
        ) * 100
        bias_display["Merton_pct_overstated"] = (
            bias_display["Merton_CVaR_5pct_fullsample_biased"] / bias_display["Merton_CVaR_5pct_walkforward"] - 1
        ) * 100
        st.dataframe(bias_display.style.format(precision=2), width='stretch')
        st.caption(
            "'Walkforward' = corrected, honest calibration. 'Fullsample_biased' = the original, "
            "look-ahead-biased approach. Positive %-overstated means the biased approach reported a "
            "HIGHER (more alarming) CVaR than the honest one actually supports — the bias was small "
            "for calm SPY but economically meaningful for volatile BTC, whose jump/vol regime shifted "
            "more over the sample."
        )

# ─────────────────────────────────────────────────────────────────────────
# TAB 2 — Data & Calibration
# ─────────────────────────────────────────────────────────────────────────
with tab_data:
    st.markdown("#### Real market data and historical (P-measure) calibration")
    st.caption(
        "Both assets' Heston and Merton parameters are fit directly from each asset's own historical "
        "return series (not from an options chain) — see the Methodology tab for why."
    )
    cols = st.columns(2)
    for col, asset_key in zip(cols, ASSET_LABELS):
        df, S0, hp, mp = DATA[asset_key]
        with col:
            st.markdown(f"##### {ASSET_LABELS[asset_key]}")
            price_fig = go.Figure()
            price_fig.add_trace(go.Scatter(x=df.index, y=df["close"], mode="lines",
                                            line=dict(color=COLOR[asset_key], width=1.6), name="Close"))
            price_fig.update_layout(height=260, margin=dict(t=10, b=10, l=10, r=10),
                                     yaxis_title="Price", xaxis_rangeslider_visible=True)
            st.plotly_chart(price_fig, width='stretch')

            vol_fig = go.Figure()
            vol_fig.add_trace(go.Scatter(x=df.index, y=df["realized_vol_21d"] * 100, mode="lines",
                                          line=dict(color="#8C8C8C", width=1.3), fill="tozeroy",
                                          fillcolor=f"rgba({','.join(str(int(COLOR[asset_key].lstrip('#')[i:i+2],16)) for i in (0,2,4))},0.15)"))
            vol_fig.update_layout(height=190, margin=dict(t=10, b=10, l=10, r=10),
                                   yaxis_title="21d realized vol (ann. %)")
            st.plotly_chart(vol_fig, width='stretch')

            log_returns = df["log_return"].values
            threshold = 3.0 * log_returns.std()
            jump_mask = np.abs(log_returns) > threshold
            hist_fig = go.Figure()
            hist_fig.add_trace(go.Histogram(x=log_returns[~jump_mask], nbinsx=80, name="Normal days",
                                             marker_color=COLOR[asset_key], opacity=0.75))
            hist_fig.add_trace(go.Histogram(x=log_returns[jump_mask], nbinsx=40, name="Flagged jumps (>3\u03c3)",
                                             marker_color=COLOR["bs"], opacity=0.9))
            hist_fig.add_vline(x=threshold, line_dash="dot", line_color="gray")
            hist_fig.add_vline(x=-threshold, line_dash="dot", line_color="gray")
            hist_fig.update_layout(height=230, barmode="overlay", margin=dict(t=25, b=10, l=10, r=10),
                                    title=dict(text="Daily log-returns — jump identification (Merton calibration)", font=dict(size=13)),
                                    legend=dict(orientation="h", y=1.25))
            st.plotly_chart(hist_fig, width='stretch')

            st.markdown("**Calibrated parameters**")
            pcol1, pcol2 = st.columns(2)
            heston_tbl = pd.DataFrame({"Heston": {k: round(v, 5) for k, v in hp.items()}})
            merton_tbl = pd.DataFrame({"Merton": {k: round(v, 5) for k, v in mp.items()}})
            pcol1.dataframe(heston_tbl, width='stretch')
            pcol2.dataframe(merton_tbl, width='stretch')
            st.caption(f"n = {len(df):,} trading days &nbsp;|&nbsp; spot S\u2080 = {S0:,.2f} &nbsp;|&nbsp; "
                       f"{jump_mask.sum()} days flagged as jumps ({jump_mask.mean()*100:.2f}% of history)")

# ─────────────────────────────────────────────────────────────────────────
# TAB — Live Market Check (Q-measure vs. P-measure)
# ─────────────────────────────────────────────────────────────────────────
with tab_live:
    st.markdown("#### Live options chain vs. this project's historical (P-measure) model")
    st.caption(
        "This project's main documented limitation is that no free HISTORICAL options-chain "
        "data exists, forcing a P-measure-only (historical) calibration. We don't have that — but "
        "we DO have free access to a live, current options-chain snapshot. This tab fetches it in "
        "real time and compares it against the historically-calibrated Merton model's implied smile, "
        "surfacing the volatility risk premium the market is pricing right now."
    )
    live_col1, live_col2 = st.columns([1, 3])
    target_dte = live_col1.slider("Target days to expiry", 21, 180, 63, 7, key="live_dte")

    try:
        chain = get_live_chain("SPY", target_dte)
        spy_df, spy_S0, spy_hp, spy_mp = DATA["spy"]
        T_live = chain["T_years"]

        calls = chain["calls"]
        near_money = calls[(calls["moneyness"] > 0.80) & (calls["moneyness"] < 1.20)].copy()
        near_money = near_money.sort_values("strike")
        strikes_live = near_money["strike"].values

        model_ivs_merton, model_ivs_bs = [], []
        for K_live in strikes_live:
            merton_p = merton_price(chain["spot"], K_live, T_live, R_FREE, spy_mp["sigma"],
                                     spy_mp["lam"], spy_mp["mu_j"], spy_mp["sigma_j"], "call")
            model_ivs_merton.append(implied_volatility(merton_p, chain["spot"], K_live, T_live, R_FREE, "call") * 100)
            model_ivs_bs.append(spy_mp["sigma"] * 100)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("SPY spot (live)", f"{chain['spot']:,.2f}")
        m2.metric("Chosen expiry", chain["expiry"], f"{chain['dte_days']} calendar days out")
        atm_idx = int(np.argmin(np.abs(near_money["moneyness"].values - 1.0)))
        atm_market_iv = near_money["impliedVolatility"].values[atm_idx] * 100
        m3.metric("Market ATM implied vol (Q-measure)", f"{atm_market_iv:.1f}%")
        vrp = atm_market_iv - spy_mp["sigma"] * 100
        m4.metric("Volatility risk premium (Q \u2212 P)", f"{vrp:+.1f} pts",
                  "Market prices MORE vol than the historical model" if vrp > 0
                  else "Market prices LESS vol than the historical model")

        fig_live = go.Figure()
        fig_live.add_trace(go.Scatter(x=near_money["moneyness"], y=near_money["impliedVolatility"] * 100,
                                       mode="lines+markers", name="Market implied vol (Q-measure, live)",
                                       line=dict(color="#111111", width=2)))
        fig_live.add_trace(go.Scatter(x=near_money["moneyness"], y=model_ivs_merton, mode="lines+markers",
                                       name="Merton model (P-measure, historical calib.)",
                                       line=dict(color=COLOR["merton"])))
        fig_live.add_trace(go.Scatter(x=near_money["moneyness"], y=model_ivs_bs, mode="lines",
                                       name="Flat historical \u03c3 (BS reference)",
                                       line=dict(color="#888888", dash="dash")))
        fig_live.add_vline(x=1.0, line_dash="dot", line_color="lightgray")
        fig_live.update_layout(height=460, xaxis_title="Moneyness (K / Spot)",
                                yaxis_title="Implied volatility (%)", legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_live, width='stretch')

        st.info(
            "The gap between the black (live market) and blue (historical Merton model) curves IS "
            "the volatility risk premium: on average, across long samples, options tend to be priced "
            "slightly above subsequently-realised volatility — compensation option SELLERS earn for "
            "bearing tail/jump risk. A large live gap on any given day is a genuine (if simple) signal "
            "a real vol-arbitrage desk would investigate further."
        )
        with st.expander("Raw live call-chain data (near the money)"):
            st.dataframe(near_money[["strike", "moneyness", "bid", "ask", "mid", "impliedVolatility",
                                      "volume", "openInterest"]].style.format(precision=4),
                         width='stretch')
    except Exception as exc:
        st.warning(
            f"Could not fetch a live options chain right now ({type(exc).__name__}: {exc}). "
            "This tab requires live internet access to Yahoo Finance and a ticker with a listed "
            "options market (SPY has one; BTC-USD does not). All other tabs are unaffected."
        )

# ─────────────────────────────────────────────────────────────────────────
# TAB 3 — Volatility Smile
# ─────────────────────────────────────────────────────────────────────────
with tab_smile:
    st.markdown("#### Implied-volatility smile: pricing validation across all three models")
    st.caption(
        "Even though Black-Scholes, Heston, and Merton are all fed the SAME flat diffusive volatility "
        "input, Heston (stochastic vol) and Merton (jumps) each generate a volatility smile across "
        "strikes purely from their own dynamics — Black-Scholes cannot, by construction."
    )
    cols = st.columns(2)
    for col, asset_key in zip(cols, ASSET_LABELS):
        smile = compute_smile(asset_key)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=smile["moneyness"], y=smile["Black-Scholes (flat)"], mode="lines",
                                  line=dict(dash="dash", color="#444444"), name="Black-Scholes (flat)"))
        fig.add_trace(go.Scatter(x=smile["moneyness"], y=smile["Heston"], mode="lines+markers",
                                  line=dict(color=COLOR["heston"]), name="Heston"))
        fig.add_trace(go.Scatter(x=smile["moneyness"], y=smile["Merton"], mode="lines+markers",
                                  line=dict(color=COLOR["merton"]), name="Merton"))
        fig.add_vline(x=1.0, line_dash="dot", line_color="lightgray")
        fig.update_layout(height=420, title=ASSET_LABELS[asset_key],
                           xaxis_title="Moneyness (K / S\u2080)", yaxis_title="Black-Scholes implied vol (%)",
                           legend=dict(orientation="h", y=1.12))
        col.plotly_chart(fig, width='stretch')

# ─────────────────────────────────────────────────────────────────────────
# TAB 4 — Headline CVaR Result
# ─────────────────────────────────────────────────────────────────────────
with tab_headline:
    st.markdown("#### The project's central experiment: CVaR of hedging loss, BS vs. Merton hedge")
    st.caption(
        "The 'true' market always evolves under each asset's own calibrated Merton jump-diffusion "
        "(a crash-realistic market). We vary only what the market maker believes and hedges with."
    )

    fig = px.bar(
        headline_df, x="asset", y="CVaR_5pct", color="hedge_model", barmode="group",
        color_discrete_map={"BS hedge (blind to jumps)": COLOR["bs"], "Merton hedge (jump-aware)": COLOR["merton"]},
        text_auto=".3s", labels={"CVaR_5pct": "CVaR of hedging loss (95%)", "asset": ""}, height=420,
    )
    fig.update_layout(legend_title_text="", margin=dict(t=30))
    st.plotly_chart(fig, width='stretch')

    st.markdown("#### Full hedging P&L distributions (live, vectorized re-simulation)")
    n_paths_h = st.select_slider("Paths to simulate for the distributions below",
                                  options=[1000, 2000, 5000, 8000], value=5000, key="headline_npaths")
    dcols = st.columns(2)
    for col, asset_key in zip(dcols, ASSET_LABELS):
        pnl_dists = compute_pnl_distributions(asset_key, n_paths=n_paths_h)
        fig2 = go.Figure()
        for name, pnls in pnl_dists.items():
            fig2.add_trace(go.Violin(y=pnls, name=name, box_visible=True, meanline_visible=True,
                                      line_color=HEDGE_COLOR_MAP[name], opacity=0.75, points=False))
        fig2.update_layout(height=420, title=f"{ASSET_LABELS[asset_key]} — terminal hedging P&L ({n_paths_h:,} paths)",
                            yaxis_title="Terminal hedging P&L", showlegend=False)
        col.plotly_chart(fig2, width='stretch')

        stat_cols = col.columns(2)
        for j, (name, pnls) in enumerate(pnl_dists.items()):
            losses = -pnls
            with stat_cols[j]:
                st.metric(name, f"CVaR(95%) = {fmt(cvar_historical(losses, 0.05))}",
                          f"mean P&L {fmt(pnls.mean())}")

    st.info(
        "Table above = the exact numbers reported in the write-up (8,000-path run). Distributions "
        "above are re-simulated live in-app (vectorized batch simulator) so you can vary the path "
        "count and immediately see how stable the CVaR estimate is."
    )

# ─────────────────────────────────────────────────────────────────────────
# TAB — Statistical Rigor (bootstrap CIs + hypothesis tests)
# ─────────────────────────────────────────────────────────────────────────
with tab_stats:
    st.markdown("#### Is the headline CVaR gap real, or could it be simulation noise?")
    st.caption(
        "A point estimate like 'CVaR(95%) = 38.86' is not a complete claim without a confidence "
        "interval, and 'the BS hedge is worse than the Merton hedge' needs a formal hypothesis test, "
        "not just one number being larger than another. This tab answers both, via the nonparametric "
        "bootstrap: resample paths with replacement thousands of times, recompute CVaR each time, and "
        "read off the empirical distribution of that statistic (see statistical_inference.py)."
    )

    st.markdown("##### Report-scale result (precomputed, 8,000 paths, 3,000 bootstrap resamples)")
    sig_display = statistical_significance_df.copy()
    fig_ci = go.Figure()
    for i, row in sig_display.iterrows():
        for model, color, offset in [("BS", COLOR["bs"], -0.15), ("Merton", COLOR["merton"], 0.15)]:
            fig_ci.add_trace(go.Scatter(
                x=[i + offset], y=[row[f"{model}_CVaR_point"]],
                error_y=dict(type="data", symmetric=False,
                              array=[row[f"{model}_CVaR_CI_upper"] - row[f"{model}_CVaR_point"]],
                              arrayminus=[row[f"{model}_CVaR_point"] - row[f"{model}_CVaR_CI_lower"]],
                              thickness=2, width=8),
                mode="markers", marker=dict(size=13, color=color),
                name=f"{model} hedge" if i == 0 else None, showlegend=(i == 0),
                legendgroup=model,
            ))
    fig_ci.update_xaxes(tickmode="array", tickvals=list(range(len(sig_display))),
                         ticktext=list(sig_display["asset"]))
    fig_ci.update_layout(height=420, yaxis_title="CVaR of hedging loss (95%), with 95% bootstrap CI",
                          legend=dict(orientation="h", y=1.12), margin=dict(t=30))
    st.plotly_chart(fig_ci, width='stretch')

    for _, row in sig_display.iterrows():
        sig_txt = "SIGNIFICANT" if row["significant_at_5pct"] else "NOT significant"
        sig_color = "success" if row["significant_at_5pct"] else "warning"
        getattr(st, sig_color)(
            f"**{row['asset']}**: paired bootstrap test, BS \u2212 Merton CVaR(95%) = "
            f"**{row['diff_point']:.2f}** (95% CI [{row['diff_CI_lower']:.2f}, {row['diff_CI_upper']:.2f}]), "
            f"p = {row['p_value']:.4f} \u2014 **{sig_txt}** at the 5% level."
        )

    st.markdown("##### Live re-bootstrap")
    st.caption(
        "Resamples PATH INDICES jointly across both hedge models (a paired bootstrap), since both "
        "were evaluated on the exact same underlying simulated paths — this correctly uses the "
        "(typically strong, positive) correlation between the two models' path-by-path outcomes, "
        "rather than treating them as independent samples."
    )
    sc1, sc2, sc3, sc4 = st.columns(4)
    stat_asset = sc1.selectbox("Asset", list(ASSET_LABELS), format_func=lambda k: ASSET_LABELS[k], key="stat_asset")
    stat_npaths = sc2.select_slider("Paths", options=[1000, 2000, 4000, 8000], value=4000, key="stat_npaths")
    stat_nboot = sc3.select_slider("Bootstrap resamples", options=[500, 1000, 2000, 3000], value=1000, key="stat_nboot")
    stat_q = sc4.select_slider("Tail level (q)", options=[0.01, 0.05, 0.10], value=0.05, key="stat_q")

    ci_bs, ci_merton, test = compute_bootstrap_live(stat_asset, stat_npaths, stat_nboot, stat_q)
    m1, m2, m3 = st.columns(3)
    m1.metric(f"BS hedge CVaR({int((1-stat_q)*100)}%)", fmt(ci_bs["point_estimate"]),
              f"SE={fmt(ci_bs['se'])}")
    m2.metric(f"Merton hedge CVaR({int((1-stat_q)*100)}%)", fmt(ci_merton["point_estimate"]),
              f"SE={fmt(ci_merton['se'])}")
    m3.metric("Paired difference (BS \u2212 Merton)", fmt(test["point_diff"]),
              f"p={test['p_value']:.4f}")
    st.caption(
        f"BS hedge 95% CI: [{fmt(ci_bs['ci_lower'])}, {fmt(ci_bs['ci_upper'])}]  |  "
        f"Merton hedge 95% CI: [{fmt(ci_merton['ci_lower'])}, {fmt(ci_merton['ci_upper'])}]  |  "
        f"Difference 95% CI: [{fmt(test['ci_lower'])}, {fmt(test['ci_upper'])}]  |  "
        f"{'**Statistically significant**' if test['significant_at_5pct'] else 'Not statistically significant'} at the 5% level."
    )

# ─────────────────────────────────────────────────────────────────────────
# TAB — Tail Risk (Extreme Value Theory)
# ─────────────────────────────────────────────────────────────────────────
with tab_evt:
    st.markdown("#### Extreme Value Theory: Peaks-Over-Threshold tail risk")
    st.caption(
        "Historical CVaR at very deep quantiles (99%+) is informed by only a handful of the worst "
        "simulated paths. EVT/POT fits a Generalized Pareto Distribution (GPD) to the tail of "
        "exceedances over a high threshold and extrapolates smoothly beyond the observed sample — "
        "the direct analogue of the Central Limit Theorem, but for tails instead of averages "
        "(Pickands-Balkema-de Haan theorem). See tail_risk.py for full validation."
    )

    ec1, ec2, ec3 = st.columns(3)
    evt_asset = ec1.selectbox("Asset", list(ASSET_LABELS), format_func=lambda k: ASSET_LABELS[k], key="evt_asset")
    evt_npaths = ec2.select_slider("Paths", options=[2000, 4000, 8000], value=8000, key="evt_npaths")
    evt_thresh_q = ec3.slider("GPD threshold (percentile of losses)", 0.80, 0.95, 0.90, 0.01, key="evt_thresh")

    evt_data = compute_evt_live(evt_asset, evt_npaths, evt_thresh_q)

    evt_cols = st.columns(2)
    for col, (name, d) in zip(evt_cols, evt_data.items()):
        fit = d["fit"]
        with col:
            st.markdown(f"**{name}** — GPD fit: \u03be={fit['xi']:.3f}, \u03b2={fit['beta']:.2f}, "
                        f"threshold={fit['threshold']:.2f} ({fit['n_exceedances']} exceedances)")
            comp = d["comparison"]
            fig_evt = go.Figure()
            fig_evt.add_trace(go.Scatter(x=comp["q"], y=comp["historical_CVaR"], mode="lines+markers",
                                          name="Historical CVaR", line=dict(color="#888888")))
            fig_evt.add_trace(go.Scatter(x=comp["q"], y=comp["EVT_CVaR"], mode="lines+markers",
                                          name="EVT (POT) CVaR", line=dict(color=HEDGE_COLOR_MAP.get(name, COLOR["merton"]))))
            fig_evt.update_xaxes(type="log", title_text="Tail probability q (log scale)")
            fig_evt.update_layout(height=340, yaxis_title="CVaR of hedging loss",
                                   legend=dict(orientation="h", y=1.15), margin=dict(t=20))
            col.plotly_chart(fig_evt, width='stretch')
            col.dataframe(comp[["q", "historical_VaR", "historical_CVaR", "EVT_VaR", "EVT_CVaR"]]
                          .style.format(precision=2), width='stretch')

    st.info(
        "At moderate quantiles (q=0.05) historical and EVT estimates agree closely — both have plenty "
        "of data. They DIVERGE at deep quantiles (q=0.001), where the historical estimator is informed "
        "by only a handful of simulated paths and EVT is instead extrapolating from the fitted tail "
        "shape — exactly the regime Extreme Value Theory exists for."
    )

    st.markdown("##### Diagnostics: mean-excess plot and Hill plot")
    diag_cols = st.columns(2)
    first_model = list(evt_data.keys())[0]
    mep = evt_data[first_model]["mean_excess"]
    hp_plot = evt_data[first_model]["hill_plot"]
    fig_mep = go.Figure()
    fig_mep.add_trace(go.Scatter(x=mep["threshold"], y=mep["mean_excess"], mode="markers+lines",
                                  line=dict(color=COLOR[evt_asset])))
    fig_mep.update_layout(height=320, title=f"Mean-excess plot ({first_model})",
                           xaxis_title="Threshold u", yaxis_title="Mean excess E[Loss-u | Loss>u]")
    diag_cols[0].plotly_chart(fig_mep, width='stretch')
    diag_cols[0].caption("Roughly linear beyond the chosen threshold is the classic (Davison & Smith, "
                          "1990) visual check that a GPD tail fit is appropriate.")

    fig_hill = go.Figure()
    fig_hill.add_trace(go.Scatter(x=hp_plot["k"], y=hp_plot["xi_hill"], mode="lines",
                                   line=dict(color=COLOR[evt_asset])))
    fig_hill.add_hline(y=evt_data[first_model]["fit"]["xi"], line_dash="dash", line_color="gray",
                        annotation_text="GPD MLE xi")
    fig_hill.update_layout(height=320, title=f"Hill plot ({first_model})",
                            xaxis_title="k (number of top order statistics used)", yaxis_title="Hill xi estimate")
    diag_cols[1].plotly_chart(fig_hill, width='stretch')
    diag_cols[1].caption("A reliable tail-index estimate looks roughly STABLE (flat) across a broad "
                          "range of k; Hill's estimator is used here as a stability cross-check "
                          "alongside the primary MLE-based GPD fit (see tail_risk.py for why).")

# ─────────────────────────────────────────────────────────────────────────
# TAB — P&L Attribution ("P&L explain")
# ─────────────────────────────────────────────────────────────────────────
with tab_attribution:
    st.markdown("#### Greeks-based P&L attribution — where does the hidden risk actually show up?")
    st.caption(
        "A real trading desk decomposes daily P&L into Delta, Gamma, Theta, and an 'unexplained' "
        "residual (\u0394V \u2248 \u0394\u00b7\u0394S + \u00bd\u0393\u00b7\u0394S\u00b2 + \u0398\u00b7\u0394t + residual). "
        "The residual is exactly where JUMP risk (or any move a smooth Taylor expansion can't see) shows up — "
        "turning the project's headline CVaR number into a day-by-day, causal explanation."
    )
    ac1, ac2, ac3, ac4 = st.columns(4)
    attr_asset = ac1.selectbox("Asset", list(ASSET_LABELS), format_func=lambda k: ASSET_LABELS[k], key="attr_asset")
    attr_hedge = ac2.selectbox("Hedge model", ["bs", "heston", "merton"],
                                format_func=lambda x: {"bs": "Black-Scholes", "heston": "Heston", "merton": "Merton"}[x],
                                key="attr_hedge")
    attr_seed = ac3.number_input("Seed", value=99, step=1, key="attr_seed")
    attr_path_idx = ac4.number_input("Path index (0-49)", min_value=0, max_value=49, value=0, step=1, key="attr_path_idx")

    attribution, path_used, K_used = compute_attribution_path(
        attr_asset, "call", attr_hedge, int(attr_seed), path_index=int(attr_path_idx))

    fig_attr = make_subplots(rows=1, cols=2, subplot_titles=(
        "Cumulative P&L attribution", "Per-step residual (\u2248 unexplained risk)"))
    for col_name, color in [("cum_delta_pnl", COLOR["merton"]), ("cum_gamma_pnl", COLOR["heston"]),
                             ("cum_theta_pnl", "#888888"), ("cum_residual_pnl", COLOR["bs"])]:
        fig_attr.add_trace(go.Scatter(x=attribution["step"], y=attribution[col_name], mode="lines",
                                       name=col_name.replace("cum_", "").replace("_pnl", "").title(),
                                       line=dict(color=color)), row=1, col=1)
    fig_attr.add_trace(go.Scatter(x=attribution["step"], y=attribution["cum_actual_dV"], mode="lines",
                                   name="Actual (total)", line=dict(color="black", dash="dot")), row=1, col=1)
    bar_colors = [COLOR["bs"] if abs(v) == attribution["residual_pnl"].abs().max() else "#B0B0B0"
                  for v in attribution["residual_pnl"]]
    fig_attr.add_trace(go.Bar(x=attribution["step"], y=attribution["residual_pnl"],
                               marker_color=bar_colors, name="Residual"), row=1, col=2)
    fig_attr.update_layout(height=440, legend=dict(orientation="h", y=1.15))
    st.plotly_chart(fig_attr, width='stretch')

    worst_step = int(attribution.loc[attribution["residual_pnl"].abs().idxmax(), "step"])
    worst_val = attribution["residual_pnl"].abs().max()
    st.caption(f"Largest single-day 'unexplained' residual: **{worst_val:,.4f}** at step {worst_step} "
               f"(highlighted in red) &mdash; this is a day the Greeks-based Taylor expansion could not "
               f"anticipate, i.e., a realised jump or other higher-order move.")

    st.markdown("##### Aggregated across many paths: whose residual has the fatter tail?")
    n_paths_attr = st.select_slider("Paths to aggregate", options=[500, 1000, 2000, 3000], value=2000, key="attr_npaths")
    residual_data = compute_attribution_batch_residuals(attr_asset, "call", n_paths_attr, int(attr_seed))

    fig_resid = go.Figure()
    for name, attr_df in residual_data.items():
        fig_resid.add_trace(go.Histogram(x=attr_df["cum_residual_pnl"], name=name, opacity=0.65,
                                          marker_color=HEDGE_COLOR_MAP[name], nbinsx=60))
    fig_resid.update_layout(height=400, barmode="overlay", xaxis_title="Total 'unexplained' residual P&L over the option's life",
                             legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig_resid, width='stretch')

    r1, r2 = st.columns(2)
    for col, (name, attr_df) in zip([r1, r2], residual_data.items()):
        resid = attr_df["cum_residual_pnl"].values
        col.metric(f"{name} — residual CVaR(95%)", fmt(cvar_historical(-resid, 0.05)),
                   f"mean {fmt(resid.mean())}")

    st.success(
        "The BS hedge's residual distribution has a visibly fatter left tail than the Merton hedge's "
        "under the SAME jumpy 'true' market — the granular, per-step confirmation of why the BS "
        "hedge's terminal CVaR is higher: it accumulates large unexplained losses specifically on jump days."
    )

# ─────────────────────────────────────────────────────────────────────────
# TAB 5 — Heston Finding
# ─────────────────────────────────────────────────────────────────────────
with tab_heston:
    st.markdown("#### Secondary finding: stochastic volatility alone is not a jump hedge")
    st.caption(
        "Adding a Heston hedge (stochastic vol, no jumps) to the comparison — NOW at full scale "
        "(8,000 paths, daily rebalancing, identical to the BS/Merton headline comparison), thanks to "
        "heston.py's COS-method Fourier pricer, which vectorizes across paths. Previously this "
        "comparison was restricted to ~150 paths / biweekly rebalancing because the original "
        "quad-based Fourier inversion could not be vectorized — the finding below is now confirmed at "
        "the same statistical scale and rebalancing frequency as every other hedge-model comparison "
        "in this project, not just a small-sample approximation to it."
    )
    fig = px.bar(
        heston_secondary_df, x="asset", y="CVaR_5pct", color="hedge_model", barmode="group",
        color_discrete_map=HEDGE_COLOR_MAP, text_auto=".3s",
        labels={"CVaR_5pct": "CVaR of hedging loss (95%)", "asset": ""}, height=440,
    )
    fig.update_layout(legend_title_text="", margin=dict(t=30))
    st.plotly_chart(fig, width='stretch')

    st.warning(
        "On BOTH assets, the Heston hedge shows **higher** CVaR than even the naive Black-Scholes "
        "hedge — despite being the more 'sophisticated' model. Stochastic volatility does not protect "
        "against jump risk, and can even slightly worsen outcomes by reallocating hedge sensitivity in "
        "a way that doesn't address the actual tail risk present. Sophistication must match the "
        "*specific* risk in the market, not just add complexity."
    )

    st.markdown("##### Full comparison table")
    st.dataframe(heston_secondary_df.style.format(precision=2), width='stretch')

# ─────────────────────────────────────────────────────────────────────────
# TAB — Vega Hedging (Delta-Vega multi-instrument hedge)
# ─────────────────────────────────────────────────────────────────────────
with tab_vega:
    st.markdown("#### Giving stochastic-volatility risk a fair fight: Delta-Vega hedging")
    st.caption(
        "Every hedge studied elsewhere trades ONLY the underlying (Delta-hedging). The underlying has "
        "zero Vega, so no amount of stock trading can hedge volatility risk — meaning a Delta-only "
        "Heston hedge can never use its own core insight (stochastic vol) at all. This adds a SECOND "
        "option instrument, sized to neutralize net Vega (n_instrument = Vega_target / Vega_instrument), "
        "with the stock then used to mop up whatever Delta remains (see hedging_engine.py's "
        "simulate_delta_vega_hedge_batch)."
    )

    st.markdown("##### Report-scale result (precomputed): true market = each asset's OWN calibrated Heston process")
    fig_vega_report = px.bar(
        vega_hedging_df, x="asset", y="CVaR_5pct", color="strategy", barmode="group",
        color_discrete_map={"Delta-only": COLOR["bs"], "Delta-Vega": COLOR["merton"]},
        text_auto=".3s", labels={"CVaR_5pct": "CVaR of hedging loss (95%)", "asset": ""}, height=400,
    )
    fig_vega_report.update_layout(legend_title_text="", margin=dict(t=30))
    st.plotly_chart(fig_vega_report, width='stretch')
    st.dataframe(vega_hedging_df.style.format(precision=2), width='stretch')

    st.markdown("##### Live re-simulation")
    vc1, vc2, vc3, vc4 = st.columns(4)
    vega_asset = vc1.selectbox("Asset", list(ASSET_LABELS), format_func=lambda k: ASSET_LABELS[k], key="vega_asset")
    vega_true_dgp = vc2.radio("True market", ["heston", "merton"], horizontal=True, key="vega_true_dgp",
                               format_func=lambda x: {"heston": "Stochastic vol (Heston)", "merton": "Jumps (Merton)"}[x])
    vega_k_mult = vc3.slider("Instrument strike (x Spot)", 1.02, 1.25, 1.10, 0.01, key="vega_k_mult")
    vega_rebal = vc4.select_slider("Rebalance every N days", options=[1, 2, 5, 10], value=5, key="vega_rebal")
    vega_npaths = st.select_slider("Paths", options=[1000, 2000, 4000], value=2000, key="vega_npaths")

    summary_live, pnls_delta_live, pnls_vega_live = compute_vega_hedge_live(
        vega_asset, vega_npaths, vega_k_mult, vega_true_dgp, vega_rebal)

    lv1, lv2 = st.columns(2)
    fig_vega_dist = go.Figure()
    fig_vega_dist.add_trace(go.Violin(y=pnls_delta_live, name="Delta-only", box_visible=True,
                                       meanline_visible=True, line_color=COLOR["bs"], points=False))
    fig_vega_dist.add_trace(go.Violin(y=pnls_vega_live, name="Delta-Vega", box_visible=True,
                                       meanline_visible=True, line_color=COLOR["merton"], points=False))
    fig_vega_dist.update_layout(height=400, yaxis_title="Terminal hedging P&L",
                                 title="Hedging P&L distribution")
    lv1.plotly_chart(fig_vega_dist, width='stretch')

    lv2.dataframe(summary_live.style.format(precision=3), width='stretch')
    cvar_delta_live = cvar_historical(-pnls_delta_live, 0.05)
    cvar_vega_live = cvar_historical(-pnls_vega_live, 0.05)
    if cvar_vega_live < cvar_delta_live:
        lv2.success(f"Delta-Vega **reduces** CVaR(95%) from {fmt(cvar_delta_live)} to {fmt(cvar_vega_live)} "
                    f"({(1 - cvar_vega_live/cvar_delta_live)*100:.1f}% lower) under this true market.")
    else:
        lv2.warning(f"Delta-Vega does **not** reduce CVaR(95%) here ({fmt(cvar_delta_live)} \u2192 "
                    f"{fmt(cvar_vega_live)}) — vega hedging adds a second instrument's transaction cost "
                    f"and idiosyncratic noise, which only pays off when there is genuine vol-of-vol risk "
                    f"to hedge against (try the Heston true-market option with a higher-xi asset).")

    st.info(
        "Try switching the true market between 'Stochastic vol (Heston)' and 'Jumps (Merton)': Vega "
        "hedging is designed for the FORMER, not the latter — and even under Heston, whether it helps "
        "depends on how much vol-of-vol (\u03be) risk the asset actually has. This is the same "
        "'sophistication must match the specific risk present' lesson as the Heston Finding tab, now "
        "extended one step further: Heston hedging DID need a Vega instrument to have a chance, and "
        "even then only helps when there is real stochastic-volatility risk to hedge."
    )

# ─────────────────────────────────────────────────────────────────────────
# TAB 6 — Efficient Frontier
# ─────────────────────────────────────────────────────────────────────────
with tab_frontier:
    st.markdown("#### Rebalancing frequency: transaction cost vs. hedging-risk tradeoff")
    st.caption("Holding the BS hedge model fixed against each asset's calibrated Merton 'true' market.")
    cols = st.columns(2)
    for col, asset_key in zip(cols, ASSET_LABELS):
        frontier = frontier_df[asset_key]
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=frontier["rebalance_every_n_steps"], y=frontier["CVaR_5pct"],
                                  mode="lines+markers", name="CVaR (95%)", line=dict(color=COLOR[asset_key], width=3)),
                      secondary_y=False)
        fig.add_trace(go.Scatter(x=frontier["rebalance_every_n_steps"], y=frontier["std_pnl"],
                                  mode="lines+markers", name="Std. dev. of P&L",
                                  line=dict(color="#8C8C8C", dash="dash")), secondary_y=True)
        fig.update_layout(height=420, title=ASSET_LABELS[asset_key],
                           xaxis_title="Rebalance every N trading days", legend=dict(orientation="h", y=1.15))
        fig.update_yaxes(title_text="CVaR of hedging loss (95%)", secondary_y=False)
        fig.update_yaxes(title_text="Std. dev. of hedging P&L", secondary_y=True)
        col.plotly_chart(fig, width='stretch')
    st.info(
        "CVaR rises monotonically with less frequent rebalancing on both assets — the classic "
        "cost-vs-risk tradeoff a real market-making desk has to manage explicitly."
    )

    st.markdown("---")
    st.markdown("#### Band (\"no-transaction region\") hedging: a smarter alternative to a rigid schedule")
    st.caption(
        "Whalley & Wilmott (1993): trade only when Delta drifts more than a tolerance band away from "
        "the current hedge, instead of on a fixed calendar. Precomputed at project-report scale below; "
        "adjust the controls to re-sweep live."
    )
    bc1, bc2, bc3 = st.columns(3)
    band_asset = bc1.selectbox("Asset", list(ASSET_LABELS), format_func=lambda k: ASSET_LABELS[k], key="band_asset")
    band_npaths = bc2.select_slider("Paths", options=[1000, 2000, 4000], value=2000, key="band_npaths")
    band_tc = bc3.slider("Transaction cost (bps)", 0, 50, 5, key="band_tc")

    precomputed = band_hedging_df[band_hedging_df["asset"] == ASSET_LABELS[band_asset]]
    live_sweep = compute_band_sweep(band_asset, "call", band_npaths, band_tc, 777)

    fbc1, fbc2 = st.columns(2)
    fig_band_report = go.Figure()
    daily_cvar = precomputed[precomputed["strategy"].str.contains("Daily")]["CVaR_5pct"].iloc[0]
    band_rows = precomputed[precomputed["strategy"].str.contains("Band")]
    fig_band_report.add_trace(go.Scatter(
        x=band_rows["mean_transaction_cost"], y=band_rows["CVaR_5pct"], mode="markers+text",
        text=band_rows["strategy"].str.extract(r"width=([\d.]+)")[0], textposition="top center",
        marker=dict(size=12, color=COLOR[band_asset]), name="Band hedging"))
    fig_band_report.add_hline(y=daily_cvar, line_dash="dash", line_color="gray",
                               annotation_text="Daily rebalancing (report-scale, 4,000 paths)")
    fig_band_report.update_layout(height=400, title="Report-scale result (precomputed)",
                                   xaxis_title="Mean transaction cost paid per option",
                                   yaxis_title="CVaR of hedging loss (95%)")
    fbc1.plotly_chart(fig_band_report, width='stretch')

    fig_band_live = go.Figure()
    fig_band_live.add_trace(go.Scatter(
        x=live_sweep["mean_transaction_cost"], y=live_sweep["CVaR_5pct"], mode="markers+lines+text",
        text=[f"w={w}" for w in live_sweep["band_width"]], textposition="top center",
        marker=dict(size=10, color=COLOR[band_asset]), line=dict(color=COLOR[band_asset], dash="dot")))
    fig_band_live.update_layout(height=400, title=f"Live sweep ({band_npaths:,} paths, {band_tc}bps)",
                                 xaxis_title="Mean transaction cost paid per option",
                                 yaxis_title="CVaR of hedging loss (95%)")
    fbc2.plotly_chart(fig_band_live, width='stretch')

    best_row = live_sweep.loc[live_sweep["CVaR_5pct"].idxmin()]
    st.success(
        f"For {ASSET_LABELS[band_asset]} at {band_tc}bps transaction cost, the lowest-CVaR band width in "
        f"this live sweep is **{best_row['band_width']}** (CVaR={best_row['CVaR_5pct']:,.2f}, mean cost="
        f"{best_row['mean_transaction_cost']:,.2f}) — band hedging is not simply 'less trading = more risk': "
        f"a well-chosen tolerance band can match or beat rigid daily rebalancing at a fraction of the trading cost."
    )

# ─────────────────────────────────────────────────────────────────────────
# TAB — American Options (binomial tree + Crank-Nicolson PDE)
# ─────────────────────────────────────────────────────────────────────────
with tab_american:
    st.markdown("#### American options: numerical PDE / tree methods")
    st.caption(
        "Every pricer used elsewhere in this project (Black-Scholes, Heston, Merton) is European-only. "
        "This adds a Cox-Ross-Rubinstein binomial tree AND a Crank-Nicolson finite-difference PDE "
        "solver for American options — cross-validated against each other, against the known European "
        "reduction (no early exercise without dividends), and checked to show a genuine early-exercise "
        "premium exactly where theory says one must exist (see american_options.py)."
    )

    aoc1, aoc2 = st.columns(2)
    amer_asset = aoc1.selectbox("Asset", list(ASSET_LABELS), format_func=lambda k: ASSET_LABELS[k], key="amer_asset")
    amer_type = aoc2.radio("Option type", ["put", "call"], horizontal=True, key="amer_type")

    prem, cn, premium_df, S0_amer, sigma_amer = compute_american_options_live(amer_asset, amer_type)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Binomial: European price", fmt(prem["european_price"]))
    m2.metric("Binomial: American price", fmt(prem["american_price"]))
    m3.metric("Early-exercise premium", fmt(prem["premium"]), f"{prem['premium_pct']:.2f}%")
    m4.metric("Crank-Nicolson American price", fmt(cn["price"]),
              f"|diff vs. tree| = {fmt(abs(cn['price'] - prem['american_price']))}")

    ac1, ac2 = st.columns(2)
    boundary = cn["boundary"]
    if boundary:
        taus, S_stars = zip(*boundary)
        fig_boundary = go.Figure()
        fig_boundary.add_trace(go.Scatter(x=np.array(taus) * 365, y=S_stars, mode="lines",
                                           line=dict(color=COLOR[amer_asset], width=3), name="Exercise boundary"))
        fig_boundary.add_hline(y=S0_amer, line_dash="dot", line_color="gray", annotation_text="Strike K")
        fig_boundary.update_layout(height=380, title="Early-exercise boundary S*(t)",
                                    xaxis_title="Calendar days from maturity (left) to today (right)",
                                    yaxis_title="Boundary price")
        ac1.plotly_chart(fig_boundary, width='stretch')
        ac1.caption("Below the boundary, immediate exercise is optimal; above it, the market maker "
                    "should hold. Computed directly off the Crank-Nicolson PDE grid at every time step "
                    "-- no extra solves needed.")

    fig_premium = go.Figure()
    fig_premium.add_trace(go.Scatter(x=premium_df["moneyness"], y=premium_df["premium"], mode="lines+markers",
                                      line=dict(color=COLOR[amer_asset])))
    fig_premium.add_vline(x=1.0, line_dash="dot", line_color="lightgray")
    fig_premium.update_layout(height=380, title="Early-exercise premium across moneyness",
                               xaxis_title="Strike / Spot (moneyness)", yaxis_title="American - European price")
    ac2.plotly_chart(fig_premium, width='stretch')
    ac2.caption("The early-exercise premium is largest for in-the-money puts (low moneyness), where "
                "immediate exercise captures time value on interest earned from the strike proceeds.")

    st.markdown("##### Report-scale validation table (both assets)")
    st.dataframe(american_options_df.style.format(precision=4), width='stretch')
    st.info(
        "Cross-validated (see american_options.py's own CLI check): the binomial tree converges to the "
        "Black-Scholes closed form as steps grow; an American call with NO dividends exactly equals its "
        "European price (early exercise is never optimal without dividends); an American call WITH "
        "dividends shows a genuine positive premium; and the binomial tree and Crank-Nicolson PDE — two "
        "completely independent numerical methods — agree with each other to within a few cents."
    )

# ─────────────────────────────────────────────────────────────────────────
# TAB — Stress Testing
# ─────────────────────────────────────────────────────────────────────────
with tab_stress:
    st.markdown("#### Deterministic scenario stress testing (Basel/CCAR-style)")
    st.caption(
        "Complements the statistical CVaR analysis with named, distribution-free adverse scenarios — "
        "run through the exact same discrete hedging engine, not a separate ad-hoc calculation. This is "
        "what regulatory (CCAR/ICAAP) and internal stress-testing frameworks require ALONGSIDE a "
        "statistical VaR/CVaR model, precisely because 'this scenario is outside the calibration "
        "window' is a risk a purely statistical model can under-price."
    )
    stress_asset = st.selectbox("Asset", list(ASSET_LABELS), format_func=lambda k: ASSET_LABELS[k], key="stress_asset")
    stress_option = st.radio("Option type", ["call", "put"], horizontal=True, key="stress_opt")
    live_stress = compute_stress_scenarios(stress_asset, stress_option)

    fig_stress = px.bar(
        live_stress, x="scenario", y="terminal_pnl", color="hedge_model", barmode="group",
        color_discrete_map=HEDGE_COLOR_MAP, text_auto=".2s",
        labels={"terminal_pnl": "Terminal hedging P&L", "scenario": ""}, height=480,
        hover_data=["price_move_pct"],
    )
    fig_stress.update_xaxes(tickangle=-15)
    fig_stress.add_hline(y=0, line_color="black", line_width=1)
    fig_stress.update_layout(legend_title_text="", margin=dict(t=30))
    st.plotly_chart(fig_stress, width='stretch')

    pivot_stress = live_stress.pivot(index="scenario", columns="hedge_model", values="terminal_pnl")
    n_better = int((pivot_stress["Merton hedge"] > pivot_stress["BS hedge"]).sum())
    st.info(
        f"The Merton (jump-aware) hedge outperforms the BS hedge in **{n_better}/{len(pivot_stress)}** "
        f"named scenarios for {ASSET_LABELS[stress_asset]} — with the largest gaps concentrated in the "
        f"crash-style scenarios (Black Monday, COVID crash), exactly where jump-awareness should matter most."
    )
    st.dataframe(pivot_stress.style.format(precision=2), width='stretch')

# ─────────────────────────────────────────────────────────────────────────
# TAB — Model Validation (VaR backtesting)
# ─────────────────────────────────────────────────────────────────────────
with tab_validation:
    st.markdown("#### VaR model backtesting: Kupiec POF test + Basel traffic light")
    st.caption(
        "The same discipline a bank's independent model-validation function applies before (and "
        "periodically after) putting a risk model into production: VaR is ESTIMATED on one half of "
        "the simulated paths and its exception rate is TESTED on the other, held-out half — an honest "
        "out-of-sample coverage test, not a look-ahead-biased in-sample check."
    )

    zone_color = {"Green": "#59A14F", "Yellow": "#F1C232", "Red": "#E15759"}
    vcols = st.columns(4)
    for i, (_, row) in enumerate(var_backtest_df.iterrows()):
        with vcols[i % 4]:
            st.markdown(
                f"<div style='border-radius:10px;padding:0.8rem;background:{zone_color[row['basel_zone']]}22;"
                f"border:1.5px solid {zone_color[row['basel_zone']]};'>"
                f"<b>{row['asset'].split(' (')[0]} \u2014 {row['hedge_model']}</b><br>"
                f"Basel zone: <b style='color:{zone_color[row['basel_zone']]}'>{row['basel_zone']}</b><br>"
                f"Exception rate: {row['exception_rate']:.4f} (target {row['target_rate']:.4f})<br>"
                f"Kupiec p-value: {row['p_value']:.4f}</div>", unsafe_allow_html=True)
    st.caption("")

    fig_var = px.bar(
        var_backtest_df, x="hedge_model", y="exception_rate", color="hedge_model", facet_col="asset",
        color_discrete_map=HEDGE_COLOR_MAP, labels={"exception_rate": "Out-of-sample exception rate"},
        height=400,
    )
    fig_var.add_hline(y=0.05, line_dash="dash", line_color="black", annotation_text="Target (5%)")
    fig_var.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig_var.update_layout(showlegend=False, margin=dict(t=40))
    st.plotly_chart(fig_var, width='stretch')

    st.markdown("##### Live re-backtest")
    lc1, lc2, lc3, lc4 = st.columns(4)
    val_asset = lc1.selectbox("Asset", list(ASSET_LABELS), format_func=lambda k: ASSET_LABELS[k], key="val_asset")
    val_npaths = lc2.select_slider("Paths", options=[2000, 4000, 8000], value=4000, key="val_npaths")
    val_q = lc3.select_slider("VaR level (q)", options=[0.01, 0.05, 0.10], value=0.05, key="val_q")
    val_train_frac = lc4.slider("Train fraction", 0.3, 0.7, 0.5, 0.05, key="val_train_frac")

    live_backtest = compute_var_backtest(val_asset, val_npaths, val_q, val_train_frac)
    st.dataframe(
        live_backtest[["hedge_model", "var_estimate", "n_train", "n_test", "exception_rate",
                       "target_rate", "LR_stat", "p_value", "reject_H0_at_95pct", "basel_zone"]]
        .style.format(precision=4), width='stretch')

    st.warning(
        "A model landing in the Yellow or Red zone, or with Kupiec's test rejecting H0, doesn't "
        "necessarily mean the hedge itself is bad — it means the SIMPLE historical-quantile VaR "
        "estimate for that hedge model is not reliably calibrated out-of-sample at this sample size, "
        "exactly the kind of finding that would trigger a model refinement in a real risk function."
    )

# ─────────────────────────────────────────────────────────────────────────
# TAB 7 — Interactive Risk Lab
# ─────────────────────────────────────────────────────────────────────────
with tab_lab:
    st.markdown("#### Live multi-parameter sweeps")
    st.caption(
        "Change any control below and every chart in this tab updates — each shows results across "
        "MULTIPLE hedge models / rebalancing frequencies / strikes simultaneously (not one run at a time)."
    )
    lc1, lc2, lc3, lc4, lc5 = st.columns(5)
    asset_key = lc1.selectbox("Asset", list(ASSET_LABELS), format_func=lambda k: ASSET_LABELS[k], key="lab_asset")
    option_type = lc2.selectbox("Option type", ["call", "put"], key="lab_opt")
    n_paths_lab = lc3.select_slider("Paths per simulation", options=[500, 1000, 2000, 3000, 5000], value=2000, key="lab_paths")
    tc_bps = lc4.slider("Transaction cost (bps)", 0, 50, 5, key="lab_tc")
    seed = lc5.number_input("Seed", value=42, step=1, key="lab_seed")
    r_free_lab = st.slider("Risk-free rate", 0.0, 0.08, 0.04, 0.005, key="lab_rfree")

    st.markdown("##### Sweep 1 — Rebalancing frequency \u00d7 hedge model")
    freq_sweep = compute_frequency_sweep(asset_key, option_type, n_paths_lab, tc_bps, r_free_lab, int(seed))
    fc1, fc2 = st.columns([3, 2])
    fig_freq = px.line(
        freq_sweep, x="rebalance_every", y="CVaR_5pct", color="hedge_model", markers=True,
        color_discrete_map=HEDGE_COLOR_MAP,
        labels={"rebalance_every": "Rebalance every N days", "CVaR_5pct": "CVaR of hedging loss (95%)"},
        height=380,
    )
    fig_freq.update_layout(legend_title_text="", margin=dict(t=20))
    fc1.plotly_chart(fig_freq, width='stretch')

    heat_pivot = freq_sweep.pivot(index="hedge_model", columns="rebalance_every", values="CVaR_5pct")
    fig_heat = px.imshow(heat_pivot, text_auto=".1f", color_continuous_scale="RdYlGn_r", aspect="auto",
                          labels=dict(x="Rebalance every N days", y="", color="CVaR(95%)"), height=380)
    fc2.plotly_chart(fig_heat, width='stretch')

    st.markdown("##### Sweep 2 — Strike / moneyness \u00d7 hedge model (daily rebalancing)")
    money_sweep = compute_moneyness_sweep(asset_key, option_type, n_paths_lab, tc_bps, r_free_lab, int(seed))
    fig_money = px.line(
        money_sweep, x="moneyness", y="CVaR_5pct", color="hedge_model", markers=True,
        color_discrete_map=HEDGE_COLOR_MAP,
        labels={"moneyness": "Strike / Spot (moneyness)", "CVaR_5pct": "CVaR of hedging loss (95%)"},
        height=380,
    )
    fig_money.add_vline(x=1.0, line_dash="dot", line_color="lightgray")
    fig_money.update_layout(legend_title_text="", margin=dict(t=20))
    st.plotly_chart(fig_money, width='stretch')

    st.markdown("##### Drill-down — one illustrative hedging path")
    dd1, dd2, dd3 = st.columns(3)
    hedge_model_dd = dd1.selectbox("Hedge model", ["bs", "heston", "merton"],
                                    format_func=lambda x: {"bs": "Black-Scholes", "heston": "Heston", "merton": "Merton"}[x],
                                    key="lab_hedge_dd")
    rebalance_dd = dd2.select_slider("Rebalance every N days", options=[1, 2, 5, 10, 21], value=1, key="lab_rebal_dd")
    path_idx_dd = dd3.number_input("Path index (0-49)", min_value=0, max_value=49, value=0, step=1, key="lab_path_dd")

    result, K = compute_illustrative_path(asset_key, option_type, hedge_model_dd, rebalance_dd, tc_bps,
                                           r_free_lab, int(seed), path_index=int(path_idx_dd))
    fig_path = make_subplots(rows=1, cols=3, subplot_titles=("Underlying price path", "Hedge Delta", "Cash account"))
    fig_path.add_trace(go.Scatter(x=result["step"], y=result["S"], mode="lines",
                                   line=dict(color=COLOR["merton"]), name="Price"), row=1, col=1)
    fig_path.add_hline(y=K, line_dash="dash", line_color="gray", row=1, col=1)
    fig_path.add_trace(go.Scatter(x=result["step"], y=result["delta"], mode="lines",
                                   line=dict(color=COLOR["bs"]), name="Delta"), row=1, col=2)
    fig_path.add_trace(go.Scatter(x=result["step"], y=result["cash"], mode="lines",
                                   line=dict(color=COLOR["heston"]), name="Cash"), row=1, col=3)
    fig_path.update_layout(height=350, showlegend=False, margin=dict(t=40, b=10))
    st.plotly_chart(fig_path, width='stretch')
    st.caption(f"Terminal hedging P&L on this path: **{result.attrs['terminal_pnl']:,.2f}** &nbsp;|&nbsp; "
               f"payoff at maturity: {result.attrs['payoff']:,.2f} &nbsp;|&nbsp; "
               f"initial premium received: {result.attrs['initial_premium']:,.2f}")

# ─────────────────────────────────────────────────────────────────────────
# TAB 8 — Methodology & Notes
# ─────────────────────────────────────────────────────────────────────────
with tab_notes:
    st.markdown("""
#### Key methodological choices (and their limitations)

**Historical (P-measure) calibration, now cross-checked against a LIVE Q-measure snapshot.** Genuinely
free, long-history options chain data does not exist for public download, so Heston/Merton parameters
are fit to each asset's own historical return series rather than to observed option prices. This tells
us "what a Heston/Merton model consistent with how this asset has actually behaved would look like," not
"what the market currently prices volatility/jump risk at." The **Live Market Check** tab partly closes
this gap: it fetches SPY's real, current listed-options chain and compares the market's live implied
vol smile against this project's historical model — surfacing the volatility risk premium directly.

**Separating the "true" market from the "hedge model".** Every experiment fixes how the underlying
actually evolves (a real historical path, or a simulated path from the asset's calibrated Merton
process) and varies only what the market maker *believes* and hedges with. The mismatch between these
two is what generates hidden tail risk — and is exactly the kind of model-risk analysis a real
derivatives risk function performs.

**Look-ahead bias in the real-data backtest — found and fixed.** An earlier version of the real-data
backtest calibrated Merton parameters once from the ENTIRE historical sample, then used those same
parameters to hedge every window, including windows years before the calibration data even ends. This
is now walk-forward: every window is hedged using parameters calibrated ONLY from data strictly prior
to that window, refreshed quarterly. Both the corrected and original (biased) numbers are reported side
by side in the Overview tab's expander, rather than silently replaced — the bias turned out to be small
for SPY but economically meaningful for BTC.

**Why CVaR — and why not ONLY CVaR, and why not ONLY historical CVaR.** Conditional Value-at-Risk
(expected loss in the worst q% of outcomes) is coherent (subadditive, respects diversification) unlike
VaR, and is the same risk measure used in the companion CVaR/EVT research project. But a real risk
function never relies on a single statistical measure at a single confidence level:
- **Greeks-based P&L attribution** explains WHERE the risk shows up, day by day.
- **Deterministic scenario stress testing** is distribution-free (Basel/CCAR-style).
- **Kupiec/Basel VaR backtesting** checks whether the statistical model is even well-calibrated
  out-of-sample.
- **Bootstrap confidence intervals and a paired hypothesis test** (Statistical Rigor tab) quantify how
  much of the headline CVaR gap could plausibly be simulation noise rather than a real effect.
- **Extreme Value Theory / Peaks-Over-Threshold** (Tail Risk tab) fits a Generalized Pareto Distribution
  to the tail and extrapolates smoothly to quantiles (99.9%+) far beyond what a finite historical sample
  can reliably estimate on its own.

This is the same triad-plus-two a bank's market-risk function actually uses together, not a single
statistic trusted in isolation.

**Vega risk: the second-biggest gap in a Delta-only hedge.** Every hedge model studied trades only the
underlying, which has zero Vega — a Delta-only Heston hedge structurally cannot use its own core
insight (stochastic volatility) at all. The **Vega Hedging** tab adds a second option instrument sized
to neutralize net Vega, giving the Heston hedge a fair fight, and shows honestly that the benefit
depends on how much genuine vol-of-vol risk is actually present (it helps materially for BTC, but can
even hurt for SPY, whose calibrated vol-of-vol is small) — the same "sophistication must match the
specific risk" lesson as the Heston Finding tab, one level deeper.

**Band ("no-transaction region") hedging.** Real desks rarely rebalance on a rigid calendar. Following
Whalley & Wilmott (1993), the hedging engine also supports trading only when Delta drifts outside a
tolerance band — often achieving a materially better cost/risk tradeoff than fixed-frequency
rebalancing at an equivalent average trading rate (see Efficient Frontier & Bands).

**American options: a numerical-methods capability, not just a European pricing library.** Every other
pricer in this project (BS, Heston, Merton) is European-only. The **American Options** tab adds a
Cox-Ross-Rubinstein binomial tree AND a Crank-Nicolson finite-difference PDE solver, cross-validated
against each other and against the known theoretical results (no early-exercise premium without
dividends; a genuine positive premium once dividends are introduced).

#### Engineering notes
- The batch hedging simulator (`simulate_delta_hedge_batch`) is vectorized across paths, looping only
  over time steps — validated **bit-identical** (max diff 0.00e+00) against the per-path simulator on
  200 test paths, then used for all 4,000–8,000 path CVaR studies (~1000x faster).
- **Heston pricing is now vectorized too**, via the Fourier-COSine (COS) method (Fang & Oosterlee,
  2008), which expands the same characteristic function in a finite series that vectorizes across
  spot prices using the log-price's scale-invariance — resolving what used to be this project's
  single most-repeated documented limitation. Benchmarked ~180x faster than the original
  `scipy.integrate.quad` Fourier inversion over 500 prices, and cross-validated to agree with it to
  within ~3e-4 relative error (see `heston.py`'s own CLI check). The Heston Finding and Vega Hedging
  tabs, and the headline Heston secondary comparison, now all run at the SAME full path-count / daily
  rebalancing scale as the BS and Merton studies.
- A real performance bug was found and fixed mid-build: the per-path simulator was computing full
  Heston Greeks (6 Fourier-integral evaluations) on every time step, not just rebalancing points. A
  `_price_only` fast path cut this to 1 evaluation per non-rebalancing step.
- The real-data backtest exploits the linear homogeneity of European option pricing in `(S, K)`:
  every historical window is rescaled by its own starting price, priced at a uniform `K=1.0`, and the
  resulting P&L is multiplied back by the original price — letting the vectorized simulator batch-process
  hundreds of overlapping historical windows that each have a different strike.
- The band hedging simulator (`simulate_delta_hedge_band_batch`) was validated two ways: bit-identical
  to its own per-path version, AND proven to reduce EXACTLY to the fixed-frequency simulator when the
  band width is set to zero (0.00e+00 difference) — confirming it's a strict generalisation, not a
  parallel, inconsistent implementation.
- The P&L attribution module (`pnl_attribution.py`) reuses the same scalar-or-array pricing dispatch
  for its per-path AND vectorized-batch functions (no duplicated logic), mirroring the vectorization
  philosophy used throughout the rest of the codebase.
- The Delta-Vega hedge (`simulate_delta_vega_hedge_batch`) has a genuine numerical subtlety, found and
  documented rather than hidden: every option's Vega collapses to 0 near expiry, and a fixed-strike
  instrument away from the money collapses FASTER than an at-the-money target's, so the naive hedge
  ratio can compound to an unreasonable size well before either Vega is literally zero. Handled by
  capping the instrument quantity at a fixed multiple of its well-behaved, full-maturity value —
  approximating how a real desk would roll to a fresher instrument rather than lever into a threadbare one.
- Monte Carlo variance reduction (`variance_reduction.py`) demonstrates antithetic variates and control
  variates cut Merton Monte Carlo pricing standard error by 10-50% at equal simulation budget, and that
  importance sampling via Poisson jump-rate exponential tilting gives a materially more precise deep-tail
  (99%+) CVaR estimate at equal path count — with an explicit, honest note that the tilting strength is
  itself a bias-variance tradeoff (too much tilting causes importance-weight degeneracy), not a free
  parameter to maximize.

#### Skills demonstrated, by module
| Area | Modules | Topics |
|---|---|---|
| Stochastic calculus / Fourier methods | `heston.py`, `merton.py` | Characteristic functions, Gil-Pelaez inversion, the COS method, compound Poisson jump-diffusions |
| Numerical PDE / tree methods | `american_options.py` | Crank-Nicolson finite differences, CRR binomial trees, free-boundary (early-exercise) problems |
| Statistics / extreme value theory | `statistical_inference.py`, `tail_risk.py` | Nonparametric bootstrap, paired hypothesis testing, Generalized Pareto tail fitting, Hill's estimator |
| Monte Carlo methods | `variance_reduction.py` | Antithetic & control variates, importance sampling / exponential tilting, likelihood-ratio reweighting |
| Financial engineering | `hedging_engine.py`, `risk_analysis.py`, `pnl_attribution.py` | Delta/Vega hedging, replication arguments, Greeks-based P&L explain, transaction-cost tradeoffs |
| Risk management practice | `stress_testing.py`, `model_validation.py` | Basel/CCAR-style scenario design, Kupiec POF testing, traffic-light VaR backtesting |
| Software / numerical engineering | all modules | Vectorization, bit-identical cross-validation between independent implementations, documented performance profiling |

#### Data
SPY and BTC-USD daily closes, 2018–2025, via Yahoo Finance (historical calibration); live SPY options
chain via Yahoo Finance (Live Market Check tab only). See `README.md` and `report/report.md` in the
project repository for the full technical write-up, all figures, and all result tables.
""")
