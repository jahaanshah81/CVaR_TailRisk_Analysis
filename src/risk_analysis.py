"""
risk_analysis.py
-----------------
The project's central empirical contribution: quantifying how much tail
risk a market maker unknowingly carries when they hedge using a
simplified model of the world.

We hold the TRUE data-generating process fixed (e.g., Merton
jump-diffusion — a market that actually has crash risk) and vary only
the HEDGE MODEL the market maker uses to compute Delta (BS, Heston, or
Merton). For each combination we build an empirical distribution of
terminal hedging P&L across many independent simulated paths, then
compute the Conditional Value-at-Risk (CVaR) of the hedging LOSS
distribution — directly reusing the same risk measure studied in the
companion CVaR/EVT research project, now applied to a market maker's
hedging book instead of a buy-and-hold return series.

The central hypothesis this tests: a market maker who hedges with
Black-Scholes Delta in a market that actually jumps will have a hedging
P&L distribution with a MUCH heavier left tail (higher CVaR) than their
BS-based risk model would predict — because BS Delta cannot see jump
risk at all. Quantifying this gap in dollar terms is exactly the kind
of model-risk analysis a real derivatives risk management function
performs.
"""

import numpy as np
import pandas as pd

from merton import merton_simulate_paths
from heston import heston_simulate_paths
from hedging_engine import (
    simulate_delta_hedge, simulate_delta_hedge_batch,
    simulate_delta_hedge_band, simulate_delta_hedge_band_batch,
)


def cvar_historical(losses: np.ndarray, q: float = 0.05) -> float:
    """
    Historical-simulation CVaR: mean of the worst q-fraction of losses.
    (Identical convention to the companion CVaR/EVT project: loss > 0 is
    bad. Here, loss = -terminal_hedging_pnl, so a large loss means the
    market maker lost money hedging the option.)
    """
    losses = np.asarray(losses, dtype=float).ravel()
    n = len(losses)
    sorted_losses = np.sort(losses)[::-1]
    k = max(1, int(np.floor(q * n)))
    return float(sorted_losses[:k].mean())


def var_historical(losses: np.ndarray, q: float = 0.05) -> float:
    """Historical-simulation VaR: the q-quantile of the loss distribution."""
    losses = np.asarray(losses, dtype=float).ravel()
    return float(np.quantile(losses, 1.0 - q))


def build_hedging_pnl_distribution(true_paths: np.ndarray, K: float, r: float,
                                    option_type: str, hedge_model: str,
                                    hedge_params: dict, rebalance_every: int,
                                    transaction_cost_rate: float, dt: float) -> np.ndarray:
    """
    Run the delta-hedging simulation across every row of `true_paths`
    (each row one independent realised price path) and collect the
    terminal hedging P&L from each.

    Uses the vectorized batch simulator (simulate_delta_hedge_batch) for
    all three hedge models -- "bs", "merton", AND (since heston.py added
    the COS-method Fourier pricer) "heston" -- orders of magnitude faster
    than a Python loop over thousands of paths. A per-path fallback is
    kept below for any OTHER future hedge model that isn't batch-vectorized.

    Returns
    -------
    np.ndarray of shape (n_paths,), one terminal P&L per path.
    """
    if hedge_model in ("bs", "merton", "heston"):
        return simulate_delta_hedge_batch(
            true_paths, K, r, option_type, hedge_model, hedge_params,
            rebalance_every=rebalance_every,
            transaction_cost_rate=transaction_cost_rate, dt=dt,
        )

    pnls = np.empty(true_paths.shape[0])
    for i in range(true_paths.shape[0]):
        result = simulate_delta_hedge(
            true_paths[i], K, r, option_type, hedge_model, hedge_params,
            rebalance_every=rebalance_every,
            transaction_cost_rate=transaction_cost_rate, dt=dt,
        )
        pnls[i] = result.attrs["terminal_pnl"]
    return pnls


def build_band_hedging_pnl_and_cost(true_paths: np.ndarray, K: float, r: float,
                                     option_type: str, hedge_model: str, hedge_params: dict,
                                     band_width: float, transaction_cost_rate: float,
                                     dt: float) -> tuple:
    """
    Same purpose as build_hedging_pnl_distribution(), but for the band
    ("no-transaction region") hedging strategy — returns BOTH the terminal
    P&L per path AND the total transaction cost paid per path, so the
    cost/risk tradeoff of band hedging can be compared directly against
    fixed-frequency rebalancing.
    """
    if hedge_model in ("bs", "merton", "heston"):
        return simulate_delta_hedge_band_batch(
            true_paths, K, r, option_type, hedge_model, hedge_params,
            band_width=band_width, transaction_cost_rate=transaction_cost_rate, dt=dt,
        )

    pnls = np.empty(true_paths.shape[0])
    costs = np.empty(true_paths.shape[0])
    for i in range(true_paths.shape[0]):
        result = simulate_delta_hedge_band(
            true_paths[i], K, r, option_type, hedge_model, hedge_params,
            band_width=band_width, transaction_cost_rate=transaction_cost_rate, dt=dt,
        )
        pnls[i] = result.attrs["terminal_pnl"]
        costs[i] = result.attrs["total_transaction_cost"]
    return pnls, costs


def compare_delta_vs_delta_vega_hedge(
        true_model_paths: np.ndarray, K_target: float, K_instrument: float, r: float,
        option_type: str, hedge_model: str, hedge_params: dict, rebalance_every: int,
        transaction_cost_rate: float, dt: float, q_levels: tuple = (0.05, 0.01)) -> pd.DataFrame:
    """
    Compares Delta-only hedging against Delta-Vega (two-instrument)
    hedging for a SINGLE hedge model, on the SAME true price paths --
    isolating the effect of adding a Vega hedge instrument, holding
    everything else (true dynamics, hedge model beliefs, rebalancing
    frequency, transaction costs) fixed. Most relevant for "heston",
    whose entire modelling premise (stochastic volatility) a Delta-only
    hedge can never actually use.

    Returns
    -------
    pd.DataFrame, one row per strategy ("Delta-only", "Delta-Vega"), with
    the same summary-statistic columns as compare_hedge_models_under_true_dgp.
    """
    from hedging_engine import simulate_delta_vega_hedge_batch

    pnls_delta = build_hedging_pnl_distribution(
        true_model_paths, K_target, r, option_type, hedge_model, hedge_params,
        rebalance_every, transaction_cost_rate, dt,
    )
    pnls_vega, costs_vega = simulate_delta_vega_hedge_batch(
        true_model_paths, K_target, K_instrument, r, option_type, hedge_model, hedge_params,
        rebalance_every=rebalance_every, transaction_cost_rate=transaction_cost_rate, dt=dt,
    )

    rows = []
    for name, pnls, mean_cost in (("Delta-only", pnls_delta, np.nan),
                                   ("Delta-Vega", pnls_vega, float(costs_vega.mean()))):
        losses = -pnls
        row = {
            "strategy": name, "mean_pnl": float(pnls.mean()), "std_pnl": float(pnls.std(ddof=1)),
            "mean_transaction_cost": mean_cost,
        }
        for q in q_levels:
            row[f"VaR_{int(q*100)}pct"] = var_historical(losses, q)
            row[f"CVaR_{int(q*100)}pct"] = cvar_historical(losses, q)
        rows.append(row)
    return pd.DataFrame(rows)


def compare_hedge_models_under_true_dgp(
        true_model_paths: np.ndarray, K: float, r: float, option_type: str,
        hedge_models: dict, rebalance_every: int, transaction_cost_rate: float,
        dt: float, q_levels: tuple = (0.05, 0.01)) -> pd.DataFrame:
    """
    The project's headline experiment: fix the TRUE price paths (however
    they were generated), and sweep across several candidate HEDGE
    MODELS, reporting the resulting hedging-P&L distribution's mean,
    std, and CVaR at each q-level for each hedge model.

    Parameters
    ----------
    true_model_paths : np.ndarray, shape (n_paths, n_steps+1)
        Price paths from the assumed-true data-generating process.
    hedge_models : dict
        {model_name: (model_type_str, params_dict)}, e.g.
        {"BS hedge": ("bs", {"sigma": 0.2}),
         "Heston hedge": ("heston", {...}),
         "Merton hedge": ("merton", {...})}

    Returns
    -------
    pd.DataFrame, one row per hedge model, with P&L distribution summary
    statistics and CVaR at each requested q-level.
    """
    rows = []
    for name, (model_type, params) in hedge_models.items():
        pnls = build_hedging_pnl_distribution(
            true_model_paths, K, r, option_type, model_type, params,
            rebalance_every, transaction_cost_rate, dt,
        )
        losses = -pnls   # loss convention: positive = market maker lost money

        row = {
            "hedge_model": name,
            "mean_pnl": float(pnls.mean()),
            "std_pnl": float(pnls.std(ddof=1)),
            "worst_loss": float(losses.max()),
        }
        for q in q_levels:
            row[f"VaR_{int(q*100)}pct"] = var_historical(losses, q)
            row[f"CVaR_{int(q*100)}pct"] = cvar_historical(losses, q)
        rows.append(row)

    return pd.DataFrame(rows)


def rebalancing_efficient_frontier(
        true_model_paths: np.ndarray, K: float, r: float, option_type: str,
        hedge_model: str, hedge_params: dict, rebalance_frequencies: list,
        transaction_cost_rate: float, dt: float, q: float = 0.05) -> pd.DataFrame:
    """
    The classic cost-vs-risk tradeoff: sweep over rebalancing frequencies
    and report total expected transaction costs vs. CVaR of hedging loss
    at each frequency, holding the hedge model and true DGP fixed.

    Returns
    -------
    pd.DataFrame, one row per rebalancing frequency.
    """
    rows = []
    for freq in rebalance_frequencies:
        pnls = build_hedging_pnl_distribution(
            true_model_paths, K, r, option_type, hedge_model, hedge_params,
            freq, transaction_cost_rate, dt,
        )
        losses = -pnls
        rows.append({
            "rebalance_every_n_steps": freq,
            "std_pnl": float(pnls.std(ddof=1)),
            f"CVaR_{int(q*100)}pct": cvar_historical(losses, q),
            "mean_pnl": float(pnls.mean()),
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# CLI Sanity Check
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Small-scale demonstration: simulate a "true" market with Merton jumps,
    then compare BS-hedge vs. Merton-hedge CVaR of the resulting hedging
    losses. We expect the BS hedge to show materially higher CVaR, since
    it cannot see jump risk at all.
    Run from repo root: python src/risk_analysis.py
    """
    print("Risk analysis sanity check\n" + "-" * 40)

    S0, K, r, T = 100.0, 100.0, 0.03, 0.25
    n_steps = 63
    n_paths = 2000   # small demo scale; run_experiment.py uses more

    # "True" market: Merton jump-diffusion with meaningful crash risk
    true_sigma, true_lam, true_mu_j, true_sigma_j = 0.18, 2.0, -0.08, 0.10
    true_paths = merton_simulate_paths(
        S0, T, r, true_sigma, true_lam, true_mu_j, true_sigma_j,
        n_paths=n_paths, n_steps=n_steps, seed=7,
    )

    hedge_models = {
        "BS hedge (blind to jumps)": ("bs", {"sigma": true_sigma}),
        "Merton hedge (jump-aware)": ("merton", {
            "sigma": true_sigma, "lam": true_lam, "mu_j": true_mu_j, "sigma_j": true_sigma_j,
        }),
    }

    summary = compare_hedge_models_under_true_dgp(
        true_paths, K, r, "call", hedge_models,
        rebalance_every=1, transaction_cost_rate=0.0005, dt=T / n_steps,
    )
    print(summary.to_string(index=False))

    cvar_bs = summary.loc[summary["hedge_model"].str.contains("BS"), "CVaR_5pct"].iloc[0]
    cvar_merton = summary.loc[summary["hedge_model"].str.contains("Merton"), "CVaR_5pct"].iloc[0]
    print(f"\nBS-hedge CVaR(5%)     : {cvar_bs:.4f}")
    print(f"Merton-hedge CVaR(5%) : {cvar_merton:.4f}")
    print(f"BS hedge shows {'HIGHER' if cvar_bs > cvar_merton else 'LOWER'} tail risk "
          f"than the jump-aware hedge, as hypothesised: {cvar_bs > cvar_merton}")

    print("\nDelta-only vs. Delta-Vega hedging (Heston hedge model, same true Merton market)...")
    heston_params_demo = {"kappa": 2.0, "theta": true_sigma ** 2, "xi": 0.5, "rho": -0.6, "v0": true_sigma ** 2}
    dv_summary = compare_delta_vs_delta_vega_hedge(
        true_paths, K, K * 1.10, r, "call", "heston", heston_params_demo,
        rebalance_every=5, transaction_cost_rate=0.0005, dt=T / n_steps,
    )
    print(dv_summary.to_string(index=False))
    print("\n(Illustrative here since the true market is Merton, not Heston -- see the dashboard's")
    print("Vega Hedging tab and report.md for the full stochastic-volatility comparison where Delta-Vega")
    print("hedging is expected to help the most.)")
