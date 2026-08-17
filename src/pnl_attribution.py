"""
pnl_attribution.py
-------------------
Greeks-based "P&L explain" — a genuine daily front-office / market-risk
practice: decompose the change in an option's mark-to-model value into
the pieces a trading desk actually reports to management:

    ΔV  ≈  Delta·ΔS  +  ½·Gamma·ΔS²  +  Theta·Δt  +  residual

The first three terms are exactly what the hedge model's own Greeks
predict (a 2nd-order Taylor expansion in the underlying and time). The
`residual` term is whatever the model's Greeks could NOT explain: higher
order price moves, and — critically for this project — actual price
JUMPS, which no smooth Taylor expansion can ever anticipate.

This turns the project's headline finding (BS hedge has higher CVaR
under a jumpy market) into something more actionable for a real risk
desk: it shows WHERE the extra risk actually shows up on the P&L
explain, step by step, rather than only in a single terminal number.

Both a per-path (any hedge model, including Heston) and a vectorized
batch (bs/merton only, matching the vectorization pattern used
throughout this project) version are provided.
"""

import numpy as np
import pandas as pd

from black_scholes import bs_price, bs_greeks
from heston import heston_price, heston_greeks_fd
from merton import merton_price


def _price_vec(model: str, S, K: float, T: float, r: float, option_type: str, params: dict):
    """Model-dispatch pricing that works for scalar OR array-valued S."""
    if T <= 1e-8:
        S_arr = np.asarray(S, dtype=float)
        if option_type == "call":
            payoff = np.where(S_arr > K, S_arr - K, 0.0)
        else:
            payoff = np.where(K > S_arr, K - S_arr, 0.0)
        return float(payoff) if payoff.ndim == 0 else payoff

    if model == "bs":
        return bs_price(S, K, T, r, params["sigma"], option_type)
    elif model == "merton":
        return merton_price(S, K, T, r, params["sigma"], params["lam"],
                             params["mu_j"], params["sigma_j"], option_type)
    elif model == "heston":
        return heston_price(S, K, T, r, params["kappa"], params["theta"],
                             params["xi"], params["rho"], params["v0"], option_type)
    else:
        raise ValueError(f"Unknown model '{model}'. Use 'bs', 'heston', or 'merton'.")


def _price_delta_gamma_vec(model: str, S, K: float, T: float, r: float,
                            option_type: str, params: dict):
    """Model-dispatch price + Delta + Gamma, scalar OR array-valued S."""
    if T <= 1e-8:
        price = _price_vec(model, S, K, T, r, option_type, params)
        zeros = np.zeros_like(np.asarray(S, dtype=float))
        zeros = float(zeros) if zeros.ndim == 0 else zeros
        return price, zeros, zeros

    if model == "bs":
        price = bs_price(S, K, T, r, params["sigma"], option_type)
        g = bs_greeks(S, K, T, r, params["sigma"], option_type)
        return price, g["delta"], g["gamma"]
    elif model == "heston":
        price = heston_price(S, K, T, r, params["kappa"], params["theta"],
                              params["xi"], params["rho"], params["v0"], option_type)
        g = heston_greeks_fd(S, K, T, r, params["kappa"], params["theta"],
                              params["xi"], params["rho"], params["v0"], option_type)
        return price, g["delta"], g["gamma"]
    elif model == "merton":
        price = merton_price(S, K, T, r, params["sigma"], params["lam"],
                              params["mu_j"], params["sigma_j"], option_type)
        h = 0.01 * np.asarray(S, dtype=float)
        p_up = merton_price(S + h, K, T, r, params["sigma"], params["lam"],
                             params["mu_j"], params["sigma_j"], option_type)
        p_down = merton_price(S - h, K, T, r, params["sigma"], params["lam"],
                               params["mu_j"], params["sigma_j"], option_type)
        delta = (p_up - p_down) / (2 * h)
        gamma = (p_up - 2 * price + p_down) / (h ** 2)
        return price, delta, gamma
    else:
        raise ValueError(f"Unknown model '{model}'. Use 'bs', 'heston', or 'merton'.")


def compute_pnl_attribution(price_path: np.ndarray, K: float, r: float, option_type: str,
                             hedge_model: str, hedge_params: dict, dt: float) -> pd.DataFrame:
    """
    Per-step Greeks-based P&L attribution along ONE price path, for ANY
    hedge model (bs, heston, or merton).

    Returns
    -------
    pd.DataFrame, one row per time step, with columns:
        step, S, dS, actual_dV, delta_pnl, gamma_pnl, theta_pnl, residual_pnl,
        and cumulative versions (cum_actual_dV, cum_delta_pnl, ...).
    `cum_actual_dV` at the last row equals the option's total mark-to-model
    value change over its life (price at maturity minus initial premium).
    """
    price_path = np.asarray(price_path, dtype=float)
    n_steps = len(price_path) - 1
    T_total = n_steps * dt

    rows = []
    for t in range(n_steps):
        S_t, S_next = float(price_path[t]), float(price_path[t + 1])
        T_t = T_total - t * dt
        T_next = max(T_total - (t + 1) * dt, 0.0)

        price_t, delta_t, gamma_t = _price_delta_gamma_vec(hedge_model, S_t, K, T_t, r,
                                                             option_type, hedge_params)
        price_next = _price_vec(hedge_model, S_next, K, T_next, r, option_type, hedge_params)
        price_t_decayed = _price_vec(hedge_model, S_t, K, T_next, r, option_type, hedge_params)

        dS = S_next - S_t
        delta_pnl = float(delta_t) * dS
        gamma_pnl = 0.5 * float(gamma_t) * dS ** 2
        theta_pnl = float(price_t_decayed) - float(price_t)
        predicted = delta_pnl + gamma_pnl + theta_pnl
        actual = float(price_next) - float(price_t)
        residual = actual - predicted

        rows.append({
            "step": t + 1, "S": S_next, "dS": dS, "actual_dV": actual,
            "delta_pnl": delta_pnl, "gamma_pnl": gamma_pnl, "theta_pnl": theta_pnl,
            "residual_pnl": residual,
        })

    df = pd.DataFrame(rows)
    for col in ["actual_dV", "delta_pnl", "gamma_pnl", "theta_pnl", "residual_pnl"]:
        df[f"cum_{col}"] = df[col].cumsum()
    return df


def compute_pnl_attribution_batch(price_paths: np.ndarray, K: float, r: float, option_type: str,
                                   hedge_model: str, hedge_params: dict, dt: float) -> pd.DataFrame:
    """
    Vectorized version of compute_pnl_attribution() across an entire batch
    of paths (bs/merton hedge models only — Heston pricing cannot be
    vectorized the same way). Returns only the path-level CUMULATIVE
    totals of each attribution bucket (not the full per-step detail, which
    would be a lot of data across thousands of paths) — exactly what's
    needed to see the DISTRIBUTION of "unexplained" (residual) P&L across
    many simulated outcomes.

    Returns
    -------
    pd.DataFrame, shape (n_paths, 4): cum_delta_pnl, cum_gamma_pnl,
    cum_theta_pnl, cum_residual_pnl.
    """
    if hedge_model not in ("bs", "merton"):
        raise ValueError("Batch attribution only supports 'bs' and 'merton' hedge models "
                          "(Heston pricing is not vectorizable this way).")

    price_paths = np.asarray(price_paths, dtype=float)
    n_paths, n_plus1 = price_paths.shape
    n_steps = n_plus1 - 1
    T_total = n_steps * dt

    cum_delta = np.zeros(n_paths)
    cum_gamma = np.zeros(n_paths)
    cum_theta = np.zeros(n_paths)
    cum_residual = np.zeros(n_paths)

    for t in range(n_steps):
        S_t = price_paths[:, t]
        S_next = price_paths[:, t + 1]
        T_t = T_total - t * dt
        T_next = max(T_total - (t + 1) * dt, 0.0)

        price_t, delta_t, gamma_t = _price_delta_gamma_vec(hedge_model, S_t, K, T_t, r,
                                                             option_type, hedge_params)
        price_next = _price_vec(hedge_model, S_next, K, T_next, r, option_type, hedge_params)
        price_t_decayed = _price_vec(hedge_model, S_t, K, T_next, r, option_type, hedge_params)

        dS = S_next - S_t
        delta_pnl = delta_t * dS
        gamma_pnl = 0.5 * gamma_t * dS ** 2
        theta_pnl = price_t_decayed - price_t
        predicted = delta_pnl + gamma_pnl + theta_pnl
        actual = price_next - price_t
        residual = actual - predicted

        cum_delta += delta_pnl
        cum_gamma += gamma_pnl
        cum_theta += theta_pnl
        cum_residual += residual

    return pd.DataFrame({
        "cum_delta_pnl": cum_delta, "cum_gamma_pnl": cum_gamma,
        "cum_theta_pnl": cum_theta, "cum_residual_pnl": cum_residual,
    })


# ══════════════════════════════════════════════════════════════════════════════
# CLI Sanity Check
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Sanity check: attribute P&L for a BS hedge along (a) a smooth GBM path
    (no jumps — the model matches reality) and (b) a path with a large
    jump (the model does NOT match reality). The residual bucket should
    be small and unremarkable in (a), and should show one large spike
    exactly at the jump date in (b) — precisely the "hidden" risk this
    project's headline result quantifies in aggregate, now visible at the
    level of an individual day's P&L explain.
    Run from repo root: python src/pnl_attribution.py
    """
    print("P&L attribution sanity check\n" + "-" * 40)

    S0, K, r, sigma, T = 100.0, 100.0, 0.03, 0.20, 0.25
    n_steps = 63
    dt = T / n_steps
    hedge_params = {"sigma": sigma}

    rng = np.random.default_rng(7)

    print("[1] Smooth GBM path (BS hedge matches reality)...")
    z = rng.standard_normal(n_steps)
    log_path = np.log(S0) + np.cumsum((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z)
    smooth_path = np.concatenate([[S0], np.exp(log_path)])
    attr_smooth = compute_pnl_attribution(smooth_path, K, r, "call", "bs", hedge_params, dt)
    print(f"    Sum |residual| over path : {attr_smooth['residual_pnl'].abs().sum():.4f}")
    print(f"    Max |single-day residual|: {attr_smooth['residual_pnl'].abs().max():.4f}")

    print("\n[2] Same path but with one large jump inserted mid-life...")
    jumpy_path = smooth_path.copy()
    jump_day = n_steps // 2
    jumpy_path[jump_day:] *= 0.85   # -15% jump, propagated through the rest of the path
    attr_jumpy = compute_pnl_attribution(jumpy_path, K, r, "call", "bs", hedge_params, dt)
    worst_day = attr_jumpy["residual_pnl"].idxmin()
    print(f"    Sum |residual| over path : {attr_jumpy['residual_pnl'].abs().sum():.4f}")
    print(f"    Largest single-day residual occurs at step "
          f"{attr_jumpy.loc[worst_day, 'step']:.0f} (jump was inserted at step {jump_day})")
    print(f"    Value of that day's residual: {attr_jumpy.loc[worst_day, 'residual_pnl']:.4f}")

    assert attr_jumpy["residual_pnl"].abs().max() > 5 * attr_smooth["residual_pnl"].abs().max(), \
        "The jump day's residual should dwarf any residual on the smooth path"
    assert abs(attr_jumpy.loc[worst_day, "step"] - jump_day) <= 1, \
        "The largest residual should occur at (or right next to) the actual jump date"
    print("\n    Confirmed: the Greeks-based P&L explain correctly isolates the jump")
    print("    as a large 'unexplained' residual on exactly the day it occurs — this")
    print("    is the same hidden risk the project's headline CVaR result quantifies,")
    print("    now visible at the level of a single day's P&L attribution.")

    print("\n[3] Cross-validating vectorized batch attribution vs. per-path...")
    n_check = 100
    z_batch = rng.standard_normal((n_check, n_steps))
    log_paths = np.log(S0) + np.cumsum(
        (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z_batch, axis=1)
    paths_batch = np.concatenate([np.full((n_check, 1), S0), np.exp(log_paths)], axis=1)

    cum_residual_perpath = np.array([
        compute_pnl_attribution(paths_batch[i], K, r, "call", "bs", hedge_params, dt)
        ["residual_pnl"].sum()
        for i in range(n_check)
    ])
    attr_batch = compute_pnl_attribution_batch(paths_batch, K, r, "call", "bs", hedge_params, dt)
    max_diff = np.max(np.abs(cum_residual_perpath - attr_batch["cum_residual_pnl"].values))
    print(f"    Max |per-path - vectorized| cumulative residual difference: {max_diff:.2e}")
    assert max_diff < 1e-6, "Vectorized batch attribution must match the per-path version"
    print("    Confirmed: vectorized batch attribution is a pure performance optimisation.")
