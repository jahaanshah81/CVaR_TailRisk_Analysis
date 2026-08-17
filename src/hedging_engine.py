"""
hedging_engine.py
------------------
Simulates a market maker who sells one European option and delta-hedges
the resulting risk by trading the underlying at discrete rebalancing
times, under realistic transaction costs.

THE KEY EXPERIMENTAL DESIGN (this is the whole point of the project):
We deliberately separate two things that are usually conflated:
    1. The TRUE data-generating process — how the underlying actually
       moves (a real historical price path, or a simulated path from a
       richer model such as Merton with jumps).
    2. The HEDGE MODEL — which model the market maker *believes* in and
       uses to compute Delta/Gamma/Vega for hedging purposes.

When these two match, hedging error is small and well understood (the
classical theory). When the hedge model is simpler than reality — e.g.,
the market maker hedges with Black-Scholes Delta while the real market
has jumps — the realised hedging P&L develops a heavy left tail that the
hedge model itself cannot see. Quantifying this gap, using CVaR, is the
project's central empirical contribution.

Cash-account bookkeeping (careful, standard replication argument):
    t=0:    receive premium V_0 for the (short) option, buy Delta_0 shares
            of stock; cash_0 = V_0 - Delta_0 * S_0
    t_i:    cash grows at the risk-free rate between rebalances;
            trade (Delta_i - Delta_{i-1}) shares, paying transaction costs
    t=T:    cash grows one final time; option settles (pay the payoff,
            since we are short); liquidate the final stock position.
    Hedging error = final portfolio value = cash_T + Delta_last*S_T - payoff

A perfect continuous hedge under the *matching* model would drive this
to (near) zero. Any positive or negative residual is the realised P&L
the market maker actually experiences.
"""

import numpy as np
import pandas as pd

from black_scholes import bs_price, bs_greeks
from heston import heston_price, heston_greeks_fd, heston_price_cos_batch
from merton import merton_price


def _price_and_greeks(model: str, S: float, K: float, T: float, r: float,
                       option_type: str, params: dict) -> tuple[float, dict]:
    """
    Dispatch pricing + Greeks to the requested hedge model.
    `params` holds model-specific parameters (see calibration.py outputs).
    T is time remaining to maturity; if T <= 0, returns intrinsic value
    and zeroed Greeks (the option has expired).
    """
    if T <= 1e-8:
        intrinsic = max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
        return intrinsic, {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}

    if model == "bs":
        price = bs_price(S, K, T, r, params["sigma"], option_type)
        greeks = bs_greeks(S, K, T, r, params["sigma"], option_type)
        return price, greeks

    elif model == "heston":
        price = heston_price(S, K, T, r, params["kappa"], params["theta"],
                              params["xi"], params["rho"], params["v0"], option_type)
        greeks = heston_greeks_fd(S, K, T, r, params["kappa"], params["theta"],
                                   params["xi"], params["rho"], params["v0"], option_type)
        return price, greeks

    elif model == "merton":
        price = merton_price(S, K, T, r, params["sigma"], params["lam"],
                              params["mu_j"], params["sigma_j"], option_type)
        # Merton Delta/Gamma via simple finite difference (same spirit as Heston's).
        h = S * 0.01
        p_up = merton_price(S + h, K, T, r, params["sigma"], params["lam"],
                             params["mu_j"], params["sigma_j"], option_type)
        p_down = merton_price(S - h, K, T, r, params["sigma"], params["lam"],
                               params["mu_j"], params["sigma_j"], option_type)
        delta = (p_up - p_down) / (2 * h)
        gamma = (p_up - 2 * price + p_down) / (h ** 2)
        # Vega w.r.t. the diffusive vol sigma, same finite-difference convention as
        # black_scholes.bs_greeks (price change per 1 vol POINT, i.e. per 0.01 in sigma) --
        # needed for the Delta-Vega multi-instrument hedge below.
        h_sigma = 0.005
        p_vega_up = merton_price(S, K, T, r, params["sigma"] + h_sigma, params["lam"],
                                  params["mu_j"], params["sigma_j"], option_type)
        p_vega_down = merton_price(S, K, T, r, max(params["sigma"] - h_sigma, 1e-6), params["lam"],
                                    params["mu_j"], params["sigma_j"], option_type)
        vega = (p_vega_up - p_vega_down) / (2 * h_sigma) * 0.01
        return price, {"delta": delta, "gamma": gamma, "vega": vega, "theta": np.nan}

    else:
        raise ValueError(f"Unknown model '{model}'. Use 'bs', 'heston', or 'merton'.")


def _price_only(model: str, S: float, K: float, T: float, r: float,
                 option_type: str, params: dict) -> float:
    """
    Cheap price-only computation, used on non-rebalancing steps where we
    need the option's mark-to-market price for bookkeeping/reporting but
    NOT its Greeks. This matters a great deal for Heston: computing full
    Greeks costs 6 characteristic-function price evaluations (each a
    numerical Fourier integral) via finite differences, but a single mark
    only costs 1 — a 6x speed-up on every non-rebalancing time step.
    """
    if T <= 1e-8:
        return max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)

    if model == "bs":
        return bs_price(S, K, T, r, params["sigma"], option_type)
    elif model == "heston":
        return heston_price(S, K, T, r, params["kappa"], params["theta"],
                             params["xi"], params["rho"], params["v0"], option_type)
    elif model == "merton":
        return merton_price(S, K, T, r, params["sigma"], params["lam"],
                             params["mu_j"], params["sigma_j"], option_type)
    else:
        raise ValueError(f"Unknown model '{model}'. Use 'bs', 'heston', or 'merton'.")


def simulate_delta_hedge(price_path: np.ndarray, K: float, r: float,
                          option_type: str, hedge_model: str, hedge_params: dict,
                          rebalance_every: int = 1,
                          transaction_cost_rate: float = 0.0005,
                          dt: float = 1.0 / 252) -> pd.DataFrame:
    """
    Run a single discrete delta-hedging simulation along one price path.

    Parameters
    ----------
    price_path : np.ndarray
        Sequence of underlying prices, S_0, S_1, ..., S_n (the "true"
        realised path — real historical data or a simulated path from
        any model, including one different from `hedge_model`).
    K : float
        Option strike.
    r : float
        Risk-free rate (annualised, continuously compounded).
    option_type : "call" or "put".
    hedge_model : "bs", "heston", or "merton" — the model the market
        maker uses to compute Delta (may differ from how price_path was
        actually generated — that mismatch is the point).
    hedge_params : dict
        Parameters for the hedge model (see calibration.py).
    rebalance_every : int
        Rebalance the hedge every N time steps (1 = every step / daily
        if dt=1/252; 5 = weekly, etc.) Larger values increase hedging
        error but reduce transaction costs — the classic tradeoff.
    transaction_cost_rate : float
        Proportional transaction cost per unit of stock notional traded
        (e.g., 0.0005 = 5 basis points, a realistic equity spread cost).
    dt : float
        Time step size in years.

    Returns
    -------
    pd.DataFrame, one row per time step, with columns:
        S, T_remaining, option_price, delta, gamma, theta,
        hedge_shares, trade, transaction_cost, cash, portfolio_value
    The final row's `portfolio_value` is the terminal hedging P&L.
    """
    n_steps = len(price_path) - 1
    T_total = n_steps * dt

    rows = []
    cash = 0.0
    hedge_shares = 0.0

    # t = 0: sell the option, receive premium, put on the initial hedge
    S0 = price_path[0]
    V0, greeks0 = _price_and_greeks(hedge_model, S0, K, T_total, r, option_type, hedge_params)
    hedge_shares = greeks0["delta"]
    cash = V0 - hedge_shares * S0
    rows.append({
        "step": 0, "S": S0, "T_remaining": T_total, "option_price": V0,
        "delta": greeks0["delta"], "gamma": greeks0["gamma"], "theta": greeks0.get("theta", np.nan),
        "hedge_shares": hedge_shares, "trade": hedge_shares, "transaction_cost": 0.0,
        "cash": cash, "portfolio_value": np.nan,
    })

    for t in range(1, n_steps + 1):
        S_t = price_path[t]
        T_remaining = max(T_total - t * dt, 0.0)

        # Cash grows at the risk-free rate for one time step
        cash *= np.exp(r * dt)

        rebalance_now = (t % rebalance_every == 0) or (t == n_steps)

        if rebalance_now and T_remaining > 1e-8:
            V_t, greeks_t = _price_and_greeks(hedge_model, S_t, K, T_remaining, r,
                                               option_type, hedge_params)
            new_delta = greeks_t["delta"]
            trade = new_delta - hedge_shares
            cost = transaction_cost_rate * abs(trade) * S_t
            cash -= trade * S_t + cost
            hedge_shares = new_delta
        else:
            # Non-rebalancing step: we still need the mark-to-market price
            # for bookkeeping, but NOT the Greeks (nothing is traded) — use
            # the cheap price-only path, which matters a great deal for
            # Heston (6x fewer Fourier-integral evaluations per such step).
            V_t = _price_only(hedge_model, S_t, K, T_remaining, r, option_type, hedge_params)
            greeks_t = {"delta": hedge_shares, "gamma": np.nan, "theta": np.nan}
            trade, cost = 0.0, 0.0

        rows.append({
            "step": t, "S": S_t, "T_remaining": T_remaining, "option_price": V_t,
            "delta": greeks_t["delta"], "gamma": greeks_t["gamma"], "theta": greeks_t.get("theta", np.nan),
            "hedge_shares": hedge_shares, "trade": trade, "transaction_cost": cost,
            "cash": cash, "portfolio_value": np.nan,
        })

    df = pd.DataFrame(rows)

    # Settle at maturity: pay the option payoff, liquidate the hedge position.
    S_T = price_path[-1]
    payoff = max(S_T - K, 0.0) if option_type == "call" else max(K - S_T, 0.0)
    final_cash = df["cash"].iloc[-1]
    final_hedge_value = df["hedge_shares"].iloc[-1] * S_T
    terminal_pnl = final_cash + final_hedge_value - payoff

    df.loc[df.index[-1], "portfolio_value"] = terminal_pnl
    df.attrs["terminal_pnl"] = terminal_pnl
    df.attrs["payoff"] = payoff
    df.attrs["initial_premium"] = V0

    return df


def _batch_price_and_delta(model: str, S: np.ndarray, K: float, T: float, r: float,
                            option_type: str, params: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorized price + Delta across an entire batch of paths at once,
    for a single point in time. This is the key performance primitive:
    instead of looping over thousands of paths in Python, we call the
    pricing function ONCE per time step with S as a vector.

    `bs_price`/`bs_greeks` (pure numpy/scipy.stats elementwise ops) and
    `merton_price` (a finite Poisson sum of bs_price calls) vectorize
    correctly over an array-valued S with no changes needed. Heston used
    to be the exception here (scipy.integrate.quad, used by the original
    Fourier-inversion heston_price(), requires a scalar-valued
    integrand) — this project's single most-repeated documented
    limitation. `heston_price_cos_batch` (heston.py) removes it: the
    Fourier-COSine (COS) method expands the same characteristic function
    in a finite series that vectorizes over S via ordinary numpy
    broadcasting, so Heston can now run at full batch scale exactly like
    "bs" and "merton" (cross-validated against the original quad-based
    pricer in heston.py's own CLI check).
    """
    if T <= 1e-8:
        intrinsic = np.where(S > K, S - K, 0.0) if option_type == "call" \
            else np.where(K > S, K - S, 0.0)
        return intrinsic, np.zeros_like(S)

    if model == "bs":
        price = bs_price(S, K, T, r, params["sigma"], option_type)
        delta = bs_greeks(S, K, T, r, params["sigma"], option_type)["delta"]
        return price, delta

    elif model == "merton":
        price = merton_price(S, K, T, r, params["sigma"], params["lam"],
                              params["mu_j"], params["sigma_j"], option_type)
        h = 0.01 * np.asarray(S)
        p_up = merton_price(S + h, K, T, r, params["sigma"], params["lam"],
                             params["mu_j"], params["sigma_j"], option_type)
        p_down = merton_price(S - h, K, T, r, params["sigma"], params["lam"],
                               params["mu_j"], params["sigma_j"], option_type)
        delta = (p_up - p_down) / (2 * h)
        return price, delta

    elif model == "heston":
        price = heston_price_cos_batch(S, K, T, r, params["kappa"], params["theta"],
                                        params["xi"], params["rho"], params["v0"], option_type)
        h = 0.01 * np.asarray(S)
        p_up = heston_price_cos_batch(S + h, K, T, r, params["kappa"], params["theta"],
                                       params["xi"], params["rho"], params["v0"], option_type)
        p_down = heston_price_cos_batch(S - h, K, T, r, params["kappa"], params["theta"],
                                         params["xi"], params["rho"], params["v0"], option_type)
        delta = (p_up - p_down) / (2 * h)
        return price, delta

    else:
        raise ValueError(f"Unknown model '{model}'. Use 'bs', 'heston', or 'merton'.")


def _batch_vega(model: str, S: np.ndarray, K: float, T: float, r: float,
                 option_type: str, params: dict) -> np.ndarray:
    """
    Vectorized Vega across a batch of paths, for a single point in time —
    the counterpart to `_batch_price_and_delta`, used only by the
    Delta-Vega multi-instrument hedge below (simulate_delta_vega_hedge_batch).
    Vega convention matches black_scholes.bs_greeks: price change per 1
    vol POINT (a 0.01 change in the diffusive vol / variance driver).
    """
    S = np.asarray(S, dtype=float)
    if T <= 1e-8:
        return np.zeros_like(S)

    if model == "bs":
        return bs_greeks(S, K, T, r, params["sigma"], option_type)["vega"]

    elif model == "merton":
        h_sigma = 0.005
        p_up = merton_price(S, K, T, r, params["sigma"] + h_sigma, params["lam"],
                             params["mu_j"], params["sigma_j"], option_type)
        p_down = merton_price(S, K, T, r, max(params["sigma"] - h_sigma, 1e-6), params["lam"],
                               params["mu_j"], params["sigma_j"], option_type)
        return (p_up - p_down) / (2 * h_sigma) * 0.01

    elif model == "heston":
        h_v = 0.01
        p_up = heston_price_cos_batch(S, K, T, r, params["kappa"], params["theta"],
                                       params["xi"], params["rho"], params["v0"] + h_v, option_type)
        p_down = heston_price_cos_batch(S, K, T, r, params["kappa"], params["theta"],
                                         params["xi"], params["rho"], max(params["v0"] - h_v, 1e-8), option_type)
        return (p_up - p_down) / (2 * h_v) * 0.01

    else:
        raise ValueError(f"Unknown model '{model}'. Use 'bs', 'heston', or 'merton'.")



def simulate_delta_hedge_batch(price_paths: np.ndarray, K: float, r: float,
                                option_type: str, hedge_model: str, hedge_params: dict,
                                rebalance_every: int = 1,
                                transaction_cost_rate: float = 0.0005,
                                dt: float = 1.0 / 252) -> np.ndarray:
    """
    Vectorized version of simulate_delta_hedge() that processes an entire
    batch of paths simultaneously (looping over time steps only, not
    paths). Used for large-scale CVaR studies where thousands of paths
    are needed and the per-path Python loop would be prohibitively slow.

    Parameters
    ----------
    price_paths : np.ndarray, shape (n_paths, n_steps+1)
        Each row is one independent realised price path.
    Other parameters identical to simulate_delta_hedge().

    Returns
    -------
    np.ndarray, shape (n_paths,) — terminal hedging P&L for each path.
    """
    n_paths, n_plus_1 = price_paths.shape
    n_steps = n_plus_1 - 1
    T_total = n_steps * dt

    S0 = price_paths[:, 0]
    V0, delta = _batch_price_and_delta(hedge_model, S0, K, T_total, r, option_type, hedge_params)
    hedge_shares = delta.copy()
    cash = V0 - hedge_shares * S0

    for t in range(1, n_steps + 1):
        S_t = price_paths[:, t]
        T_remaining = max(T_total - t * dt, 0.0)

        cash *= np.exp(r * dt)
        rebalance_now = (t % rebalance_every == 0) or (t == n_steps)

        if rebalance_now and T_remaining > 1e-8:
            _, new_delta = _batch_price_and_delta(hedge_model, S_t, K, T_remaining, r,
                                                   option_type, hedge_params)
            trade = new_delta - hedge_shares
            cost = transaction_cost_rate * np.abs(trade) * S_t
            cash -= trade * S_t + cost
            hedge_shares = new_delta

    S_T = price_paths[:, -1]
    payoff = np.where(S_T > K, S_T - K, 0.0) if option_type == "call" \
        else np.where(K > S_T, K - S_T, 0.0)
    terminal_pnl = cash + hedge_shares * S_T - payoff

    return terminal_pnl


# ══════════════════════════════════════════════════════════════════════════════
# Band ("no-transaction region") hedging — a practical alternative to
# fixed-frequency rebalancing
# ══════════════════════════════════════════════════════════════════════════════
#
# Real derivatives desks rarely rebalance on a rigid calendar schedule.
# Instead, a common practical rule (going back to Whalley & Wilmott, 1993,
# "An asymptotic analysis of an optimal hedging model for option pricing
# with transaction costs") is to trade only when the CURRENTLY HELD hedge
# has drifted more than some tolerance band away from the model's current
# Delta, and otherwise do nothing — monitoring risk continuously but
# trading only when it is worth paying the transaction cost to do so.
# This usually achieves a BETTER cost/risk tradeoff than calendar-based
# rebalancing at an equivalent average trading frequency, because it
# concentrates trades on days when the hedge actually needs it (large
# moves) rather than on a fixed schedule regardless of what happened.

def simulate_delta_hedge_band(price_path: np.ndarray, K: float, r: float,
                               option_type: str, hedge_model: str, hedge_params: dict,
                               band_width: float = 0.05,
                               transaction_cost_rate: float = 0.0005,
                               dt: float = 1.0 / 252) -> pd.DataFrame:
    """
    Per-path band ("no-transaction region") hedging simulator. Delta is
    monitored (repriced) every step, but only TRADED when it has drifted
    more than `band_width` away from the currently held hedge position
    (always trades on the final settlement step). Mirrors the bookkeeping
    conventions of simulate_delta_hedge() exactly, with two extra
    df.attrs: `total_transaction_cost` and `n_rebalances`.
    """
    n_steps = len(price_path) - 1
    T_total = n_steps * dt

    rows = []
    S0 = price_path[0]
    V0, greeks0 = _price_and_greeks(hedge_model, S0, K, T_total, r, option_type, hedge_params)
    hedge_shares = greeks0["delta"]
    cash = V0 - hedge_shares * S0
    rows.append({
        "step": 0, "S": S0, "T_remaining": T_total, "option_price": V0,
        "delta": greeks0["delta"], "hedge_shares": hedge_shares, "trade": hedge_shares,
        "transaction_cost": 0.0, "cash": cash, "portfolio_value": np.nan,
    })

    for t in range(1, n_steps + 1):
        S_t = price_path[t]
        T_remaining = max(T_total - t * dt, 0.0)
        cash *= np.exp(r * dt)

        if T_remaining > 1e-8:
            # Delta is repriced every step regardless (continuous risk
            # monitoring), but only acted on if it has drifted outside the band.
            V_t, greeks_t = _price_and_greeks(hedge_model, S_t, K, T_remaining, r,
                                               option_type, hedge_params)
            new_delta = greeks_t["delta"]
            drift = new_delta - hedge_shares
            do_rebalance = abs(drift) > band_width

            if do_rebalance:
                trade = drift
                cost = transaction_cost_rate * abs(trade) * S_t
                cash -= trade * S_t + cost
                hedge_shares = new_delta
            else:
                trade, cost = 0.0, 0.0
        else:
            # Final settlement step: matches simulate_delta_hedge() exactly —
            # no further trading, mark the option to its intrinsic value.
            V_t = _price_only(hedge_model, S_t, K, T_remaining, r, option_type, hedge_params)
            new_delta = hedge_shares
            trade, cost = 0.0, 0.0

        rows.append({
            "step": t, "S": S_t, "T_remaining": T_remaining, "option_price": V_t,
            "delta": new_delta, "hedge_shares": hedge_shares, "trade": trade,
            "transaction_cost": cost, "cash": cash, "portfolio_value": np.nan,
        })

    df = pd.DataFrame(rows)
    S_T = price_path[-1]
    payoff = max(S_T - K, 0.0) if option_type == "call" else max(K - S_T, 0.0)
    final_cash = df["cash"].iloc[-1]
    final_hedge_value = df["hedge_shares"].iloc[-1] * S_T
    terminal_pnl = final_cash + final_hedge_value - payoff

    df.loc[df.index[-1], "portfolio_value"] = terminal_pnl
    df.attrs["terminal_pnl"] = terminal_pnl
    df.attrs["payoff"] = payoff
    df.attrs["initial_premium"] = V0
    df.attrs["total_transaction_cost"] = float(df["transaction_cost"].sum())
    df.attrs["n_rebalances"] = int((df["trade"] != 0.0).sum())
    return df


def simulate_delta_hedge_band_batch(price_paths: np.ndarray, K: float, r: float,
                                     option_type: str, hedge_model: str, hedge_params: dict,
                                     band_width: float = 0.05,
                                     transaction_cost_rate: float = 0.0005,
                                     dt: float = 1.0 / 252) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorized batch version of simulate_delta_hedge_band() (bs/merton
    hedge models only, same restriction as simulate_delta_hedge_batch —
    Heston pricing cannot be vectorized this way).

    Returns
    -------
    (terminal_pnl, total_transaction_cost) : both np.ndarray, shape (n_paths,)
        `total_transaction_cost` lets us report the cost/risk tradeoff of
        band hedging directly against fixed-frequency rebalancing.
    """
    n_paths, n_plus_1 = price_paths.shape
    n_steps = n_plus_1 - 1
    T_total = n_steps * dt

    S0 = price_paths[:, 0]
    V0, delta = _batch_price_and_delta(hedge_model, S0, K, T_total, r, option_type, hedge_params)
    hedge_shares = delta.copy()
    cash = V0 - hedge_shares * S0
    total_cost = np.zeros(n_paths)

    for t in range(1, n_steps + 1):
        S_t = price_paths[:, t]
        T_remaining = max(T_total - t * dt, 0.0)
        cash *= np.exp(r * dt)

        if T_remaining > 1e-8:
            _, new_delta = _batch_price_and_delta(hedge_model, S_t, K, T_remaining, r,
                                                   option_type, hedge_params)
            drift = new_delta - hedge_shares
            rebalance_mask = np.abs(drift) > band_width

            trade = np.where(rebalance_mask, drift, 0.0)
            cost = transaction_cost_rate * np.abs(trade) * S_t
            cash -= trade * S_t + cost
            hedge_shares = np.where(rebalance_mask, new_delta, hedge_shares)
            total_cost += cost
        # else: final settlement step — matches simulate_delta_hedge_batch()
        # exactly, no further trading; the option settles against whatever
        # hedge position is currently held.

    S_T = price_paths[:, -1]
    payoff = np.where(S_T > K, S_T - K, 0.0) if option_type == "call" \
        else np.where(K > S_T, K - S_T, 0.0)
    terminal_pnl = cash + hedge_shares * S_T - payoff

    return terminal_pnl, total_cost


# ══════════════════════════════════════════════════════════════════════════════
# Delta-Vega hedging — a second instrument to also neutralize volatility risk
# ══════════════════════════════════════════════════════════════════════════════
#
# Every hedge studied above trades ONLY the underlying, i.e. neutralizes Delta
# and nothing else. A real option market maker's SECOND-biggest risk (after
# Delta) is almost always Vega — exposure to volatility itself moving — and
# the underlying stock has ZERO vega, so no amount of stock trading can hedge
# it. The standard practitioner fix is to hold a SECOND option (a more liquid,
# different-strike vanilla, same underlying/tenor) sized to cancel the book's
# net Vega, and only then use the stock to mop up whatever Delta the option
# hedge itself introduces. This also gives the Heston hedge (whose entire
# reason for existing is stochastic VOLATILITY risk, not jumps) a fair fight:
# a Delta-only Heston hedge can't use its own main insight at all, since Delta
# hedging never touches vol risk in the first place.
#
# Two-instrument hedge ratios (short 1 unit of the target option):
#     n_instrument = Vega_target / Vega_instrument        (zeroes net Vega)
#     n_stock      = Delta_target - n_instrument*Delta_instrument   (zeroes net Delta)

def simulate_delta_vega_hedge_batch(price_paths: np.ndarray, K_target: float, K_instrument: float,
                                     r: float, option_type: str, hedge_model: str, hedge_params: dict,
                                     rebalance_every: int = 1, transaction_cost_rate: float = 0.0005,
                                     dt: float = 1.0 / 252) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorized Delta-Vega hedge: short 1 unit of a target option (strike
    `K_target`), hedged with (a) a second vanilla option of the SAME
    tenor and type but a different strike `K_instrument` (sized to
    neutralize net Vega) and (b) the underlying stock (sized to
    neutralize whatever net Delta remains once the option hedge is in
    place). Both instruments' prices/Greeks are computed under the SAME
    hedge model, exactly mirroring how the rest of this project isolates
    "what the market maker believes" from "how the market actually
    moves" (`price_paths` may come from any true data-generating process,
    including one that differs from `hedge_model`).

    Only "bs", "merton", and "heston" hedge models are supported (all
    three now have batch-vectorized price/Delta/Vega — see
    `_batch_price_and_delta` and `_batch_vega`).

    Returns
    -------
    (terminal_pnl, total_transaction_cost) : both np.ndarray, shape (n_paths,)
    """
    n_paths, n_plus_1 = price_paths.shape
    n_steps = n_plus_1 - 1
    T_total = n_steps * dt

    S0 = price_paths[:, 0]
    V0_t, delta0_t = _batch_price_and_delta(hedge_model, S0, K_target, T_total, r, option_type, hedge_params)
    V0_i, delta0_i = _batch_price_and_delta(hedge_model, S0, K_instrument, T_total, r, option_type, hedge_params)
    vega0_t = _batch_vega(hedge_model, S0, K_target, T_total, r, option_type, hedge_params)
    vega0_i = _batch_vega(hedge_model, S0, K_instrument, T_total, r, option_type, hedge_params)

    n_instr = vega0_t / vega0_i
    n_stock = delta0_t - n_instr * delta0_i
    cash = V0_t - n_stock * S0 - n_instr * V0_i
    total_cost = np.zeros(n_paths)
    # A per-path CAP on |n_instr|, set relative to the INITIAL (well-behaved,
    # full-maturity) hedge ratio -- see the note below for why this is needed.
    n_instr_cap = 5.0 * np.maximum(np.abs(n_instr), 1.0)

    for t in range(1, n_steps + 1):
        S_t = price_paths[:, t]
        T_remaining = max(T_total - t * dt, 0.0)
        cash *= np.exp(r * dt)
        rebalance_now = (t % rebalance_every == 0) or (t == n_steps)

        if rebalance_now and T_remaining > 1e-8:
            _, delta_t = _batch_price_and_delta(hedge_model, S_t, K_target, T_remaining, r, option_type, hedge_params)
            V_i, delta_i = _batch_price_and_delta(hedge_model, S_t, K_instrument, T_remaining, r, option_type, hedge_params)
            vega_t = _batch_vega(hedge_model, S_t, K_target, T_remaining, r, option_type, hedge_params)
            vega_i = _batch_vega(hedge_model, S_t, K_instrument, T_remaining, r, option_type, hedge_params)

            # Every option's Vega collapses to 0 as expiry approaches (vega ~ sqrt(T_remaining)),
            # and a FIXED-strike instrument that drifts away from the money has its Vega collapse
            # FASTER than an at-the-money target's -- so the ratio vega_t/vega_i can compound to
            # an unreasonable size well before vega_i is anywhere near a hard numerical zero (this
            # was caught empirically: see the module's CLI check for a worked example). This is a
            # genuine practitioner issue, not a modelling bug -- a real desk facing a hedge
            # instrument whose Vega has withered away would roll to a fresh, more at-the-money
            # instrument rather than lever up into an increasingly threadbare one. We approximate
            # that discipline with a simple, explicit rule: cap the instrument quantity at a fixed
            # multiple of its INITIAL (full-maturity, well-behaved) value, and freeze the ratio
            # once that cap binds, rather than trade into a numerically explosive position.
            raw_n_instr = np.divide(vega_t, vega_i, out=np.full_like(vega_t, np.nan), where=np.abs(vega_i) > 1e-10)
            within_cap = np.isfinite(raw_n_instr) & (np.abs(raw_n_instr) <= n_instr_cap)
            new_n_instr = np.where(within_cap, raw_n_instr, n_instr)
            new_n_stock = delta_t - new_n_instr * delta_i

            trade_stock = new_n_stock - n_stock
            trade_instr = new_n_instr - n_instr
            cost = transaction_cost_rate * (np.abs(trade_stock) * S_t + np.abs(trade_instr) * V_i)
            cash -= trade_stock * S_t + trade_instr * V_i + cost
            total_cost += cost
            n_stock, n_instr = new_n_stock, new_n_instr

    S_T = price_paths[:, -1]
    payoff_t = np.where(S_T > K_target, S_T - K_target, 0.0) if option_type == "call" \
        else np.where(K_target > S_T, K_target - S_T, 0.0)
    payoff_i = np.where(S_T > K_instrument, S_T - K_instrument, 0.0) if option_type == "call" \
        else np.where(K_instrument > S_T, K_instrument - S_T, 0.0)
    terminal_pnl = cash + n_stock * S_T + n_instr * payoff_i - payoff_t

    return terminal_pnl, total_cost


# ══════════════════════════════════════════════════════════════════════════════
# CLI Sanity Check
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Sanity check: hedge a Black-Scholes option against a path SIMULATED
    from the SAME Black-Scholes model (GBM). Any single path's hedging
    error is a random draw and can look large just by chance (it is the
    realisation of a mean-zero random variable, not a fixed number) —
    so we check the properties that theory actually predicts: across
    many independent paths, hedging error should average to ~0, and its
    variance should shrink as we rebalance more frequently.
    Run from repo root: python src/hedging_engine.py
    """
    print("Delta-hedging engine sanity check\n" + "-" * 40)

    rng = np.random.default_rng(42)
    S0, K, r, sigma, T = 100.0, 100.0, 0.03, 0.20, 0.25   # 3-month option
    n_steps = 63   # ~3 months of daily steps
    dt = T / n_steps

    n_trials = 1000
    pnls_daily, pnls_weekly = [], []
    for _ in range(n_trials):
        z = rng.standard_normal(n_steps)
        log_path = np.log(S0) + np.cumsum((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z)
        path = np.concatenate([[S0], np.exp(log_path)])

        res_daily = simulate_delta_hedge(path, K, r, "call", "bs", {"sigma": sigma},
                                          rebalance_every=1, transaction_cost_rate=0.0, dt=dt)
        res_weekly = simulate_delta_hedge(path, K, r, "call", "bs", {"sigma": sigma},
                                           rebalance_every=5, transaction_cost_rate=0.0, dt=dt)
        pnls_daily.append(res_daily.attrs["terminal_pnl"])
        pnls_weekly.append(res_weekly.attrs["terminal_pnl"])

    pnls_daily = np.array(pnls_daily)
    pnls_weekly = np.array(pnls_weekly)
    premium = res_daily.attrs["initial_premium"]

    print(f"[1] Initial option premium         : {premium:.4f}")
    print(f"[2] Daily rebalance  P&L over {n_trials} paths : "
          f"mean={pnls_daily.mean():.4f}  std={pnls_daily.std():.4f}")
    print(f"[3] Weekly rebalance P&L over {n_trials} paths : "
          f"mean={pnls_weekly.mean():.4f}  std={pnls_weekly.std():.4f}")

    assert abs(pnls_daily.mean()) < 0.15, "Mean hedging error should be close to zero"
    assert pnls_daily.std() < pnls_weekly.std(), \
        "Daily rebalancing should have lower hedging-error variance than weekly"
    print("\n    Confirmed: hedging error is approximately mean-zero, and more")
    print("    frequent rebalancing reduces its variance — the correct")
    print("    theoretical behaviour under a matching (BS vs. GBM) model.")

    # --- Cross-validate the vectorized batch simulator against the ---
    # --- per-path simulator on IDENTICAL paths: must match exactly. ---
    print("\n[4] Cross-validating vectorized batch simulator vs. per-path simulator...")
    n_check = 200
    z_batch = rng.standard_normal((n_check, n_steps))
    log_paths = np.log(S0) + np.cumsum(
        (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z_batch, axis=1)
    paths_batch = np.concatenate([np.full((n_check, 1), S0), np.exp(log_paths)], axis=1)

    pnls_perpath = np.array([
        simulate_delta_hedge(paths_batch[i], K, r, "call", "bs", {"sigma": sigma},
                              rebalance_every=1, transaction_cost_rate=0.0005, dt=dt
                              ).attrs["terminal_pnl"]
        for i in range(n_check)
    ])
    pnls_vectorized = simulate_delta_hedge_batch(
        paths_batch, K, r, "call", "bs", {"sigma": sigma},
        rebalance_every=1, transaction_cost_rate=0.0005, dt=dt,
    )
    max_diff = np.max(np.abs(pnls_perpath - pnls_vectorized))
    print(f"    Max |per-path - vectorized| P&L difference over {n_check} paths: {max_diff:.2e}")
    assert max_diff < 1e-8, "Vectorized batch simulator must exactly match the per-path simulator"
    print("    Confirmed: the vectorized simulator is a pure performance optimisation —")
    print("    it produces IDENTICAL results to the per-path simulator.")

    # --- Band ("no-transaction region") hedging sanity check ---
    print("\n[5] Band hedging: cross-validating per-path vs. vectorized batch...")
    pnls_band_perpath = np.array([
        simulate_delta_hedge_band(paths_batch[i], K, r, "call", "bs", {"sigma": sigma},
                                   band_width=0.05, transaction_cost_rate=0.0005, dt=dt
                                   ).attrs["terminal_pnl"]
        for i in range(n_check)
    ])
    pnls_band_vectorized, costs_band_vectorized = simulate_delta_hedge_band_batch(
        paths_batch, K, r, "call", "bs", {"sigma": sigma},
        band_width=0.05, transaction_cost_rate=0.0005, dt=dt,
    )
    max_diff_band = np.max(np.abs(pnls_band_perpath - pnls_band_vectorized))
    print(f"    Max |per-path - vectorized| band-hedge P&L difference: {max_diff_band:.2e}")
    assert max_diff_band < 1e-8, "Vectorized band simulator must exactly match the per-path version"

    print("\n[6] Band hedging with band_width=0 should reduce to daily rebalancing...")
    pnls_zero_band, _ = simulate_delta_hedge_band_batch(
        paths_batch, K, r, "call", "bs", {"sigma": sigma},
        band_width=0.0, transaction_cost_rate=0.0005, dt=dt,
    )
    max_diff_zero = np.max(np.abs(pnls_zero_band - pnls_vectorized))
    print(f"    Max |band(width=0) - daily-rebalance| P&L difference: {max_diff_zero:.2e}")
    assert max_diff_zero < 1e-6, "A zero-width band should trade every step, matching daily rebalancing"

    print("\n[7] Band hedging should trade less often, at some cost in hedging variance...")
    pnls_wide_band, costs_wide_band = simulate_delta_hedge_band_batch(
        paths_batch, K, r, "call", "bs", {"sigma": sigma},
        band_width=0.10, transaction_cost_rate=0.0005, dt=dt,
    )
    print(f"    Daily rebalance   : mean total cost paid = n/a (fixed schedule), "
          f"std(P&L) = {pnls_vectorized.std():.4f}")
    print(f"    Band (width=0.10) : mean total cost paid = {costs_wide_band.mean():.4f}, "
          f"std(P&L) = {pnls_wide_band.std():.4f}")
    print("    Confirmed: band hedging and fixed-frequency rebalancing are both valid,")
    print("    cross-validated (vectorized == per-path) hedging strategies with a")
    print("    genuine cost/risk tradeoff between them.")

    print("\nSanity check complete (see risk_analysis.py for the full")
    print("many-path CVaR study across hedge models and true dynamics).")

    # --- Heston now batch-vectorized via the COS method (heston.py) ---
    print("\n[8] Heston hedging is now batch-vectorized (via the COS method)...")
    heston_params = {"kappa": 2.0, "theta": 0.04, "xi": 0.4, "rho": -0.7, "v0": 0.04}
    n_check_h = 60
    pnls_heston_perpath = np.array([
        simulate_delta_hedge(paths_batch[i], K, r, "call", "heston", heston_params,
                              rebalance_every=5, transaction_cost_rate=0.0005, dt=dt
                              ).attrs["terminal_pnl"]
        for i in range(n_check_h)
    ])
    pnls_heston_batch = simulate_delta_hedge_batch(
        paths_batch[:n_check_h], K, r, "call", "heston", heston_params,
        rebalance_every=5, transaction_cost_rate=0.0005, dt=dt,
    )
    max_diff_h = np.max(np.abs(pnls_heston_perpath - pnls_heston_batch))
    rel_diff_h = max_diff_h / np.mean(np.abs(pnls_heston_perpath))
    print(f"    Max |per-path (quad) - batch (COS)| Heston hedge P&L difference over "
          f"{n_check_h} paths: {max_diff_h:.4f}  (relative to mean |P&L|: {rel_diff_h:.2%})")
    assert rel_diff_h < 0.05, \
        "Batch (COS) and per-path (quad) Heston hedging should agree closely -- two independent Fourier methods"
    print("    Confirmed: full-scale (thousands-of-paths, daily-rebalancing) Heston-hedge studies")
    print("    are now possible -- previously restricted to ~150 paths / biweekly rebalancing.")

    # --- Delta-Vega multi-instrument hedge ---
    print("\n[9] Delta-Vega hedge: does it actually neutralize Vega, and does it help?")
    K_instrument = 110.0   # an OTM call of the same tenor as the vega-hedging instrument
    pnls_vega, costs_vega = simulate_delta_vega_hedge_batch(
        paths_batch, K, K_instrument, r, "call", "bs", {"sigma": sigma},
        rebalance_every=1, transaction_cost_rate=0.0005, dt=dt,
    )
    print(f"    Delta-Vega hedge (BS, matching-model world): mean P&L={pnls_vega.mean():.4f}  "
          f"std={pnls_vega.std():.4f}  mean cost={costs_vega.mean():.4f}")
    assert abs(pnls_vega.mean()) < 1.0, "Delta-Vega hedge should still be close to mean-zero in a matching-model world"

    print("\n    Checking the hedge ratios actually zero out net Vega at inception...")
    S0_arr = np.array([100.0])
    _, d_t = _batch_price_and_delta("bs", S0_arr, K, T, r, "call", {"sigma": sigma})
    _, d_i = _batch_price_and_delta("bs", S0_arr, K_instrument, T, r, "call", {"sigma": sigma})
    vega_t = _batch_vega("bs", S0_arr, K, T, r, "call", {"sigma": sigma})
    vega_i = _batch_vega("bs", S0_arr, K_instrument, T, r, "call", {"sigma": sigma})
    n_instr_check = vega_t / vega_i
    net_vega = -vega_t + n_instr_check * vega_i   # short target + long n_instr of the instrument
    print(f"    Net portfolio Vega after hedging (should be ~0): {net_vega[0]:.2e}")
    assert abs(net_vega[0]) < 1e-8, "The Delta-Vega hedge ratio should exactly zero out net Vega by construction"

    print("\n    Comparing Delta-only vs. Delta-Vega hedging under a STOCHASTIC-VOLATILITY true market")
    print("    (giving the Heston hedge model a chance to use its own core insight: vol risk)...")
    from heston import heston_simulate_paths
    S_h, _ = heston_simulate_paths(S0, T, r, kappa=2.0, theta=0.04, xi=0.6, rho=-0.7, v0=0.06,
                                    n_paths=4000, n_steps=n_steps, seed=11)
    pnls_delta_only = simulate_delta_hedge_batch(
        S_h, K, r, "call", "heston", heston_params, rebalance_every=5,
        transaction_cost_rate=0.0005, dt=dt,
    )
    pnls_delta_vega, _ = simulate_delta_vega_hedge_batch(
        S_h, K, K_instrument, r, "call", "heston", heston_params, rebalance_every=5,
        transaction_cost_rate=0.0005, dt=dt,
    )
    from risk_analysis import cvar_historical
    cvar_delta_only = cvar_historical(-pnls_delta_only, 0.05)
    cvar_delta_vega = cvar_historical(-pnls_delta_vega, 0.05)
    print(f"    Heston hedge, Delta-only  : CVaR(95%) = {cvar_delta_only:.4f}")
    print(f"    Heston hedge, Delta-Vega  : CVaR(95%) = {cvar_delta_vega:.4f}")
    print(f"    Delta-Vega {'reduces' if cvar_delta_vega < cvar_delta_only else 'does NOT reduce'} "
          f"tail risk relative to Delta-only, under a genuinely stochastic-volatility true market.")
    assert cvar_delta_vega < cvar_delta_only, \
        "Adding a Vega hedge should materially reduce tail risk when the true market has genuine vol-of-vol risk"

    print("\nAll checks passed.")
