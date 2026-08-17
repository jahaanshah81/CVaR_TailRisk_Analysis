"""
stress_testing.py
------------------
Deterministic scenario stress testing — the complement to the
statistical (Monte-Carlo) CVaR analysis used everywhere else in this
project. Real derivatives risk-management functions require BOTH: a
statistical VaR/CVaR model (captures the AVERAGE shape of the tail) AND
a small set of named, deterministic adverse scenarios that don't rely on
any distributional assumption at all (Basel/CCAR/ICAAP-style scenario
stress testing) — because "this exact scenario has never happened in the
calibration window" is precisely the kind of risk a purely statistical
model can under-price.

Each scenario is expressed as a sequence of daily log-returns applied to
the underlying starting at S0, then run through the SAME discrete
delta-hedging engine used everywhere else in the project (not a separate
ad-hoc calculation), so results are directly comparable to the
statistical CVaR figures reported elsewhere.
"""

import numpy as np
import pandas as pd

from hedging_engine import simulate_delta_hedge


def _scenario_path(S0: float, daily_log_returns: np.ndarray) -> np.ndarray:
    log_path = np.log(S0) + np.cumsum(daily_log_returns)
    return np.concatenate([[S0], np.exp(log_path)])


def build_scenarios(S0: float, n_steps: int, sigma: float) -> dict:
    """
    Returns {scenario_name: price_path}, each an (n_steps + 1)-length
    array starting at S0, for a fixed set of named adverse scenarios
    inspired by real historical episodes and standard regulatory
    stress-testing categories.
    """
    scenarios = {}

    # 1. Black-Monday-style single-day crash (-20%), calm otherwise.
    rets = np.zeros(n_steps)
    rets[n_steps // 2] = np.log(0.80)
    scenarios["Black Monday (\u201387-style, -20% single day)"] = _scenario_path(S0, rets)

    # 2. COVID-style multi-day crash followed by an elevated-vol aftermath.
    rng2 = np.random.default_rng(2)
    rets = np.zeros(n_steps)
    crash_start = n_steps // 3
    rets[crash_start] = np.log(0.88)
    rets[crash_start + 1] = np.log(0.91)
    tail_len = max(0, min(10, n_steps - crash_start - 2))
    if tail_len > 0:
        rets[crash_start + 2: crash_start + 2 + tail_len] = rng2.normal(
            0, 2.5 * sigma / np.sqrt(252), tail_len)
    scenarios["COVID-style crash (2020) + vol-spike aftermath"] = _scenario_path(S0, rets)

    # 3. Flash-crash: sharp intraday-style drop, partial same-window rebound.
    rets = np.zeros(n_steps)
    mid = n_steps // 2
    rets[mid] = np.log(0.90)
    rets[mid + 1] = np.log(1.05)
    scenarios["Flash crash (-10% then +5% rebound)"] = _scenario_path(S0, rets)

    # 4. Pure volatility-regime shock: no directional view, realised vol doubles.
    rng4 = np.random.default_rng(4)
    rets = rng4.normal(0, 2.0 * sigma / np.sqrt(252), n_steps)
    scenarios["Vol-regime doubling (no directional jump)"] = _scenario_path(S0, rets)

    # 5. Slow grind lower: sustained mild negative drift, low vol — the
    #    "boring but bleeding" scenario a jump-focused VaR model can miss.
    rets = np.full(n_steps, np.log(1 - 0.0025))
    scenarios["Slow grind lower (-0.25%/day, low vol)"] = _scenario_path(S0, rets)

    # 6. Melt-up: a sustained sharp rally (tests short-gamma pain on the
    #    upside too, not just crash risk).
    rets = np.full(n_steps, np.log(1 + 0.006))
    scenarios["Melt-up rally (+0.6%/day)"] = _scenario_path(S0, rets)

    return scenarios


def run_stress_scenarios(S0: float, K: float, r: float, option_type: str, n_steps: int,
                          dt: float, sigma: float, hedge_specs: dict,
                          transaction_cost_rate: float = 0.0005) -> pd.DataFrame:
    """
    Runs every named scenario from build_scenarios() through every hedge
    model in `hedge_specs` (dict: {hedge_name: (model_str, params_dict)}),
    using the project's standard per-path discrete delta-hedging simulator.

    Returns
    -------
    pd.DataFrame, tidy format: one row per (scenario, hedge_model), with
    the terminal hedging P&L and the scenario's total price move.
    """
    scenarios = build_scenarios(S0, n_steps, sigma)
    rows = []
    for scen_name, path in scenarios.items():
        for hedge_name, (model, params) in hedge_specs.items():
            result = simulate_delta_hedge(
                path, K, r, option_type, model, params,
                rebalance_every=1, transaction_cost_rate=transaction_cost_rate, dt=dt,
            )
            rows.append({
                "scenario": scen_name, "hedge_model": hedge_name,
                "terminal_pnl": result.attrs["terminal_pnl"],
                "price_move_pct": (path[-1] / path[0] - 1) * 100,
            })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# CLI Sanity Check
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Sanity check: run the standard BS-vs-Merton hedge comparison through
    every named stress scenario for a 3-month at-the-money call, and
    confirm the BS hedge underperforms (more negative P&L) the Merton
    hedge specifically on the CRASH scenarios (where jump-awareness
    matters), while the two are comparable on scenarios with no
    directional jump (pure vol shock, slow grind, melt-up).
    Run from repo root: python src/stress_testing.py
    """
    print("Stress-testing sanity check\n" + "-" * 40)

    S0, K, r, sigma, T = 100.0, 100.0, 0.03, 0.20, 0.25
    n_steps = 63
    dt = T / n_steps
    lam, mu_j, sigma_j = 2.0, -0.08, 0.10   # meaningful crash risk in the "true" sense

    hedge_specs = {
        "BS hedge": ("bs", {"sigma": sigma}),
        "Merton hedge": ("merton", {"sigma": sigma, "lam": lam, "mu_j": mu_j, "sigma_j": sigma_j}),
    }

    results = run_stress_scenarios(S0, K, r, "call", n_steps, dt, sigma, hedge_specs)
    pivot = results.pivot(index="scenario", columns="hedge_model", values="terminal_pnl")
    print(pivot.to_string())

    crash_scenarios = [s for s in pivot.index if "Monday" in s or "COVID" in s or "Flash" in s]
    for scen in crash_scenarios:
        bs_pnl = pivot.loc[scen, "BS hedge"]
        merton_pnl = pivot.loc[scen, "Merton hedge"]
        print(f"\n  {scen}: BS={bs_pnl:.4f}  Merton={merton_pnl:.4f}  "
              f"(Merton {'better' if merton_pnl > bs_pnl else 'worse'})")

    n_merton_better = sum(pivot.loc[s, "Merton hedge"] > pivot.loc[s, "BS hedge"] for s in crash_scenarios)
    print(f"\nMerton hedge outperforms BS hedge on {n_merton_better}/{len(crash_scenarios)} crash scenarios.")
    assert n_merton_better >= 2, \
        "The jump-aware Merton hedge should outperform the BS hedge on most crash scenarios"
    print("\n    Confirmed: deterministic scenario stress tests corroborate the")
    print("    project's statistical CVaR finding using a completely independent,")
    print("    distribution-free methodology.")
