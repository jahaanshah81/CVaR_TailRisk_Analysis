"""
black_scholes.py
-----------------
Black-Scholes-Merton (BSM) European option pricing and Greeks.

This is the baseline model every option pricer must reduce to: constant
volatility, continuous trading, log-normal underlying dynamics
    dS_t = mu*S_t*dt + sigma*S_t*dW_t
Prices and every Greek here are closed-form. Heston (stochastic vol) and
Merton (jump-diffusion) are validated against this baseline in the limit
of zero vol-of-vol / zero jump intensity, and against each other via
Monte Carlo cross-checks (see heston.py, merton.py).

Sign / parameter convention used throughout this project:
    S      : current underlying price
    K      : strike price
    T      : time to maturity, in years
    r      : continuously-compounded risk-free rate
    q      : continuous dividend yield (0 for non-dividend-paying assets)
    sigma  : annualised volatility (BSM) or reference vol (other models)
    option_type : "call" or "put"

Reference: Black & Scholes (1973), Merton (1973).
"""

import numpy as np
from scipy.stats import norm


def _d1_d2(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0):
    """Compute the standard d1, d2 terms shared across price and Greeks."""
    if T <= 0 or sigma <= 0:
        raise ValueError("T and sigma must be strictly positive.")
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bs_price(S: float, K: float, T: float, r: float, sigma: float,
             option_type: str = "call", q: float = 0.0) -> float:
    """
    Black-Scholes-Merton European option price.

    Parameters
    ----------
    S, K, T, r, sigma, q : see module docstring.
    option_type : "call" or "put"

    Returns
    -------
    float : option price.
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    disc_q = np.exp(-q * T)
    disc_r = np.exp(-r * T)

    if option_type == "call":
        return S * disc_q * norm.cdf(d1) - K * disc_r * norm.cdf(d2)
    elif option_type == "put":
        return K * disc_r * norm.cdf(-d2) - S * disc_q * norm.cdf(-d1)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type}")


def bs_greeks(S: float, K: float, T: float, r: float, sigma: float,
              option_type: str = "call", q: float = 0.0) -> dict:
    """
    Compute all standard Black-Scholes Greeks.

    Returns
    -------
    dict with keys:
        delta : d(price)/d(S)               — per $1 move in underlying
        gamma : d(delta)/d(S)                — same for call and put
        vega  : d(price)/d(sigma) / 100      — per 1 percentage-point vol move
        theta : d(price)/d(t) / 365          — per calendar day (time decay)
        rho   : d(price)/d(r) / 100          — per 1 percentage-point rate move
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    disc_q = np.exp(-q * T)
    disc_r = np.exp(-r * T)
    pdf_d1 = norm.pdf(d1)

    gamma = disc_q * pdf_d1 / (S * sigma * np.sqrt(T))
    vega = S * disc_q * pdf_d1 * np.sqrt(T) / 100.0   # per 1 vol point (0.01)

    if option_type == "call":
        delta = disc_q * norm.cdf(d1)
        theta = (
            -disc_q * S * pdf_d1 * sigma / (2 * np.sqrt(T))
            - r * K * disc_r * norm.cdf(d2)
            + q * S * disc_q * norm.cdf(d1)
        ) / 365.0
        rho = K * T * disc_r * norm.cdf(d2) / 100.0
    elif option_type == "put":
        delta = disc_q * (norm.cdf(d1) - 1.0)
        theta = (
            -disc_q * S * pdf_d1 * sigma / (2 * np.sqrt(T))
            + r * K * disc_r * norm.cdf(-d2)
            - q * S * disc_q * norm.cdf(-d1)
        ) / 365.0
        rho = -K * T * disc_r * norm.cdf(-d2) / 100.0
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type}")

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


def implied_volatility(price: float, S: float, K: float, T: float, r: float,
                        option_type: str = "call", q: float = 0.0,
                        tol: float = 1e-8, max_iter: int = 100) -> float:
    """
    Recover Black-Scholes implied volatility from an observed option price
    via Newton-Raphson (falling back to bisection if Newton diverges).
    Used later to compare model-implied prices against a common vol scale.
    """
    sigma = 0.3   # reasonable starting guess
    for _ in range(max_iter):
        model_price = bs_price(S, K, T, r, sigma, option_type, q)
        vega = bs_greeks(S, K, T, r, sigma, option_type, q)["vega"] * 100.0  # undo /100
        diff = model_price - price
        if abs(diff) < tol:
            return sigma
        if vega < 1e-8:
            break
        sigma -= diff / vega
        if sigma <= 0:
            sigma = 1e-4

    # Bisection fallback for robustness
    lo, hi = 1e-4, 5.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        model_price = bs_price(S, K, T, r, mid, option_type, q)
        if model_price > price:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


# ══════════════════════════════════════════════════════════════════════════════
# CLI Sanity Check
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Validate against well-known benchmark values and put-call parity.
    Run from repo root: python src/black_scholes.py
    """
    print("Black-Scholes sanity checks\n" + "-" * 40)

    # Benchmark: Hull's textbook example (S=42, K=40, T=0.5, r=0.10, sigma=0.20)
    S, K, T, r, sigma = 42.0, 40.0, 0.5, 0.10, 0.20
    call = bs_price(S, K, T, r, sigma, "call")
    put = bs_price(S, K, T, r, sigma, "put")
    print(f"[1] Call price = {call:.4f}  (Hull benchmark ~= 4.76)")
    print(f"[2] Put price  = {put:.4f}  (Hull benchmark ~= 0.81)")

    # Put-call parity: C - P = S*exp(-qT) - K*exp(-rT)
    parity_lhs = call - put
    parity_rhs = S - K * np.exp(-r * T)
    print(f"[3] Put-call parity: C-P={parity_lhs:.6f}  S-Ke^-rT={parity_rhs:.6f}  "
          f"(match: {np.isclose(parity_lhs, parity_rhs)})")

    # Greeks sanity: call delta in (0,1), put delta in (-1,0)
    greeks_c = bs_greeks(S, K, T, r, sigma, "call")
    greeks_p = bs_greeks(S, K, T, r, sigma, "put")
    print(f"\n[4] Call Greeks: {greeks_c}")
    print(f"[5] Put  Greeks: {greeks_p}")
    assert 0 < greeks_c["delta"] < 1, "Call delta out of range"
    assert -1 < greeks_p["delta"] < 0, "Put delta out of range"
    assert np.isclose(greeks_c["gamma"], greeks_p["gamma"]), "Gamma should match for call/put"
    assert np.isclose(greeks_c["vega"], greeks_p["vega"]), "Vega should match for call/put"

    # Implied vol recovery: price at sigma=0.20, recover sigma from that price
    recovered = implied_volatility(call, S, K, T, r, "call")
    print(f"\n[6] Implied vol recovery: input sigma=0.20, recovered={recovered:.6f}")
    assert np.isclose(recovered, sigma, atol=1e-4), "IV recovery failed"

    print("\nAll checks passed.")
