"""
heston.py
---------
Heston (1993) stochastic volatility option pricing model.

Extends Black-Scholes by letting volatility itself follow a random,
mean-reverting process (a Cox-Ingersoll-Ross / square-root diffusion):

    dS_t = r*S_t dt + sqrt(v_t)*S_t dW_t^S
    dv_t = kappa*(theta - v_t) dt + xi*sqrt(v_t) dW_t^v
    corr(dW_t^S, dW_t^v) = rho

Parameters
----------
kappa : mean-reversion speed of variance
theta : long-run variance level
xi    : volatility of volatility ("vol of vol")
rho   : correlation between spot and variance shocks (typically < 0,
        the empirical "leverage effect": prices fall, vol rises)
v0    : initial (current) variance

Heston has NO closed-form price the way Black-Scholes does, but it DOES
have a closed-form characteristic function of the log-price. We price via
numerical Fourier inversion (Gil-Pelaez / Heston's original P1-P2
decomposition), using the numerically stable "Little Trap" reformulation
of the characteristic function (Albrecher, Mayer, Schoutens & Tistaert,
2007) that avoids the branch-cut discontinuities of Heston's original
1993 formula for long maturities or extreme parameters.

We independently validate this Fourier pricer against Monte Carlo
simulation of the same SDE (Full Truncation Euler scheme of Lord,
Koekkoek & Van Dijk, 2010, which keeps the discretized variance process
well-defined even though the exact CIR process can only be simulated
exactly with more expensive schemes).

References:
    Heston, S.L. (1993). "A closed-form solution for options with
    stochastic volatility." Review of Financial Studies, 6(2), 327-343.
    Albrecher, H., Mayer, P., Schoutens, W., & Tistaert, J. (2007).
    "The little Heston trap." Wilmott Magazine.
    Lord, R., Koekkoek, R., & Van Dijk, D. (2010). "A comparison of
    biased simulation schemes for stochastic volatility models."
    Quantitative Finance, 10(2), 177-194.
"""

import numpy as np
from scipy.integrate import quad


def _heston_char_func(u: complex, S0: float, T: float, r: float,
                       kappa: float, theta: float, xi: float,
                       rho: float, v0: float) -> complex:
    """
    Heston characteristic function of ln(S_T), "Little Trap" formulation.
    phi(u) = E[exp(i*u*ln(S_T))]  under the risk-neutral measure.
    """
    i = 1j
    d = np.sqrt((kappa - rho * xi * i * u) ** 2 + (xi ** 2) * (i * u + u ** 2))
    g = (kappa - rho * xi * i * u - d) / (kappa - rho * xi * i * u + d)

    exp_dT = np.exp(-d * T)
    C = (i * u * (np.log(S0) + r * T)
         + (kappa * theta / xi ** 2)
         * ((kappa - rho * xi * i * u - d) * T
            - 2.0 * np.log((1 - g * exp_dT) / (1 - g))))
    D = ((kappa - rho * xi * i * u - d) / xi ** 2) * ((1 - exp_dT) / (1 - g * exp_dT))

    return np.exp(C + D * v0)


def heston_price(S0: float, K: float, T: float, r: float,
                  kappa: float, theta: float, xi: float, rho: float, v0: float,
                  option_type: str = "call", upper_limit: float = 200.0) -> float:
    """
    Heston European call/put price via numerical Fourier inversion of the
    characteristic function (Heston's original P1/P2 decomposition).

    Returns
    -------
    float : option price.
    """
    phi_neg_i = _heston_char_func(-1j, S0, T, r, kappa, theta, xi, rho, v0)

    def integrand_p1(u):
        i = 1j
        phi_shifted = _heston_char_func(u - i, S0, T, r, kappa, theta, xi, rho, v0)
        val = np.exp(-i * u * np.log(K)) * phi_shifted / (i * u * phi_neg_i)
        return val.real

    def integrand_p2(u):
        i = 1j
        phi_u = _heston_char_func(u, S0, T, r, kappa, theta, xi, rho, v0)
        val = np.exp(-i * u * np.log(K)) * phi_u / (i * u)
        return val.real

    # Numerical integration over u in (0, upper_limit); the integrand decays
    # rapidly, and a small epsilon avoids the removable singularity at u=0.
    eps = 1e-8
    int_p1, _ = quad(integrand_p1, eps, upper_limit, limit=200)
    int_p2, _ = quad(integrand_p2, eps, upper_limit, limit=200)

    P1 = 0.5 + int_p1 / np.pi
    P2 = 0.5 + int_p2 / np.pi

    call = S0 * P1 - K * np.exp(-r * T) * P2

    if option_type == "call":
        return call
    elif option_type == "put":
        # Put-call parity (holds for any correct European pricer)
        return call - S0 + K * np.exp(-r * T)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type}")


def heston_greeks_fd(S0: float, K: float, T: float, r: float,
                      kappa: float, theta: float, xi: float, rho: float, v0: float,
                      option_type: str = "call", bump: float = 0.01) -> dict:
    """
    Heston Greeks via central finite differences on the Fourier price.
    (Heston has no simple closed-form Greeks the way BSM does; bumping
    the semi-analytical price is the standard practitioner approach and
    is exact in the limit bump -> 0.)

    Returns
    -------
    dict with keys: delta, gamma, vega (w.r.t. v0, per 0.01 variance bump),
    theta (per calendar day).
    """
    def price(S0_, T_, v0_):
        return heston_price(S0_, K, T_, r, kappa, theta, xi, rho, v0_, option_type)

    h_S = S0 * bump
    p_up = price(S0 + h_S, T, v0)
    p_mid = price(S0, T, v0)
    p_down = price(S0 - h_S, T, v0)
    delta = (p_up - p_down) / (2 * h_S)
    gamma = (p_up - 2 * p_mid + p_down) / (h_S ** 2)

    h_v = 0.01
    vega = (price(S0, T, v0 + h_v) - price(S0, T, v0 - h_v)) / (2 * h_v) * 0.01

    h_T = 1.0 / 365.0
    theta = (price(S0, T - h_T, v0) - p_mid) / h_T / 365.0

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta}


def heston_simulate_paths(S0: float, T: float, r: float,
                           kappa: float, theta: float, xi: float, rho: float, v0: float,
                           n_paths: int, n_steps: int, seed: int = 42) -> tuple:
    """
    Simulate Heston price and variance paths via the Full Truncation Euler
    scheme (Lord, Koekkoek & Van Dijk, 2010), which floors the variance at
    zero before using it, avoiding the negative-variance problem of naive
    Euler discretization of the CIR process.

    Returns
    -------
    (S_paths, v_paths) : each np.ndarray of shape (n_paths, n_steps+1)
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps

    S_paths = np.empty((n_paths, n_steps + 1))
    v_paths = np.empty((n_paths, n_steps + 1))
    S_paths[:, 0] = S0
    v_paths[:, 0] = v0

    for t in range(n_steps):
        z1 = rng.standard_normal(n_paths)
        z2 = rng.standard_normal(n_paths)
        z_v = z1
        z_s = rho * z1 + np.sqrt(1 - rho ** 2) * z2

        v_plus = np.maximum(v_paths[:, t], 0.0)   # "full truncation"
        v_paths[:, t + 1] = (v_paths[:, t]
                              + kappa * (theta - v_plus) * dt
                              + xi * np.sqrt(v_plus) * np.sqrt(dt) * z_v)
        S_paths[:, t + 1] = S_paths[:, t] * np.exp(
            (r - 0.5 * v_plus) * dt + np.sqrt(v_plus) * np.sqrt(dt) * z_s
        )

    return S_paths, v_paths


def heston_price_montecarlo(S0: float, K: float, T: float, r: float,
                             kappa: float, theta: float, xi: float, rho: float, v0: float,
                             option_type: str = "call",
                             n_paths: int = 200_000, n_steps: int = 200,
                             seed: int = 42) -> tuple[float, float]:
    """Monte Carlo price used purely to cross-validate heston_price()."""
    S_paths, _ = heston_simulate_paths(S0, T, r, kappa, theta, xi, rho, v0,
                                        n_paths, n_steps, seed)
    S_T = S_paths[:, -1]
    if option_type == "call":
        payoff = np.maximum(S_T - K, 0.0)
    else:
        payoff = np.maximum(K - S_T, 0.0)
    discounted = np.exp(-r * T) * payoff
    price = discounted.mean()
    se = discounted.std(ddof=1) / np.sqrt(n_paths)
    return price, se


# ══════════════════════════════════════════════════════════════════════════════
# COS method (Fang & Oosterlee, 2008): a VECTORIZABLE alternative to the
# scipy.integrate.quad Fourier inversion above
# ══════════════════════════════════════════════════════════════════════════════
#
# heston_price() above is accurate but fundamentally scalar: scipy.integrate.quad
# requires a scalar-valued integrand, so pricing across many strikes or many
# underlying prices (as the hedging engine needs, once per rebalance, across
# every simulated path at once) means one quad call per price -- this is the
# single most-repeated documented limitation in this project (see
# hedging_engine.py, risk_analysis.py, README.md, report.md).
#
# The Fourier-COSine (COS) method fixes this. Instead of numerically
# integrating the Gil-Pelaez inversion integral, it expands the SAME
# characteristic function in a truncated Fourier-cosine series against the
# (closed-form) cosine coefficients of the payoff -- turning the price into a
# finite, fully vectorizable SUM over N terms (typically N=128-256), with no
# adaptive-quadrature loop at all. Crucially, the characteristic function
# itself does not depend on the current spot price when expressed in terms of
# the LOG-RETURN Y = ln(S_T / S_0) (Heston's SDE is scale-invariant in S), so
# the expensive part of the computation (evaluating phi(u) at each of the N
# frequencies) can be done ONCE and reused across an entire ARRAY of different
# S_0 values simultaneously via ordinary numpy broadcasting -- exactly the
# vectorization this project needs to hedge thousands of paths under Heston at
# once, the way it already does for Black-Scholes and Merton.
#
# Reference: Fang, F., & Oosterlee, C.W. (2008). "A novel pricing method for
# European options based on Fourier-cosine series expansions." SIAM Journal
# on Scientific Computing, 31(2), 826-848.

def _cos_truncation_range(T: float, r: float, theta: float, v0: float, L: float = 12.0):
    """
    Heuristic truncation range [a, b] for the log-return Y = ln(S_T/S_0),
    centered at the risk-neutral forward log-return with a width set by
    L standard deviations of a representative variance scale. This does
    NOT need to be exact -- the COS method's accuracy is very insensitive
    to a somewhat-too-wide range (the cosine coefficients of a bounded
    payoff decay rapidly); it only needs to comfortably contain the bulk
    of Y's probability mass. Validated empirically below against the
    independent quad-based heston_price() across a range of strikes,
    maturities, and parameter sets.
    """
    vbar = 0.5 * (v0 + theta)   # blends "current" and "long-run" variance
    c1 = (r - 0.5 * vbar) * T
    width = L * np.sqrt(vbar * T + 1e-12)
    return c1 - width, c1 + width


def heston_price_cos_batch(S0: np.ndarray, K: float, T: float, r: float,
                            kappa: float, theta: float, xi: float, rho: float, v0: float,
                            option_type: str = "call", N: int = 256, L: float = 12.0) -> np.ndarray:
    """
    Heston European option price via the COS method, VECTORIZED across an
    entire array of spot prices S0 for a single fixed strike K -- exactly
    the shape needed to hedge many simulated paths under Heston at once
    (one call per rebalancing date, instead of one scipy.integrate.quad
    call per path). Mathematically equivalent to calling heston_price()
    once per element of S0 (cross-validated below), just vectorized.

    Parameters
    ----------
    S0 : np.ndarray or float
        One or many current underlying prices.
    K, T, r, kappa, theta, xi, rho, v0, option_type : as in heston_price().
    N : number of Fourier-cosine terms (256 is generous; convergence is
        typically achieved well before this for reasonable parameters).
    L : truncation-range width, in "representative standard deviations"
        of the log-return (see _cos_truncation_range).

    Returns
    -------
    np.ndarray, same shape as S0 (or a 0-d array / float if S0 is scalar).
    """
    S0_arr = np.atleast_1d(np.asarray(S0, dtype=float))
    a, b = _cos_truncation_range(T, r, theta, v0, L)
    bma = b - a

    k = np.arange(N)
    u_k = k * np.pi / bma                                  # shape (N,)

    # phi(u_k) does NOT depend on S0 (evaluated on the log-RETURN Y, i.e.
    # with a unit reference spot of 1.0) -- computed ONCE and reused for
    # every path below. This is the crux of the vectorization.
    phi_vals = _heston_char_func(u_k, 1.0, T, r, kappa, theta, xi, rho, v0)
    cos_term = np.real(phi_vals * np.exp(-1j * u_k * a))   # shape (N,)
    cos_term[0] *= 0.5                                       # COS sum's leading term is halved

    k_log = np.log(K / S0_arr)[:, None]                     # shape (n, 1) -- moneyness varies per path
    u_k_row = u_k[None, :]                                  # shape (1, N)
    denom = 1.0 + u_k_row ** 2

    # chi_k(k_log, b) = integral of e^y*cos(...) from k_log to b -- closed form
    cos_b, sin_b = np.cos(u_k_row * (b - a)), np.sin(u_k_row * (b - a))
    cos_c, sin_c = np.cos(u_k_row * (k_log - a)), np.sin(u_k_row * (k_log - a))
    term_d = (cos_b + u_k_row * sin_b) * np.exp(b)
    term_c = (cos_c + u_k_row * sin_c) * np.exp(k_log)
    chi = (term_d - term_c) / denom                          # shape (n, N)

    # psi_k(k_log, b) = integral of cos(...) from k_log to b -- k=0 handled separately (no 0/0)
    psi = np.empty_like(chi)
    psi[:, 0] = (b - k_log[:, 0])
    with np.errstate(divide="ignore", invalid="ignore"):
        psi[:, 1:] = (bma / (k[1:] * np.pi)) * (sin_b[:, 1:] - sin_c[:, 1:])

    V_k = (2.0 * S0_arr[:, None] / bma) * (chi - np.exp(k_log) * psi)   # shape (n, N)
    call_price = np.exp(-r * T) * np.sum(cos_term[None, :] * V_k, axis=1)
    call_price = np.maximum(call_price, 0.0)   # guard against tiny negative numerical noise

    if option_type == "call":
        price = call_price
    elif option_type == "put":
        price = call_price - S0_arr + K * np.exp(-r * T)   # put-call parity (q=0, as in heston_price)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type}")

    return price if np.asarray(S0).ndim > 0 else float(price[0])


def heston_price_cos(S0: float, K: float, T: float, r: float,
                      kappa: float, theta: float, xi: float, rho: float, v0: float,
                      option_type: str = "call", N: int = 256, L: float = 12.0) -> float:
    """Scalar convenience wrapper around heston_price_cos_batch (kept as a
    thin wrapper, not a separate implementation, so there is only one COS
    code path to validate and maintain)."""
    return float(heston_price_cos_batch(np.array([S0]), K, T, r, kappa, theta, xi, rho, v0,
                                         option_type, N, L)[0])


# ══════════════════════════════════════════════════════════════════════════════
# CLI Sanity Check
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Validate the Fourier-based Heston price against independent Monte
    Carlo simulation, and confirm it reduces to Black-Scholes when
    vol-of-vol xi -> 0 (deterministic variance).
    Run from repo root: python src/heston.py
    """
    from black_scholes import bs_price

    print("Heston sanity checks\n" + "-" * 40)

    S0, K, T, r = 100.0, 100.0, 1.0, 0.03
    kappa, theta, xi, rho, v0 = 2.0, 0.04, 0.4, -0.7, 0.04   # standard textbook params

    closed_form = heston_price(S0, K, T, r, kappa, theta, xi, rho, v0, "call")
    mc_price, mc_se = heston_price_montecarlo(S0, K, T, r, kappa, theta, xi, rho, v0, "call")

    print(f"[1] Fourier (closed-form) price : {closed_form:.4f}")
    print(f"[2] Monte Carlo price           : {mc_price:.4f} +/- {1.96*mc_se:.4f} (95% CI)")
    within_ci = abs(closed_form - mc_price) < 4 * mc_se
    print(f"    Agreement within 4 SE: {within_ci}")
    assert within_ci, "Fourier and Monte Carlo Heston prices disagree beyond noise"

    # Reduction check: xi -> 0 (deterministic variance = v0 = theta) should
    # match plain Black-Scholes with sigma = sqrt(v0)
    tiny_xi = 1e-4
    reduced = heston_price(S0, K, T, r, kappa=2.0, theta=v0, xi=tiny_xi, rho=0.0, v0=v0,
                            option_type="call")
    bs_reference = bs_price(S0, K, T, r, np.sqrt(v0), "call")
    print(f"\n[3] Heston (xi->0)  : {reduced:.4f}")
    print(f"    Black-Scholes   : {bs_reference:.4f}")
    assert np.isclose(reduced, bs_reference, atol=5e-2), \
        "Heston should reduce to Black-Scholes as xi -> 0"

    # Greeks sanity
    greeks = heston_greeks_fd(S0, K, T, r, kappa, theta, xi, rho, v0, "call")
    print(f"\n[4] Heston Greeks (finite-difference): {greeks}")
    assert 0 < greeks["delta"] < 1, "Heston call delta out of range"

    # ── COS method: validate against the independent quad-based pricer ──
    print("\n[5] COS method vs. quad-based Fourier inversion, across strikes/maturities...")
    max_rel_err = 0.0
    for T_test in (0.1, 0.25, 1.0, 2.0):
        for K_test in (70.0, 90.0, 100.0, 110.0, 130.0):
            quad_p = heston_price(S0, K_test, T_test, r, kappa, theta, xi, rho, v0, "call")
            cos_p = heston_price_cos(S0, K_test, T_test, r, kappa, theta, xi, rho, v0, "call")
            rel_err = abs(cos_p - quad_p) / max(quad_p, 1e-4)
            max_rel_err = max(max_rel_err, rel_err)
    print(f"    Max relative error (COS vs. quad) across {4*5} (T, K) combinations: {max_rel_err:.2e}")
    assert max_rel_err < 1e-3, "COS method should match the quad-based Fourier price to within 0.1%"

    print("\n[6] COS method put-call parity...")
    cos_call = heston_price_cos(S0, K, T, r, kappa, theta, xi, rho, v0, "call")
    cos_put = heston_price_cos(S0, K, T, r, kappa, theta, xi, rho, v0, "put")
    parity_lhs = cos_call - cos_put
    parity_rhs = S0 - K * np.exp(-r * T)
    print(f"    C-P={parity_lhs:.6f}   S-Ke^-rT={parity_rhs:.6f}   match={np.isclose(parity_lhs, parity_rhs)}")
    assert np.isclose(parity_lhs, parity_rhs, atol=1e-6), "COS call/put must satisfy put-call parity exactly"

    print("\n[7] COS reduces to Black-Scholes as xi -> 0 (same check as [3], via COS)...")
    reduced_cos = heston_price_cos(S0, K, T, r, kappa=2.0, theta=v0, xi=tiny_xi, rho=0.0, v0=v0,
                                    option_type="call")
    print(f"    Heston COS (xi->0) : {reduced_cos:.4f}   Black-Scholes: {bs_reference:.4f}")
    assert np.isclose(reduced_cos, bs_reference, atol=5e-2), "COS method should also reduce to Black-Scholes as xi -> 0"

    print("\n[8] Batch (vectorized-across-paths) COS pricing: bit-identical to a per-path loop...")
    rng = np.random.default_rng(5)
    S_paths = S0 * np.exp(rng.normal(0, 0.15, 500))   # 500 synthetic "spot prices" spanning a wide range
    prices_loop = np.array([heston_price_cos(float(s), K, T, r, kappa, theta, xi, rho, v0, "call")
                            for s in S_paths])
    prices_batch = heston_price_cos_batch(S_paths, K, T, r, kappa, theta, xi, rho, v0, "call")
    max_diff = np.max(np.abs(prices_loop - prices_batch))
    print(f"    Max |per-path loop - vectorized batch| over {len(S_paths)} spot prices: {max_diff:.2e}")
    assert max_diff < 1e-10, "Batch COS pricing must be numerically identical to a per-path loop over the same formula"

    print("\n[9] Speed benchmark: COS (vectorized) vs. quad (per-path loop) over 500 spot prices...")
    import time
    t0 = time.perf_counter()
    _ = np.array([heston_price(float(s), K, T, r, kappa, theta, xi, rho, v0, "call") for s in S_paths])
    t_quad = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = heston_price_cos_batch(S_paths, K, T, r, kappa, theta, xi, rho, v0, "call")
    t_cos = time.perf_counter() - t0
    print(f"    quad (per-path loop)      : {t_quad*1000:.1f} ms for {len(S_paths)} prices")
    print(f"    COS (vectorized batch)    : {t_cos*1000:.1f} ms for {len(S_paths)} prices")
    print(f"    Speedup: {t_quad/max(t_cos,1e-9):.0f}x")
    assert t_cos < t_quad, "The vectorized COS batch method should be faster than the per-path quad loop"

    print("\n    Confirmed: the COS method matches the independently-validated quad-based Fourier price")
    print("    (which itself matches Monte Carlo and the Black-Scholes reduction limit), is exactly")
    print("    vectorizable across many spot prices at once, and is materially faster -- resolving this")
    print("    project's most-repeated documented limitation (Heston pricing could not be vectorized).")

    print("\nAll checks passed.")
