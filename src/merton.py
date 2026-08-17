"""
merton.py
---------
Merton (1976) jump-diffusion option pricing model.

Extends Black-Scholes by adding a compound Poisson jump process to the
underlying's dynamics, capturing the fat-tailed, crash-prone behaviour
that continuous GBM cannot:

    dS_t / S_t = (r - lambda*k - 0.5*sigma^2) dt + sigma dW_t + dJ_t

where J_t is a compound Poisson process: jumps arrive at rate `lambda`
(expected number of jumps per year), and each jump's log-size is
    ln(1 + Y) ~ Normal(mu_J, sigma_J^2)
k = E[Y] = exp(mu_J + 0.5*sigma_J^2) - 1  is the expected relative jump
size, used to compensate the drift so the discounted price process
remains a martingale under the risk-neutral measure.

Merton showed the resulting option price has a quasi-closed-form solution:
a Poisson-weighted mixture of Black-Scholes prices, one for each possible
number of jumps during the option's life. This avoids Monte Carlo for
pricing, though we still cross-validate against Monte Carlo below.

Reference: Merton, R.C. (1976). "Option pricing when underlying stock
returns are discontinuous." Journal of Financial Economics, 3(1-2), 125-144.
"""

import numpy as np
from scipy.special import factorial

from black_scholes import bs_price


def merton_price(S: float, K: float, T: float, r: float, sigma: float,
                  lam: float, mu_j: float, sigma_j: float,
                  option_type: str = "call", n_terms: int = 50) -> float:
    """
    Merton jump-diffusion European option price via the Poisson-weighted
    sum of Black-Scholes prices.

    Parameters
    ----------
    S, K, T, r, sigma, option_type : as in black_scholes.py (sigma here is
        the *diffusive* volatility, excluding the jump contribution).
    lam : float
        Jump intensity (expected number of jumps per year), lambda > 0.
    mu_j : float
        Mean of the log jump size, ln(1+Y) ~ N(mu_j, sigma_j^2).
    sigma_j : float
        Std. dev. of the log jump size.
    n_terms : int
        Number of terms in the Poisson sum to truncate at. 50 is more
        than sufficient for any realistic (lambda*T) since Poisson
        probabilities decay factorially.

    Returns
    -------
    float : option price.
    """
    k = np.exp(mu_j + 0.5 * sigma_j ** 2) - 1.0     # expected relative jump size
    lam_prime = lam * (1.0 + k)                      # risk-neutral compensated intensity

    price = 0.0
    for n in range(n_terms):
        # Poisson probability of exactly n jumps over [0, T]
        poisson_weight = np.exp(-lam_prime * T) * (lam_prime * T) ** n / factorial(n)
        if poisson_weight < 1e-14 and n > 5:
            break   # remaining terms are numerically negligible

        # Adjusted rate and volatility for the n-jump scenario
        r_n = r - lam * k + n * np.log(1.0 + k) / T
        sigma_n = np.sqrt(sigma ** 2 + n * sigma_j ** 2 / T)
        sigma_n = max(sigma_n, 1e-6)   # numerical floor to avoid sigma=0 when n=0, sigma=0

        price += poisson_weight * bs_price(S, K, T, r_n, sigma_n, option_type)

    return price


def merton_simulate_paths(S0: float, T: float, r: float, sigma: float,
                           lam: float, mu_j: float, sigma_j: float,
                           n_paths: int, n_steps: int,
                           seed: int = 42) -> np.ndarray:
    """
    Simulate Merton jump-diffusion price paths via Euler discretization
    with a Poisson-thinned jump increment at each step. Used both for
    Monte Carlo cross-validation of merton_price(), and later as the
    "true" data-generating process for hedging simulations.

    Returns
    -------
    np.ndarray of shape (n_paths, n_steps+1), each row a simulated path.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    k = np.exp(mu_j + 0.5 * sigma_j ** 2) - 1.0
    drift = (r - lam * k - 0.5 * sigma ** 2) * dt

    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = S0

    for t in range(n_steps):
        z = rng.standard_normal(n_paths)
        diffusion = sigma * np.sqrt(dt) * z

        n_jumps = rng.poisson(lam * dt, size=n_paths)
        jump_sizes = np.zeros(n_paths)
        has_jump = n_jumps > 0
        if has_jump.any():
            # Sum of n_jumps i.i.d. N(mu_j, sigma_j^2) log-jumps per path
            for idx in np.where(has_jump)[0]:
                jump_sizes[idx] = rng.normal(mu_j, sigma_j, size=n_jumps[idx]).sum()

        paths[:, t + 1] = paths[:, t] * np.exp(drift + diffusion + jump_sizes)

    return paths


def merton_price_montecarlo(S: float, K: float, T: float, r: float, sigma: float,
                             lam: float, mu_j: float, sigma_j: float,
                             option_type: str = "call",
                             n_paths: int = 200_000, n_steps: int = 100,
                             seed: int = 42) -> tuple[float, float]:
    """
    Monte Carlo price of a Merton jump-diffusion option, used purely to
    cross-validate the closed-form merton_price() above.

    Returns
    -------
    (price, standard_error) : tuple of float
    """
    paths = merton_simulate_paths(S, T, r, sigma, lam, mu_j, sigma_j,
                                   n_paths, n_steps, seed)
    S_T = paths[:, -1]

    if option_type == "call":
        payoff = np.maximum(S_T - K, 0.0)
    else:
        payoff = np.maximum(K - S_T, 0.0)

    discounted = np.exp(-r * T) * payoff
    price = discounted.mean()
    se = discounted.std(ddof=1) / np.sqrt(n_paths)
    return price, se


# ══════════════════════════════════════════════════════════════════════════════
# CLI Sanity Check
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Validate the closed-form Merton price against independent Monte Carlo
    simulation, and confirm it reduces to Black-Scholes as lambda -> 0.
    Run from repo root: python src/merton.py
    """
    print("Merton jump-diffusion sanity checks\n" + "-" * 40)

    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.03, 0.20
    lam, mu_j, sigma_j = 0.5, -0.10, 0.15    # moderate jump risk, negative skew

    closed_form = merton_price(S, K, T, r, sigma, lam, mu_j, sigma_j, "call")
    mc_price, mc_se = merton_price_montecarlo(S, K, T, r, sigma, lam, mu_j, sigma_j, "call")

    print(f"[1] Closed-form price : {closed_form:.4f}")
    print(f"[2] Monte Carlo price : {mc_price:.4f} +/- {1.96*mc_se:.4f} (95% CI)")
    within_ci = abs(closed_form - mc_price) < 3 * mc_se
    print(f"    Agreement within 3 SE: {within_ci}")
    assert within_ci, "Closed-form and Monte Carlo Merton prices disagree beyond noise"

    # Reduction check: lambda=0 (no jumps) should equal plain Black-Scholes
    no_jump_price = merton_price(S, K, T, r, sigma, lam=1e-8, mu_j=0.0, sigma_j=1e-8,
                                  option_type="call")
    bs_reference = bs_price(S, K, T, r, sigma, "call")
    print(f"\n[3] Merton (lambda->0) : {no_jump_price:.4f}")
    print(f"    Black-Scholes      : {bs_reference:.4f}")
    assert np.isclose(no_jump_price, bs_reference, atol=1e-3), \
        "Merton should reduce to Black-Scholes as lambda -> 0"

    print("\nAll checks passed.")
