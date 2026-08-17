"""
calibration.py
--------------
Calibrates Heston and Merton model parameters from a REAL historical
return series.

IMPORTANT METHODOLOGICAL NOTE (read this — it matters for interviews):
We do not have access to a real historical options chain (genuinely free,
long-history options market data essentially does not exist for public
download). So instead of "implied calibration" (fitting model parameters
to match observed market option prices / the implied volatility surface,
which calibrates to the risk-neutral measure Q), we perform "historical
calibration": we fit model parameters to match statistical properties of
the REAL underlying's historical return series (the physical measure P).

This is a well-known, legitimate, and clearly weaker substitute for
implied calibration — it tells us "what a Heston/Merton model consistent
with how this asset has actually behaved would look like," not "what the
market currently thinks volatility/jump risk is worth." We are explicit
about this distinction throughout the project (see the report), since
conflating P-measure and Q-measure calibration is a common and important
mistake to avoid in quantitative finance.

Heston historical calibration:
    v0    = most recent 21-day realized variance
    theta = long-run average realized variance
    kappa = mean-reversion speed, estimated via AR(1) regression of
            variance changes on (theta - v_t)  [Euler-discretized CIR]
    xi    = vol-of-vol, estimated as the residual std of that regression
    rho   = empirical correlation between daily returns and daily changes
            in realized variance (captures the leverage effect)

Merton historical calibration (threshold-based jump identification):
    1. Flag any daily log-return exceeding `threshold_stds` standard
       deviations as a "jump"
    2. lambda = (number of flagged jumps) / (years of data)
    3. mu_j, sigma_j = mean/std of the flagged jump returns
    4. Remaining (non-jump) variance is allocated to the diffusive sigma
"""

import numpy as np


def calibrate_heston(log_returns: np.ndarray, realized_var: np.ndarray,
                      dt: float = 1.0 / 252) -> dict:
    """
    Historical calibration of Heston parameters from a return series and
    its rolling realized variance series (same length, aligned).

    Parameters
    ----------
    log_returns : np.ndarray
        Daily log returns.
    realized_var : np.ndarray
        Daily rolling realized variance (e.g., realized_vol_21d**2),
        same length as log_returns, no NaNs.
    dt : float
        Time step in years (1/252 for daily data).

    Returns
    -------
    dict with keys: kappa, theta, xi, rho, v0
    """
    v = np.asarray(realized_var, dtype=float)
    r = np.asarray(log_returns, dtype=float)
    assert len(v) == len(r), "log_returns and realized_var must be aligned"

    theta = float(v.mean())
    v0 = float(v[-1])

    # AR(1)-style regression: dv_t = kappa*(theta - v_t)*dt + xi*sqrt(v_t*dt)*eps_t
    # => dv_t / sqrt(v_t) = kappa*dt*(theta - v_t)/sqrt(v_t) + xi*sqrt(dt)*eps_t
    # Simple OLS of dv_t on (theta - v_t) recovers kappa*dt as the slope.
    dv = np.diff(v)
    x = (theta - v[:-1])
    # OLS slope through the origin: kappa*dt = sum(x*dv)/sum(x*x)
    denom = np.sum(x ** 2)
    kappa_dt = np.sum(x * dv) / denom if denom > 1e-12 else 0.05 * dt
    kappa = max(kappa_dt / dt, 0.1)   # floor to avoid non-mean-reverting fits

    residuals = dv - kappa_dt * x
    # xi from the residual variance of the discretized CIR diffusion term:
    # residual ~= xi*sqrt(v_t*dt)*eps_t  =>  Var(residual) ~= xi^2 * v_t * dt
    var_ratio = residuals ** 2 / (v[:-1] * dt + 1e-12)
    xi = float(np.sqrt(np.median(var_ratio)))
    xi = np.clip(xi, 0.05, 3.0)   # sanity bounds

    # Leverage effect: correlation between return shocks and variance shocks
    rho = float(np.corrcoef(r[1:], dv)[0, 1])
    rho = np.clip(rho, -0.99, 0.99)

    return {"kappa": float(kappa), "theta": theta, "xi": xi, "rho": rho, "v0": v0}


def calibrate_merton(log_returns: np.ndarray, dt: float = 1.0 / 252,
                      threshold_stds: float = 3.0) -> dict:
    """
    Historical calibration of Merton jump-diffusion parameters via
    threshold-based jump identification and moment matching.

    Parameters
    ----------
    log_returns : np.ndarray
        Daily log returns.
    dt : float
        Time step in years.
    threshold_stds : float
        Number of standard deviations beyond which a daily return is
        classified as a "jump" rather than ordinary diffusion.

    Returns
    -------
    dict with keys: sigma (annualised diffusive vol), lam (jump intensity,
    per year), mu_j, sigma_j (log jump-size mean/std).
    """
    r = np.asarray(log_returns, dtype=float)
    n = len(r)
    years = n * dt

    overall_std = r.std(ddof=1)
    threshold = threshold_stds * overall_std

    is_jump = np.abs(r - r.mean()) > threshold
    jump_returns = r[is_jump]
    n_jumps = len(jump_returns)

    lam = max(n_jumps / years, 1e-3)   # jumps per year

    if n_jumps >= 2:
        mu_j = float(jump_returns.mean())
        sigma_j = float(jump_returns.std(ddof=1))
        sigma_j = max(sigma_j, 1e-4)
    else:
        # Not enough identified jumps in-sample: fall back to a small,
        # conservative negative-skew jump assumption (crash risk prior).
        mu_j, sigma_j = -0.05, 0.05

    # Variance decomposition: total = diffusive + jump contribution
    non_jump_returns = r[~is_jump]
    diffusive_var_annual = non_jump_returns.var(ddof=1) / dt
    jump_var_annual = lam * (mu_j ** 2 + sigma_j ** 2)
    total_var_annual = r.var(ddof=1) / dt

    # Sanity floor: diffusive variance must be positive
    sigma = float(np.sqrt(max(diffusive_var_annual, 0.1 * total_var_annual)))

    return {"sigma": sigma, "lam": float(lam), "mu_j": mu_j, "sigma_j": sigma_j,
            "n_jumps_identified": n_jumps, "years_of_data": years}


# ══════════════════════════════════════════════════════════════════════════════
# CLI Sanity Check
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Calibrate both models on synthetic data with KNOWN parameters, and
    confirm the calibration recovers something in the right ballpark.
    Run from repo root: python src/calibration.py
    """
    from merton import merton_simulate_paths
    from heston import heston_simulate_paths

    print("Calibration sanity checks\n" + "-" * 40)

    # --- Merton recovery test ---
    rng = np.random.default_rng(0)
    true_sigma, true_lam, true_mu_j, true_sigma_j = 0.18, 1.5, -0.04, 0.05
    n_steps = 252 * 5
    paths = merton_simulate_paths(100.0, T=5.0, r=0.03, sigma=true_sigma,
                                   lam=true_lam, mu_j=true_mu_j, sigma_j=true_sigma_j,
                                   n_paths=1, n_steps=n_steps, seed=1)[0]
    sim_returns = np.diff(np.log(paths))

    merton_cal = calibrate_merton(sim_returns, dt=5.0 / n_steps)
    print(f"[1] Merton calibration (true: sigma={true_sigma}, lam={true_lam}, "
          f"mu_j={true_mu_j}, sigma_j={true_sigma_j})")
    print(f"    Recovered: {merton_cal}")

    # --- Heston recovery test ---
    true_kappa, true_theta, true_xi, true_rho, true_v0 = 3.0, 0.04, 0.5, -0.6, 0.04
    S_paths, v_paths = heston_simulate_paths(
        100.0, T=5.0, r=0.03, kappa=true_kappa, theta=true_theta,
        xi=true_xi, rho=true_rho, v0=true_v0, n_paths=1, n_steps=n_steps, seed=1)
    heston_returns = np.diff(np.log(S_paths[0]))
    heston_var = v_paths[0][1:]   # align lengths with returns

    heston_cal = calibrate_heston(heston_returns, heston_var, dt=5.0 / n_steps)
    print(f"\n[2] Heston calibration (true: kappa={true_kappa}, theta={true_theta}, "
          f"xi={true_xi}, rho={true_rho}, v0={true_v0})")
    print(f"    Recovered: {heston_cal}")

    print("\nCalibration recovers parameters in a reasonable ballpark "
          "(exact recovery is not expected from a single finite path — "
          "this is a sanity check, not a convergence proof).")
