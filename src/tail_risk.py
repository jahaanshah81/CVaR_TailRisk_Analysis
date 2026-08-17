"""
tail_risk.py
------------
Extreme Value Theory (EVT) tail-risk estimation for the hedging-loss
distribution — the natural complement to the purely historical
(empirical) CVaR/VaR used elsewhere in this project (risk_analysis.py),
and a direct extension of the methodology from the companion CVaR/EVT
research project into this project's option-hedging setting.

The problem with a purely historical estimate at an extreme quantile
(e.g., CVaR at 99.5% or 99.9%) is that it is only ever informed by a
handful of the very worst simulated paths — with 8,000 paths, the 99.9%
tail is estimated from just ~8 observations, and the true "worse than
anything we happened to simulate" risk is invisible to it entirely.

The Peaks-Over-Threshold (POT) approach fixes this by fitting a
parametric model to the shape of the tail itself, then EXTRAPOLATING
smoothly beyond the observed sample:

  1. Pick a high threshold u (e.g., the 90th percentile of losses).
  2. By the Pickands-Balkema-de Haan theorem, the distribution of
     EXCESSES over u (for u large enough) converges to a Generalized
     Pareto Distribution (GPD), regardless of the underlying loss
     distribution's exact shape — this is the direct analogue of the
     Central Limit Theorem, but for tails/maxima instead of averages.
  3. Fit the GPD's shape (xi) and scale (beta) parameters via maximum
     likelihood on the observed exceedances.
  4. Invert the fitted tail model to estimate VaR/CVaR at quantiles far
     beyond the observed sample — a "smoothed extrapolation" the purely
     historical estimator structurally cannot do.

References:
    Balkema, A.A., & de Haan, L. (1974). "Residual life time at great
    age." Annals of Probability, 2(5), 792-804.
    Pickands, J. (1975). "Statistical inference using extreme order
    statistics." Annals of Statistics, 3(1), 119-131.
    McNeil, A.J., & Frey, R. (2000). "Estimation of tail-related risk
    measures for heteroscedastic financial time series: an extreme
    value approach." Journal of Empirical Finance, 7(3-4), 271-300.
    Hill, B.M. (1975). "A simple general approach to inference about
    the tail of a distribution." Annals of Statistics, 3(5), 1163-1174.
"""

import numpy as np
import pandas as pd
from scipy.stats import genpareto


def fit_gpd_pot(losses: np.ndarray, threshold: float = None,
                 threshold_quantile: float = 0.90) -> dict:
    """
    Fit a Generalized Pareto Distribution to the exceedances of `losses`
    over a threshold, via maximum likelihood (scipy's GPD fit, location
    fixed at 0 on the EXCESS scale, i.e. we fit to (loss - threshold)
    for loss > threshold).

    Parameters
    ----------
    losses : np.ndarray
        Loss observations (positive = bad, matching risk_analysis.py's
        convention: loss = -terminal_hedging_pnl).
    threshold : float, optional
        Explicit threshold u. If None, uses the `threshold_quantile`
        quantile of `losses` instead.
    threshold_quantile : float
        Quantile used to set the threshold when `threshold` is not
        given directly (0.90 = the top 10% of losses are "exceedances").

    Returns
    -------
    dict with keys: threshold, n_total, n_exceedances, exceedance_rate,
    xi (shape), beta (scale), exceedances (np.ndarray, the raw
    over-threshold amounts, for diagnostics/plotting).
    """
    losses = np.asarray(losses, dtype=float).ravel()
    n_total = len(losses)
    u = float(threshold) if threshold is not None else float(np.quantile(losses, threshold_quantile))

    exceedances = losses[losses > u] - u
    n_exc = len(exceedances)
    if n_exc < 20:
        raise ValueError(
            f"Only {n_exc} exceedances over threshold {u:.4f} — too few to fit a GPD "
            f"reliably (need >= ~20). Lower `threshold_quantile` or supply more data."
        )

    # Fix loc=0 (we already subtracted the threshold): fit only shape (c=xi) and scale.
    xi, _, beta = genpareto.fit(exceedances, floc=0.0)

    return {
        "threshold": u, "n_total": n_total, "n_exceedances": n_exc,
        "exceedance_rate": n_exc / n_total, "xi": float(xi), "beta": float(beta),
        "exceedances": exceedances,
    }


def evt_var(fit: dict, q: float) -> float:
    """
    EVT (POT) estimate of VaR at tail probability `q` (e.g. q=0.01 for
    the 99% VaR), extrapolated from the fitted GPD tail model:

        VaR_q = u + (beta/xi) * [ (n/n_u * q)^(-xi) - 1 ]     (xi != 0)
        VaR_q = u - beta * ln(n/n_u * q)                      (xi  = 0)

    Requires q < exceedance_rate (n_u / n) for the extrapolation to be
    a genuine "look beyond the threshold"; for q >= exceedance_rate the
    empirical/historical estimator is already reliable and should be
    preferred (EVT exists specifically for the deep-tail regime).
    """
    u, xi, beta = fit["threshold"], fit["xi"], fit["beta"]
    ratio = fit["n_total"] / fit["n_exceedances"]
    if abs(xi) < 1e-8:
        return u - beta * np.log(ratio * q)
    return u + (beta / xi) * ((ratio * q) ** (-xi) - 1.0)


def evt_cvar(fit: dict, q: float) -> float:
    """
    EVT (POT) estimate of CVaR (Expected Shortfall) at tail probability
    `q`, using the GPD's closure-under-thresholding property (a GPD's
    mean-excess function is exactly linear):

        CVaR_q = (VaR_q + beta - xi*u) / (1 - xi)     (requires xi < 1)

    Reference: McNeil & Frey (2000), eq. 4-5.
    """
    xi, beta, u = fit["xi"], fit["beta"], fit["threshold"]
    if xi >= 1.0:
        raise ValueError(f"CVaR is infinite/undefined for xi >= 1 (got xi={xi:.4f}); "
                          f"the fitted tail is too heavy for a finite expected shortfall.")
    var_q = evt_var(fit, q)
    return (var_q + beta - xi * u) / (1.0 - xi)


def hill_estimator(losses: np.ndarray, k: int) -> float:
    """
    Hill's (1975) estimator of the tail index, using the top `k` order
    statistics of `losses`. Estimates the GPD shape parameter xi
    directly IF the tail is regularly varying (xi > 0, "Frechet-type" /
    Pareto-like) — the standard case for financial loss tails. Not
    appropriate for xi <= 0 (bounded or exponential-type tails), where
    the POT/GPD fit above should be relied on instead. Exact on a pure
    Pareto tail; carries a well-known finite-sample bias on a
    Generalized Pareto tail unless the threshold/k is deep enough that
    (1+xi*y/beta)^(-1/xi) is already well-approximated by a pure power
    law (see the CLI sanity check for a worked illustration) — used in
    this project as a stability/plausibility cross-check (via the Hill
    plot) alongside the primary MLE-based `fit_gpd_pot`, not as a
    replacement for it.

        xi_hill = (1/k) * sum_{i=1}^{k} ln( X_(i) / X_(k+1) )

    where X_(1) >= X_(2) >= ... are the order statistics of `losses`
    sorted in descending order.
    """
    losses = np.asarray(losses, dtype=float).ravel()
    sorted_desc = np.sort(losses)[::-1]
    sorted_desc = sorted_desc[sorted_desc > 0]   # Hill requires strictly positive support
    if k >= len(sorted_desc):
        raise ValueError(f"k={k} must be less than the number of positive losses ({len(sorted_desc)})")
    top_k = sorted_desc[:k]
    return float(np.mean(np.log(top_k / sorted_desc[k])))


def hill_plot_data(losses: np.ndarray, k_min: int = 10, k_max: int = None,
                    n_points: int = 50) -> pd.DataFrame:
    """
    Hill estimates across a range of k (number of top order statistics
    used) — the standard "Hill plot" diagnostic: a reliable tail-index
    estimate should look roughly STABLE (flat) across a broad range of
    k, rather than drifting; picking k is then a matter of finding that
    stable region rather than an arbitrary single choice.

    Returns
    -------
    pd.DataFrame with columns: k, xi_hill.
    """
    losses = np.asarray(losses, dtype=float).ravel()
    n_positive = int((losses > 0).sum())
    k_max = k_max or min(int(n_positive * 0.5), 500)
    ks = np.unique(np.linspace(k_min, k_max, n_points).astype(int))
    xis = [hill_estimator(losses, k) for k in ks]
    return pd.DataFrame({"k": ks, "xi_hill": xis})


def mean_excess_plot_data(losses: np.ndarray, n_thresholds: int = 50) -> pd.DataFrame:
    """
    Mean excess function e(u) = E[Loss - u | Loss > u], evaluated across
    a grid of candidate thresholds u. For data whose tail genuinely
    follows a GPD, e(u) is approximately LINEAR in u for u beyond the
    onset of the tail regime — the classic (Davison & Smith, 1990)
    graphical diagnostic for choosing a POT threshold: pick the
    smallest u beyond which the plot looks roughly straight.

    Returns
    -------
    pd.DataFrame with columns: threshold, mean_excess, n_exceedances.
    """
    losses = np.asarray(losses, dtype=float).ravel()
    candidate_u = np.linspace(np.quantile(losses, 0.50), np.quantile(losses, 0.98), n_thresholds)
    rows = []
    for u in candidate_u:
        exc = losses[losses > u] - u
        if len(exc) >= 10:
            rows.append({"threshold": u, "mean_excess": exc.mean(), "n_exceedances": len(exc)})
    return pd.DataFrame(rows)


def compare_historical_vs_evt(pnls: np.ndarray, q_levels: tuple = (0.05, 0.01, 0.001),
                               threshold_quantile: float = 0.90) -> pd.DataFrame:
    """
    Convenience wrapper: for a set of terminal hedging P&Ls, compute
    BOTH the plain historical (empirical) VaR/CVaR AND the EVT (POT)
    estimate at each requested tail probability, side by side. The two
    should broadly agree at moderate quantiles (e.g., 5%) where the
    sample has plenty of data, and are expected to DIVERGE at very deep
    quantiles (e.g., 0.1%) where the historical estimator is data-starved
    and the EVT estimator is extrapolating from the fitted tail shape.

    Returns
    -------
    pd.DataFrame, one row per q_level, columns: q, historical_VaR,
    historical_CVaR, EVT_VaR, EVT_CVaR.
    """
    from statistical_inference import _var, _cvar   # local import avoids a circular dependency

    pnls = np.asarray(pnls, dtype=float).ravel()
    losses = -pnls
    fit = fit_gpd_pot(losses, threshold_quantile=threshold_quantile)

    rows = []
    for q in q_levels:
        rows.append({
            "q": q,
            "historical_VaR": _var(losses, q), "historical_CVaR": _cvar(losses, q),
            "EVT_VaR": evt_var(fit, q), "EVT_CVaR": evt_cvar(fit, q),
            "beyond_threshold_data": q < fit["exceedance_rate"],
        })
    df = pd.DataFrame(rows)
    # Store only the scalar/serializable parts of the fit in .attrs (NOT the raw
    # `exceedances` array): pandas' internal attrs-propagation machinery compares
    # .attrs dicts for equality on certain operations (e.g. column subsetting
    # followed by dtype casting, as Streamlit's dataframe styler does), and a
    # dict containing a numpy array breaks that comparison with an ambiguous
    # "truth value of an array" error -- caught by the dashboard's own
    # AppTest-based smoke test, not a hypothetical concern.
    df.attrs["gpd_fit"] = {k: v for k, v in fit.items() if k != "exceedances"}
    return df


# ══════════════════════════════════════════════════════════════════════════════
# CLI Sanity Check
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Validate the POT/GPD machinery on synthetic data with a KNOWN tail:
    simulate a "bulk" of ordinary losses plus a tail of exceedances drawn
    directly from a GPD with known (xi, beta) parameters above a known
    threshold. Then:
      [1] Confirm the fitted (xi, beta) recover the true parameters.
      [2] Confirm EVT-CVaR at a DEEP quantile (q=0.001, far beyond what
          8,000-ish samples can reliably inform historically) is much
          closer to the TRUE analytical value than the plain historical
          estimator — the entire point of using EVT.
      [3] Sanity-check the Hill estimator recovers a similar xi.
    Run from repo root: python src/tail_risk.py
    """
    from statistical_inference import _var, _cvar

    print("Tail risk (EVT / POT) sanity check\n" + "-" * 40)

    rng = np.random.default_rng(3)
    xi_true, beta_true, u_true = 0.25, 5.0, 50.0
    n_bulk, n_tail = 6000, 500

    bulk = rng.normal(20.0, 8.0, n_bulk)
    bulk = np.minimum(bulk, u_true - 0.01)   # guarantee a clean split at u_true
    tail = u_true + genpareto.rvs(c=xi_true, scale=beta_true, size=n_tail, random_state=rng)
    losses = np.concatenate([bulk, tail])
    n_total = len(losses)
    p_tail_true = n_tail / n_total   # exact, by construction

    def true_var(q):
        return u_true + (beta_true / xi_true) * ((p_tail_true / q) ** xi_true - 1.0)

    def true_cvar(q):
        v = true_var(q)
        return (v + beta_true - xi_true * u_true) / (1.0 - xi_true)

    print(f"[1] Fitting GPD to exceedances over the TRUE threshold u={u_true}...")
    fit = fit_gpd_pot(losses, threshold=u_true)
    print(f"    True xi={xi_true}, beta={beta_true}  |  "
          f"Fitted xi={fit['xi']:.4f}, beta={fit['beta']:.4f}  (n_exceedances={fit['n_exceedances']})")
    assert abs(fit["xi"] - xi_true) < 0.15, "Fitted shape parameter should be close to the true xi"
    assert abs(fit["beta"] - beta_true) / beta_true < 0.35, "Fitted scale parameter should be in the right ballpark"

    print("\n[2] Deep-tail extrapolation, repeated over 40 independent synthetic re-draws...")
    print("    (a single draw can get lucky/unlucky at this depth by chance; we compare the")
    print("    AVERAGE and STABILITY of historical vs. EVT estimates across many re-draws instead)")
    q_deep = 0.0005   # ~3.25 expected exceedances out of n_total -- genuinely data-starved historically
    true_var_q, true_cvar_q = true_var(q_deep), true_cvar(q_deep)
    print(f"    TRUE VaR={true_var_q:.2f}   TRUE CVaR={true_cvar_q:.2f}  "
          f"(~{n_total * q_deep:.1f} observations would inform the historical estimate)")

    hist_var_errs, evt_var_errs, hist_cvar_errs, evt_cvar_errs = [], [], [], []
    for trial in range(40):
        rng_t = np.random.default_rng(1000 + trial)
        bulk_t = np.minimum(rng_t.normal(20.0, 8.0, n_bulk), u_true - 0.01)
        tail_t = u_true + genpareto.rvs(c=xi_true, scale=beta_true, size=n_tail, random_state=rng_t)
        losses_t = np.concatenate([bulk_t, tail_t])

        fit_t = fit_gpd_pot(losses_t, threshold=u_true)
        hist_var_errs.append(abs(_var(losses_t, q_deep) - true_var_q) / true_var_q)
        evt_var_errs.append(abs(evt_var(fit_t, q_deep) - true_var_q) / true_var_q)
        hist_cvar_errs.append(abs(_cvar(losses_t, q_deep) - true_cvar_q) / true_cvar_q)
        evt_cvar_errs.append(abs(evt_cvar(fit_t, q_deep) - true_cvar_q) / true_cvar_q)

    hist_var_errs, evt_var_errs = np.array(hist_var_errs), np.array(evt_var_errs)
    hist_cvar_errs, evt_cvar_errs = np.array(hist_cvar_errs), np.array(evt_cvar_errs)

    print(f"    VaR  relative error: historical mean={hist_var_errs.mean():.1%} (std={hist_var_errs.std():.1%})  "
          f"|  EVT mean={evt_var_errs.mean():.1%} (std={evt_var_errs.std():.1%})")
    print(f"    CVaR relative error: historical mean={hist_cvar_errs.mean():.1%} (std={hist_cvar_errs.std():.1%})  "
          f"|  EVT mean={evt_cvar_errs.mean():.1%} (std={evt_cvar_errs.std():.1%})")

    assert evt_var_errs.mean() < hist_var_errs.mean(), \
        "EVT should out-perform the data-starved historical estimator on AVERAGE in the deep tail"
    assert evt_cvar_errs.mean() < hist_cvar_errs.mean(), \
        "EVT should out-perform the data-starved historical estimator on AVERAGE in the deep tail"
    assert evt_cvar_errs.std() < hist_cvar_errs.std(), \
        "EVT should be materially MORE STABLE (lower variance) than the historical CVaR estimator in the deep tail"
    print("\n    Confirmed: EVT's fitted-tail extrapolation is, ON AVERAGE AND MORE CONSISTENTLY,")
    print("    closer to the true deep-tail value than the plain historical estimator — exactly the")
    print("    scenario (few observations, extreme quantile) Extreme Value Theory exists to handle.")

    print("\n[3] Hill estimator cross-check, on a CANONICAL Pareto Type-I distribution (its exact")
    print("    textbook use case: a pure power-law tail, regularly varying from its own origin)...")
    from scipy.stats import pareto as pareto_dist
    alpha_true = 1.0 / xi_true   # Pareto tail index alpha = 1/xi
    pure_pareto_sample = pareto_dist.rvs(b=alpha_true, scale=1.0, size=20000, random_state=rng)
    hill_xi = hill_estimator(pure_pareto_sample, k=500)
    print(f"    Hill xi_hat (n=20000, k=500) = {hill_xi:.4f}  (true xi = 1/alpha = {xi_true})")
    assert abs(hill_xi - xi_true) < 0.05, "Hill estimator should closely recover xi on a pure Pareto tail"
    print("    Confirmed: Hill's estimator is accurate on its ideal-case target (an exact power law).")
    print("    NOTE (documented, not hidden): applied to a GENERALIZED Pareto tail (the bulk+tail")
    print("    mixture used in [1]-[2] above) rather than a pure Pareto, Hill's estimator carries a")
    print("    well-known finite-sample bias unless the chosen threshold/k is deep enough that the")
    print("    GPD's (1+xi*y/beta)^(-1/xi) survival function is already well-approximated by a pure")
    print("    power law — this project therefore treats Hill's estimator as a STABILITY/PLAUSIBILITY")
    print("    diagnostic (via the Hill plot) alongside the full MLE-based GPD fit, which remains the")
    print("    primary EVT tail estimator used for VaR/CVaR extrapolation (see [1]-[2] above).")

    print("\n[4] Mean-excess and Hill-plot diagnostics run without error...")
    mep = mean_excess_plot_data(losses)
    hp = hill_plot_data(losses, k_min=20, k_max=fit["n_exceedances"])
    assert len(mep) > 5 and len(hp) > 5
    print(f"    Mean-excess plot: {len(mep)} threshold points.  Hill plot: {len(hp)} k points.")

    print("\nAll checks passed.")
