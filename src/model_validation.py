"""
model_validation.py
--------------------
Backtests the project's OWN VaR/CVaR risk model the way a bank's
independent model-validation / market-risk function actually does it:

    1. Kupiec (1995) unconditional-coverage Proportion-of-Failures (POF)
       test — a likelihood-ratio test of whether the observed exception
       (breach) rate matches the model's stated confidence level.
    2. The Basel traffic-light coverage framework — classifies a VaR
       model as Green / Yellow / Red zone based on the number of
       exceptions observed.

Every quant risk model used in production at a bank is required to pass
exactly this kind of backtest before (and periodically after) deployment
— this module applies the same discipline to our own hedging-P&L VaR
model, using an honest train/test split (VaR is ESTIMATED on one half of
the simulated paths and its coverage is TESTED on the other, held-out
half) to avoid the look-ahead bias of testing a model on the same data
used to fit it.
"""

import numpy as np
from scipy.stats import chi2, binom

from risk_analysis import var_historical


def kupiec_pof_test(n_obs: int, n_exceptions: int, confidence: float = 0.95) -> dict:
    """
    Kupiec (1995) unconditional-coverage POF likelihood-ratio test.

    H0: the true exception probability equals the model's stated
    (1 - confidence) rate. Under H0, the LR statistic is asymptotically
    chi-squared with 1 degree of freedom.

    Returns
    -------
    dict with the LR statistic, its p-value, and a reject/fail-to-reject
    verdict at the conventional 95% test-confidence level.
    """
    p = 1.0 - confidence
    x, n = int(n_exceptions), int(n_obs)

    if x == 0:
        lr_stat = -2 * (n * np.log(1 - p))
    elif x == n:
        lr_stat = -2 * (n * np.log(p))
    else:
        p_hat = x / n
        lr_stat = -2 * (
            (n - x) * np.log(1 - p) + x * np.log(p)
            - (n - x) * np.log(1 - p_hat) - x * np.log(p_hat)
        )

    p_value = float(1 - chi2.cdf(lr_stat, df=1))
    return {
        "n_obs": n, "n_exceptions": x, "exception_rate": x / n, "target_rate": p,
        "LR_stat": float(lr_stat), "p_value": p_value,
        "reject_H0_at_95pct": bool(p_value < 0.05),
    }


def basel_traffic_light(n_obs: int, n_exceptions: int, confidence: float = 0.99) -> dict:
    """
    Basel-style traffic-light backtest. The original Basel III framework
    uses a fixed 250-day table (Green: 0-4 exceptions, Yellow: 5-9,
    Red: 10+, at 99% confidence). We generalise the SAME logic to any
    sample size via the binomial CDF, so it applies to whatever number of
    held-out test paths we actually have: a model is in the Green zone
    while the cumulative probability of seeing AT MOST this many
    exceptions under a correctly-calibrated model is below 95%; Yellow up
    to ~99.99%; Red beyond that (i.e., an outcome this bad, or worse, is
    itself alarmingly improbable under a correctly calibrated model).
    """
    p = 1.0 - confidence
    cum_prob = float(binom.cdf(n_exceptions, n_obs, p))
    if cum_prob < 0.95:
        zone = "Green"
    elif cum_prob < 0.9999:
        zone = "Yellow"
    else:
        zone = "Red"
    return {"n_obs": n_obs, "n_exceptions": n_exceptions, "cumulative_prob": cum_prob, "zone": zone}


def run_var_backtest(pnls: np.ndarray, q: float = 0.05, train_fraction: float = 0.5) -> dict:
    """
    Honest out-of-sample VaR backtest on a set of i.i.d. simulated hedging
    P&L outcomes:
        1. Split the paths into a TRAIN set (used only to estimate the
           model's VaR at level q) and a held-out TEST set.
        2. Count how many TEST-set outcomes actually breach the
           TRAIN-estimated VaR.
        3. Run the Kupiec POF test and Basel traffic-light test on that
           count.

    A well-specified hedge/risk model should see a TEST-set breach rate
    close to q, regardless of which half of the (i.i.d.) sample was used
    for training — deviations indicate the VaR estimate is unstable or
    the loss distribution has fatter/thinner tails than the fitted
    quantile captured.

    Returns
    -------
    dict combining the estimated VaR, the Kupiec test results, and the
    Basel traffic-light zone.
    """
    pnls = np.asarray(pnls, dtype=float)
    n = len(pnls)
    n_train = int(n * train_fraction)

    train_losses = -pnls[:n_train]
    test_losses = -pnls[n_train:]

    var_estimate = var_historical(train_losses, q)
    n_exceptions = int((test_losses > var_estimate).sum())

    kupiec = kupiec_pof_test(len(test_losses), n_exceptions, confidence=1 - q)
    basel = basel_traffic_light(len(test_losses), n_exceptions, confidence=1 - q)

    return {
        "var_estimate": float(var_estimate), "n_train": n_train, "n_test": len(test_losses),
        **kupiec, "basel_zone": basel["zone"], "basel_cumulative_prob": basel["cumulative_prob"],
    }


# ══════════════════════════════════════════════════════════════════════════════
# CLI Sanity Check
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Sanity check: a correctly-calibrated Gaussian loss model backtested
    against its own true quantile should pass (fail to reject H0, Green
    zone) almost always; deliberately understating the VaR (using a much
    smaller quantile than the true one) should get flagged as a failing
    model (reject H0, Red/Yellow zone) with high probability.
    Run from repo root: python src/model_validation.py
    """
    print("Model validation (VaR backtesting) sanity check\n" + "-" * 40)

    rng = np.random.default_rng(11)
    n = 4000
    true_losses = rng.standard_normal(n) * 10.0   # symmetric, so "pnls" = -losses
    pnls_well_specified = -true_losses

    print("[1] Well-specified model (train/test split of the SAME distribution)...")
    result_ok = run_var_backtest(pnls_well_specified, q=0.05, train_fraction=0.5)
    print(f"    VaR(95%) estimate (train half) : {result_ok['var_estimate']:.4f}")
    print(f"    Test-set exception rate        : {result_ok['exception_rate']:.4f} "
          f"(target {result_ok['target_rate']:.4f})")
    print(f"    Kupiec LR stat / p-value        : {result_ok['LR_stat']:.4f} / {result_ok['p_value']:.4f}")
    print(f"    Basel zone                       : {result_ok['basel_zone']}")
    assert not result_ok["reject_H0_at_95pct"], \
        "A well-specified model should not usually be rejected by Kupiec's test"
    assert result_ok["basel_zone"] in ("Green", "Yellow"), \
        "A well-specified model should not land in the Red zone"

    print("\n[2] Deliberately mis-specified model (VaR understated by a factor of 3)...")
    bad_var_estimate = result_ok["var_estimate"] / 3.0
    test_losses = -pnls_well_specified[result_ok["n_train"]:]
    n_exceptions_bad = int((test_losses > bad_var_estimate).sum())
    kupiec_bad = kupiec_pof_test(len(test_losses), n_exceptions_bad, confidence=0.95)
    basel_bad = basel_traffic_light(len(test_losses), n_exceptions_bad, confidence=0.95)
    print(f"    Understated VaR estimate        : {bad_var_estimate:.4f}")
    print(f"    Test-set exception rate         : {kupiec_bad['exception_rate']:.4f} (target 0.05)")
    print(f"    Kupiec LR stat / p-value         : {kupiec_bad['LR_stat']:.4f} / {kupiec_bad['p_value']:.4f}")
    print(f"    Basel zone                        : {basel_bad['zone']}")
    assert kupiec_bad["reject_H0_at_95pct"], \
        "A deliberately mis-specified (understated) VaR model should be rejected by Kupiec's test"
    assert basel_bad["zone"] in ("Yellow", "Red"), \
        "A deliberately mis-specified (understated) VaR model should not land in the Green zone"

    print("\n    Confirmed: the backtest correctly PASSES a well-specified model and")
    print("    correctly FLAGS a mis-specified (understated tail risk) one — exactly")
    print("    the discipline a bank's model-validation function applies in practice.")
