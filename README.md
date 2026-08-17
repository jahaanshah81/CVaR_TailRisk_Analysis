# Options Market-Making & Multi-Model Delta-Hedging Simulator

### Quantifying Hidden Tail Risk from Model Misspecification in Derivatives Hedging

---

## What This Project Does

A market maker who sells an option must hedge the resulting risk by trading the
underlying asset. The textbook theory (Black-Scholes) assumes markets move smoothly
and continuously. Real markets jump — crashes, earnings surprises, flash crashes.

**This project asks and quantitatively answers:** *how much hidden tail risk does a
market maker carry when they hedge using a model that cannot see jumps, compared to
one that can?*

We price options under three increasingly realistic models of the market
(Black-Scholes, Heston stochastic volatility, Merton jump-diffusion), calibrate all
three to **real historical SPY and BTC-USD data**, simulate a market maker
delta-hedging a short option position under each hedge model, and measure the
**Conditional Value-at-Risk (CVaR)** of the resulting hedging losses — directly
reusing the same tail-risk framework from the companion CVaR/EVT research project.

Beyond the core finding, the project also demonstrates the broader numerical-methods
and risk-management toolkit a quantitative researcher is expected to have: **bootstrap
confidence intervals and hypothesis tests** on every headline number, **Extreme Value
Theory** for deep-tail risk, a **Delta-Vega multi-instrument hedge**, **American options**
via two independent numerical PDE/tree methods, a **vectorized Heston pricer** (the
Fourier-COSine / COS method) that resolved this project's original biggest performance
limitation, and **Monte Carlo variance reduction / importance sampling** — see
[Beyond the Headline](#beyond-the-headline-a-practitioners-risk-toolkit) below.

## Headline Result

| Asset | BS-hedge CVaR(95%) | Merton-hedge CVaR(95%) | Hidden tail risk |
|---|---|---|---|
| SPY (calm equity, simulated) | 38.86 | 31.89 | +22% |
| BTC-USD (volatile, simulated) | 10,025.16 | 8,184.50 | +22% |
| SPY (calm equity, **real historical data**) | 16.28 | 12.40 | +31% |
| BTC-USD (volatile, **real historical data**) | 1,838.60 | 1,118.38 | +64% |

The real-historical-data backtest (not just simulation) confirms the finding: a
market maker hedging with Black-Scholes Delta carries materially more tail risk,
and a more negatively biased average P&L, than one using a jump-aware Merton hedge
— on both a calm equity and a volatile crypto asset. Every number above now also
carries a **bootstrap 95% confidence interval and a paired hypothesis test**
(both gaps are statistically significant, p < 0.001 — see the Statistical Rigor
tab / `results/tables/statistical_significance.csv`), and the real-data backtest is
now **walk-forward** (calibrated only on data available at the time, refreshed
quarterly) rather than calibrated once on the full sample — a look-ahead-bias fix
described in [Methodology Notes](#methodology-notes-read-before-citing-numbers).

A secondary experiment further shows that Heston (stochastic volatility, no jumps)
does **not** help against jump risk and is sometimes worse than plain Black-Scholes
— sophistication must match the *specific* risk present, not just be "more complex."
This comparison now runs at the **same full 8,000-path, daily-rebalancing scale** as
the headline result (previously reduced to ~150 paths), thanks to a new vectorized
Heston pricer. See `report/report.md` for the full write-up.

## Beyond the Headline: A Practitioner's Risk Toolkit

The core CVaR result is corroborated many further, independent ways — the kind of
multi-angle validation a real derivatives risk function requires before trusting a
single number:

| Tool | What it adds | Key finding |
|---|---|---|
| **Greeks-based P&L attribution** (`pnl_attribution.py`) | Decomposes hedging P&L into Delta/Gamma/Theta/residual, day by day | BS hedge's "unexplained" residual CVaR is measurably fatter-tailed than Merton's — the *causal*, granular version of the headline result |
| **Deterministic stress testing** (`stress_testing.py`) | Named, distribution-free adverse scenarios (Black Monday, COVID crash, flash crash, vol-regime shock) | Merton hedge beats BS hedge in every named crash scenario, on both assets |
| **Basel/Kupiec VaR backtesting** (`model_validation.py`) | Honest out-of-sample coverage test of the VaR model itself | SPY's BS-hedge VaR model lands in the Basel **Yellow zone** (borderline miscalibrated); the Merton-hedge model is solidly **Green** |
| **Band ("no-transaction region") hedging** (`hedging_engine.py`) | Whalley–Wilmott-style trade-only-when-Delta-drifts strategy | Can match or beat daily rebalancing's CVaR at a fraction of the transaction cost — not simply "less trading = more risk" |
| **Live options-chain check** (`market_data_live.py`) | Compares the historical (P-measure) model smile against SPY's real, live (Q-measure) options chain | Surfaces the live volatility risk premium the market is pricing right now |
| **Bootstrap inference** (`statistical_inference.py`) | Confidence intervals + a paired hypothesis test on every CVaR estimate and gap | The headline BS-vs-Merton CVaR gap is statistically significant (p < 0.001) on both assets — not just a larger point estimate |
| **Extreme Value Theory** (`tail_risk.py`) | Peaks-Over-Threshold GPD tail fit, extrapolating far beyond what a finite sample can estimate directly | At deep quantiles (99.9%), EVT and historical CVaR diverge exactly where the historical estimator is data-starved — the reason EVT exists |
| **Delta-Vega hedging** (`hedging_engine.py`) | A second option instrument sized to neutralize net Vega, giving the Heston hedge a fair fight | Cuts CVaR by more than half for BTC under a genuine stochastic-vol market, but can even hurt for SPY (whose calibrated vol-of-vol is small) — sophistication must match the risk, again |
| **American options** (`american_options.py`) | Binomial tree + Crank-Nicolson PDE, cross-validated against each other and against theory | Genuine early-exercise premium and boundary, absent from every other (European-only) pricer in this project |
| **Vectorized Heston (COS method)** (`heston.py`) | Fourier-COSine pricer, ~180x faster than the original quad-based inversion | Resolved this project's single most-repeated documented limitation, enabling full-scale Heston studies |
| **Monte Carlo variance reduction** (`variance_reduction.py`) | Antithetic + control variates for pricing; importance sampling (jump-rate tilting) for tail-risk estimation | 10–50% lower Monte Carlo standard error at equal simulation budget; more precise deep-tail CVaR at equal path count |

All of the above are fully wired into the interactive dashboard (see below) with
their own tabs, live re-computation controls, and — for the batch-vectorizable
pieces — cross-validated bit-identical (or near-identical, where two genuinely
different numerical methods are being compared) against independent implementations.

## Project Structure

```
Project_1/
├── README.md
├── requirements.txt
├── pytest.ini
├── data/                        ← real historical SPY & BTC-USD price data
├── tests/                       ← pytest suite (see "Testing" below)
│   ├── conftest.py
│   ├── test_cli_scripts.py       ← runs every module's own validation suite, asserts success
│   ├── test_core_functions.py    ← direct unit tests for key functions (parametrized, fixtures)
│   └── test_dashboard_smoke.py   ← headless Streamlit AppTest smoke test (all 16 tabs)
├── src/
│   ├── data_loader.py           ← fetches real price data from Yahoo Finance
│   ├── black_scholes.py         ← BSM pricing, Greeks, implied vol (validated vs. Hull)
│   ├── heston.py                 ← Heston stochastic vol: Fourier inversion AND a vectorized
│   │                                COS-method pricer (validated vs. MC, quad, BS reduction limit)
│   ├── merton.py                 ← Merton jump-diffusion (validated vs. Monte Carlo
│   │                                + BS reduction limit)
│   ├── calibration.py           ← historical (P-measure) calibration of both models
│   ├── american_options.py      ← binomial tree + Crank-Nicolson PDE for American options
│   │                                (cross-validated against each other and against theory)
│   ├── hedging_engine.py        ← discrete delta-hedging simulator: fixed-frequency,
│   │                                band ("no-transaction region"), AND Delta-Vega
│   │                                multi-instrument rebalancing, each with a vectorized
│   │                                batch mode (validated identical to the per-path
│   │                                simulator, ~1000x faster)
│   ├── risk_analysis.py         ← CVaR of hedging P&L across hedge models
│   ├── pnl_attribution.py       ← Greeks-based P&L attribution / "P&L explain"
│   ├── stress_testing.py        ← deterministic scenario stress tests (Basel/CCAR-style)
│   ├── model_validation.py      ← Kupiec POF test + Basel traffic-light VaR backtesting
│   ├── statistical_inference.py ← bootstrap confidence intervals + paired hypothesis tests
│   ├── tail_risk.py             ← Extreme Value Theory: GPD Peaks-Over-Threshold, Hill estimator
│   ├── variance_reduction.py    ← antithetic/control variates; importance sampling for CVaR
│   ├── market_data_live.py      ← live options-chain fetch (Q-measure vs. P-measure)
│   ├── run_experiment.py        ← MAIN SCRIPT — runs the full 14-step pipeline
│   └── dashboard.py             ← interactive Streamlit app (16 tabs, live demo)
├── results/
│   ├── figures/                 ← all report figures (PDFs)
│   └── tables/                  ← all report tables (CSVs)
└── report/
    └── report.md                 ← full write-up
```

## How to Run

```powershell
pip install -r requirements.txt
cd Project_1
python src/data_loader.py        # fetch real SPY + BTC-USD data
python src/run_experiment.py     # run the full 14-step pipeline (a few minutes)
```

To explore interactively instead:

```powershell
streamlit run src/dashboard.py
```

This opens a 16-tab browser dashboard: overview (now with an explicit look-ahead-bias
transparency panel), real data & calibration, a **live options-chain check against the
real market**, volatility smiles, the headline CVaR result (with live-adjustable path
counts), **bootstrap statistical rigor**, **Extreme Value Theory tail risk**,
**Greeks-based P&L attribution**, the Heston finding (now full-scale), **Delta-Vega
hedging**, an **efficient frontier with band-hedging cost/risk sweeps**, **American
options** (binomial tree + Crank-Nicolson PDE), **deterministic stress testing**,
**Basel/Kupiec VaR backtesting**, a free-form multi-parameter risk lab, and full
methodology notes — almost everything is recomputed live and cached, not just a
static report viewer.

Every source file also runs its own validation suite standalone
(`python src/black_scholes.py`, `python src/heston.py`, etc.) — each independently
cross-checks its pricer against a known benchmark, Monte Carlo simulation, or a
reduction to a simpler nested model, before being trusted for the main experiment.

## Testing

```powershell
pip install -r requirements.txt   # includes pytest
pytest
```

The pytest suite (`tests/`) wraps every module's own extensive CLI validation script
into automated, CI-discoverable tests (`test_cli_scripts.py`), adds direct unit tests
for the newer statistical/numerical modules with fixtures and parametrization
(`test_core_functions.py`), and headlessly smoke-tests the entire 16-tab Streamlit
dashboard using Streamlit's official `AppTest` framework (`test_dashboard_smoke.py`)
— which, during development, caught a genuine bug (a numpy array stored in a
DataFrame's `.attrs` broke pandas' internal attrs-comparison inside Streamlit's
dataframe styler) that a plain import-and-run smoke test would have missed.

## Methodology Notes (read before citing numbers)

- **Historical vs. implied calibration — partly bridged.** We calibrate to the
  *historical* (physical, P-measure) statistical properties of each asset's returns,
  not to a real market options chain (genuinely free HISTORICAL options data
  essentially does not exist). This is a well-known, clearly weaker substitute for
  implied (Q-measure) calibration. The dashboard's **Live Market Check** tab partly
  offsets this: it fetches SPY's actual, current listed-options chain and compares
  its live implied-vol smile directly against the historical model's — a genuine,
  point-in-time Q-vs-P comparison, even though it can't be baked into a reproducible
  historical backtest the way the rest of the pipeline is.
- **Heston is now vectorized.** The original Fourier-inversion pricer
  (`scipy.integrate.quad`) could not be vectorized across paths, restricting the
  Heston-hedge comparison to ~150 paths / biweekly rebalancing. A new COS-method
  pricer (Fang & Oosterlee, 2008) exploits the log-price's scale-invariance to
  vectorize across paths via ordinary numpy broadcasting — cross-validated to agree
  with the original quad-based pricer to within ~3e-4 relative error and benchmarked
  ~180x faster. The Heston secondary comparison, the Delta-Vega hedging study, and
  the headline batch hedging simulator all now support "heston" at full scale, on
  equal footing with "bs" and "merton".
- **Look-ahead bias in the real-data backtest — found and fixed.** An earlier
  version calibrated Merton parameters once from an asset's *entire* history, then
  used those parameters to hedge every historical window, including windows years
  before the calibration data ends. The backtest is now **walk-forward**: every
  window is hedged using parameters calibrated only from data strictly prior to
  that window, refreshed quarterly (with the first ~2 years used only to seed the
  initial calibration). Both the corrected and original (biased) numbers are
  reported side by side (`results/tables/real_data_backtest_lookahead_bias_check.csv`
  and the dashboard's Overview tab) — the bias was small for SPY but reduced BTC's
  reported CVaR by roughly 30%, an economically meaningful correction.
- **Statistical CVaR is not the whole risk picture — now with uncertainty
  quantification.** A single tail-risk statistic, however carefully estimated, is
  never sufficient on its own — this is why the project includes distribution-free
  stress scenarios, an explicit backtest of the VaR model's own out-of-sample
  calibration (Kupiec/Basel), a bootstrap confidence interval and paired hypothesis
  test on every headline CVaR gap, and an Extreme Value Theory tail model for
  quantiles beyond what any finite historical sample can estimate directly.
- **The Delta-Vega hedge has a genuine, documented numerical subtlety.** Every
  option's Vega collapses to 0 near expiry, and a fixed-strike instrument away from
  the money collapses faster than an at-the-money target's — so a naive Vega-hedge
  ratio can compound to an unreasonable size well before expiry. Handled by capping
  the instrument quantity at a fixed multiple of its well-behaved, full-maturity
  value (see `hedging_engine.py`'s `simulate_delta_vega_hedge_batch`), approximating
  how a real desk would roll to a fresher instrument rather than lever into a
  threadbare one — found empirically during development and documented, not hidden.
- **The importance-sampling tilting parameter is a bias-variance tradeoff, not a
  free parameter.** Too little tilting under-samples the tail (little benefit over
  plain Monte Carlo); too much creates importance-weight degeneracy, which
  *increases* estimator variance again (verified empirically in
  `variance_reduction.py`'s own CLI check).
- **Every pricer and simulator is independently validated** against Monte Carlo
  simulation of its own SDE, a nested simpler model in the appropriate parameter
  limit, a bit-identical cross-check between its per-path and vectorized-batch
  implementations, and/or (for American options and the COS-method Heston pricer)
  a second, completely independent numerical method — never trust a single
  untested pricer or simulator.
