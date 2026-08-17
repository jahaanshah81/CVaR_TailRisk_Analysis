"""
test_cli_scripts.py
--------------------
Wraps every module's own extensive, already-written CLI validation suite
(the `if __name__ == "__main__":` block at the bottom of nearly every
src/*.py file) into a proper, automated, CI-discoverable pytest suite.

Every one of these scripts independently cross-validates its own pricer,
simulator, or estimator against an analytical benchmark, Monte Carlo
simulation, a nested-model reduction limit, or a bit-identical
per-path-vs-vectorized comparison (see each module's own docstring for
specifics). This test does not duplicate that logic; it turns "run this
script and read the printed output" into "run this script and assert it
succeeds", so a regression in ANY of these cross-checks fails an
automated build instead of requiring a human to notice a changed number
in a terminal.

`data_loader.py` and `market_data_live.py` are intentionally excluded:
both require live internet access to Yahoo Finance (one fetches
historical price history, the other a live options chain) and are I/O
utilities rather than numerical logic to validate -- appropriate to
exercise manually / in an integration environment with network access,
not in a fast, deterministic unit-test run. `run_experiment.py` and
`dashboard.py` are excluded for the same "orchestration, not a unit"
reason; see test_dashboard_smoke.py for the dashboard's own dedicated
smoke test.
"""

import subprocess
import sys
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"

SELF_VALIDATING_MODULES = [
    "black_scholes.py",
    "heston.py",
    "merton.py",
    "calibration.py",
    "hedging_engine.py",
    "risk_analysis.py",
    "pnl_attribution.py",
    "stress_testing.py",
    "model_validation.py",
    "statistical_inference.py",
    "tail_risk.py",
    "american_options.py",
    "variance_reduction.py",
]


@pytest.mark.parametrize("module_name", SELF_VALIDATING_MODULES)
def test_module_cli_validation_passes(module_name):
    """Run `python src/<module_name>.py` and require a clean (exit code 0)
    run -- i.e., every internal cross-check / assertion in that module's
    own validation suite passed. Exit code (not a specific printed
    string) is the success signal: an uncaught AssertionError from a
    failed internal check exits non-zero with a traceback, which this
    test would catch regardless of that module's own chosen wording for
    a successful run (most, but not all, print "All checks passed.")."""
    script_path = SRC_DIR / module_name
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True, text=True, timeout=300, cwd=str(SRC_DIR),
    )
    assert result.returncode == 0, (
        f"{module_name} exited with code {result.returncode}.\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
