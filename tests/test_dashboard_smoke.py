"""
test_dashboard_smoke.py
-------------------------
Smoke-tests the Streamlit dashboard using Streamlit's own official
testing framework (`streamlit.testing.v1.AppTest`), which executes the
ENTIRE script (every `st.tabs()` block runs on every script execution --
Streamlit only toggles which tab is *displayed*, not which tab's code
*runs*) headlessly, with no browser or server required, and surfaces any
exception raised while building the page.

This is a genuine regression test, not a placeholder: it caught a real
bug during development (a numpy array stored in a DataFrame's `.attrs`
broke pandas' internal attrs-comparison when Streamlit's dataframe
styler cast a sliced copy to strings) that a plain `import dashboard`
smoke test would have missed entirely, since that bug only manifested
once the actual `st.dataframe(...)` rendering path executed.

Requires `results/tables/*.csv` to already exist (run
`python src/run_experiment.py` first) and internet access for the Live
Market Check tab (that tab degrades gracefully with a warning if the
live fetch fails, rather than raising, so it does not fail this test
either way).
"""

import os

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

DASHBOARD_PATH = os.path.join(os.path.dirname(__file__), "..", "src", "dashboard.py")


@pytest.fixture(scope="module")
def app():
    at = AppTest.from_file(DASHBOARD_PATH, default_timeout=180)
    at.run()
    return at


def test_dashboard_loads_without_exception(app):
    assert not app.exception, f"Dashboard raised on initial load: {[e.message for e in app.exception]}"


def test_dashboard_renders_all_tabs_content(app):
    # Every tab's content block executes on every script run; a healthy
    # dashboard should have produced a substantial number of markdown,
    # dataframe, and chart elements across all 16 tabs.
    assert len(app.markdown) > 20
    assert len(app.dataframe) > 5


@pytest.mark.parametrize("widget_key,new_value", [
    ("vega_true_dgp", "Jumps (Merton)"),
    ("amer_type", "call"),
])
def test_dashboard_radio_alternate_branch(app, widget_key, new_value):
    """Exercise the NON-default branch of key radio widgets (the default-value
    run above only covers one code path per if/else)."""
    app.radio(key=widget_key).set_value(new_value).run()
    assert not app.exception, (
        f"Setting {widget_key}={new_value} raised: {[e.message for e in app.exception]}"
    )


@pytest.mark.parametrize("widget_key,asset", [("evt_asset", "btc"), ("stat_asset", "btc"), ("vega_asset", "btc")])
def test_dashboard_selectbox_alternate_asset(app, widget_key, asset):
    app.selectbox(key=widget_key).set_value(asset).run()
    assert not app.exception, (
        f"Setting {widget_key}={asset} raised: {[e.message for e in app.exception]}"
    )
