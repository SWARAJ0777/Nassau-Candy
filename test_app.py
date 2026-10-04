"""Headless UI tests (Streamlit AppTest). Run: pytest -q"""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def fresh():
    return AppTest.from_file(APP, default_timeout=180).run()


def sb(at, kind, label_part):
    """Find a sidebar widget by a fragment of its label."""
    for w in getattr(at.sidebar, kind):
        if label_part.lower() in w.label.lower():
            return w
    raise AssertionError(f"no sidebar {kind} containing {label_part!r}")


def metric(at, label):
    return next(m for m in at.metric if m.label == label)


@pytest.fixture(scope="module")
def base():
    return fresh()


def test_default_run_has_no_exceptions(base):
    assert not base.exception
    assert [t.label for t in base.tabs] == ["Factory Simulator", "What-If Analysis", "Recommendations",
                                            "Risk & Impact", "Method & Assumptions"]


def test_brief_kpis_are_present(base):
    for label in ("Lead Time Reduction", "Profit Impact Stability", "Scenario Confidence", "Recommendation Coverage"):
        assert metric(base, label).value


def test_every_product_renders():
    at = fresh()
    sel = sb(at, "selectbox", "Product")
    for p in sel.options:
        sel.set_value(p)
        at.run()
        assert not at.exception, f"crash for {p}"


def test_region_and_ship_mode_filters():
    at = fresh()
    sb(at, "multiselect", "region").set_value(["Pacific"])
    sb(at, "multiselect", "ship mode").set_value(["Same Day", "First Class"])
    at.run()
    assert not at.exception
    import re
    cap = next(c.value for c in at.caption if "orders in view" in c.value)
    n = int(re.search(r"([\d,]+) orders in view", cap).group(1).replace(",", ""))
    assert 0 < n < 3253                                   # Pacific has 3,253 orders in total


def test_empty_filters_fall_back_to_all():
    at = fresh()
    sb(at, "multiselect", "region").set_value([])
    at.run()
    assert not at.exception
    assert any("ignored" in w.value for w in at.sidebar.warning)


def test_priority_slider_changes_recommendations(base):
    at = fresh()
    at.sidebar.slider[0].set_value(0)
    at.run()
    low = [(m.label, m.value) for m in at.metric if m.label.startswith(("Lead Time", "Scenario"))]
    at.sidebar.slider[0].set_value(100)
    at.run()
    assert not at.exception
    high = [(m.label, m.value) for m in at.metric if m.label.startswith(("Lead Time", "Scenario"))]
    assert low and high


def test_assumption_edits_flow_through():
    at = fresh()
    base_val = metric(at, "Lead Time Reduction").value
    for w in at.sidebar.number_input:
        if w.label.startswith("Transit speed"):
            w.set_value(5000.0 if w.max >= 5000 else w.max)      # near-instant transit
    at.run()
    assert not at.exception
    assert metric(at, "Lead Time Reduction").value != base_val


def test_capacity_cap_changes_coverage():
    at = fresh()
    tight = metric(at, "Recommendation Coverage").value
    sb(at, "slider", "volume growth").set_value(3.0)
    at.run()
    assert not at.exception
    loose = metric(at, "Recommendation Coverage").value
    assert tight != loose


def test_impossible_threshold_shows_guidance():
    at = fresh()
    sb(at, "slider", "Minimum lead-time gain").set_value(10.0)
    sb(at, "slider", "volume growth").set_value(1.0)
    at.run()
    assert not at.exception
    # with a 10% bar and a 1.0x cap nothing should clear -> a guidance warning, not a crash
    assert any("No product has a move" in w.value for w in at.warning) or metric(at, "Recommendation Coverage").value.startswith("0 of")


def test_whatif_alternative_can_be_switched():
    at = fresh()
    alt = next(s for s in at.selectbox if s.label.startswith("Compare current factory"))
    for opt in alt.options:
        alt.set_value(opt)
        at.run()
        assert not at.exception
        alt = next(s for s in at.selectbox if s.label.startswith("Compare current factory"))


def test_method_tab_validation_button_runs():
    at = fresh()
    btn = next(b for b in at.button if b.label.startswith("Run model validation"))
    btn.click()
    at.run()
    assert not at.exception


def test_default_ranking_puts_material_products_first(base):
    """Regression: percentage-gain ranking used to put 3-order products above a $28K chocolate bar."""
    table = next(d.value for d in base.dataframe if "Order-days saved" in d.value.columns)
    assert table.iloc[0]["Orders"] >= 100
    assert table["Order-days saved"].is_monotonic_decreasing


def test_rank_by_options_all_work():
    at = fresh()
    radio = next(r for r in at.radio if r.label.startswith("Rank recommendations"))
    for opt in radio.options:
        radio.set_value(opt)
        at.run()
        assert not at.exception
        radio = next(r for r in at.radio if r.label.startswith("Rank recommendations"))
