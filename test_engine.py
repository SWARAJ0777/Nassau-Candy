"""Engine invariants + regression against the Stage 8 numbers. Run: pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import engine as E  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data" / "Nassau_Candy_Distributor.csv"


@pytest.fixture(scope="session")
def full():
    df = E.load_data(DATA)
    return E.add_counterfactuals(df, E.fit_models(df))


@pytest.fixture(scope="session")
def C(full):
    return E.scenarios(full, full, E.Assumptions())


# ---------------------------------------------------------------- data + features
def test_lead_time_is_offset_corrected(full):
    assert full["raw_gap_days"].min() > 800            # raw gaps are impossible (~900-1,640 days)
    assert full["lead_time"].min() == 0
    assert full["lead_time"].max() <= 15
    assert 3.5 < full["lead_time"].mean() < 5.0


def test_ship_modes_order_correctly_after_correction(full):
    m = full.groupby("Ship Mode")["lead_time"].mean()
    assert m["Same Day"] < m["First Class"] < m["Second Class"] < m["Standard Class"]


def test_no_missing_values_in_model_inputs(full):
    cols = ["lat", "lon", "dist_mi", "base_lin", "base_gb", "factory"] + ["dist_" + f for f in E.FACTORY_NAMES]
    assert not full[cols].isna().any().any()


def test_distance_columns(full):
    for f in E.FACTORY_NAMES:
        assert (full["dist_" + f] >= 0).all() and (full["dist_" + f] < 4000).all()
        rows = full[full["factory"] == f]
        assert np.allclose(rows["dist_mi"], rows["dist_" + f])
    # spot check: Houston customers are ~1,000 miles from the Arizona factory and ~800 from Georgia
    h = full[full["City"] == "Houston"].iloc[0]
    assert 900 < h["dist_Lot's O' Nuts"] < 1100
    assert 750 < h["dist_Wicked Choccy's"] < 900


def test_every_product_maps_to_one_factory(full):
    assert full.groupby("Product Name")["factory"].nunique().max() == 1
    assert full["Product Name"].nunique() == 15


# ---------------------------------------------------------------- counterfactuals
def test_swapping_to_own_factory_changes_nothing(full):
    for f in E.FACTORY_NAMES:
        own = full[full["factory"] == f]
        assert own["dl_" + f].abs().max() < 1e-9
        assert own["dg_" + f].abs().max() < 1e-9


def test_dispatch_effects_are_small(full):
    """Stage 5/7 finding: factory and distance barely move dispatch time."""
    for f in E.FACTORY_NAMES:
        assert full["dl_" + f].abs().max() < 0.5


# ---------------------------------------------------------------- scenarios
def test_scenario_grid_is_complete(C):
    assert len(C) == 15 * 4
    assert (C["from"] != C["to"]).all()
    assert C.groupby("product").size().eq(4).all()


def test_profit_sign_follows_distance(C):
    """With production cost equal, moving farther away can only cost money."""
    assert ((C["dGP"] * C["freight_k"]) <= 1e-9).all()


def test_scores_are_bounded(C):
    assert C["conf"].between(0, 100).all()
    assert C["stability"].between(0, 1).all()
    assert C["data_share"].between(0, 1).all()
    assert not C[["red_pct", "dGP", "conf", "vol_ratio", "risk_red_pct"]].isna().any().any()


def test_flags_are_known(C):
    known = set(E.FLAG_LABELS)
    for s in C["flags"]:
        assert set(filter(None, s.split(","))) <= known


def test_empty_view_returns_empty(full):
    assert E.scenarios(full.iloc[0:0], full, E.Assumptions()).empty


def test_filtered_view_runs_and_shrinks_n(full):
    v = full[full["Region"] == "Pacific"]
    C2 = E.scenarios(v, full, E.Assumptions())
    assert not C2.empty
    assert C2.groupby("product")["n"].first().sum() == len(v)


def test_transit_assumption_scales_transit_delta(full):
    A1 = E.Assumptions()
    A2 = E.Assumptions(speed={k: v * 2 for k, v in A1.speed.items()})
    c1, c2 = E.scenarios(full, full, A1), E.scenarios(full, full, A2)
    assert np.allclose(c2["d_transit"], c1["d_transit"] / 2)
    assert np.allclose(c2["d_dispatch"], c1["d_dispatch"])          # learned part is assumption-free


def test_infinite_speed_means_data_only(full):
    A = E.Assumptions(speed={k: 1e9 for k in E.SHIP_ORDER})
    c = E.scenarios(full, full, A)
    assert np.allclose(c["d_transit"], 0, atol=1e-5)
    assert np.allclose(c["red_pct"], c["red_pct_data"], atol=1e-3)


# ---------------------------------------------------------------- recommendations
def test_unconstrained_matches_stage8(full, C):
    A = E.Assumptions(capacity_limit=1e9)
    R = E.recommend(C, A, full, 0.5, 0.0).set_index("product")
    expect = {"Wonka Bar - Fudge Mallows": ("The Other Factory", 10.78),
              "Wonka Bar - Nutty Crunch Surprise": ("The Other Factory", 10.61),
              "Wonka Bar -Scrumdiddlyumptious": ("The Other Factory", 10.54),
              "Wonka Bar - Milk Chocolate": ("Secret Factory", 6.37),
              "Wonka Bar - Triple Dazzle Caramel": ("Secret Factory", 6.62)}
    for p, (fac, red) in expect.items():
        assert R.loc[p, "action"] == "MOVE" and R.loc[p, "to"] == fac
        assert R.loc[p, "red_pct"] == pytest.approx(red, abs=0.05)


@pytest.mark.parametrize("cap", [1.0, 1.2, 1.5, 2.0, 5.0])
def test_capacity_is_respected_jointly(full, C, cap):
    A = E.Assumptions(capacity_limit=cap)
    R = E.recommend(C, A, full, 0.5, 0.15)
    mv = R[R["action"] == "MOVE"]
    assert (mv["joint_vol_ratio"] <= cap + 1e-9).all()
    base = full.groupby("factory")["Units"].sum()
    net = base.copy()
    for _, r in mv.iterrows():
        net[r["to"]] += r["prod_units"]
        net[r["from"]] -= r["prod_units"]
    assert ((net / base) <= cap + 1e-9).all()                       # independent re-computation


def test_moves_meet_the_stated_rules(full, C):
    A = E.Assumptions()
    R = E.recommend(C, A, full, 0.5, 0.15)
    mv = R[R["action"] == "MOVE"]
    assert (mv["red_pct"] >= A.min_reduction_pct).all()
    assert (mv["dGP"] >= 0).all()
    assert (R[R["action"] == "KEEP"]["red_pct"] == 0).all()
    assert len(R) == 15


def test_priority_slider_trades_speed_for_profit(full, C):
    A = E.Assumptions(capacity_limit=1e9)
    speed = E.recommend(C, A, full, 1.0, 0.0).set_index("product")
    profit = E.recommend(C, A, full, 0.0, 0.0).set_index("product")
    assert (speed["red_pct"] >= profit["red_pct"] - 1e-9).all()
    assert (profit["dGP_pct"] >= speed["dGP_pct"] - 1e-9).all()


def test_impossible_threshold_yields_no_moves(full, C):
    R = E.recommend(C, E.Assumptions(min_reduction_pct=99.0), full, 0.5, 0.15)
    assert (R["action"] == "KEEP").all()
    K = E.kpis(R)
    assert K["moves"] == 0 and K["lead_time_reduction_pct"] == 0 and K["coverage_sales"] == 0


def test_kpis_are_consistent(full, C):
    R = E.recommend(C, E.Assumptions(), full, 0.5, 0.15)
    K = E.kpis(R)
    assert 0 <= K["coverage_sales"] <= 1 and 0 <= K["coverage_products"] <= 1
    assert K["moves"] == (R["action"] == "MOVE").sum()
    assert K["portfolio_reduction_pct"] <= K["lead_time_reduction_pct"] + 1e-9
    assert K["d_profit"] == pytest.approx(R["dGP"].sum())


def test_assumption_driven_gain_lowers_confidence(C):
    big = C[(C["data_share"] < 0.2) & (C["n"] >= 100)]
    assert len(big) > 0
    # same data support, but confidence must sit well below the pre-fix 70-97 range
    assert big["conf"].max() < 70


# ---------------------------------------------------------------- dashboard views
def test_factory_profile_current_row_is_baseline(full):
    d = full[full["Product Name"] == "Wonka Bar - Milk Chocolate"]
    P = E.factory_profile(d, full, E.Assumptions())
    cur = P[P["current"]].iloc[0]
    assert cur["factory"] == "Wicked Choccy's" and cur["d_days"] == 0 and cur["dGP"] == 0
    assert len(P) == 5


def test_breakdown_orders_ship_modes(full):
    d = full[full["Product Name"] == "Wonka Bar - Milk Chocolate"]
    B = E.breakdown(d, "Secret Factory", E.Assumptions(), "Ship Mode")
    assert list(B["Ship Mode"].astype(str)) == E.SHIP_ORDER
    assert B["current"].iloc[0] < B["current"].iloc[-1]             # Same Day < Standard


def test_profit_grid_matches_scenario(full, C):
    A = E.Assumptions()
    prod, alt = "Wonka Bar - Milk Chocolate", "Secret Factory"
    d = full[full["Product Name"] == prod]
    rates, prems, Z = E.profit_grid(d, alt, A)
    i, j = list(np.round(prems, 2)).index(0.0), list(np.round(rates, 2)).index(0.10)
    expected = C[(C["product"] == prod) & (C["to"] == alt)]["dGP"].iloc[0]
    assert Z[i, j] == pytest.approx(expected)
