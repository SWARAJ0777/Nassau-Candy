"""
Nassau Candy - Factory Reallocation & Shipping Optimization engine.

Pipeline (mirrors the mentoring stages):
  load_data()            Stage 3/5  cleaning + feature engineering (lead time, distances)
  fit_models()           Stage 6/7  Ridge (primary, interpretable) + Gradient Boosting (challenger)
  add_counterfactuals()  Stage 8    per-row predicted change in dispatch time for every factory swap
  scenarios()            Stage 8    one row per (product, alternative factory)
  recommend()            Stage 8    rule-based, capacity-aware recommendation per product
  kpis()                 Stage 8    the four KPIs from the project brief

LEARNED vs ASSUMED
  * Dispatch time (Order Date -> Ship Date, offset-corrected) is LEARNED from the data.
  * Transit time and freight cost are ASSUMPTIONS (the dataset has no delivery date and
    no freight cost). They live in `Assumptions` and are editable in the dashboard.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import zipcodes
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --------------------------------------------------------------------------- reference data
FACTORIES = {
    "Lot's O' Nuts": (32.881893, -111.768036),
    "Wicked Choccy's": (32.076176, -81.088371),
    "Sugar Shack": (48.11914, -96.18115),
    "Secret Factory": (41.446333, -90.565487),
    "The Other Factory": (35.1175, -89.971107),
}
FACTORY_NAMES = list(FACTORIES)

PRODUCT_FACTORY = {
    "Wonka Bar - Nutty Crunch Surprise": "Lot's O' Nuts",
    "Wonka Bar - Fudge Mallows": "Lot's O' Nuts",
    "Wonka Bar -Scrumdiddlyumptious": "Lot's O' Nuts",
    "Wonka Bar - Milk Chocolate": "Wicked Choccy's",
    "Wonka Bar - Triple Dazzle Caramel": "Wicked Choccy's",
    "Laffy Taffy": "Sugar Shack",
    "SweeTARTS": "Sugar Shack",
    "Nerds": "Sugar Shack",
    "Fun Dip": "Sugar Shack",
    "Fizzy Lifting Drinks": "Sugar Shack",
    "Everlasting Gobstopper": "Secret Factory",
    "Hair Toffee": "The Other Factory",
    "Lickable Wallpaper": "Secret Factory",
    "Wonka Gum": "Secret Factory",
    "Kazookles": "The Other Factory",
}

# The dataset only carries a city for Canadian customers -> city-centre coordinates.
CANADA_CITIES = {
    "Calgary": (51.0447, -114.0719), "Edmonton": (53.5461, -113.4938),
    "Vancouver": (49.2827, -123.1207), "Winnipeg": (49.8951, -97.1384),
    "Moncton": (46.0878, -64.7782), "St. John's": (47.5615, -52.7126),
    "Halifax": (44.6488, -63.5752), "Toronto": (43.6532, -79.3832),
    "Charlottetown": (46.2382, -63.1311), "Montreal": (45.5017, -73.5673),
    "Quebec City": (46.8139, -71.2080), "Regina": (50.4452, -104.6189),
}

SHIP_ORDER = ["Same Day", "First Class", "Second Class", "Standard Class"]
REGIONS = ["Atlantic", "Gulf", "Interior", "Pacific"]

LIN_NUM = ["dist_mi", "Units", "price_per_unit", "is_canada"]
LIN_CAT = ["Ship Mode", "factory", "Region", "id_year", "order_month_c"]
GB_NUM = ["ship_ord", "dist_mi", "Units", "price_per_unit", "order_month", "is_canada"]
GB_CAT = ["factory", "Region", "id_year"]

FLAG_LABELS = {
    "LOW_DATA": "Low data (<30 orders)",
    "CAPACITY": "Capacity stress",
    "PROFIT_NEG": "Profit loss",
    "MODEL_DISAGREE": "Models disagree",
    "ASSUMPTION_DRIVEN": "Mostly assumption-driven",
}

# Grid used for the "Profit Impact Stability" KPI (share of grid cells where profit does not fall)
STABILITY_RATES = (0.05, 0.10, 0.20)
STABILITY_PREMIUMS = (-0.02, 0.0, 0.02, 0.05)


# --------------------------------------------------------------------------- assumptions
@dataclass
class Assumptions:
    """Everything the data cannot tell us. Replace with carrier / finance numbers when available."""
    speed: dict = field(default_factory=lambda: {          # miles per day in transit
        "Same Day": 1500.0, "First Class": 900.0, "Second Class": 650.0, "Standard Class": 500.0})
    freight_mult: dict = field(default_factory=lambda: {   # freight cost multiplier by ship mode
        "Same Day": 3.0, "First Class": 1.8, "Second Class": 1.3, "Standard Class": 1.0})
    freight_rate: float = 0.10        # USD per unit per 1,000 miles (Standard Class)
    prod_cost_premium: float = 0.0    # alt-factory unit production cost vs current (+0.02 = 2% dearer)
    capacity_limit: float = 1.5       # max (new volume / original volume) at the receiving factory
    min_reduction_pct: float = 3.0    # only recommend a move if it is at least this much faster
    low_data_rows: int = 30
    disagree_days: float = 0.15
    assumption_share: float = 0.25    # flag if learned part is < this share of the predicted gain


# --------------------------------------------------------------------------- data + features
def haversine_mi(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * 3958.8 * np.arcsin(np.sqrt(a))


def _geocode(df: pd.DataFrame) -> pd.DataFrame:
    zips = df.loc[df["Country/Region"] == "United States", "Postal Code"].str.zfill(5).unique()
    zmap = {}
    for z in zips:
        m = zipcodes.matching(z)
        if m:
            zmap[z] = (float(m[0]["lat"]), float(m[0]["long"]))
    lat = np.full(len(df), np.nan)
    lon = np.full(len(df), np.nan)
    is_ca = (df["Country/Region"] == "Canada").to_numpy()
    zip5 = df["Postal Code"].str.zfill(5)
    for i, (ca, city, z) in enumerate(zip(is_ca, df["City"], zip5)):
        c = CANADA_CITIES.get(city) if ca else zmap.get(z)
        if c:
            lat[i], lon[i] = c
    out = df.assign(lat=lat, lon=lon)
    # fall back to the state mean for any ZIP the lookup could not resolve
    for col in ("lat", "lon"):
        out[col] = out[col].fillna(out.groupby("State/Province")[col].transform("mean")).fillna(out[col].mean())
    return out


def load_data(path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"Postal Code": str})
    df["od"] = pd.to_datetime(df["Order Date"], format="%d-%m-%Y")
    df["sd"] = pd.to_datetime(df["Ship Date"], format="%d-%m-%Y")
    df["raw_gap_days"] = (df["sd"] - df["od"]).dt.days
    df["id_year"] = df["Order ID"].str.extract(r"^[A-Z]{2}-(\d{4})")[0].astype(int)
    # Stage 3: ship dates carry a constant offset per Order-ID year -> subtract the tier minimum.
    df["lead_time"] = df["raw_gap_days"] - df.groupby("id_year")["raw_gap_days"].transform("min")
    df["factory"] = df["Product Name"].map(PRODUCT_FACTORY)
    if df["factory"].isna().any():
        raise ValueError("Unmapped product(s): " + ", ".join(df.loc[df.factory.isna(), "Product Name"].unique()))
    df = _geocode(df)
    for f, (la, lo) in FACTORIES.items():
        df["dist_" + f] = haversine_mi(df["lat"], df["lon"], la, lo)
    # distance from the product's CURRENT factory to the customer
    df["dist_mi"] = np.select([df["factory"] == f for f in FACTORY_NAMES], [df["dist_" + f] for f in FACTORY_NAMES])
    df["ship_ord"] = df["Ship Mode"].map({m: i for i, m in enumerate(SHIP_ORDER)})
    df["price_per_unit"] = df["Sales"] / df["Units"]
    df["is_canada"] = (df["Country/Region"] == "Canada").astype(int)
    df["order_month"] = df["od"].dt.month
    df["order_month_c"] = df["order_month"].astype(str)
    return df.reset_index(drop=True)


def data_checks(df: pd.DataFrame) -> dict:
    """Facts shown in the Method tab so the date correction is auditable."""
    tiers = df.groupby("id_year")["raw_gap_days"].agg(["min", "max", "count"])
    return {
        "rows": len(df), "orders": df["Order ID"].nunique(),
        "raw_gap_min": int(df.raw_gap_days.min()), "raw_gap_max": int(df.raw_gap_days.max()),
        "lead_mean": float(df.lead_time.mean()), "lead_max": int(df.lead_time.max()),
        "tiers": tiers,
    }


# --------------------------------------------------------------------------- models
def _pipe(est, num, cat):
    return Pipeline([
        ("p", ColumnTransformer([("n", StandardScaler(), num),
                                 ("c", OneHotEncoder(handle_unknown="ignore"), cat)])),
        ("m", est)])


def make_linear(alpha: float = 1.0):
    return _pipe(Ridge(alpha=alpha), LIN_NUM, LIN_CAT)


def make_boosting():
    return _pipe(GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3,
                                           subsample=0.8, min_samples_leaf=20, random_state=42),
                 GB_NUM, GB_CAT)


def fit_models(df: pd.DataFrame) -> dict:
    y = df["lead_time"]
    return {"lin": make_linear().fit(df, y), "gb": make_boosting().fit(df, y)}


def add_counterfactuals(df: pd.DataFrame, models: dict) -> pd.DataFrame:
    """For every row and every factory: baseline prediction and predicted change if produced there."""
    out = df.copy()
    out["base_lin"] = models["lin"].predict(out)
    out["base_gb"] = models["gb"].predict(out)
    for f in FACTORY_NAMES:
        alt = out.assign(factory=f, dist_mi=out["dist_" + f])
        out["dl_" + f] = models["lin"].predict(alt) - out["base_lin"]
        out["dg_" + f] = models["gb"].predict(alt) - out["base_gb"]
    return out


def validation_metrics(df: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    """Order-grouped cross-validation (rows of one order share dates, so they must stay together)."""
    y = df["lead_time"].to_numpy()
    groups = df["Order ID"].to_numpy()
    cv = GroupKFold(n_splits=n_splits)
    models = {
        "Rule: average by ship mode": _pipe(LinearRegression(), [], ["Ship Mode"]),
        "Ridge (primary)": make_linear(),
        "Gradient Boosting (challenger)": make_boosting(),
    }
    rows = []
    for name, m in models.items():
        p = cross_val_predict(m, df, y, groups=groups, cv=cv)
        e = y - p
        rows.append({"Model": name, "RMSE (days)": np.sqrt(np.mean(e ** 2)), "MAE (days)": np.mean(np.abs(e)),
                     "R²": 1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2),
                     "Within ±1 day": np.mean(np.abs(e) <= 1), "Within ±2 days": np.mean(np.abs(e) <= 2)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- simulation
def _with_assumptions(view: pd.DataFrame, A: Assumptions) -> pd.DataFrame:
    return view.assign(speed=view["Ship Mode"].map(A.speed), mult=view["Ship Mode"].map(A.freight_mult))


def scenarios(view: pd.DataFrame, full: pd.DataFrame, A: Assumptions) -> pd.DataFrame:
    """One row per (product, alternative factory), evaluated on the orders in `view`.

    `full` (unfiltered) supplies capacity context: a reassignment moves a product's whole
    volume, whatever slice of customers the user is currently looking at.
    """
    if view.empty:
        return pd.DataFrame()
    v = _with_assumptions(view, A)
    base_units = full.groupby("factory")["Units"].sum()
    prod_units = full.groupby("Product Name")["Units"].sum()
    fac_rows = full.groupby("factory").size()
    rows = []
    for prod, d in v.groupby("Product Name"):
        cur = d["factory"].iloc[0]
        speed = d["speed"].to_numpy()
        units = d["Units"].to_numpy()
        mult = d["mult"].to_numpy()
        dist_cur = d["dist_mi"].to_numpy()
        base_lin = d["base_lin"].to_numpy()
        tr_cur = dist_cur / speed
        tot_cur = base_lin + tr_cur
        bt = tot_cur.mean()
        p90_cur = np.percentile(tot_cur, 90)
        cost, gp, sales = d["Cost"].sum(), d["Gross Profit"].sum(), d["Sales"].sum()
        for alt in FACTORY_NAMES:
            if alt == cur:
                continue
            dist_alt = d["dist_" + alt].to_numpy()
            tr_alt = dist_alt / speed
            dl_row = d["dl_" + alt].to_numpy()
            dL, dG = dl_row.mean(), d["dg_" + alt].mean()
            dT = (tr_alt - tr_cur).mean()
            tot_alt = base_lin + dl_row + tr_alt
            K = (units * mult * (dist_alt - dist_cur) / 1000.0).sum()      # freight: USD per (USD/unit/1000mi)
            dGP = -(A.freight_rate * K + A.prod_cost_premium * cost)
            grid = [-(r * K + c * cost) for r in STABILITY_RATES for c in STABILITY_PREMIUMS]
            n = len(d)
            vol_ratio = (base_units[alt] + prod_units[prod]) / base_units[alt]
            share = abs(dL) / (abs(dL) + abs(dT) + 1e-9)
            s_prod, s_fac = min(1.0, n / 100), min(1.0, fac_rows[alt] / 500)
            agree = max(0.0, 1 - abs(dL - dG) / 0.25)
            conf = 100 * (0.4 * s_prod + 0.3 * s_fac + 0.3 * agree) * (0.5 + 0.5 * min(1.0, share / 0.5))
            flags = [k for k, c in [
                ("LOW_DATA", n < A.low_data_rows), ("CAPACITY", vol_ratio > A.capacity_limit),
                ("PROFIT_NEG", dGP < 0), ("MODEL_DISAGREE", abs(dL - dG) > A.disagree_days),
                ("ASSUMPTION_DRIVEN", share < A.assumption_share)] if c]
            rows.append({
                "product": prod, "from": cur, "to": alt, "n": n, "sales": sales, "gp": gp, "cost": cost,
                "base_days": bt, "d_dispatch": dL, "d_dispatch_gb": dG, "d_transit": dT, "d_total": dL + dT,
                "red_pct": -(dL + dT) / bt * 100, "red_pct_data": -dL / bt * 100,
                "risk_red_pct": (p90_cur - np.percentile(tot_alt, 90)) / p90_cur * 100,
                "d_dist": float((dist_alt - dist_cur).mean()),
                "freight_k": K, "dGP": dGP, "dGP_pct": dGP / gp * 100 if gp else 0.0,
                "breakeven_premium_pct": -(A.freight_rate * K) / cost * 100 if cost else np.nan,
                "stability": float(np.mean([g >= 0 for g in grid])),
                "conf": conf, "data_share": share, "vol_ratio": vol_ratio, "prod_units": float(prod_units[prod]),
                "flags": ",".join(flags)})
    return pd.DataFrame(rows)


def _norm(s: pd.Series) -> pd.Series:
    rng = s.max() - s.min()
    return (s - s.min()) / rng if rng > 0 else s * 0.0


def add_scores(C: pd.DataFrame, priority: float, risk_weight: float = 0.15) -> pd.DataFrame:
    """priority: 0 = profit only ... 1 = speed only. Risk (tail lead time) gets a fixed side weight."""
    C = C.copy()
    main = priority * _norm(C["red_pct"]) + (1 - priority) * _norm(C["dGP_pct"])
    C["score"] = (1 - risk_weight) * main + risk_weight * _norm(C["risk_red_pct"])
    return C


def recommend(C: pd.DataFrame, A: Assumptions, full: pd.DataFrame, priority: float = 0.5,
              risk_weight: float = 0.15) -> pd.DataFrame:
    """Best feasible move per product, or KEEP.

    Feasible = at least `min_reduction_pct` faster AND profit not lower at the chosen assumptions AND
    the receiving factory stays within `capacity_limit` counting every move already accepted
    (products are processed largest-sales first, so big products get first claim on capacity).
    """
    if C.empty:
        return pd.DataFrame()
    C = add_scores(C, priority, risk_weight)
    base_units = full.groupby("factory")["Units"].sum().to_dict()
    cur_units = dict(base_units)
    out = []
    for prod in C.groupby("product")["sales"].first().sort_values(ascending=False).index:
        d = C[C["product"] == prod].sort_values("score", ascending=False)
        ok = d[(d["red_pct"] >= A.min_reduction_pct) & (d["dGP"] >= 0)]
        chosen, blocked = None, False
        for _, r in ok.iterrows():
            joint = (cur_units[r["to"]] + r["prod_units"]) / base_units[r["to"]]
            if joint <= A.capacity_limit:
                chosen = r.copy()
                chosen["joint_vol_ratio"] = joint
                cur_units[r["to"]] += r["prod_units"]
                cur_units[r["from"]] -= r["prod_units"]
                break
            blocked = True
        if chosen is not None:
            out.append({**chosen.to_dict(), "action": "MOVE",
                        "reason": "Meets speed, profit and capacity rules"})
        else:
            top = d.iloc[0]
            reason = ("Best option would overload the receiving factory" if blocked else
                      "No alternative is fast enough without losing profit")
            out.append({**{k: top[k] for k in ("product", "from", "n", "sales", "gp", "base_days", "conf")},
                        "to": top["from"], "action": "KEEP", "red_pct": 0.0, "red_pct_data": 0.0, "dGP": 0.0,
                        "dGP_pct": 0.0, "stability": np.nan, "flags": "", "joint_vol_ratio": np.nan,
                        "risk_red_pct": 0.0, "score": np.nan, "d_total": 0.0, "reason": reason})
    R = pd.DataFrame(out)
    return R


def _wavg(x, w):
    w = np.asarray(w, dtype=float)
    return float(np.average(x, weights=w)) if len(w) and w.sum() > 0 else float("nan")


def kpis(R: pd.DataFrame) -> dict:
    """The four KPIs from the project brief (+ context)."""
    if R.empty:
        return {}
    mv = R[R["action"] == "MOVE"]
    return {
        "lead_time_reduction_pct": _wavg(mv["red_pct"], mv["sales"]) if len(mv) else 0.0,
        "portfolio_reduction_pct": _wavg(R["red_pct"], R["sales"]),
        "profit_impact_stability": _wavg(mv["stability"], mv["sales"]) if len(mv) else float("nan"),
        "scenario_confidence": _wavg(mv["conf"], mv["sales"]) if len(mv) else float("nan"),
        "coverage_products": len(mv) / len(R), "coverage_sales": mv["sales"].sum() / R["sales"].sum(),
        "moves": len(mv), "products": len(R),
        "d_profit": float(R["dGP"].sum()), "d_profit_pct": float(R["dGP"].sum() / R["gp"].sum() * 100),
    }


# --------------------------------------------------------------------------- views for the dashboard
def factory_profile(d: pd.DataFrame, full: pd.DataFrame, A: Assumptions) -> pd.DataFrame:
    """Predicted order-to-delivery (dispatch learned + transit assumed) if this product were made at each factory."""
    v = _with_assumptions(d, A)
    cur = v["factory"].iloc[0]
    base_units = full.groupby("factory")["Units"].sum()
    pu = full.loc[full["Product Name"] == v["Product Name"].iloc[0], "Units"].sum()
    cost = v["Cost"].sum()
    speed, units, mult = v["speed"].to_numpy(), v["Units"].to_numpy(), v["mult"].to_numpy()
    rows = []
    for f in FACTORY_NAMES:
        dist = v["dist_" + f].to_numpy()
        K = (units * mult * (dist - v["dist_mi"].to_numpy()) / 1000.0).sum()
        rows.append({
            "factory": f, "current": f == cur, "avg_dist": dist.mean(),
            "dispatch": (v["base_lin"] + v["dl_" + f]).mean(), "transit": (dist / speed).mean(),
            "dGP": 0.0 if f == cur else -(A.freight_rate * K + A.prod_cost_premium * cost),
            "vol_ratio": 1.0 if f == cur else (base_units[f] + pu) / base_units[f]})
    P = pd.DataFrame(rows)
    P["total"] = P["dispatch"] + P["transit"]
    ct = P.loc[P["current"], "total"].iloc[0]
    P["d_days"] = P["total"] - ct
    P["d_pct"] = P["d_days"] / ct * 100
    return P


def breakdown(d: pd.DataFrame, alt: str, A: Assumptions, by: str) -> pd.DataFrame:
    """Current vs alternative predicted total days, split by ship mode or region."""
    v = _with_assumptions(d, A)
    v = v.assign(cur_total=v["base_lin"] + v["dist_mi"] / v["speed"],
                 alt_total=v["base_lin"] + v["dl_" + alt] + v["dist_" + alt] / v["speed"])
    g = v.groupby(by).agg(orders=("Row ID", "size"), current=("cur_total", "mean"), alternative=("alt_total", "mean"),
                          avg_dist_cur=("dist_mi", "mean"), avg_dist_alt=("dist_" + alt, "mean")).reset_index()
    g["change_days"] = g["alternative"] - g["current"]
    order = SHIP_ORDER if by == "Ship Mode" else REGIONS
    g[by] = pd.Categorical(g[by], order, ordered=True)
    return g.sort_values(by).reset_index(drop=True)


def profit_grid(d: pd.DataFrame, alt: str, A: Assumptions, rates=None, premiums=None):
    """Profit change (USD) over a grid of freight rates x production-cost premiums."""
    v = _with_assumptions(d, A)
    K = (v["Units"] * v["mult"] * (v["dist_" + alt] - v["dist_mi"]) / 1000.0).sum()
    cost = v["Cost"].sum()
    rates = np.round(np.arange(0.0, 0.3001, 0.05), 2) if rates is None else rates
    premiums = np.round(np.arange(-0.02, 0.1001, 0.02), 2) if premiums is None else premiums
    Z = np.array([[-(r * K + c * cost) for r in rates] for c in premiums])
    return rates, premiums, Z
