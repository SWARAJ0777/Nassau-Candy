"""Nassau Candy - Factory Reallocation & Shipping Optimization (Streamlit dashboard)."""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import engine as E

st.set_page_config(page_title="Nassau Candy Factory Optimizer", page_icon="🍬", layout="wide")

DATA = Path(__file__).parent / "data" / "Nassau_Candy_Distributor.csv"

# Palette validated with the dataviz validator (first three categorical slots, light surface #fcfcfb)
SURFACE = "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"           # series 1 = current / learned, series 2 = alternative
BLUE_TINT = "#cde2fb"
INK, MUTED = "#0b0b0b", "#898781"
DIV_NEG, DIV_MID, DIV_POS = "#e34948", "#f0efec", "#2a78d6"


# --------------------------------------------------------------------------- cached engine
@st.cache_resource(show_spinner="Loading data and training models (first run only)...")
def get_data() -> pd.DataFrame:
    df = E.load_data(DATA)
    models = E.fit_models(df)
    return E.add_counterfactuals(df, models)


@st.cache_resource(show_spinner="Running order-grouped cross-validation (about 20 seconds)...")
def get_validation() -> pd.DataFrame:
    return E.validation_metrics(get_data())


def style(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=48, b=8), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        font=dict(size=13), bargap=0.35)
    fig.update_xaxes(showgrid=False, linecolor="#c3c2b7")
    fig.update_yaxes(gridcolor="#e1e0d9", zeroline=True, zerolinecolor="#c3c2b7")
    try:
        fig.update_layout(barcornerradius=4)
    except Exception:
        pass
    return fig


def fmt_flags(s: str) -> str:
    return "; ".join(E.FLAG_LABELS[f] for f in s.split(",")) if s else "—"


full = get_data()

# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Controls")
    products = full.groupby("Product Name")["Sales"].sum().sort_values(ascending=False).index.tolist()
    product = st.selectbox("Product", products, format_func=lambda p: f"{p}  ·  {E.PRODUCT_FACTORY[p]}")
    regions = st.multiselect("Customer region", E.REGIONS, default=E.REGIONS)
    modes = st.multiselect("Ship mode", E.SHIP_ORDER, default=E.SHIP_ORDER)
    priority = st.slider("Optimization priority", 0, 100, 50,
                         help="0 = protect profit, 100 = maximize speed. Re-ranks options; "
                              "a tail-risk term keeps a small fixed weight.")
    st.caption("Profit ◄ 0 ········· 100 ► Speed")

    A0 = E.Assumptions()
    with st.expander("Assumptions (editable)", expanded=False):
        st.caption("The data has no delivery date and no freight cost, so transit time and freight are "
                   "**assumptions**. Replace them with carrier and finance data when you have it.")
        speed = {m: st.number_input(f"Transit speed, {m} (miles/day)", 100.0, 3000.0, A0.speed[m], 50.0)
                 for m in E.SHIP_ORDER}
        freight_rate = st.number_input("Freight cost, \\$ per unit per 1,000 miles (Standard Class)",
                                       0.0, 2.0, A0.freight_rate, 0.01)
        mult = {m: st.number_input(f"Freight multiplier, {m}", 0.5, 10.0, A0.freight_mult[m], 0.1)
                for m in E.SHIP_ORDER}
        premium = st.slider("Production cost at new factory vs current (%)", -5.0, 15.0, 0.0, 0.5) / 100
        cap = st.slider("Max volume growth at a receiving factory (×)", 1.0, 3.0, A0.capacity_limit, 0.1,
                        help="Counts every accepted move together, not one at a time.")
        min_red = st.slider("Minimum lead-time gain to recommend a move (%)", 0.0, 10.0, A0.min_reduction_pct, 0.5)
        risk_w = st.slider("Weight on tail-risk reduction", 0.0, 0.5, 0.15, 0.05,
                           help="Tail risk = 90th-percentile predicted lead time.")

A = E.Assumptions(speed=speed, freight_mult=mult, freight_rate=freight_rate, prod_cost_premium=premium,
                  capacity_limit=cap, min_reduction_pct=min_red)

if not regions or not modes:
    st.sidebar.warning("An empty filter was ignored (all values used).")
view = full[full["Region"].isin(regions or E.REGIONS) & full["Ship Mode"].isin(modes or E.SHIP_ORDER)]

C = E.scenarios(view, full, A)
if C.empty:
    st.warning("No orders match the current filters.")
    st.stop()
R = E.recommend(C, A, full, priority / 100, risk_w)
C = E.add_scores(C, priority / 100, risk_w)

# --------------------------------------------------------------------------- header
st.title("🍬 Nassau Candy · Factory Optimizer")
st.caption(f"Factory reallocation & shipping optimization · {len(view):,} orders in view · "
           f"{view['Product Name'].nunique()} products · customer regions: {', '.join(regions or E.REGIONS)}")
st.info("**How to read this.** Dispatch time (order → shipped) is **learned from your data**. "
        "Transit time and freight cost are **assumptions** you can edit in the sidebar. "
        "Charts mark learned parts solid and assumed parts hatched, so you can always see which is which.",
        icon="ℹ️")

tab_sim, tab_what, tab_rec, tab_risk, tab_method = st.tabs(
    ["Factory Simulator", "What-If Analysis", "Recommendations", "Risk & Impact", "Method & Assumptions"])

d = view[view["Product Name"] == product]
cur = E.PRODUCT_FACTORY[product]

# --------------------------------------------------------------------------- 1. simulator
with tab_sim:
    st.subheader(f"{product}: predicted performance by factory")
    if d.empty:
        st.info("This product has no orders under the current filters.")
    else:
        P = E.factory_profile(d, full, A)
        labels = [f"{r.factory}{' (current)' if r.current else ''}" for r in P.itertuples()]
        fig = go.Figure()
        fig.add_bar(x=labels, y=P["dispatch"], name="Dispatch time (learned from data)",
                    marker=dict(color=BLUE, line=dict(color=SURFACE, width=2)),
                    hovertemplate="%{x}<br>Dispatch (learned): %{y:.2f} days<extra></extra>")
        fig.add_bar(x=labels, y=P["transit"], name="Transit time (assumed)",
                    marker=dict(color=BLUE_TINT, line=dict(color=SURFACE, width=2),
                                pattern=dict(shape="/", fgcolor=BLUE, bgcolor=BLUE_TINT, size=7, solidity=0.35)),
                    hovertemplate="%{x}<br>Transit (assumed): %{y:.2f} days<extra></extra>")
        fig.add_scatter(x=labels, y=P["total"], mode="text", text=[f"{t:.2f}" for t in P["total"]],
                        textposition="top center", showlegend=False, hoverinfo="skip",
                        textfont=dict(color=INK, size=14))
        fig.update_layout(barmode="stack", yaxis_title="Predicted days, order to delivery")
        fig.update_yaxes(range=[0, P["total"].max() * 1.18])
        st.plotly_chart(style(fig, 400), width="stretch")

        spread = P["dispatch"].max() - P["dispatch"].min()
        best = P.loc[P["total"].idxmin()]
        if best["current"]:
            best_note = " (already the current factory)"
        else:
            best_note = " ({:+.2f} days, {:+.1f}% vs current)".format(best["d_days"], best["d_pct"])
        st.markdown(
            "- Fastest predicted option: **{}**{}.\n".format(best["factory"], best_note)
            + "- Across all five factories the **learned** dispatch time differs by only **{:.2f} days**; ".format(spread)
            + "differences beyond that come from the **assumed** transit layer.")

        T = P.assign(Factory=labels)[["Factory", "avg_dist", "dispatch", "transit", "total", "d_days", "d_pct", "dGP", "vol_ratio"]]
        T.columns = ["Factory", "Avg distance (mi)", "Dispatch days (learned)", "Transit days (assumed)",
                     "Total days", "Δ days vs current", "Δ % vs current", "Δ gross profit ($)", "Volume at factory (×)"]
        st.dataframe(T, hide_index=True, width="stretch", column_config={
            "Avg distance (mi)": st.column_config.NumberColumn(format="%.0f"),
            "Dispatch days (learned)": st.column_config.NumberColumn(format="%.2f"),
            "Transit days (assumed)": st.column_config.NumberColumn(format="%.2f"),
            "Total days": st.column_config.NumberColumn(format="%.2f"),
            "Δ days vs current": st.column_config.NumberColumn(format="%+.2f"),
            "Δ % vs current": st.column_config.NumberColumn(format="%+.1f%%"),
            "Δ gross profit ($)": st.column_config.NumberColumn(format="%+,.0f"),
            "Volume at factory (×)": st.column_config.NumberColumn(format="%.2f×")})
        st.caption(f"Gross-profit change uses your freight and production-cost assumptions, on this product's "
                   f"{len(d):,} orders in view. Volume (×) is the receiving factory's total units after the move.")

# --------------------------------------------------------------------------- 2. what-if
alt = None
with tab_what:
    st.subheader(f"What-if: {product}")
    sp = C[C["product"] == product].sort_values("score", ascending=False)
    if sp.empty:
        st.info("This product has no orders under the current filters.")
    else:
        alts = sp["to"].tolist()
        alt = st.selectbox("Compare current factory with", alts, index=0, key=f"alt_{product}",
                           help="Options are ordered by your priority-weighted score (best first).")
        r = sp[sp["to"] == alt].iloc[0]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Predicted lead time", f"{r['base_days'] + r['d_total']:.2f} days",
                  f"{r['d_total']:+.2f} days vs current", delta_color="inverse")
        c2.metric("Lead-time change", f"{-r['red_pct']:+.1f}%", help="Negative = faster. Includes assumed transit time.")
        c3.metric("Gross profit change", f"${r['dGP']:+,.0f}", f"{r['dGP_pct']:+.2f}% of product GP", delta_color="normal")
        c4.metric("Scenario confidence", f"{r['conf']:.0f} / 100",
                  help="Data support x model agreement, scaled down when the gain is mostly assumed.")
        c5.metric("Volume at new factory", f"{r['vol_ratio']:.1f}×", help=f"Your capacity cap is {A.capacity_limit:.1f}×.")
        st.caption(f"Learned (data-only) part of the lead-time change: **{-r['red_pct_data']:+.1f}%**  ·  "
                   f"Flags: **{fmt_flags(r['flags'])}**")

        col_a, col_b = st.columns(2)
        for col, by, title in [(col_a, "Ship Mode", "By ship mode"), (col_b, "Region", "By customer region")]:
            B = E.breakdown(d, alt, A, by)
            fg = go.Figure()
            fg.add_bar(x=B[by].astype(str), y=B["current"], name=f"Current: {cur}", marker=dict(color=BLUE, line=dict(color=SURFACE, width=2)),
                       text=[f"{v:.2f}" for v in B["current"]], textposition="outside",
                       hovertemplate="%{x}<br>Current: %{y:.2f} days<extra></extra>")
            fg.add_bar(x=B[by].astype(str), y=B["alternative"], name=f"Alternative: {alt}", marker=dict(color=ORANGE, line=dict(color=SURFACE, width=2)),
                       text=[f"{v:.2f}" for v in B["alternative"]], textposition="outside",
                       hovertemplate="%{x}<br>Alternative: %{y:.2f} days<extra></extra>")
            fg.update_layout(barmode="group", yaxis_title="Predicted days")
            fg.update_yaxes(range=[0, max(B["current"].max(), B["alternative"].max()) * 1.2])
            col.markdown(f"**{title}**")
            col.plotly_chart(style(fg, 360), width="stretch")
        st.caption("Totals include assumed transit time. Under the data-only (learned) view, the gap between "
                   "current and alternative is "
                   f"{r['d_dispatch']:+.2f} days on average.")

# --------------------------------------------------------------------------- 3. recommendations
with tab_rec:
    K = E.kpis(R)
    st.subheader("Recommendations")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Lead Time Reduction", f"{K['lead_time_reduction_pct']:.1f}%",
              help="Sales-weighted, over products with a recommended move.")
    k2.metric("Profit Impact Stability", "n/a" if np.isnan(K["profit_impact_stability"]) else f"{K['profit_impact_stability']*100:.0f}%",
              help="Share of 12 freight-rate x production-cost cases (freight \\$0.05-0.20, cost -2% to +5%) in which a move does not lose money. Sales-weighted.")
    k3.metric("Scenario Confidence", "n/a" if np.isnan(K["scenario_confidence"]) else f"{K['scenario_confidence']:.0f} / 100",
              help="Data support x model agreement, scaled down when the gain is mostly assumed. Sales-weighted.")
    k4.metric("Recommendation Coverage", f"{K['moves']} of {K['products']} products",
              help="Products with a move that is fast enough, profit-safe and within capacity.")
    st.caption(f"Across **all** products in view the reduction is **{K['portfolio_reduction_pct']:.2f}%**; recommended moves cover "
               f"**{K['coverage_sales']*100:.0f}%** of sales. Total gross-profit change: **\\${K['d_profit']:+,.0f}** "
               f"({K['d_profit_pct']:+.2f}% of gross profit in view) at your freight and production-cost assumptions.")

    moves = R[R["action"] == "MOVE"].copy()
    # Materiality: a 15% gain on 3 orders matters far less than 5% on 2,000. Rank by order-days saved by default.
    moves["impact"] = -moves["d_total"] * moves["n"]
    rank_label = st.radio("Rank recommendations by", ["Impact (order-days saved)", "Speed gain (%)", "Priority score"],
                          horizontal=True,
                          help="Impact = predicted days saved per order x orders in view, so high-volume products "
                               "rank first. Speed gain % alone favours tiny products with a handful of orders.")
    rank_col = {"Impact (order-days saved)": "impact", "Speed gain (%)": "red_pct", "Priority score": "score"}[rank_label]
    moves = moves.sort_values(rank_col, ascending=False)
    top_n = st.slider("Show top N recommendations", 1, max(1, len(moves)), min(5, max(1, len(moves)))) if len(moves) > 1 else 1
    if moves.empty:
        st.warning("No product has a move that is fast enough, profit-safe and within capacity. "
                   "Try lowering the minimum gain, raising the capacity cap, or editing the assumptions.")
    else:
        top = moves.head(top_n).copy()
        top["rank"] = range(1, len(top) + 1)
        fig = go.Figure()
        short = [p.replace("Wonka Bar - ", "").replace("Wonka Bar -", "") for p in top["product"]]
        fig.add_bar(y=short, x=top["red_pct_data"], orientation="h", name="Learned from data",
                    marker=dict(color=BLUE, line=dict(color=SURFACE, width=2)),
                    hovertemplate="%{y}<br>Learned: %{x:+.2f}%<extra></extra>")
        fig.add_bar(y=short, x=top["red_pct"] - top["red_pct_data"], orientation="h", name="Assumed (transit)",
                    marker=dict(color=BLUE_TINT, line=dict(color=SURFACE, width=2),
                                pattern=dict(shape="/", fgcolor=BLUE, bgcolor=BLUE_TINT, size=7, solidity=0.35)),
                    hovertemplate="%{y}<br>Assumed: %{x:+.2f}%<extra></extra>")
        # label sits just past the right end of the positive stack (the learned part can be negative)
        right = np.maximum(top["red_pct_data"], 0) + np.maximum(top["red_pct"] - top["red_pct_data"], 0)
        fig.add_scatter(y=short, x=right, mode="text", text=[f"  net {v:.1f}%  → {t}" for v, t in zip(top["red_pct"], top["to"])],
                        textposition="middle right", showlegend=False, hoverinfo="skip", textfont=dict(color=INK, size=13))
        fig.update_layout(barmode="relative", xaxis_title="Lead-time reduction (%)", yaxis=dict(autorange="reversed"))
        fig.update_xaxes(range=[min(0, top["red_pct_data"].min()) * 1.3 - 0.5, top["red_pct"].max() * 1.9 + 1],
                         showgrid=True, gridcolor="#e1e0d9")
        fig.update_yaxes(showgrid=False)
        st.markdown("**Predicted lead-time reduction, split into learned and assumed parts**")
        st.plotly_chart(style(fig, 130 + 60 * len(top)), width="stretch")

        T = top.assign(Move=top["from"] + "  →  " + top["to"], Flags=top["flags"].map(fmt_flags))[
            ["rank", "product", "Move", "n", "red_pct", "red_pct_data", "impact", "dGP", "stability", "conf", "joint_vol_ratio", "Flags"]]
        T.columns = ["Rank", "Product", "Move", "Orders", "Faster by (%)", "…of which learned (%)", "Order-days saved",
                     "Δ gross profit ($)", "Profit stability", "Confidence", "Receiving factory volume (×)", "Flags"]
        st.dataframe(T, hide_index=True, width="stretch", column_config={
            "Orders": st.column_config.NumberColumn(format="%,d"),
            "Order-days saved": st.column_config.NumberColumn(format="%,.0f"),
            "Faster by (%)": st.column_config.NumberColumn(format="%.1f"),
            "…of which learned (%)": st.column_config.NumberColumn(format="%+.2f"),
            "Δ gross profit ($)": st.column_config.NumberColumn(format="%+,.0f"),
            "Profit stability": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent"),
            "Confidence": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f"),
            "Receiving factory volume (×)": st.column_config.NumberColumn(format="%.2f×")})

    with st.expander("All products: decisions, including KEEP"):
        T = R.assign(Decision=np.where(R["action"] == "MOVE", "MOVE → " + R["to"], "KEEP at " + R["from"]),
                     Flags=R["flags"].map(fmt_flags))[
            ["product", "Decision", "red_pct", "dGP", "conf", "n", "sales", "reason", "Flags"]].sort_values("sales", ascending=False)
        T.columns = ["Product", "Decision", "Faster by (%)", "Δ gross profit ($)", "Confidence", "Orders", "Sales ($)", "Why", "Flags"]
        st.dataframe(T, hide_index=True, width="stretch", column_config={
            "Faster by (%)": st.column_config.NumberColumn(format="%.1f"),
            "Δ gross profit ($)": st.column_config.NumberColumn(format="%+,.0f"),
            "Confidence": st.column_config.NumberColumn(format="%.0f"),
            "Sales ($)": st.column_config.NumberColumn(format="%,.0f")})
    export = R.assign(flags=R["flags"].map(fmt_flags))[["product", "action", "from", "to", "red_pct", "red_pct_data", "dGP", "stability", "conf", "joint_vol_ratio", "flags", "reason", "n", "sales"]]
    st.download_button("Download recommendations (CSV)", export.to_csv(index=False).encode(), "nassau_recommendations.csv", "text/csv")

# --------------------------------------------------------------------------- 4. risk & impact
with tab_risk:
    st.subheader("Risk & impact")
    st.markdown("**Alerts on the recommended moves**")
    mv = R[R["action"] == "MOVE"]
    if mv.empty:
        st.success("No moves are recommended under the current settings, so there is nothing to flag.")
    else:
        groups = {}
        for _, r in mv.sort_values("sales", ascending=False).iterrows():
            for f in (r["flags"].split(",") if r["flags"] else []):
                groups.setdefault(f, []).append(r["product"].replace("Wonka Bar - ", "").replace("Wonka Bar -", ""))
        why = {"PROFIT_NEG": ("🔴", st.error, "Loses money at your freight and production-cost assumptions"),
               "CAPACITY": ("🟠", st.warning, "Would push the receiving factory past your volume cap"),
               "MODEL_DISAGREE": ("⚠️", st.warning, "Ridge and Gradient Boosting disagree by more than 0.15 days"),
               "LOW_DATA": ("⚠️", st.warning, "Fewer than 30 orders behind the prediction; treat as indicative only"),
               "ASSUMPTION_DRIVEN": ("⚠️", st.warning, "Under 25% of the predicted gain is learned from data; the rest is the transit assumption")}
        for f in ["PROFIT_NEG", "CAPACITY", "MODEL_DISAGREE", "LOW_DATA", "ASSUMPTION_DRIVEN"]:
            if f in groups:
                icon, box, msg = why[f]
                box(f"**{E.FLAG_LABELS[f]}** ({len(groups[f])}): {', '.join(groups[f])}. {msg}.", icon=icon)
        if not groups:
            st.success("No flags on any recommended move.")
        else:
            st.caption("Test assumption-driven moves in the sensitivity grid below: they only hold if your transit-speed assumption does.")
    blocked = R[(R["action"] == "KEEP") & (R["reason"].str.contains("overload", na=False))]
    if len(blocked):
        st.warning("Capacity blocked the best option for: " + ", ".join(blocked["product"]) +
                   ". Raise the capacity cap in the sidebar to see what the unconstrained answer would be.", icon="🏭")

    if alt is not None:
        rr = C[(C["product"] == product) & (C["to"] == alt)].iloc[0]
        st.markdown(f"**Profit sensitivity**: {product}, {cur} → {alt}")
        st.caption("Change the alternative in the What-If tab. Cells show the change in gross profit (US dollars) over this product's "
                   "orders in view; blue = profit gain, red = loss.")
        rates, prems, Z = E.profit_grid(d, alt, A)
        m = max(abs(Z).max(), 1e-9)
        xl = [f"${x:.2f}" for x in rates]
        yl = [f"{p*100:+.0f}%" for p in prems]
        hm = go.Figure(go.Heatmap(z=Z, x=xl, y=yl, zmin=-m, zmax=m,
                                  colorscale=[[0, DIV_NEG], [0.5, DIV_MID], [1, DIV_POS]], showscale=False,
                                  hovertemplate="Freight $%{x} per unit per 1,000 mi<br>Production cost %{y}<br>Profit change $%{z:,.0f}<extra></extra>"))
        for i, p in enumerate(prems):
            for j, rt in enumerate(rates):
                hm.add_annotation(x=xl[j], y=yl[i], text="0" if abs(Z[i, j]) < 0.5 else f"{Z[i, j]:+,.0f}", showarrow=False,
                                  font=dict(size=12, color="white" if abs(Z[i, j]) > 0.62 * m else INK))
        hm.update_layout(xaxis_title="Freight cost ($ per unit per 1,000 miles, Standard Class)",
                         yaxis_title="Production cost at new factory vs current")
        hm.update_xaxes(type="category", showgrid=False)
        hm.update_yaxes(type="category", showgrid=False, zeroline=False)
        st.plotly_chart(style(hm, 360), width="stretch")
        be = rr["breakeven_premium_pct"]
        st.markdown(f"- Break-even: the new factory can cost up to **{be:+.1f}%** more per unit before this move loses money "
                    f"(at \\${A.freight_rate:.2f} freight).\n"
                    f"- Profit stability: **{rr['stability']*100:.0f}%** of the freight × cost scenarios (freight \\$0.05 to \\$0.20, cost −2% to +5%) show no loss.")

    st.markdown("**All 60 scenarios, with risk flags**")
    flag_pick = st.multiselect("Show only scenarios carrying any of these flags",
                               list(E.FLAG_LABELS), format_func=lambda f: E.FLAG_LABELS[f])
    S = C.copy()
    if flag_pick:
        S = S[S["flags"].apply(lambda s: any(f in s.split(",") for f in flag_pick))]
    S = S.sort_values(["sales", "score"], ascending=[False, False])
    T = S.assign(Move=S["from"] + "  →  " + S["to"], Flags=S["flags"].map(fmt_flags))[
        ["product", "Move", "red_pct", "red_pct_data", "dGP", "conf", "vol_ratio", "Flags"]]
    T.columns = ["Product", "Move", "Faster by (%)", "…of which learned (%)", "Δ gross profit ($)", "Confidence", "Volume at new factory (×)", "Flags"]
    st.dataframe(T, hide_index=True, width="stretch", column_config={
        "Faster by (%)": st.column_config.NumberColumn(format="%.1f"),
        "…of which learned (%)": st.column_config.NumberColumn(format="%+.2f"),
        "Δ gross profit ($)": st.column_config.NumberColumn(format="%+,.0f"),
        "Confidence": st.column_config.NumberColumn(format="%.0f"),
        "Volume at new factory (×)": st.column_config.NumberColumn(format="%.1f×")})

# --------------------------------------------------------------------------- 5. method
with tab_method:
    st.subheader("Method, assumptions and limits")
    chk = E.data_checks(full)
    st.markdown(f"""
**What is learned from data.** Dispatch time (Order Date → Ship Date), predicted by a Ridge regression on ship mode,
factory, distance, region, order year tier and month, with Gradient Boosting as a challenger. Rows of one order stay
together in every validation split.

**What is assumed.** Transit time (distance ÷ miles per day by ship mode), freight cost and any production-cost change at
a new factory. None of these exist in the dataset. Edit them in the sidebar.

**Date correction.** Raw ship-minus-order gaps run from {chk['raw_gap_min']:,} to {chk['raw_gap_max']:,} days because ship dates carry a
constant offset per Order-ID year. Subtracting each year tier's minimum gives a lead time of 0–{chk['lead_max']} days
(mean {chk['lead_mean']:.2f}), and ship modes then order correctly (Same Day fastest, Standard slowest). Confirm with the
data owner that this is how the dates were shifted.
""")
    st.dataframe(chk["tiers"].rename(columns={"min": "Raw gap min (days)", "max": "Raw gap max (days)", "count": "Rows"})
                 .rename_axis("Order-ID year"), width="stretch")
    st.markdown("""
**What the data says about factories.** Distance, factory and region add almost nothing to dispatch time
(distance effect ≈ 0.01 days per 1,000 miles, statistically indistinguishable from zero). Any gain beyond that depends on the
transit assumption, which is why the dashboard separates learned from assumed.

**Limits.**
- One product has only ever shipped from one factory, so alternative-factory predictions are extrapolations.
- Seven products have fewer than 15 orders; their recommendations are flagged *Low data* and should not drive decisions.
- Capacity is unknown. The volume cap is a stand-in; replace it with real plant capacity.
- Customer mix and ship-mode mix are held fixed in every scenario.
- The confidence score is a heuristic: data support × model agreement, scaled down when the gain is mostly assumed.
""")
    if st.button("Run model validation (order-grouped 5-fold CV)"):
        V = get_validation()
        st.dataframe(V, hide_index=True, width="stretch", column_config={
            "RMSE (days)": st.column_config.NumberColumn(format="%.3f"), "MAE (days)": st.column_config.NumberColumn(format="%.3f"),
            "R²": st.column_config.NumberColumn(format="%.3f"),
            "Within ±1 day": st.column_config.NumberColumn(format="percent"),
            "Within ±2 days": st.column_config.NumberColumn(format="percent")})
        st.caption("Ship mode explains most of the predictable variance; the models differ by under an hour of average error.")
