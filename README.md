# Nassau Candy · Factory Optimizer

Streamlit decision-support app for the *Factory Reallocation & Shipping Optimization* project.
It predicts shipping lead time, simulates moving each product to another factory, and recommends
moves that are faster, profit-safe and within factory capacity.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
pytest -q          # 42 tests: engine invariants, Stage 8 regression, headless UI tests
```

The first load takes a few seconds (it trains the models); after that every control is instant.
To use new data, replace `data/Nassau_Candy_Distributor.csv` (same columns).

## What is learned vs assumed

| Layer | Source |
|---|---|
| Dispatch time (Order Date → Ship Date) | **Learned** from the data (Ridge regression; Gradient Boosting as challenger) |
| Transit time | **Assumed**: distance ÷ miles/day by ship mode (sidebar) |
| Freight cost, production-cost change at a new factory | **Assumed** (sidebar) |
| Factory capacity | **Assumed**: a volume cap on the receiving factory (sidebar) |

The dataset has no delivery date, no freight cost and no capacity data, so these three are inputs you
should replace with carrier and finance numbers. Charts draw learned parts solid and assumed parts hatched.

## Data correction

Ship dates carry a constant offset per Order-ID year (raw gaps of 900-1,640 days). The engine subtracts each
year tier's minimum gap, leaving a 0-11 day lead time in which ship modes order correctly. Confirm this
with the data owner. See the *Method & Assumptions* tab.

## Dashboard

| Tab | Purpose |
|---|---|
| Factory Simulator | Pick a product; predicted days at each of the five factories (learned vs assumed) |
| What-If Analysis | Current vs alternative factory, by ship mode and customer region |
| Recommendations | Brief's four KPIs, ranked moves (by impact, speed or priority score), CSV export |
| Risk & Impact | Grouped alerts, profit-sensitivity grid, all 60 scenarios with risk flags |
| Method & Assumptions | Data correction, limits, on-demand model validation |

Sidebar: product, customer region, ship mode, speed-vs-profit priority slider, and an editable assumptions panel.

## Decision rules (`engine.recommend`)

A move is recommended only if it is at least *min gain* % faster (default 3%), does not lose profit at the chosen
assumptions, and keeps the receiving factory within the capacity cap (default 1.5×) **counting all accepted
moves together**. Products are processed largest-sales first. Otherwise the product stays (KEEP) with a reason.

Risk flags: low data (<30 orders), capacity stress, profit loss, models disagree (>0.15 days), mostly
assumption-driven (<25% of the gain is learned).

## Known limits

- Each product has only ever shipped from one factory, so alternative-factory effects are extrapolations.
- Seven products have fewer than 15 orders; their recommendations are indicative only.
- The scenario confidence score is a heuristic, not a calibrated probability.
- Customer mix and ship-mode mix are held fixed in every scenario.

## Files

```
app.py        Streamlit dashboard
engine.py     data prep, models, simulation, recommendations, KPIs
tests/        pytest suite (engine + UI)
data/         Nassau_Candy_Distributor.csv
.streamlit/   light theme (colors validated for colorblind separation)
```
