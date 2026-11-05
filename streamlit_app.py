"""Streamlit dashboard: historical 2024 test-season predictions vs the closing spread.

Run:  streamlit run streamlit_app.py
Research visualization only. Not betting advice; no live odds are fetched.
"""
import altair as alt
import pandas as pd
import streamlit as st

from sfl.pipeline import MODEL_NAMES, run_pipeline

st.set_page_config(page_title="Sports Forecasting Lab", layout="wide")


@st.cache_resource(show_spinner="Running pipeline from data/sports.db ...")
def load():
    return run_pipeline(log_mlflow=False)


r = load()
pred = r["pred_test"]

st.title("Sports Forecasting Lab: NFL 2024 out-of-sample predictions")
st.caption("Personal research project on public nflverse data. Models trained on 2021-2023, "
           "evaluated on 2024. Historical evaluation only, not betting advice; no live odds.")

week = st.selectbox("2024 week", sorted(pred["week"].unique()))
wk = pred[pred["week"] == week].copy()
wk["matchup"] = wk["away_team"] + " @ " + wk["home_team"]
wk["model_side_vs_line"] = (wk["margin_spread_ridge"] > wk["spread_line"]).map(
    {True: "home", False: "away"})
wk["home_covered"] = (wk["result"] > wk["spread_line"]).map({True: "home", False: "away"})
wk.loc[wk["result"] == wk["spread_line"], "home_covered"] = "push"
table = wk[["matchup", "game_type", "p_logreg", "p_xgboost", "p_spread_ridge", "p_market_spread",
            "margin_spread_ridge", "spread_line", "model_side_vs_line", "result", "home_covered"]]
table = table.rename(columns={
    "p_logreg": "P(home) logreg", "p_xgboost": "P(home) xgboost",
    "p_spread_ridge": "P(home) spread model", "p_market_spread": "P(home) from closing line",
    "margin_spread_ridge": "predicted home margin", "spread_line": "closing line (home fav +)",
    "model_side_vs_line": "model side vs line", "result": "actual home margin",
    "home_covered": "covering side"})
st.subheader(f"Week {week}")
st.dataframe(table.style.format(precision=3), width="stretch", hide_index=True)

c1, c2 = st.columns(2)
with c1:
    st.subheader("Test-season metrics (2024)")
    st.dataframe(r["metrics"][["model", "accuracy", "log_loss", "brier", "ece", "ats_hit_rate"]]
                 .style.format(precision=4), hide_index=True)
with c2:
    st.subheader("Calibration (10 bins)")
    cal = pd.concat([df.assign(model=name) for name, df in r["calibration"].items()])
    cal = cal[cal["n"] > 0]
    diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(
        strokeDash=[4, 4], color="gray").encode(x="x", y="y")
    pts = alt.Chart(cal).mark_line(point=True).encode(
        x=alt.X("mean_pred", title="mean predicted P(home win)", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("actual_rate", title="actual home win rate", scale=alt.Scale(domain=[0, 1])),
        color="model", tooltip=["model", "bin", "n", "mean_pred", "actual_rate"])
    st.altair_chart(diag + pts, width="stretch")

st.subheader("Monte Carlo season simulation (10,000 sims, seed 42)")
st.caption(f"2024 regular season simulated from week {r['sim_metrics']['sim_cutoff_week']} "
           "cutoff with features frozen at the cutoff; compared to actual final wins.")
st.dataframe(r["simulation"].style.format(precision=2), hide_index=True, width="stretch")
