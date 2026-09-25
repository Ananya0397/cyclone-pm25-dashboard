from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(
    page_title="Cyclone-Aware PM2.5 Forecasting",
    page_icon="🌪️",
    layout="wide",
)

MODEL_FEATURES = [
    "PM25", "PM25_Lag1", "PM25_Lag2", "PM25_Lag3",
    "Hour", "DayOfYear", "Hour_Sin", "Hour_Cos",
    "AOD", "Wind_Speed", "Pressure", "Rainfall",
    "Phase_Phase 1", "Phase_Phase 2", "Phase_Phase 3",
]

DATA_FILE = Path(__file__).with_name("cyclone_dashboard_master_data.csv")


def cpcb_category(pm25):
    pm25 = float(pm25)
    if pm25 <= 30:
        return "Good"
    elif pm25 <= 60:
        return "Satisfactory"
    elif pm25 <= 90:
        return "Moderately polluted"
    elif pm25 <= 120:
        return "Poor"
    elif pm25 <= 250:
        return "Very poor"
    else:
        return "Severe"


@st.cache_data
def load_master():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Missing dataset: {DATA_FILE.name}")
    df = pd.read_csv(DATA_FILE)
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    return df.dropna(subset=["Time", "Cyclone"]).copy()


def prepare(df):
    df = df.copy()
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")

    phase = df["Cyclone_Phase"].astype(str)
    df["Phase_Phase 1"] = phase.eq("Phase 1").astype(int)
    df["Phase_Phase 2"] = phase.eq("Phase 2").astype(int)
    df["Phase_Phase 3"] = phase.eq("Phase 3").astype(int)

    for col in MODEL_FEATURES + ["PM25_Next_Hour"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.sort_values(["Cyclone", "Time"]).reset_index(drop=True)


def run_forecast(master, selected_cyclone, current_time):
    master = prepare(master)
    selected = master[master["Cyclone"].astype(str) == str(selected_cyclone)].copy()
    train = master[master["Cyclone"].astype(str) != str(selected_cyclone)].copy()
    train = train.dropna(subset=MODEL_FEATURES + ["PM25_Next_Hour"])

    row = selected[selected["Time"] == pd.Timestamp(current_time)]
    if row.empty:
        raise ValueError("Selected time was not found.")

    current = row.iloc[0]
    missing = current[MODEL_FEATURES].index[current[MODEL_FEATURES].isna()].tolist()
    if missing:
        raise ValueError("Missing model inputs: " + ", ".join(missing))

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(train[MODEL_FEATURES], train["PM25_Next_Hour"])

    X_current = current[MODEL_FEATURES].to_frame().T
    pred = max(0.0, float(model.predict(X_current)[0]))

    actual = current["PM25_Next_Hour"]
    actual = None if pd.isna(actual) else float(actual)

    return current, pred, actual, model, train


st.title("🌪️ Cyclone-Aware PM2.5 Forecasting Dashboard")
st.caption("Leakage-safe next-hour forecasting using the project's Phase 37 Random Forest workflow.")

master = prepare(load_master())
cyclones = sorted(master["Cyclone"].dropna().astype(str).unique())

with st.sidebar:
    st.header("🌪️ Forecast Controls")
    st.subheader("Select Cyclone")
    selected = st.selectbox("Cyclone", cyclones, label_visibility="collapsed")
    st.caption("Choose a cyclone from the project's standardized dataset.")

    st.divider()
    st.subheader("📁 Optional: New Cyclone Data")
    uploaded = st.file_uploader(
        "Upload cyclone CSV",
        type=["csv"],
        help="Optional future-use feature. The current project workflow uses the standardized dataset above.",
    )
    if uploaded is not None:
        st.info("Upload received. The current forecasting workflow remains based on the standardized project dataset.")
    else:
        st.caption("No upload is required for the current project workflow.")


event_rows = master[master["Cyclone"].astype(str) == str(selected)].copy()
valid_times = event_rows.dropna(subset=MODEL_FEATURES)["Time"].sort_values()

current_time = st.selectbox(
    "Select current observation time",
    list(valid_times),
    format_func=lambda x: pd.Timestamp(x).strftime("%Y-%m-%d %H:%M"),
)

if st.button("🚀 RUN NEXT-HOUR FORECAST", type="primary", use_container_width=True):
    try:
        current, pred, actual, model, train = run_forecast(master, selected, current_time)
    except Exception as e:
        st.error(str(e))
        st.stop()

    error = abs(actual - pred) if actual is not None else None
    category = cpcb_category(pred)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Current PM2.5", f"{float(current['PM25']):.2f} µg/m³")
    c2.metric("Predicted Next Hour", f"{pred:.2f} µg/m³")
    c3.metric("Historical Actual", "N/A" if actual is None else f"{actual:.2f} µg/m³")
    c4.metric("Absolute Error", "N/A" if error is None else f"{error:.2f} µg/m³")
    c5.metric("CPCB Category", category)

    st.info(
        f"**Leakage check:** {selected} is excluded from training. "
        f"The Random Forest uses 300 trees and the 15 project features from the Phase 37 workflow."
    )

    left, right = st.columns(2)

    with left:
        st.subheader("Actual vs Predicted PM2.5")
        event = master[master["Cyclone"].astype(str) == str(selected)].copy().sort_values("Time")
        event["Predicted"] = np.nan
        valid = event.dropna(subset=MODEL_FEATURES)
        if len(valid):
            preds = model.predict(valid[MODEL_FEATURES])
            event.loc[valid.index, "Predicted"] = np.maximum(preds, 0)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=event["Time"], y=event["PM25_Next_Hour"], mode="lines", name="Actual"))
        fig.add_trace(go.Scatter(x=event["Time"], y=event["Predicted"], mode="lines", name="Predicted"))
        fig.update_layout(height=420, xaxis_title="Time", yaxis_title="PM2.5 (µg/m³)", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Environmental Drivers")
        driver_cols = ["AOD", "Wind_Speed", "Pressure", "Rainfall"]
        driver = event[["Time"] + driver_cols].melt("Time", var_name="Driver", value_name="Value")
        fig2 = go.Figure()
        for name in driver["Driver"].unique():
            d = driver[driver["Driver"] == name]
            fig2.add_trace(go.Scatter(x=d["Time"], y=d["Value"], mode="lines", name=name))
        fig2.update_layout(height=420, xaxis_title="Time", yaxis_title="Value", hovermode="x unified")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Cyclone Phase")
    phase_counts = event["Cyclone_Phase"].value_counts().reindex(["Phase 1", "Phase 2", "Phase 3"]).dropna()
    st.bar_chart(phase_counts)

    st.subheader("Forecast Record")
    record = pd.DataFrame([{
        "Cyclone": selected,
        "Current Time": pd.Timestamp(current_time),
        "Forecast Time": pd.Timestamp(current_time) + pd.Timedelta(hours=1),
        "Current PM2.5": float(current["PM25"]),
        "Predicted PM2.5": pred,
        "Historical Actual PM2.5": actual,
        "Absolute Error": error,
        "CPCB Category": category,
        "Training Cyclones": train["Cyclone"].nunique(),
    }])
    st.dataframe(record, use_container_width=True)

    csv = record.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Forecast Result", csv, "forecast_result.csv", "text/csv")

st.divider()
st.caption("Historical actual next-hour PM2.5 is displayed only for validation. It is not supplied to the model as an input.")
