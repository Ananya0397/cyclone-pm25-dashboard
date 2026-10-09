
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="Cyclone-Aware PM2.5 Forecasting",
    page_icon="🌪️",
    layout="wide"
)

MODEL_FEATURES = [
    "PM25", "PM25_Lag1", "PM25_Lag2", "PM25_Lag3",
    "Hour", "DayOfYear", "Hour_Sin", "Hour_Cos",
    "AOD", "Wind_Speed", "Pressure", "Rainfall",
    "Phase_Phase 1", "Phase_Phase 2", "Phase_Phase 3"
]

REQUIRED = [
    "Time", "Cyclone", "PM25", "PM25_Next_Hour",
    "PM25_Lag1", "PM25_Lag2", "PM25_Lag3",
    "Hour", "DayOfYear", "Hour_Sin", "Hour_Cos",
    "AOD", "Wind_Speed", "Pressure", "Rainfall",
    "Cyclone_Phase"
]

@st.cache_data
def load_master(path="cyclone_dashboard_master_data.csv"):
    df = pd.read_csv(path)
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    return df.dropna(subset=["Time", "Cyclone"]).copy()

def prepare(df):
    df = df.copy()
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    dummies = pd.get_dummies(df["Cyclone_Phase"], prefix="Phase")
    for c in ["Phase_Phase 1", "Phase_Phase 2", "Phase_Phase 3"]:
        if c not in dummies:
            dummies[c] = 0
    df = pd.concat([df.reset_index(drop=True),
                    dummies[["Phase_Phase 1","Phase_Phase 2","Phase_Phase 3"]].reset_index(drop=True)], axis=1)
    for c in MODEL_FEATURES + ["PM25_Next_Hour"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values(["Cyclone","Time"]).reset_index(drop=True)

def cpcb_category(x):
    if x <= 30: return "Good"
    if x <= 60: return "Satisfactory"
    if x <= 90: return "Moderate"
    if x <= 120: return "Poor"
    if x <= 250: return "Very Poor"
    return "Severe"


def run_forecast(master, selected_cyclone, current_time, event_data=None):
    master = prepare(master)

    # Train on the other historical cyclones for known events.
    # For a new cyclone, train on all historical cyclones.
    selected_in_master = (
        master["Cyclone"].astype(str) == str(selected_cyclone)
    ).any()

    if selected_in_master:
        train = master[
            master["Cyclone"].astype(str) != str(selected_cyclone)
        ].copy()
    else:
        train = master.copy()

    train = train.dropna(
        subset=MODEL_FEATURES + ["PM25_Next_Hour"]
    )

    # Use uploaded event data when provided.
    source = event_data.copy() if event_data is not None else master[
        master["Cyclone"].astype(str) == str(selected_cyclone)
    ].copy()

    source = prepare(source)
    source["Time"] = pd.to_datetime(source["Time"], errors="coerce")
    row = source[source["Time"] == pd.Timestamp(current_time)]

    if row.empty:
        raise ValueError("Selected observation time was not found in the uploaded data.")

    current = row.iloc[-1].copy()

    # Build lag features from observations available up to current_time.
    history = source[source["Time"] <= pd.Timestamp(current_time)].sort_values("Time")
    history = history.drop_duplicates(subset=["Time"], keep="last")

    if len(history) < 4:
        raise ValueError(
            "At least 4 hourly observations are needed to construct PM2.5 lag features."
        )

    for lag in (1, 2, 3):
        current[f"PM25_Lag{lag}"] = float(history.iloc[-(lag + 1)]["PM25"])

    timestamp = pd.Timestamp(current_time)
    current["Hour"] = timestamp.hour
    current["DayOfYear"] = timestamp.dayofyear
    current["Hour_Sin"] = np.sin(2 * np.pi * timestamp.hour / 24)
    current["Hour_Cos"] = np.cos(2 * np.pi * timestamp.hour / 24)

    # Preserve the phase supplied by the data, if available.
    phase = str(current.get("Cyclone_Phase", "Phase 2"))
    for p in ["Phase 1", "Phase 2", "Phase 3"]:
        current[f"Phase_{p}"] = int(phase == p)

    missing = [
        feature for feature in MODEL_FEATURES
        if pd.isna(current.get(feature, np.nan))
    ]
    if missing:
        raise ValueError("Missing model inputs: " + ", ".join(missing))

    model = RandomForestRegressor(
        n_estimators=300, random_state=42, n_jobs=-1
    )
    model.fit(train[MODEL_FEATURES], train["PM25_Next_Hour"])

    X = pd.DataFrame(
        [[current[f] for f in MODEL_FEATURES]],
        columns=MODEL_FEATURES
    )
    pred = max(0.0, float(model.predict(X)[0]))

    # Actual next-hour value is only available for historical data.
    actual = current.get("PM25_Next_Hour", None)
    actual = None if actual is None or pd.isna(actual) else float(actual)

    return current, pred, actual, model, train

st.title("🌪️ Cyclone-Aware PM2.5 Forecasting Dashboard")
st.caption("Leakage-safe next-hour forecasting using the project's Phase 37 Random Forest workflow.")

if not Path("cyclone_dashboard_master_data.csv").exists():
    st.error("Place cyclone_dashboard_master_data.csv in the same folder as app.py.")
    st.stop()

master = load_master()
master = prepare(master)
cyclones = sorted(master["Cyclone"].dropna().astype(str).unique())

with st.sidebar:
    st.header("Input")
    uploaded = st.file_uploader(
        "Upload a cyclone CSV (optional)",
        type=["csv"],
        help="For the exact project workflow, the uploaded file should contain the same engineered columns as the exported project dataset."
    )
    if uploaded is not None:
        upload_df = pd.read_csv(uploaded)
        upload_df = prepare(upload_df)
        uploaded_names = sorted(upload_df["Cyclone"].dropna().astype(str).unique()) if "Cyclone" in upload_df else []
        if uploaded_names:
            selected = st.selectbox("Cyclone from upload", uploaded_names)
            working = upload_df
        else:
            st.warning("Cyclone column not found in the uploaded file.")
            selected = st.selectbox("Cyclone", cyclones)
            working = master
    else:
        selected = st.selectbox("Cyclone", cyclones)
        working = master

# For LOCO training, the master dataset is authoritative. If an uploaded
# cyclone exists in the master dataset, its uploaded rows are used for the
# displayed selected event while all other master cyclones provide training.
event_rows = working[working["Cyclone"].astype(str) == str(selected)].copy()
if event_rows.empty:
    st.error("No rows found for the selected cyclone.")
    st.stop()

valid_times = event_rows.dropna(subset=MODEL_FEATURES)["Time"].sort_values()
current_time = st.selectbox("Select current observation time", list(valid_times), format_func=lambda x: pd.Timestamp(x).strftime("%Y-%m-%d %H:%M"))

if st.button("🚀 RUN NEXT-HOUR FORECAST", type="primary", use_container_width=True):
    try:
        # Use master data for the LOCO training population.
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
        f"The Random Forest uses 300 trees and the 15 project features shown in the notebook's Phase 37."
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
    phase_counts = event["Cyclone_Phase"].value_counts().reindex(["Phase 1","Phase 2","Phase 3"]).dropna()
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
        "Training Cyclones": train["Cyclone"].nunique()
    }])
    st.dataframe(record, use_container_width=True)

    csv = record.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Forecast Result", csv, "forecast_result.csv", "text/csv")

st.divider()
st.caption("Historical actual next-hour PM2.5 is displayed only for validation. It is not supplied to the model as an input.")
