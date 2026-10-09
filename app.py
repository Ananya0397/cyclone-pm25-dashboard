import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

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
RAW_REQUIRED = ["Time", "PM25", "AOD", "Wind_Speed", "Pressure", "Rainfall"]
PHASE_DUMMIES = ["Phase_Phase 1", "Phase_Phase 2", "Phase_Phase 3"]


def parse_times(series):
    """Parse ISO timestamps and common DD-MM-YYYY HH:MM timestamps safely."""
    parsed = pd.to_datetime(series, errors="coerce")
    # If most values failed, retry day-first (e.g. 21-09-2026 00:00).
    if parsed.notna().sum() < max(1, int(0.7 * len(series))):
        parsed_dayfirst = pd.to_datetime(series, dayfirst=True, errors="coerce")
        if parsed_dayfirst.notna().sum() > parsed.notna().sum():
            parsed = parsed_dayfirst
    return parsed


@st.cache_data

def load_master_bytes(data_bytes):
    df = pd.read_csv(io.BytesIO(data_bytes))
    if "Time" not in df or "Cyclone" not in df:
        raise ValueError("Historical dataset must contain Time and Cyclone columns.")
    df["Time"] = parse_times(df["Time"])
    return df.dropna(subset=["Time", "Cyclone"]).copy()


def add_time_features(df):
    df = df.copy()
    df["Time"] = parse_times(df["Time"])
    df["Hour"] = df["Time"].dt.hour
    df["DayOfYear"] = df["Time"].dt.dayofyear
    df["Hour_Sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["Hour_Cos"] = np.cos(2 * np.pi * df["Hour"] / 24)
    return df


def add_phase_features(df):
    df = df.copy()
    if "Cyclone_Phase" not in df.columns:
        df["Cyclone_Phase"] = "Phase 2"
    phase = df["Cyclone_Phase"].astype(str).str.strip()
    phase = phase.where(phase.isin(["Phase 1", "Phase 2", "Phase 3"]), "Phase 2")
    df["Cyclone_Phase"] = phase
    for phase_name, col in zip(["Phase 1", "Phase 2", "Phase 3"], PHASE_DUMMIES):
        df[col] = (df["Cyclone_Phase"] == phase_name).astype(int)
    return df


def prepare_historical(df):
    df = add_time_features(df)
    required_base = ["PM25", "AOD", "Wind_Speed", "Pressure", "Rainfall"]
    for col in required_base + ["PM25_Next_Hour", "PM25_Lag1", "PM25_Lag2", "PM25_Lag3"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Recalculate lags only if the historical export does not contain them.
    df = df.sort_values(["Cyclone", "Time"]).copy()
    for lag in (1, 2, 3):
        col = f"PM25_Lag{lag}"
        calculated = df.groupby("Cyclone")["PM25"].shift(lag)
        df[col] = df[col].where(df[col].notna(), calculated)
    df = add_phase_features(df)
    for col in MODEL_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def prepare_uploaded(raw, cyclone_name):
    df = raw.copy()
    # Normalize column names gently while preserving the original labels for matching.
    df.columns = [str(c).strip() for c in df.columns]
    # Support a few common target/feature aliases.
    aliases = {
        "PM2.5": "PM25", "PM2_5": "PM25", "PM2.5 (µg/m³)": "PM25",
        "AOD_550nm": "AOD", "WindSpeed": "Wind_Speed",
        "Wind Speed": "Wind_Speed", "Surface_Pressure": "Pressure",
        "Total_Precipitation": "Rainfall", "DateTime": "Time", "Timestamp": "Time",
    }
    df = df.rename(columns={k: v for k, v in aliases.items() if k in df.columns and v not in df.columns})
    missing = [c for c in RAW_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(
            "Your uploaded cyclone CSV is missing these required columns: " + ", ".join(missing) +
            ". Required columns are Time, PM25, AOD, Wind_Speed, Pressure and Rainfall."
        )
    df["Time"] = parse_times(df["Time"])
    for col in ["PM25", "AOD", "Wind_Speed", "Pressure", "Rainfall"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Time", "PM25", "AOD", "Wind_Speed", "Pressure", "Rainfall"])
    if df.empty:
        raise ValueError("No usable rows remain after parsing timestamps and numeric measurements.")
    df = df.sort_values("Time").drop_duplicates(subset=["Time"], keep="last").reset_index(drop=True)
    if "Cyclone" not in df.columns or df["Cyclone"].isna().all():
        df["Cyclone"] = cyclone_name
    else:
        df["Cyclone"] = df["Cyclone"].fillna(cyclone_name).astype(str)
    # If user uploaded an engineered dataset with phase labels, retain them.
    if "Cyclone_Phase" not in df.columns:
        n = len(df)
        # Transparent fallback: split the uploaded event timeline into three equal sections.
        phase_idx = np.minimum((np.arange(n) * 3 / max(n, 1)).astype(int) + 1, 3)
        df["Cyclone_Phase"] = [f"Phase {i}" for i in phase_idx]
    else:
        phase = df["Cyclone_Phase"].astype(str).str.strip()
        df["Cyclone_Phase"] = phase.where(phase.isin(["Phase 1", "Phase 2", "Phase 3"]), "Phase 2")
    df = add_time_features(df)
    # Lags are created only from observations in the uploaded file, never future rows.
    for lag in (1, 2, 3):
        col = f"PM25_Lag{lag}"
        if col not in df.columns:
            df[col] = df["PM25"].shift(lag)
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").where(
                pd.to_numeric(df[col], errors="coerce").notna(), df["PM25"].shift(lag)
            )
    df = add_phase_features(df)
    for col in MODEL_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    latest = df.iloc[-1].copy()
    missing_features = latest[MODEL_FEATURES].index[latest[MODEL_FEATURES].isna()].tolist()
    if missing_features:
        raise ValueError(
            "The latest row does not have all model inputs. Missing values in: " +
            ", ".join(missing_features) +
            ". Upload at least four consecutive hourly rows with numeric measurements, including AOD."
        )
    return df, latest


def cpcb_category(value):
    if value <= 30:
        return "Good"
    if value <= 60:
        return "Satisfactory"
    if value <= 90:
        return "Moderate"
    if value <= 120:
        return "Poor"
    if value <= 250:
        return "Very Poor"
    return "Severe"


def train_model(master):
    training = master.dropna(subset=MODEL_FEATURES + ["PM25_Next_Hour"]).copy()
    if training.empty:
        raise ValueError("No complete historical training rows found. Check PM25_Next_Hour and feature columns.")
    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(training[MODEL_FEATURES], training["PM25_Next_Hour"])
    return model, training


st.title("🌪️ Cyclone-Aware PM2.5 Forecasting Dashboard")
st.caption("Leakage-aware next-hour PM2.5 forecasting using the project’s historical cyclone data and Random Forest workflow.")

with st.sidebar:
    st.header("1. Historical training data")
    default_path = Path(__file__).with_name("cyclone_dashboard_master_data.csv")
    if default_path.exists():
        st.success("Bundled historical dataset found.")
        use_uploaded_master = st.checkbox("Use a different historical CSV", value=False)
    else:
        st.warning("Upload the historical master CSV to train the model.")
        use_uploaded_master = True
    master_file = st.file_uploader(
        "Historical CSV (14 cyclones)", type=["csv"],
        help="Expected to contain the historical project features and PM25_Next_Hour target.",
        disabled=not use_uploaded_master,
    )
    st.header("2. New cyclone data")
    new_cyclone_file = st.file_uploader(
        "Upload partial new-cyclone CSV", type=["csv"],
        help="Upload observations only up to the latest hour available. The app predicts the following hour.",
    )
    cyclone_name = st.text_input("Cyclone name", value="New Cyclone")

if use_uploaded_master and master_file is None:
    st.info("Upload your historical master CSV in the sidebar to continue.")
    st.stop()

try:
    if use_uploaded_master:
        master_bytes = master_file.getvalue()
    else:
        master_bytes = default_path.read_bytes()
    historical_raw = load_master_bytes(master_bytes)
    historical = prepare_historical(historical_raw)
except Exception as exc:
    st.error(f"Could not load historical training data: {exc}")
    st.stop()

training_cyclones = sorted(historical["Cyclone"].dropna().astype(str).unique())
info1, info2, info3 = st.columns(3)
info1.metric("Historical cyclones", len(training_cyclones))
info2.metric("Historical rows", f"{len(historical):,}")
info3.metric("Random Forest trees", "300")
with st.expander("Historical training data details"):
    st.write(", ".join(training_cyclones))

if new_cyclone_file is None:
    st.info("Upload a partial new-cyclone CSV in the sidebar to generate a next-hour forecast.")
    st.markdown("**Required columns:** `Time`, `PM25`, `AOD`, `Wind_Speed`, `Pressure`, `Rainfall`. The CSV should contain at least four consecutive observations so the three PM2.5 lag features can be calculated.")
    st.stop()

try:
    uploaded_raw = pd.read_csv(new_cyclone_file)
    event, current = prepare_uploaded(uploaded_raw, cyclone_name.strip() or "New Cyclone")
    model, training = train_model(historical)
    feature_row = pd.DataFrame([[current[c] for c in MODEL_FEATURES]], columns=MODEL_FEATURES)
    prediction = max(0.0, float(model.predict(feature_row)[0]))
except Exception as exc:
    st.error(f"Could not generate the forecast: {exc}")
    st.stop()

current_time = pd.Timestamp(current["Time"])
# The project is hourly; forecast timestamp is the next clock hour.
forecast_time = current_time + pd.Timedelta(hours=1)
actual = None
if "PM25_Next_Hour" in event.columns:
    maybe_actual = pd.to_numeric(pd.Series([current.get("PM25_Next_Hour", np.nan)]), errors="coerce").iloc[0]
    if pd.notna(maybe_actual):
        actual = float(maybe_actual)
absolute_error = abs(actual - prediction) if actual is not None else None

st.success(f"Forecast generated for **{forecast_time:%d %b %Y, %H:%M}** using observations through **{current_time:%d %b %Y, %H:%M}**.")
if len(event) < 4:
    st.warning("Fewer than four observations were uploaded. Lag values may be missing or insufficient for a reliable forecast.")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Current PM2.5", f"{float(current['PM25']):.2f} µg/m³")
c2.metric("Predicted Next Hour", f"{prediction:.2f} µg/m³")
c3.metric("Historical Actual", "N/A" if actual is None else f"{actual:.2f} µg/m³")
c4.metric("Absolute Error", "N/A" if absolute_error is None else f"{absolute_error:.2f} µg/m³")
c5.metric("CPCB Category", cpcb_category(prediction))

st.caption(f"Training used {training['Cyclone'].nunique()} cyclone(s) and {len(training):,} complete rows from the historical dataset. The uploaded new cyclone is forecast using only its latest observed row.")

left, right = st.columns(2)
with left:
    st.subheader("PM2.5 observations and next-hour forecast")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=event["Time"], y=event["PM25"], mode="lines+markers", name="Observed PM2.5"))
    fig.add_trace(go.Scatter(x=[forecast_time], y=[prediction], mode="markers+text", name="Next-hour forecast", text=[f"{prediction:.2f}"], textposition="top center", marker={"size": 12, "symbol": "diamond"}))
    fig.update_layout(height=420, xaxis_title="Time", yaxis_title="PM2.5 (µg/m³)", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
with right:
    st.subheader("Environmental measurements")
    drivers = event[["Time", "AOD", "Wind_Speed", "Pressure", "Rainfall"]].melt("Time", var_name="Driver", value_name="Value")
    fig2 = go.Figure()
    for name in drivers["Driver"].unique():
        series = drivers[drivers["Driver"] == name]
        fig2.add_trace(go.Scatter(x=series["Time"], y=series["Value"], mode="lines", name=name))
    fig2.update_layout(height=420, xaxis_title="Time", yaxis_title="Value (original units)", hovermode="x unified")
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("Cyclone Phase")
phase_counts = event["Cyclone_Phase"].value_counts().reindex(["Phase 1", "Phase 2", "Phase 3"]).fillna(0)
st.bar_chart(phase_counts)

st.subheader("Uploaded cyclone observations")
st.dataframe(event[["Time", "PM25", "AOD", "Wind_Speed", "Pressure", "Rainfall", "Cyclone_Phase"]].tail(20), use_container_width=True)

record = pd.DataFrame([{
    "Cyclone": str(current.get("Cyclone", cyclone_name)),
    "Latest observed time": current_time,
    "Forecast time": forecast_time,
    "Latest observed PM2.5 (µg/m³)": float(current["PM25"]),
    "Predicted next-hour PM2.5 (µg/m³)": prediction,
    "Actual next-hour PM2.5 (µg/m³)": actual,
    "Absolute error (µg/m³)": absolute_error,
    "CPCB category": cpcb_category(prediction),
    "Training cyclones": training["Cyclone"].nunique(),
    "Training rows": len(training),
}])
st.subheader("Forecast record")
st.dataframe(record, use_container_width=True)
st.download_button(
    "⬇️ Download forecast result CSV",
    data=record.to_csv(index=False).encode("utf-8"),
    file_name="cyclone_pm25_next_hour_forecast.csv",
    mime="text/csv",
    use_container_width=True,
)

with st.expander("Model and data assumptions"):
    st.markdown("""
- The Random Forest is retrained from the bundled historical master dataset when the app session runs.
- The uploaded CSV must end at the latest hour the model is allowed to see; later observations must not be included if you want a genuine one-step-ahead test.
- If `Cyclone_Phase` is absent, the app divides the uploaded rows into three approximately equal timeline sections as a fallback. Replace this with your exact project phase rule for final evaluation.
- Missing phase labels in the historical data are treated as Phase 2. Confirm that this is consistent with the source pipeline.
- AOD product and units should match the historical training dataset. Predictions are estimates, not official air-quality measurements.
""")
st.caption("Cyclone-Aware PM2.5 Forecasting | Random Forest (300 trees) | Next-hour prediction")
