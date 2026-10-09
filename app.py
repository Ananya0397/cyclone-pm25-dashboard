import os
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title='Cyclone-Aware PM2.5 Forecasting', page_icon='🌪️', layout='wide')
MODEL_FEATURES = ['PM25','PM25_Lag1','PM25_Lag2','PM25_Lag3','Hour','DayOfYear','Hour_Sin','Hour_Cos','AOD','Wind_Speed','Pressure','Rainfall','Phase_Phase 1','Phase_Phase 2','Phase_Phase 3']
RAW_REQUIRED = ['Time','PM25','AOD','Wind_Speed','Pressure','Rainfall']

@st.cache_data
def load_master(path):
    df = pd.read_csv(path)
    df['Time'] = pd.to_datetime(df['Time'], errors='coerce', dayfirst=True)
    return df.dropna(subset=['Time','Cyclone']).copy()

def phase_from_times(df):
    # Assign three temporal phases across the uploaded event; use supplied labels if present.
    if 'Cyclone_Phase' not in df.columns or df['Cyclone_Phase'].isna().all():
        duration = (df['Time'].max() - df['Time'].min()).total_seconds()
        if duration <= 0:
            df['Cyclone_Phase'] = 'Phase 2'
        else:
            frac = (df['Time'] - df['Time'].min()).dt.total_seconds() / duration
            df['Cyclone_Phase'] = np.select([frac < 1/3, frac < 2/3], ['Phase 1','Phase 2'], default='Phase 3')
    return df

def engineer_raw(df, cyclone_name):
    df = df.copy()
    missing = [c for c in RAW_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError('Missing required columns: ' + ', '.join(missing))
    df['Time'] = pd.to_datetime(df['Time'], errors='coerce', dayfirst=True)
    for c in ['PM25','AOD','Wind_Speed','Pressure','Rainfall']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['Time']).sort_values('Time').drop_duplicates('Time').reset_index(drop=True)
    df['Cyclone'] = cyclone_name
    df = phase_from_times(df)
    df['Hour'] = df['Time'].dt.hour
    df['DayOfYear'] = df['Time'].dt.dayofyear
    df['Hour_Sin'] = np.sin(2*np.pi*df['Hour']/24)
    df['Hour_Cos'] = np.cos(2*np.pi*df['Hour']/24)
    for lag in [1,2,3]: df[f'PM25_Lag{lag}'] = df['PM25'].shift(lag)
    dummies = pd.get_dummies(df['Cyclone_Phase'].astype(str), prefix='Phase')
    for c in ['Phase_Phase 1','Phase_Phase 2','Phase_Phase 3']:
        df[c] = dummies[c] if c in dummies else 0
    return df

def cpcb_category(x):
    if x <= 30: return 'Good'
    if x <= 60: return 'Satisfactory'
    if x <= 90: return 'Moderate'
    if x <= 120: return 'Poor'
    if x <= 250: return 'Very Poor'
    return 'Severe'

st.title('🌪️ Cyclone-Aware PM2.5 Forecasting Dashboard')
st.caption("Leakage-safe next-hour forecasting using the project's historical cyclone data and Random Forest workflow.")

master_path = Path(__file__).with_name('cyclone_dashboard_master_data.csv')
if not master_path.exists():
    st.error('Historical dataset not found. Keep cyclone_dashboard_master_data.csv in the same folder as app.py.')
    st.stop()
master = load_master(str(master_path))
cyclones = sorted(master['Cyclone'].dropna().astype(str).unique())

with st.sidebar:
    st.header('Input')
    uploaded = st.file_uploader('Upload a cyclone CSV (optional)', type=['csv'], help='Raw CSV columns: Time, PM25, AOD, Wind_Speed, Pressure, Rainfall. A Cyclone column is optional.')
    selected = st.selectbox('Cyclone', cyclones + ['New Cyclone'])
    custom_name = st.text_input('New cyclone name', 'New Cyclone') if selected == 'New Cyclone' else selected

st.markdown('')
c1, c2, c3 = st.columns(3)
c1.metric('Historical cyclones', master['Cyclone'].nunique())
c2.metric('Historical rows', f'{len(master):,}')
c3.metric('Random Forest trees', '300')
with st.expander('Historical training data details'):
    st.write('Cyclones used:', ', '.join(cyclones))
    st.write('Training target: `PM25_Next_Hour`; model features: 15 project features.')

if uploaded is None:
    st.info('Upload a partial new-cyclone CSV in the sidebar to generate a next-hour forecast.')
    st.markdown('**Required columns:** `Time`, `PM25`, `AOD`, `Wind_Speed`, `Pressure`, `Rainfall`. At least four consecutive observations are needed to calculate three PM2.5 lag features.')
    st.stop()

try:
    raw = pd.read_csv(uploaded)
    if 'Cyclone' in raw.columns and selected == 'New Cyclone':
        names = raw['Cyclone'].dropna().astype(str).unique()
        if len(names): custom_name = names[0]
    event = engineer_raw(raw, custom_name)
except Exception as e:
    st.error(f'Could not prepare uploaded CSV: {e}')
    st.stop()

valid = event.dropna(subset=MODEL_FEATURES).copy()
if valid.empty:
    st.warning('At least four valid consecutive PM2.5 observations are needed. Check timestamps and PM25 values.')
    st.stop()

# If the chosen cyclone is a known historical event, exclude it to preserve leave-one-cyclone-out validation.
if custom_name in cyclones:
    train = master[master['Cyclone'].astype(str) != custom_name].copy()
    train_note = f'{custom_name} is excluded from training.'
else:
    train = master.copy()
    train_note = f'New event {custom_name} is not in the historical training set; all 14 historical cyclones are used for training.'
train = train.dropna(subset=MODEL_FEATURES + ['PM25_Next_Hour'])
if train.empty:
    st.error('No usable historical training rows were found.')
    st.stop()

valid_times = valid['Time'].sort_values().tolist()
current_time = st.selectbox('Select current observation time', valid_times, index=len(valid_times)-1, format_func=lambda x: pd.Timestamp(x).strftime('%Y-%m-%d %H:%M'))
if st.button('🚀 RUN NEXT-HOUR FORECAST', type='primary', use_container_width=True):
    try:
        model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
        model.fit(train[MODEL_FEATURES], train['PM25_Next_Hour'])
        current = valid.loc[valid['Time'] == current_time].iloc[-1]
        xrow = pd.DataFrame([current[MODEL_FEATURES].astype(float).values], columns=MODEL_FEATURES)
        pred = max(0.0, float(model.predict(xrow)[0]))
        future_row = event.loc[event['Time'] == (pd.Timestamp(current_time) + pd.Timedelta(hours=1))]
        actual = float(future_row['PM25'].iloc[0]) if not future_row.empty and pd.notna(future_row['PM25'].iloc[0]) else None
        error = abs(actual-pred) if actual is not None else None
        category = cpcb_category(pred)
        a,b,c,d,e = st.columns(5)
        a.metric('Current PM2.5', f"{float(current['PM25']):.2f} µg/m³")
        b.metric('Predicted Next Hour', f'{pred:.2f} µg/m³')
        c.metric('Historical Actual', 'N/A' if actual is None else f'{actual:.2f} µg/m³')
        d.metric('Absolute Error', 'N/A' if error is None else f'{error:.2f} µg/m³')
        e.metric('CPCB Category', category)
        st.info(f'**Leakage check:** {train_note} Random Forest uses 300 trees and 15 project features. Forecast time: **{(pd.Timestamp(current_time)+pd.Timedelta(hours=1)).strftime("%Y-%m-%d %H:%M")}**. Phase labels for the uploaded event are used if supplied; otherwise, phases are divided across the event duration.')
        event_chart = event.copy()
        event_chart['Predicted Next-Hour PM2.5'] = np.nan
        ok = event_chart.dropna(subset=MODEL_FEATURES)
        if len(ok): event_chart.loc[ok.index, 'Predicted Next-Hour PM2.5'] = np.maximum(model.predict(ok[MODEL_FEATURES]), 0)
        left,right = st.columns(2)
        with left:
            st.subheader('Actual vs Predicted PM2.5')
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=event_chart['Time'], y=event_chart['PM25'], mode='lines+markers', name='Observed PM2.5'))
            fig.add_trace(go.Scatter(x=event_chart['Time'], y=event_chart['Predicted Next-Hour PM2.5'], mode='lines+markers', name='Model forecast'))
            fig.update_layout(height=420, xaxis_title='Time', yaxis_title='PM2.5 (µg/m³)', hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)
        with right:
            st.subheader('Environmental Drivers')
            drivers = event_chart[['Time','AOD','Wind_Speed','Pressure','Rainfall']].melt('Time', var_name='Driver', value_name='Value')
            fig2 = go.Figure()
            for name in drivers['Driver'].unique():
                subset = drivers[drivers['Driver']==name]
                fig2.add_trace(go.Scatter(x=subset['Time'], y=subset['Value'], mode='lines', name=name))
            fig2.update_layout(height=420, xaxis_title='Time', yaxis_title='Value', hovermode='x unified')
            st.plotly_chart(fig2, use_container_width=True)
        st.subheader('Cyclone Phase')
        phase_counts = event['Cyclone_Phase'].value_counts().reindex(['Phase 1','Phase 2','Phase 3']).fillna(0)
        st.bar_chart(phase_counts)
        st.subheader('Forecast Record')
        record = pd.DataFrame([{'Cyclone':custom_name,'Current Time':pd.Timestamp(current_time),'Forecast Time':pd.Timestamp(current_time)+pd.Timedelta(hours=1),'Current PM2.5':float(current['PM25']),'Predicted PM2.5':pred,'Historical Actual PM2.5':actual,'Absolute Error':error,'CPCB Category':category,'Training Cyclones':train['Cyclone'].nunique()}])
        st.dataframe(record, use_container_width=True)
        st.download_button('⬇️ Download Forecast Result', record.to_csv(index=False).encode('utf-8'), 'forecast_result.csv', 'text/csv')
    except Exception as e:
        st.error(f'Forecast failed: {e}')

st.divider()
st.caption('The next-hour forecast uses the latest selected observation and preceding PM2.5 lag values. Actual next-hour PM2.5 is shown only if that timestamp exists in the uploaded CSV.')
