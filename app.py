import streamlit as st
import pandas as pd
import plotly.express as px
from statsmodels.tsa.seasonal import seasonal_decompose

st.set_page_config(page_title="Climate Trend Analyzer", layout="wide")
st.title("🌍 Climate Trend & Anomaly Analyzer")

# Load Data
try:
    df = pd.read_csv('data/climate_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
except FileNotFoundError:
    st.error("Dataset not found! Please run 'python data_gen.py' first.")
    st.stop()

# --- 1. TREND ANALYSIS ---
st.subheader("📈 Long-term Temperature Trend")
df_resampled = df.set_index('Date').resample('MS').mean()
decomposition = seasonal_decompose(df_resampled['AverageTemperature'], model='additive', period=12)

trend_fig = px.line(x=df_resampled.index, y=decomposition.trend, title="Extracted Warming Trend (Smoothed)")
st.plotly_chart(trend_fig, use_container_width=True)

# --- 2. ANOMALY DETECTION ---
st.subheader("🚨 Climate Anomaly Detection")
residuals = decomposition.resid.dropna()
threshold = 3 * residuals.std()
anomalies = residuals[abs(residuals) > threshold]

col1, col2 = st.columns([3, 1])
with col1:
    fig_ano = px.scatter(x=anomalies.index, y=df_resampled.loc[anomalies.index, 'AverageTemperature'], 
                         title="Detected Anomalies (Extreme Weather Events)", color_discrete_sequence=['red'])
    fig_ano.add_trace(px.line(x=df_resampled.index, y=df_resampled['AverageTemperature']).data[0])
    st.plotly_chart(fig_ano, use_container_width=True)

with col2:
    st.write("**Detected Anomaly Dates:**")
    st.write(anomalies.index.date)

st.info("💡 **Insight:** The red dots represent years where the temperature deviated significantly from the expected seasonal cycle and long-term trend.")