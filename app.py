import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Ethereum ARIMA Forecast", layout="wide")

# Sidebar navigation
with st.sidebar:
    st.title("🔧 Settings")
    start_date = st.date_input("Start Date", pd.to_datetime("2020-01-01"))
    end_date = st.date_input("End Date", pd.to_datetime("2024-12-31"))
    st.markdown("Model Order (ARIMA p, d, q)")
    p = st.slider("p (AR)", 0, 5, 1)
    d = st.slider("d (Diff)", 0, 2, 1)
    q = st.slider("q (MA)", 0, 5, 1)
    forecast_days = st.slider("Forecast Days", 7, 90, 30)
    st.markdown("---")
    section = st.radio("📌 Go to Section", ["📈 Forecast", "📊 EDA", "📃 Model Summary"])

# Load data
@st.cache_data
def load_data(start, end):
    df = yf.download("ETH-USD", start=start, end=end)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index = pd.to_datetime(df.index)
    df = df.asfreq("D").fillna(method="ffill")
    return df

eth = load_data(start_date, end_date)
eth["Close_diff"] = eth["Close"].diff()
eth["Rolling_Mean"] = eth["Close"].rolling(window=30).mean()

# ARIMA Fit (on train/test)
train = eth["Close"][:-forecast_days]
test = eth["Close"][-forecast_days:]
model = ARIMA(train, order=(p, d, q)).fit()
forecast = model.forecast(steps=forecast_days)
rmse = np.sqrt(mean_squared_error(test, forecast))
mape = mean_absolute_percentage_error(test, forecast) * 100

# Final model for future forecast
final_model = ARIMA(eth["Close"], order=(p, d, q)).fit()
future_forecast = final_model.get_forecast(steps=forecast_days)
forecast_df = future_forecast.summary_frame()

# Section: EDA
if section == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")

    st.subheader("Raw ETH-USD Data")
    st.dataframe(eth.tail())

    st.subheader("Ethereum Closing Price")
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(eth["Close"], label="Close")
    ax1.set_ylabel("Price (USD)")
    ax1.set_title("ETH Closing Price")
    st.pyplot(fig1)

    st.subheader("30-Day Rolling Mean")
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.plot(eth["Close"], label="Close")
    ax2.plot(eth["Rolling_Mean"], label="30-Day Mean")
    ax2.legend()
    st.pyplot(fig2)

    # ADF Test
    def adf_test(series):
        result = adfuller(series.dropna())
        return result[0], result[1]

    stat_orig, pval_orig = adf_test(eth["Close"])
    stat_diff, pval_diff = adf_test(eth["Close_diff"])

    st.subheader("ADF Stationarity Test")
    st.write(f"**Original Series**: ADF = {stat_orig:.4f}, p-value = {pval_orig:.4f}")
    st.write(f"**Differenced Series**: ADF = {stat_diff:.4f}, p-value = {pval_diff:.4f}")
    st.markdown("✅ If p-value < 0.05, the series is stationary (suitable for ARIMA modeling).")

    st.subheader("ACF & PACF Plots (Differenced Series)")
    fig3, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 4))
    plot_acf(eth["Close_diff"].dropna(), lags=30, ax=ax3)
    plot_pacf(eth["Close_diff"].dropna(), lags=30, ax=ax4)
    st.pyplot(fig3)

# Section: Forecast
elif section == "📈 Forecast":
    st.title("📈 Ethereum Price Forecast")

    st.subheader("Model Evaluation")
    st.write(f"**RMSE**: {rmse:.2f}")
    st.write(f"**MAPE**: {mape:.2f}%")

    st.subheader("Actual vs Forecast")
    fig4, ax5 = plt.subplots(figsize=(12, 5))
    ax5.plot(test.index, test, label="Actual")
    ax5.plot(test.index, forecast, label="Forecast")
    ax5.legend()
    st.pyplot(fig4)

    st.subheader(f"Forecasting Next {forecast_days} Days")
    fig5, ax6 = plt.subplots(figsize=(12, 5))
    ax6.plot(eth["Close"], label="Historical")
    ax6.plot(forecast_df["mean"], label="Forecast", color="green")
    ax6.fill_between(
        forecast_df.index,
        forecast_df["mean_ci_lower"],
        forecast_df["mean_ci_upper"],
        alpha=0.3,
        color="green",
    )
    ax6.set_title("Ethereum Forecast")
    ax6.legend()
    st.pyplot(fig5)

    st.success("✅ Forecast Complete")

    # 📥 Export Forecast Data
    st.subheader("📥 Export Forecast Data")
    export_df = forecast_df.copy()
    export_df.reset_index(inplace=True)
    export_df.rename(columns={"index": "Date"}, inplace=True)
    csv = export_df.to_csv(index=False)
    st.download_button(
        label="Download Forecast as CSV",
        data=csv,
        file_name="ethereum_forecast.csv",
        mime="text/csv",
    )

# Section: Model Summary
elif section == "📃 Model Summary":
    st.title("📃 ARIMA Model Summary")
    st.text(final_model.summary())
