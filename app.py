import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

# Hide Streamlit sidebar toggle icon
st.markdown("""
    <style>
        [data-testid="collapsedControl"] {
            display: none;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar navigation
with st.sidebar:
    st.title("🔍 Navigation")
    show_section = st.radio("Go to", ["📈 Forecast", "📊 EDA", "📃 Model Summary"])

# Load Ethereum data
@st.cache_data
def load_data():
    eth = yf.download('ETH-USD', start='2020-01-01', end='2024-12-31')
    eth = eth[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    eth.index = pd.to_datetime(eth.index)
    eth = eth.asfreq('D').fillna(method='ffill')
    return eth

eth = load_data()

# Section 1: EDA
if show_section == "📊 EDA":
    st.header("📊 Exploratory Data Analysis")
    
    st.subheader("Closing Price")
    fig, ax = plt.subplots(figsize=(14, 5))
    eth['Close'].plot(ax=ax, title='Ethereum Closing Price')
    st.pyplot(fig)

    st.subheader("30-Day Rolling Mean")
    eth['Rolling_Mean'] = eth['Close'].rolling(window=30).mean()
    fig, ax = plt.subplots(figsize=(14, 5))
    eth[['Close', 'Rolling_Mean']].plot(ax=ax, title='30-Day Rolling Mean')
    st.pyplot(fig)

    st.subheader("Trading Volume")
    fig, ax = plt.subplots(figsize=(14, 5))
    eth['Volume'].plot(ax=ax, title='Ethereum Trading Volume')
    st.pyplot(fig)

    st.subheader("Basic Statistics")
    st.dataframe(eth.describe())

# Section 2: Forecast
elif show_section == "📈 Forecast":
    st.header("📈 Ethereum Price Forecast (ARIMA)")

    st.subheader("Stationarity Check (ADF Test)")
    def adf_test(series):
        result = adfuller(series.dropna())
        return result[0], result[1]

    adf_stat, p_val = adf_test(eth['Close'])
    st.write(f"ADF Statistic: {adf_stat:.4f}")
    st.write(f"p-value: {p_val:.4f}")
    st.write("✅ Data is stationary" if p_val <= 0.05 else "⚠️ Data is non-stationary")

    st.subheader("ACF and PACF")
    fig1 = plot_acf(eth['Close'].diff().dropna(), lags=30)
    st.pyplot(fig1.figure)
    fig2 = plot_pacf(eth['Close'].diff().dropna(), lags=30)
    st.pyplot(fig2.figure)

    st.subheader("Train/Test Forecast")
    train = eth['Close'][:-30]
    test = eth['Close'][-30:]
    model = ARIMA(train, order=(1, 1, 1))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=30)

    rmse = np.sqrt(mean_squared_error(test, forecast))
    mape = mean_absolute_percentage_error(test, forecast) * 100

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(test.index, test, label='Actual')
    ax.plot(test.index, forecast, label='Forecast', color='green')
    ax.set_title('Actual vs Forecast')
    ax.legend()
    st.pyplot(fig)

    st.write(f"**RMSE:** {rmse:.2f}")
    st.write(f"**MAPE:** {mape:.2f}%")

    st.subheader("📅 Forecast Next 30 Days")
    final_model = ARIMA(eth['Close'], order=(1, 1, 1)).fit()
    forecast_next = final_model.get_forecast(steps=30)
    forecast_df = forecast_next.summary_frame()

    fig, ax = plt.subplots(figsize=(14, 5))
    eth['Close'].plot(ax=ax, label='Historical')
    forecast_df['mean'].plot(ax=ax, label='Forecast', color='green')
    ax.fill_between(forecast_df.index, forecast_df['mean_ci_lower'], forecast_df['mean_ci_upper'], color='green', alpha=0.2)
    ax.legend()
    ax.set_title('Ethereum Forecast for Next 30 Days')
    st.pyplot(fig)

# Section 3: Model Summary
elif show_section == "📃 Model Summary":
    st.header("📃 ARIMA Model Summary")
    model = ARIMA(eth['Close'], order=(1, 1, 1)).fit()
    st.text(model.summary())
