import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Ethereum Forecasting App", layout="wide")

# Main App Title
st.title("🚀 Ethereum Forecast App")
st.markdown("---")

# Sidebar
with st.sidebar:
    model_choice = st.selectbox("Select Forecasting Model", ["ARIMA", "Prophet", "LSTM"])
    st.title(f"{model_choice} Forecast Settings")
    start_date = st.date_input("Start Date", pd.to_datetime("2020-01-01"))
    end_date = st.date_input("End Date", pd.to_datetime("2024-12-31"))
    forecast_days = st.slider("Forecast Days", 7, 180, 30)
    section = st.radio("📄 Go to Section", ["Forecast", "EDA", "Model Summary"])

@st.cache_data
def load_data(start, end):
    df = yf.download("ETH-USD", start=start, end=end)
    df = df[["Close"]].dropna()
    df.index = pd.to_datetime(df.index)
    df = df.asfreq("D").fillna(method="ffill")
    return df

data = load_data(start_date, end_date)

# Data validation
if len(data) < 50:
    st.error(f"Not enough data points ({len(data)}). Please select a longer date range. Minimum 50 data points required.")
    st.stop()

if forecast_days >= len(data):
    st.error(f"Forecast days ({forecast_days}) cannot be greater than or equal to available data points ({len(data)}). Please reduce forecast days.")
    st.stop()

data["Close_diff"] = data["Close"].diff()
data["Rolling_Mean"] = data["Close"].rolling(window=30).mean()

def plot_series(title, series_dict):
    fig, ax = plt.subplots(figsize=(12, 5))
    for label, series in series_dict.items():
        ax.plot(series, label=label)
    ax.set_title(title)
    ax.legend()
    st.pyplot(fig)

def export_forecast(df):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "🔗 Download Forecast as CSV",
        csv,
        "forecast.csv",
        "text/csv",
        key="download-forecast"
    )

# ARIMA
if model_choice == "ARIMA":
    p, d, q = 1, 1, 1
    
    # Ensure we have enough data for train/test split
    min_train_size = max(50, forecast_days + 30)  # At least 50 points or forecast_days + 30
    
    if len(data) < min_train_size:
        st.error(f"Not enough data for ARIMA model. Need at least {min_train_size} data points, but only have {len(data)}.")
        st.stop()
    
    train = data["Close"][:-forecast_days]
    test = data["Close"][-forecast_days:]
    
    # Additional validation
    if len(train) < 30:
        st.error(f"Training data too small ({len(train)} points). Please reduce forecast days or increase date range.")
        st.stop()
    
    try:
        model = ARIMA(train, order=(p, d, q)).fit()
        forecast = model.forecast(steps=forecast_days)
        rmse = np.sqrt(mean_squared_error(test, forecast))
        mape = mean_absolute_percentage_error(test, forecast) * 100
        final_model = ARIMA(data["Close"], order=(p, d, q)).fit()
        future = final_model.get_forecast(steps=forecast_days)
        forecast_df = future.summary_frame().reset_index()
        forecast_df.rename(columns={"index": "Date", "mean": "Forecast"}, inplace=True)
    except Exception as e:
        st.error(f"ARIMA model failed to fit. Try adjusting the date range or forecast days. Error: {str(e)}")
        st.stop()

    if section == "Forecast":
        st.title("ARIMA Forecast")
        st.write(f"RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
        plot_series("Actual vs Forecast", {"Actual": test, "Forecast": forecast})
        plot_series("Ethereum Price Forecast", {
            "Historical": data["Close"],
            "Forecast": forecast_df.set_index("Date")["Forecast"]
        })
        export_forecast(forecast_df)
    elif section == "EDA":
        st.title("ARIMA - Exploratory Data Analysis")
        st.dataframe(data.tail())
        plot_series("ETH Price with Rolling Mean", {
            "Close": data["Close"],
            "30-Day Mean": data["Rolling_Mean"]
        })
        adf_orig = adfuller(data["Close"].dropna())
        adf_diff = adfuller(data["Close_diff"].dropna())
        st.write(f"ADF Original: p={adf_orig[1]:.4f}")
        st.write(f"ADF Differenced: p={adf_diff[1]:.4f}")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
        plot_acf(data["Close_diff"].dropna(), lags=30, ax=ax1)
        plot_pacf(data["Close_diff"].dropna(), lags=30, ax=ax2)
        st.pyplot(fig)
    elif section == "Model Summary":
        st.title("ARIMA Summary")
        st.text(final_model.summary())

# Prophet
elif model_choice == "Prophet":
    df = data.reset_index()[["Date", "Close"]]  # Ensure both columns are selected
    df.columns = ["ds", "y"]  # Now safe to rename
    df["ds"] = pd.to_datetime(df["ds"])
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df.dropna(inplace=True)

    model = Prophet(daily_seasonality=True)
    model.fit(df)
    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)  # Keep original forecast with Prophet column names
    
    # Create a copy for export with renamed columns
    forecast_df_export = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    forecast_df_export.rename(columns={"ds": "Date", "yhat": "Forecast"}, inplace=True)

    if section == "Forecast":
        st.title("Prophet Forecast")
        fig1 = model.plot(forecast)  # Use original forecast with Prophet column names
        st.pyplot(fig1)
        export_forecast(forecast_df_export.tail(forecast_days))
    elif section == "EDA":
        st.title("Prophet - EDA")
        st.dataframe(data.tail())
        plot_series("Close vs Rolling Mean", {
            "Close": data["Close"],
            "Rolling Mean": data["Rolling_Mean"]
        })
    elif section == "Model Summary":
        st.title("Prophet - Components")
        fig2 = model.plot_components(forecast)  # Use original forecast
        st.pyplot(fig2)

# LSTM
elif model_choice == "LSTM":
    # LSTM needs at least 60 data points for the lookback window
    if len(data) < 120:  # 60 for lookback + some for training
        st.error(f"Not enough data for LSTM model. Need at least 120 data points, but only have {len(data)}.")
        st.stop()
    
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data[["Close"]])
    X, y = [], []
    for i in range(60, len(scaled)):
        X.append(scaled[i - 60:i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    
    if len(X) == 0:
        st.error("Not enough data to create training sequences for LSTM. Please increase the date range.")
        st.stop()
    
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)

    input_seq = scaled[-60:].reshape(1, 60, 1)
    future_preds = []
    for _ in range(forecast_days):
        pred = model.predict(input_seq)[0, 0]
        future_preds.append(pred)
        input_seq = np.append(input_seq[:, 1:, :], [[[pred]]], axis=1)

    forecast_values = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1)).flatten()
    future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
    forecast_df = pd.DataFrame({"Date": future_dates, "Forecast": forecast_values})

    if section == "Forecast":
        st.title("LSTM Forecast")
        plot_series("Ethereum Forecast", {
            "Historical": data["Close"],
            "Forecast": forecast_df.set_index("Date")["Forecast"]
        })
        export_forecast(forecast_df)
    elif section == "EDA":
        st.title("LSTM - EDA")
        st.dataframe(data.tail())
        plot_series("Close & Rolling Mean", {
            "Close": data["Close"],
            "Rolling Mean": data["Rolling_Mean"]
        })
    elif section == "Model Summary":
        st.title("LSTM Summary")
        st.text("LSTM with 2 layers, trained on last 60 days windows.")
