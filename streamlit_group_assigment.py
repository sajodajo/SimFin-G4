import streamlit as st
import simfin as sf
from simfin.names import *
import polars as pl
import pandas as pd
import numpy as np
import itertools
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.simplefilter("ignore")

# Set up your SimFin data directory and API key
sf.set_data_dir('~/simfin_data/')
sf.set_api_key(api_key='70d5d920-9f9e-4062-9311-1b4df7c98ba4')

st.title("Stock Analysis and Next-Day Price Prediction")

@st.cache_data
def load_data():
    """
    Loads share prices (daily) and companies data from SimFin, 
    and returns them as pandas DataFrames.
    """
    shareprices_pd = sf.load(dataset='shareprices', variant='daily', market='us')
    companies_pd = sf.load_companies(market='us')
    return shareprices_pd, companies_pd

# Load data once and cache
shareprices_pd, companies_pd = load_data()

# Create a list of all tickers available in the shareprices dataset
all_tickers = sorted(shareprices_pd["Ticker"].unique())

# Let the user pick which ticker to analyze
selected_ticker = st.selectbox("Select a Ticker to Analyze", all_tickers, index=all_tickers.index("AAPL") if "AAPL" in all_tickers else 0)

# When the user clicks "Analyze", we'll run the ARIMA-based prediction
if st.button("Analyze"):
    # Convert shareprices to Polars for the filtering steps (optional, but matches your original code)
    shareprices = pl.from_pandas(shareprices_pd)
    # Make sure the Date column is properly converted to datetime
    shareprices = shareprices.with_columns(pl.col('Date').str.to_datetime('%Y-%m-%d'))

    # Filter to the chosen ticker
    ts_prices = shareprices.filter(pl.col("Ticker") == selected_ticker).select(['Date', 'Close'])

    if ts_prices.height == 0:
        st.warning(f"No data found for ticker {selected_ticker}. Please choose another.")
    else:
        st.write(f"### Raw Share Price Data for {selected_ticker}")
        st.dataframe(ts_prices.to_pandas().head(10))  # Show first 10 rows

        # Convert to pandas for statsmodels usage and set Date as index
        df = ts_prices.to_pandas()
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
        df.sort_index(inplace=True)

        # Show a simple line chart of closing prices
        st.line_chart(df["Close"], height=250)

        # Seasonal Decomposition requires a proper time series index
        # period=12 is just an example; daily data might need something else (e.g., 252 for ~1 year)
        additive_decomposition = seasonal_decompose(df["Close"], model="additive", period=12)
        multiplicative_decomposition = seasonal_decompose(df["Close"], model="multiplicative", period=12)

        # Extract residuals, trend, and seasonal from multiplicative decomposition
        residuals = multiplicative_decomposition.resid.dropna()
        trend = multiplicative_decomposition.trend.dropna()
        seasonal = multiplicative_decomposition.seasonal.dropna()

        # Show decomposition components
        st.write("### Decomposition (Multiplicative)")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("**Trend**")
            st.line_chart(trend)
        with col2:
            st.write("**Seasonal**")
            st.line_chart(seasonal)
        with col3:
            st.write("**Residual**")
            st.line_chart(residuals)

        # Define range of p, d, q for ARMA (d=0) search
        p = range(0, 10)
        d = [0]  # ARMA model assumption (already differenced/stationary)
        q = range(0, 10)

        results_list = []
        for param in itertools.product(p, d, q):
            try:
                model = ARIMA(residuals, order=param)
                model_fit = model.fit()
                results_list.append([param, model_fit.aic, model_fit.bic])
            except:
                pass

        # If no valid models found, stop
        if len(results_list) == 0:
            st.error("No valid ARIMA model could be fit. Try a different ticker or period.")
        else:
            # Convert results to a pandas DataFrame
            results_df = pd.DataFrame(results_list, columns=["order", "AIC", "BIC"])
            # Find the row with the minimum AIC
            best_row = results_df.loc[results_df["AIC"].idxmin()]
            best_order = best_row["order"]

            st.write("### Best ARIMA( p, d, q ) by AIC")
            st.write(f"**Order**: {best_order}, **AIC**: {best_row['AIC']:.2f}, **BIC**: {best_row['BIC']:.2f}")

            # Fit the best ARIMA model
            best_model = ARIMA(residuals, order=best_order)
            arma_results = best_model.fit()

            # Forecast the next residual
            next_residual = arma_results.forecast(steps=1)[0]
            # Use the latest trend and seasonal values
            next_trend = trend.iloc[-1]
            next_seasonal = seasonal.iloc[-1]

            # Compute the next day price prediction (multiplicative)
            next_value = next_residual * next_trend * next_seasonal

            st.write("### Next-Day Price Prediction")
            st.write(f"Predicted next-day close price component: **{next_value:.2f}**")

            st.success(f"Based on ARIMA residual forecasting and the latest trend/seasonality, the next-day price component is {next_value:.2f}.")
