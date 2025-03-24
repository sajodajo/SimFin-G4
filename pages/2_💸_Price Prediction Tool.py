import streamlit as st
import datetime
import numpy as np
import arch
import pandas as pd
from pySimFinLIB import pySimFin
import plotly.graph_objects as go

st.set_page_config(
    page_title="Price Prediction Tool",
    page_icon="💸",
    layout = 'wide'
)

st.title('Price Prediction Tool')


psf = pySimFin()

ticker = 'GOOG'
startDate = '2023-01-01'
endDate = '2025-03-24'

pricesDF = psf.getStockPrices(ticker,startDate,endDate)

model, mae, mse, r2 = psf.train_linear_model(pricesDF)

fig, newDF = psf.run_forecasts_and_plot(pricesDF)

st.write(fig)

forecastDF, model = psf.run_one_step_forecast_new_df(pricesDF)

st.dataframe(forecastDF[-3:])