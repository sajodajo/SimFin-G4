import streamlit as st
import datetime
import numpy as np
import arch
import pandas as pd
from pySimFinLIB import pySimFin
import plotly.graph_objects as go
from datetime import datetime
import requests
from PIL import Image
from io import BytesIO


st.set_page_config(
    page_title="Price Prediction Tool",
    page_icon="💸",
    layout = 'wide'
)



st.title('Price Prediction Tool')

psf = pySimFin()

companyDF = psf.getCompanyList()

col1, col2 = st.columns(2)

## STOCK SELECTOR ##
selected_stocks = st.multiselect('Select up to 5 stocks to visualise:', companyDF['name'].sort_values())

tickerList = psf.tickerFind(selected_stocks,companyDF)  

st.markdown(f"""
    <div style='text-align: center;'>
        <h2>Next Day Price Predictions</span></h2>
    </div>
""", unsafe_allow_html=True)

columns = st.columns(len(tickerList))


pricesDFs = {}

for idx, ticker in enumerate(tickerList):
    today = datetime.today().date()
    pricesDF = psf.getStockPrices(ticker, '2020-01-01', today)
    

   # mse, r2, rmse, percentage_rmse, nextDay, df, latest_data, next_day_prediction = psf.linearRegression(pricesDF)
    mse, r2, rmse, percentage_rmse, df  = psf.linearRegression(pricesDF)

    nextDayPrice = df.iloc[-1, -1]
    todayPrice = df.iloc[-2, -1]
    delta = nextDayPrice - todayPrice

    pricesDFs[ticker] = pricesDF
 
    with columns[idx]:

        logo_url = psf.getLogo(ticker)
        response = requests.get(logo_url)
        image = Image.open(BytesIO(response.content))
        max_height = 50
        original_width, original_height = image.size
        aspect_ratio = original_width / original_height
        new_width = int(max_height * aspect_ratio)
        image = image.resize((new_width, max_height))
        col1, col2, col3 = st.columns([1, 4, 1])
        with col2:
            st.image(image)



        if delta > 0:
            
            st.markdown(f"""
                <div style='text-align: center;'>
                    <h2> {ticker} \n\n  <span style='color: green;'>${round(nextDayPrice, 2)}</span></h2>
                </div>
            """, unsafe_allow_html=True)
        elif delta < 0:
            st.markdown(f"""
                <div style='text-align: center;'>
                    <h2> {ticker} \n\n  <span style='color: red;'>${round(nextDayPrice, 2)}</span></h2>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div style='text-align: center;'>
                    <h2> {ticker} \n\n  <span style='color: grey;'>${round(nextDayPrice, 2)}</span></h2>
                </div>
            """, unsafe_allow_html=True)


zipped = list(zip(selected_stocks, tickerList))

with st.expander("Under the Hood 🔧"):
    for idx, ticker in enumerate(tickerList):
        st.markdown(f"<h3 style='text-align: center;'>{ticker}</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<h4 style='text-align: center;'>Prediction Model Comparison</h4>", unsafe_allow_html=True)
            compareModels = psf.multiModelTest(pricesDFs.get(ticker))
            st.dataframe(compareModels)
        with col2:
            st.markdown("<h4 style='text-align: center;'>Prices - Last 3 Days</h4>", unsafe_allow_html=True)
            mse, r2, rmse, percentage_rmse, df  = psf.linearRegression(pricesDFs.get(ticker))
            st.dataframe(df.tail(3))
        st.markdown("<hr style='border: 1px solid #000;'>", unsafe_allow_html=True)
