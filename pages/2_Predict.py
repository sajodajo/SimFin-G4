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
    page_title="FlexiTradePredict",
    page_icon='Media/flexiTradeIcon.png',
    layout = 'wide'
)

col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.image("Media/ftPredict.png", width=800)

psf = pySimFin()

companyDF = psf.getCompanyList()

col1, col2 = st.columns(2)
with col1:
    st.markdown(f"""
        <div style='text-align: center;'>
            <h3 style='margin-bottom: 0rem;'><br>Choose up to 5 companies to analyse:</h3>
        </div>
    """, unsafe_allow_html=True)

with col2:
    ## STOCK SELECTOR ##
    selected_stocks = st.multiselect('', companyDF['name'].sort_values())

tickerList = psf.tickerFind(selected_stocks,companyDF)  


st.markdown("""
    <style>
        html, body, [class*="css"]  {
            font-family: 'Segoe UI', sans-serif;
        }
    </style>
    """, unsafe_allow_html=True)

if len(tickerList) > 0:
    st.markdown(f"""
        <div style='text-align: center;'>
            <h1>Next Day Price Predictions<br></span></h2>
        </div>
    """, unsafe_allow_html=True)

try:
    columns = st.columns(len(tickerList))
except:
    pass

pricesDFs = {}
bestModels = {}

for idx, ticker in enumerate(tickerList):
    today = datetime.today().date()

    pricesDF = psf.getStockPrices(ticker, '2020-01-01', today)
    pricesDFs[ticker] = pricesDF  

    compareModels = psf.multiModelTest(pricesDFs.get(ticker))
    bestModels[ticker] = compareModels.index[0]

    mse, r2, rmse, percentage_rmse, df  = psf.runBestModel(pricesDF,bestModels[ticker])

    nextDayPrice = df.iloc[-1, -1]
    todayPrice = df.iloc[-2, -1]
    delta = nextDayPrice - todayPrice


 
    with columns[idx]:
        try:
            logo_url = psf.getLogo(ticker)
            st.empty().markdown(f"""
                <div style="text-align: center;">
                    <img src="{logo_url}" style="max-height: 50px;" />
                </div>
            """, unsafe_allow_html=True)
        except: 
            logo_url = 'https://drive.google.com/file/d/1lR3jAKMkjE5tlRwKRHZsd2le0f_uAwOq/'
            st.empty().markdown(f"""
                <div style="text-align: center;">
                    <img src="{logo_url}" style="max-height: 50px;" />
                </div>
            """, unsafe_allow_html=True)

        if delta > 0:          
            st.markdown(f"""
                <div style='text-align: center;'>
                    <h2><span style='color: grey;'>{ticker} <br><span style='color: black;'>${round(nextDayPrice, 2)}<br><span style='color: green;'>+${round(delta, 2)}</span></h2>
                </div>
            """, unsafe_allow_html=True)
        elif delta < 0:
            st.markdown(f"""
                <div style='text-align: center;'>
                    <h2><span style='color: grey;'>{ticker} <br><span style='color: black;'>${round(nextDayPrice, 2)}<br><span style='color: red;'>-${abs(round(delta, 2))}</span></h2>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div style='text-align: center;'>
                    <h2><span style='color: grey;'>{ticker}<br><span style='color: black;'>${round(nextDayPrice, 2)}<br><span style='color: grey;'>+${round(delta, 2)}</span></h2>
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
            mse, r2, rmse, percentage_rmse, df  = psf.runBestModel(pricesDFs.get(ticker),bestModels[ticker])
            st.dataframe(df.tail(3))
        st.markdown("<hr style='border: 1px solid #000;'>", unsafe_allow_html=True)
