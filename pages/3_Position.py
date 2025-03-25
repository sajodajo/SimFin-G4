import streamlit as st
import datetime
import numpy as np
import arch
import pandas as pd
from pySimFinLIB import pySimFin
import plotly.graph_objects as go
from datetime import datetime

psf = pySimFin()

st.set_page_config(
    page_title="FlexiTradePosition",
    page_icon='Media/flexiTradeIcon.png',
    layout = 'wide'
)

col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.image("Media/ftPosition.png", width=800)


st.markdown("""
    <style>
        html, body, [class*="css"]  {
            font-family: 'Segoe UI', sans-serif;
        }
    </style>
    """, unsafe_allow_html=True)


try:
    companyDF = psf.getCompanyList()
except:
    st.rerun()




col1, spacer1, col2, spacer2, col3 = st.columns([3, 0.3, 3, 0.3, 3])

with col1:
    st.write("#### Cash Available ($)")
    cash_available = st.slider(
        "", 
        min_value=0, 
        max_value=1000000, 
        value=1000, 
        step=100
    )

with col2:
    st.write("#### Desired Risk Level (%)")
    risk_percent = st.slider(
        "", 
        min_value=0, 
        max_value=100, 
        value=50, 
        step=5
    )
    risk_level = risk_percent / 100

with col3:
    st.write("#### Stocks to Include")
    selected_stocks = st.multiselect('', companyDF['name'].sort_values())

    tickerList = psf.tickerFind(selected_stocks,companyDF)  


pricesDFs = {}
todayPricesDict = {}
bestModels = {}
predictions = {}
current_holdings = {}

if len(tickerList) > 0:
    st.write('### Current Holdings')

try:
    cols = st.columns(len(tickerList))
except:
    pass

for idx, ticker in enumerate(tickerList):
    with cols[idx]:
        st.write(f"##### How much {ticker} stock do you currently hold?")

        shares = st.text_input("", key=f"holdings_{ticker}")
        try:
            current_holdings[ticker] = int(shares)
        except:
            current_holdings[ticker] = 0

        today = datetime.today().date()

        if shares != '':
            pricesDF = psf.getStockPrices(ticker, '2020-01-01', today)
            pricesDFs[ticker] = pricesDF  

            compareModels = psf.multiModelTest(pricesDFs.get(ticker))
            bestModels[ticker] = compareModels.index[0]

            mse, r2, rmse, percentage_rmse, df  = psf.runBestModel(pricesDF,bestModels[ticker])

            nextDayPrice = df.iloc[-1, -1]
            todayPrice = df.iloc[-2, -1]
            delta = nextDayPrice - todayPrice

            if delta > 0:
                predictions[ticker] = "UP"
            elif delta < 0:
                predictions[ticker] = "DOWN"
            else:
                predictions[ticker] = "FLAT"

            todayPricesDict[ticker] = todayPrice

try:
    signals = psf.generate_trade_signals(predictions, current_holdings, cash_available, todayPricesDict, risk_level)
    st.dataframe(signals)
except:
    pass