import streamlit as st
import datetime
import numpy as np
import arch
import pandas as pd
from pySimFinLIB import pySimFin
import plotly.graph_objects as go

st.set_page_config(
    page_title="Stock Price Analysis",
    page_icon="📈",
    layout = 'wide'
)

st.markdown(
    """
    <style>
    .streamlit-expanderHeader {
        padding: 20px;
    }
    .stColumn {
        padding: 40px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.title('Stock Price Analysis')

psf = pySimFin()

companyDF = psf.getCompanyList()

col1, col2 = st.columns(2)

with col1:
    ## TIMEFRAME SELECTOR ##

    minDate = datetime.date.today() - datetime.timedelta(days=1800)
    maxDate = datetime.date.today() - datetime.timedelta(days=1)

    startDate, endDate = st.slider(
        "Select Date Range",
        min_value=minDate,
        max_value=maxDate,
        value=(minDate, maxDate), 
        format="YYYY-MM-DD"
    )

with col2:
## STOCK SELECTOR ##
    selected_stocks = st.multiselect('Select up to 5 stocks to visualise:', companyDF['name'].sort_values())

tickerList = psf.tickerFind(selected_stocks,companyDF)

stocksDF = psf.selectMultipleStocks(tickerList, startDate, endDate)

stockChart = psf.plotlyMultipleStocks(stocksDF)

if len(selected_stocks)==0:
    pass
elif len(selected_stocks)==1:
    stockNames = selected_stocks[0]
else:
    stockNames = ', '.join([stock.title() for stock in selected_stocks[0:-1]]) + " & " + selected_stocks[-1].title()

try:
    st.write(f'### Stock Price Analysis for {stockNames} from {startDate.strftime("%B %Y")} to {endDate.strftime("%B %Y")}')
    st.plotly_chart(stockChart)
except:
    pass



