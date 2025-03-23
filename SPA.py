import streamlit as st
import datetime
import numpy as np
import arch
import pandas as pd
from pySimFinLIB import pySimFin


st.set_page_config(layout = 'wide')

psf = pySimFin()

## SIDEBAR SELECTORS ##
st.sidebar.title("Filter")

companyDF = psf.getCompanyList()

companyName = st.sidebar.selectbox("Select a company:", companyDF['name'].sort_values(),index=94)
ticker = companyDF.loc[companyDF['name'] == companyName, 'ticker'].values[0]

minDate = '2019-04-15'
today1yrAgo = datetime.date.today() - datetime.timedelta(days=1)
try:
    startDate, endDate = st.sidebar.date_input("Select a date range", [minDate, today1yrAgo],min_value=minDate,max_value=today1yrAgo)
except (ValueError,TypeError,NameError):
    pass 

infoDF = psf.getCompanyInfo(ticker)
pricesDF = psf.getStockPrices(ticker,startDate,endDate)

selected_stocks = st.multiselect('Select up to 5 stocks to visualise:', companyDF['name'].sort_values())

psf.plotMultipleStocks(selected_stocks,startDate,endDate)


