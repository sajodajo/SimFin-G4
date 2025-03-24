# app.py

import streamlit as st
import pandas as pd
from simfin_api import PySimFin
from dotenv import load_dotenv
import os
from pathlib import Path

# Load API key from .env or set directly

dotenv_path = Path("u.env")
load_dotenv(dotenv_path=dotenv_path)

api_token = os.getenv("API_KEY")
print(api_token)

st.set_page_config(page_title="SimFin Trading System", layout="centered")

st.title("SimFin Data Viewer")

# Sidebar: User input
st.sidebar.header("Fetch Data")
#ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2022-12-31"))

option = st.sidebar.radio("Data Type", ["Share Prices", "Financial Statement"])

# Fetch button
if st.sidebar.button("Fetch Data"):
    with st.spinner("Fetching data from SimFin..."):
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        if option == "Share Prices":
            df = api.get_share_prices(ticker, start_str, end_str)
        else:
            df = api.get_financial_statement(ticker, start_str, end_str)

        # Display results
        if not df.empty:
            st.success("Data loaded successfully!")
            st.write(df.head(50))  # show first 50 rows
        else:
            st.error("No data found. Please check the ticker or date range.")

