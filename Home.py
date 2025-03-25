import streamlit as st
import datetime
import numpy as np
import arch
import pandas as pd
from pySimFinLIB import pySimFin
import plotly.graph_objects as go


st.set_page_config(
    page_title="Stock Price Analysis",
    page_icon='Media/flexiTradeIcon.png',
    layout = 'wide'
)


col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.image("Media/flexiTrade.png", width=800)


st.markdown("""
    <style>
        html, body, [class*="css"]  {
            font-family: 'Segoe UI', sans-serif;
        }
    </style>
    """, unsafe_allow_html=True)

# Custom CSS styling
st.markdown("""
    <style>
    .header {
        background-color: grey;
        color: white;
        padding: 15px;
        text-align: center;
        font-size: 32px;
        font-weight: bold;
    }
    .sub-header {
        color: white;
        font-size: 28px;
        margin-top: 20px;
        font-weight: bold;
    }
    .feature {
        font-size: 18px;
        margin: 10px 0;
    }
    .feature-title {
        font-weight: bold;
        font-size: 20px;
    }
    .emoji {
        font-size: 24px;
        margin-right: 10px;
    }
    </style>
""", unsafe_allow_html=True)


st.markdown('''
<div style="
    text-align: center;
    font-size: 30px;
    max-width: 1200px;
    margin: 0 auto;
">
    <strong>FlexiTrade</strong> is your all-in-one platform for stock market analysis, prediction, and trading strategy optimization. 
    Whether you’re a seasoned trader or just starting your investment journey, we provide powerful tools to enhance your decision-making.
</div>
''', unsafe_allow_html=True)


st.markdown('''
<div style="
    text-align: center;
    font-size: 40px;
    max-width: 1200px;
    margin: 0 auto;
    color: grey; 
">
    <strong><br>Our Tools:</strong> 
</div>
''', unsafe_allow_html=True)

col1, spacer1, col2, spacer2, col3 = st.columns([3,0.4,3,0.4,3])


with col1:
    with st.container():
        st.image('Media/ftAnalyse.png')
        st.markdown('''
        <div style="font-size: 22px; text-align: justify; margin-top: 10px;">
            <strong>FlexiTradeAnalyse</strong> tracks historical stock prices with data going back up to five years, allowing users to analyse long-term trends and patterns. 
            It supports the selection of up to five stocks at a time for side-by-side comparison, making it easy to monitor a personalized set of assets. 
            For each selected stock, the platform provides clear indications of volatility and assesses risk levels, helping users make informed investment decisions based on their risk appetite and market behavior.
        </div>
        ''', unsafe_allow_html=True)

with col2:
    with st.container():
        st.image('Media/ftPredict.png')
        st.markdown('''
        <div style="font-size: 22px; text-align: justify; margin-top: 10px;">
            <strong>FlexiTradePredict</strong> forecasts the next day’s closing price for selected stocks by automatically selecting the best-performing prediction model based on recent performance. 
            It generates the expected price for the upcoming trading day and calculates the price delta from the current day, offering users a clear view of potential short-term movement and helping guide timely trading decisions.
        </div>
        ''', unsafe_allow_html=True)

with col3:
    with st.container():
        st.image('Media/ftPosition.png')
        st.markdown('''
        <div style="font-size: 22px; text-align: justify; margin-top: 10px;">
            <strong>FlexiTradePosition</strong> provides trading recommendations driven by in-depth analysis, allowing users to define their own custom risk levels. 
            It tailors each recommendation to the user’s current portfolio position, ensuring that suggestions are relevant and context-aware. 
            By adapting its strategy to align with user-defined risk preferences, the system delivers personalized guidance that supports smarter, more confident trading decisions.
        </div>
        ''', unsafe_allow_html=True)




st.markdown('''
<div style="
    font-size: 34px; 
    font-weight: bold; 
    color: #0393D0; 
    margin-top: 40px; 
    text-align: center;
">
    <span style="font-size: 28px;">🚀</span> 
    Start exploring today and take your stock trading to the next level with FlexiTrade! 🚀
</div>
''', unsafe_allow_html=True)






