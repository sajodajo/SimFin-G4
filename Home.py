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
            font-family: 'Helvetica', sans-serif;
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


# Introduction text with sub-headings and emojis
st.write('''
StockHub is your all-in-one platform for advanced stock market analysis, prediction, and strategy optimization. Whether you’re a seasoned trader or just starting your investment journey, we provide powerful tools to enhance your decision-making.
''')

# Features Section with Emojis and Styled Headers
st.markdown('<div class="sub-header"><span class="emoji">📈</span> Explore Our Features:</div>', unsafe_allow_html=True)

# Historical Stock Price Analyzer
st.markdown('<div class="feature-title"><span class="emoji">📊</span> Historical Stock Price Analyzer</div>', unsafe_allow_html=True)
st.write('''
Dive deep into the past performance of over 5,000 companies. Analyze the stock prices of up to five companies at a time and uncover valuable insights to help guide your investment choices.
''')

# Machine Learning-based Stock Prediction Tool
st.markdown('<div class="feature-title"><span class="emoji">🤖</span> Machine Learning-based Stock Prediction Tool</div>', unsafe_allow_html=True)
st.write('''
Harness the power of AI to forecast stock prices for the next day. Our cutting-edge machine learning model provides accurate predictions, giving you an edge in the fast-paced world of stock trading.
''')

# Trading Strategy Tool
st.markdown('<div class="feature-title"><span class="emoji">💡</span> Trading Strategy Tool</div>', unsafe_allow_html=True)
st.write('''
Looking for smart trading advice? Our algorithm-driven strategy tool generates personalized trading recommendations, helping you execute trades with greater confidence and precision.
''')

# About the Team
st.markdown('<div class="feature-title"><span class="emoji">👥</span> About the Team</div>', unsafe_allow_html=True)
st.write('''
Learn more about the brilliant minds behind Stock Insights Hub. Get to know our team of experts who are dedicated to providing you with the most advanced tools to stay ahead in the stock market.
''')

# Closing Call to Action
st.markdown('<div class="sub-header"><span class="emoji">🚀</span> Start exploring today and take your stock trading to the next level with Stock Insights Hub!</div>', unsafe_allow_html=True)











