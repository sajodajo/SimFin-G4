import streamlit as st
import datetime
import numpy as np
import arch
import pandas as pd
from pySimFinLIB import pySimFin
import plotly.graph_objects as go
import modellingLIB


st.set_page_config(
    page_title="FlexiTradeAnalyse",
    page_icon='Media/flexiTradeIcon.png',
    layout = 'wide'
)

col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.image("Media/ftAnalyse.png", width=800)

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


st.markdown("""
    <style>
        html, body, [class*="css"]  {
            font-family: 'Segoe UI', sans-serif;
        }
    </style>
    """, unsafe_allow_html=True)


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
    st.markdown(f'<h3 style="text-align: center;">Stock Price Analysis for {stockNames}</h3>', unsafe_allow_html=True)
    st.markdown(f'<h4 style="text-align: center;">{startDate.strftime("%B %Y")} to {endDate.strftime("%B %Y")}</h4>', unsafe_allow_html=True)   
except:
    pass

st.plotly_chart(stockChart)

try:
    cols = st.columns(len(tickerList))
except:
    pass

for idx, ticker in enumerate(tickerList):
    with cols[idx]:
        today = datetime.datetime.today().date()

        df = psf.getStockPrices(ticker, '2020-01-01', today)
        df = modellingLIB.calcColumns(df)
        returns = df['Log_Return'].dropna()

        # Fit models
        resultARCH, bicARCH, aicARCH, llARCH = modellingLIB.fit_arch_model(returns)
        resultGARCH, bicGARCH, aicGARCH, llGARCH = modellingLIB.fit_garch_model(returns)
        resultGJRGARCH, bicGJRGARCH, aicGJRGARCH, llGJRGARCH = modellingLIB.fit_gjr_garch_model(returns)
        resultEGARCH, bicEGARCH, aicEGARCH, llEGARCH = modellingLIB.fit_egarch_model(returns)
        resultARCHt, bicARCHt, aicARCHt, llARCHt = modellingLIB.fit_arch_t_model(returns)
        resultGARCHt, bicGARCHt, aicGARCHt, llGARCHt = modellingLIB.fit_garch_t_model(returns)
        resultGJRGARCHt, bicGJRGARCHt, aicGJRGARCHt, llGJRGARCHt = modellingLIB.fit_gjr_garch_t_model(returns)
        resultEGARCHt, bicEGARCHt, aicEGARCHt, llEGARCHt = modellingLIB.fit_egarch_t_model(returns)

        # Model comparison
        comparison_df = pd.DataFrame({
            "Model": ["ARCH", "GARCH", "GJR-GARCH", "EGARCH",
                      "ARCH-t", "GARCH-t", "GJR-GARCH-t", "EGARCH-t"],
            "AIC": [aicARCH, aicGARCH, aicGJRGARCH, aicEGARCH,
                    aicARCHt, aicGARCHt, aicGJRGARCHt, aicEGARCHt],
            "BIC": [bicARCH, bicGARCH, bicGJRGARCH, bicEGARCH,
                    bicARCHt, bicGARCHt, bicGJRGARCHt, bicEGARCHt],
            "Log-Likelihood": [llARCH, llGARCH, llGJRGARCH, llEGARCH,
                               llARCHt, llGARCHt, llGJRGARCHt, llEGARCHt]
        }).set_index("Model")

        message, best_model = modellingLIB.modelChoice(comparison_df)

        if best_model == "ARCH":
            bestResult = resultARCH
        elif best_model == "ARCH-t":
            bestResult = resultARCHt
        elif best_model == "GARCH":
            bestResult = resultGARCH
        elif best_model == "GARCH-t":
            bestResult = resultGARCHt
        elif best_model == "GJR-GARCH":
            bestResult = resultGJRGARCH
        elif best_model == "GJR-GARCH-t":
            bestResult = resultGJRGARCHt
        elif best_model == "EGARCH":
            bestResult = resultEGARCH
        elif best_model == "EGARCH-t":
            bestResult = resultEGARCHt
        else:
            raise ValueError(f"Unknown model type: {best_model}")

        # Compute and display VaR
        fig, confidence_levels, VaR_hist, VaR_norm, VaR_t = modellingLIB.VaR(returns, 'N/A')

        # Choose a specific confidence level index to display (e.g., 99%)
        target_level = 0.99
        confidence_index = np.argmin(np.abs(confidence_levels - target_level))

        st.subheader(f"{ticker} Risk Summary")
        st.markdown(f"- **Assuming Typical Market Behavior:** Losses might reach <span style='color:red; font-weight:bold;'>{abs(VaR_norm[confidence_index]/100):.2%}</span>.", unsafe_allow_html=True)
        st.markdown(f"- **Including Rare Crashes:** Extreme events could lead to losses up to <span style='color:red; font-weight:bold;'>{abs(VaR_t[confidence_index]/100):.2%}</span>.", unsafe_allow_html=True)




