import pandas as pd
import requests
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import plotly.graph_objects as go


class pySimFin:

    def __init__(self):
        load_dotenv()
        self.API_KEY = os.getenv('API_KEY')
        
        self.headers = {
            "accept": "application/json",
            "Authorization": self.API_KEY
        }
        
        self.base_url = "https://backend.simfin.com/api/v3/"

    def getCompanyInfo(self, ticker):
        url = f"{self.base_url}companies/general/compact?ticker={ticker}"
        response = requests.get(url, headers=self.headers).json()

        return pd.DataFrame(response['data'], columns=response['columns'])
    
    def getStockPrices(self, ticker, start_date, end_date):
        url = f"{self.base_url}companies/prices/compact?ticker={ticker}&start={start_date}&end={end_date}"
        response = requests.get(url, headers=self.headers).json()

        if response and len(response) > 0:
            df_prices = pd.DataFrame(response[0]['data'], columns=response[0]['columns']).drop(columns=['Dividend Paid'])
            df_prices['Date'] = pd.to_datetime(df_prices['Date'])
            df_prices.set_index('Date', inplace=True)
            df_prices.index = pd.to_datetime(df_prices.index)
            return df_prices
        else:
            print("No data received for the stock price.")
            df_prices = pd.DataFrame() 
            return df_prices
    
    def getCompanyList(self):
        url = f"{self.base_url}companies/list"
        response = requests.get(url, headers=self.headers).json()

        raw = pd.DataFrame(response)
        cleaned = raw[~raw['isin'].isna()]

        return cleaned
    

    # def plotFinancialIndex(self, ticker, startDate, endDate):
    #     pricesDF = self.getStockPrices(ticker, startDate, endDate)
        
    #     pricesDF['Date'] = pd.to_datetime(pricesDF['Date'])
        
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(pricesDF['Date'], pricesDF['Last Closing Price'], label=f'{ticker} Stock Price', color='b', lw=2)
        
    #     plt.title(f'{ticker} Stock Price from {startDate} to {endDate}', fontsize=14)
        
    #     plt.gca().xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1, interval=3)) 
    #     plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m')) 
    
    #     plt.xlabel('Date', fontsize=12)
    #     plt.ylabel('Stock Price (USD)', fontsize=12)
    #     plt.grid(True)
    #     plt.legend()
        
    #     plt.xticks(rotation=45)

    #     plt.tight_layout()
    #     plt.show()

    def selectMultipleStocks(self,selected_stocks, startDate, endDate):
        selectedStocks = {}

        for ticker in selected_stocks:
            stockData = self.getStockPrices(ticker, startDate, endDate)
            StockDataPrice = stockData['Last Closing Price']
            
            selectedStocks[ticker] = StockDataPrice

        return pd.DataFrame(selectedStocks)
    
    
    # def plotMultipleStocks(self,df):
    #     fig = plt.figure(figsize=(10, 6))

    #     for column in df.columns:
    #         plt.plot(df.index, df[column], label=column)

    #     plt.title('Selected Stock Prices Over Time', fontsize=14)
    #     plt.xlabel('Date', fontsize=12)
    #     plt.ylabel('Price (USD)', fontsize=12)
    #     plt.legend()
    #     plt.grid(True)

    #     plt.xticks(rotation=45)
    #     plt.tight_layout()
    #     plt.show()

    #     return fig
    

    def plotlyMultipleStocks(self, df):
        # Create a Plotly figure
        fig = go.Figure()

        # Add traces for each column in the DataFrame
        for column in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[column], mode='lines', name=column))

        # Customize the layout
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            xaxis_tickangle=-45,  # Rotate x-axis labels
            template='plotly',  # Use a clean, default plotly template
            showlegend=True,  # Show legend
            legend=dict(
                font=dict(size=24, color='white'),  # Customize the font size and color of the legend
                bgcolor='black',  # Background color of the legend
                bordercolor='black',  # Border color
                borderwidth=2  # Border width of the legend
            ),
            height=400
        )

        # Display the plot in Streamlit
        return fig

    def tickerFind(self,nameList,companyDF):
        tickerList = []
        for name in nameList:
            for index, company in companyDF.iterrows():
                if name == company['name']:
                    tickerList.append(company['ticker'])
        return tickerList