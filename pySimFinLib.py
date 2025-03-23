import pandas as pd
import requests
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

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

        df_prices = pd.DataFrame(response[0]['data'], columns=response[0]['columns']).drop(columns=['Dividend Paid'])
        return df_prices
    
    def getCompanyList(self):
        url = f"{self.base_url}companies/list"
        response = requests.get(url, headers=self.headers).json()

        raw = pd.DataFrame(response)
        cleaned = raw[~raw['isin'].isna()]

        return cleaned
    

    def plotFinancialIndex(self, ticker, startDate, endDate):
        pricesDF = self.getStockPrices(ticker, startDate, endDate)
        
        # Ensure 'Date' column is in datetime format
        pricesDF['Date'] = pd.to_datetime(pricesDF['Date'])
        
        plt.figure(figsize=(10, 6))
        plt.plot(pricesDF['Date'], pricesDF['Last Closing Price'], label=f'{ticker} Stock Price', color='b', lw=2)
        
        plt.title(f'{ticker} Stock Price from {startDate} to {endDate}', fontsize=14)
        
        # Set x-axis locator for quarters and formatter for the date
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1, interval=3)) 
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m')) 
    
    
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Stock Price (USD)', fontsize=12)
        plt.grid(True)
        plt.legend()
        
        # Rotate the x-ticks to prevent overlap
        plt.xticks(rotation=45)
        
        # Adjust layout for better spacing
        plt.tight_layout()
        plt.show()

def plotMultipleStocks(self,selected_stocks, startDate, endDate):
    plt.figure(figsize=(10, 6))

    for ticker in selected_stocks:
        stock_data = self.getStockPrices(ticker, startDate, endDate)
        stock_data['Date'] = pd.to_datetime(stock_data['Date'], errors='coerce') 
        stock_data = stock_data.dropna(subset=['Date'])
        plt.plot(stock_data['Date'], stock_data['Last Closing Price'], label=ticker)

    startDate = datetime.strptime(startDate, '%Y-%m-%d') 
    endDate = datetime.strptime(endDate, '%Y-%m-%d') 

    plt.title(f"Selected Stock Prices from {startDate.strftime('%B %Y')} to {endDate.strftime('%B %Y')}", fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Stock Price (USD)', fontsize=12)
    plt.legend() 
    plt.grid(True)
    
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))  
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%B %Y')) 

    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()