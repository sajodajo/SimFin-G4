import pandas as pd
import requests
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import plotly.graph_objects as go
import simfin as sf
from simfin.names import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import xgboost as xgb
import statsmodels.api as sm



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
    

    def selectSingleStock(self, selected_stock, startDate, endDate):
        stockData = self.getStockPrices(selected_stock, startDate, endDate)
        stockDataPrice = stockData['Last Closing Price']
        
        return pd.Series(stockDataPrice)
        

    def selectMultipleStocks(self,selected_stocks, startDate, endDate):
        selectedStocks = {}

        for ticker in selected_stocks:
            stockData = self.getStockPrices(ticker, startDate, endDate)
            StockDataPrice = stockData['Last Closing Price']
            
            selectedStocks[ticker] = StockDataPrice

        return pd.DataFrame(selectedStocks)

    

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
                font=dict(size=24, color='black'),  # Customize the font size and color of the legend
                bgcolor='white',  # Background color of the legend
                bordercolor='white',  # Border color
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
    

    def getFinancialStatements(self,ticker,type,startDate,endDate):
        url = self.base_url + f"companies/statements/compact?ticker={ticker}&statements={type}&period=&start={startDate}&end={endDate}"

        response = requests.get(url, headers=self.headers).json()

        columns = response[0]['statements'][0]['columns']
        data = response[0]['statements'][0]['data']

        return pd.DataFrame(data,columns=columns)


    def multiModelTest(self,df):
            df['Rolling Mean Closing Price'] = df['Last Closing Price'].rolling(window=30).mean()
            df['Next Day Closing Price'] = df['Last Closing Price'].shift(-1)

            df = df.dropna()
            
            X = df[['Common Shares Outstanding', 'Last Closing Price', 'Adjusted Closing Price', 
                    'Highest Price', 'Lowest Price', 'Opening Price', 'Trading Volume','Rolling Mean Closing Price']]
            y = df['Next Day Closing Price']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            lr_model = LinearRegression()
            rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
            xgb_model = XGBRegressor(
                n_estimators=1000, 
                learning_rate=0.01, 
                max_depth=5, 
                min_child_weight=10, 
                gamma=0.1, 
                subsample=0.8, 
                colsample_bytree=0.8, 
                random_state=42,
                early_stopping_rounds=50
            )


            lr_model.fit(X_train, y_train)
            rf_model.fit(X_train, y_train)
            xgb_model.fit(
                X_train, y_train, 
                eval_set=[(X_test, y_test)], 
                verbose=False
        )

            models = {
                'Linear Regression': lr_model,
                'Random Forest': rf_model,
                'XGBoost': xgb_model
            }

            results = {}
            for name, model in models.items():
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
                
                train_mae = mean_absolute_error(y_train, y_train_pred)
                test_mae = mean_absolute_error(y_test, y_test_pred)
                
                train_mse = mean_squared_error(y_train, y_train_pred)
                test_mse = mean_squared_error(y_test, y_test_pred)
                
                train_r2 = r2_score(y_train, y_train_pred)
                test_r2 = r2_score(y_test, y_test_pred)
                
                results[name] = {
                    'Train MAE': train_mae, 'Test MAE': test_mae,
                    'Train MSE': train_mse, 'Test MSE': test_mse,
                    'Train R²': train_r2, 'Test R²': test_r2
                }

            results_df = pd.DataFrame(results).T.sort_values(by='Test MAE')

            return results_df
    
    def runBestModel(self,df,model_type):
        df['Rolling Mean Closing Price'] = df['Last Closing Price'].rolling(window=30).mean()
        df['Next Day Closing Price'] = df['Last Closing Price'].shift(-1)

        df = df.dropna()
        
        X = df[['Common Shares Outstanding', 'Last Closing Price', 'Adjusted Closing Price', 
                'Highest Price', 'Lowest Price', 'Opening Price', 'Trading Volume','Rolling Mean Closing Price']]
        y = df['Next Day Closing Price']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if model_type == 'Linear Regression':
            model = LinearRegression()
        elif model_type == 'XGBoost':
            model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
        elif model_type == 'Random Forest':
            model = RandomForestRegressor(random_state=42)
        else:
            raise ValueError("Invalid model_type. Choose from 'linear_regression', 'xgboost', or 'random_forest'.")
        
        # Fit model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = model.score(X_test, y_test)
        rmse = np.sqrt(mse)
        percentage_rmse = (rmse / y_test.mean()) * 100
        
        return mse, r2, rmse, percentage_rmse, df
    
    def getLogo(self,ticker):
        api_url = f'https://api.api-ninjas.com/v1/logo?ticker={ticker}'
        response = requests.get(api_url, headers={'X-Api-Key': 'TOPXVVT0OqevUxpYwXiMPA==d6gb3gYVVrMK6Pne'})
        if response.status_code == requests.codes.ok:
            return response.json()[0]['image']
        
    def generate_trade_signals(self,predictions, current_holdings, cash_available, prices, risk_level):
        decisions = {}
        max_invest_per_stock = risk_level * cash_available / len(predictions)
        
        for stock, prediction in predictions.items():
            price = prices[stock]
            if prediction == "UP":
                budget = max_invest_per_stock
                shares_to_buy = int(budget / price)
                if shares_to_buy > 0:
                    decisions[stock] = {"action": "BUY", "shares": shares_to_buy}
                else:
                    decisions[stock] = {"action": "HOLD"}
            
            elif prediction == "DOWN":
                if current_holdings.get(stock, 0) > 0:
                    shares_to_sell = int(current_holdings[stock] * risk_level)
                    decisions[stock] = {"action": "SELL", "shares": shares_to_sell}
                else:
                    decisions[stock] = {"action": "HOLD"}
            
            else:
                decisions[stock] = {"action": "HOLD"}
        
        return decisions
        
    

    