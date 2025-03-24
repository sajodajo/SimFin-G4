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
    
    def load_income_data(self, dataset: str='income', variant: str='annual', market: str='us') -> pd.DataFrame:

        df = sf.load(dataset=dataset, variant=variant, market=market)
        return df
    

    def getStatementsData(self,ticker,type,startDate,endDate):
        url = self.base_url + f"companies/statements/compact?ticker={ticker}&statements={type}&period=&start={startDate}&end={endDate}"

        headers = self.headers

        response = requests.get(url, headers=headers)

        print(response.text)

    def statements(self,ticker,type,startDate,endDate):
        url = self.base_url + f"companies/statements/compact?ticker={ticker}&statements={type}&period=&start={startDate}&end={endDate}"

        response = requests.get(url, headers=self.headers).json()

        columns = response[0]['statements'][0]['columns']
        data = response[0]['statements'][0]['data']

        return pd.DataFrame(data,columns=columns)
    
def train_linear_model(pricesDF, test_size=0.2):
    """
    Train and evaluate a linear regression model to predict the next day's Close.

    :param pricesDF: DataFrame containing columns:
        - 'Date'
        - 'Last Closing Price'
        - 'Opening Price'
        - 'Highest Price'
        - 'Lowest Price'
        - 'Trading Volume'
    :param test_size: float, fraction of data for testing (default 0.2)
    :return: (model, mae, mse, r2)
    """

    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    # 1. Copy and rename columns
    df = pricesDF.copy()
    df = df.reset_index(drop=True)  # or drop=False if you want to keep old index
    df.rename(columns={
        'Last Closing Price': 'Close',
        'Opening Price': 'Open',
        'Highest Price': 'High',
        'Lowest Price': 'Low',
        'Trading Volume': 'Volume'
    }, inplace=True)

    # 2. Create next day's closing price as target
    df['Next_Close'] = df['Close'].shift(-1)

    # 3. Drop rows with missing values (last row will be NaN after shift)
    df.dropna(inplace=True)

    # 4. Convert Date column to datetime if needed
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])

    # 5. Define features and target
    X = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    y = df['Next_Close']

    # 6. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False  # keep chronological order
    )

    # 7. Initialize and train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 8. Make predictions
    y_pred = model.predict(X_test)

    # 9. Evaluate
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R² Score: {r2:.4f}")

    # 10. Plot Actual vs Predicted
    plt.figure(figsize=(12,6))
    plt.plot(y_test.values, label='Actual')
    plt.plot(y_pred, label='Predicted', linestyle='--')
    plt.title('Actual vs Predicted Close Prices')
    plt.xlabel('Test Samples')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Return the trained model and metrics if you want
    return model, mae, mse, r2