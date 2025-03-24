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


def run_forecasts_and_plot(pricesDF):
    """
    Creates a new DataFrame from pricesDF, prepares it, performs rolling one-step and static forecasts,
    plots the results, and returns the new DataFrame with forecasts.
    
    Expects pricesDF to have at least the following columns:
      - 'Last Closing Price'
      - 'Opening Price'
      - 'Highest Price'
      - 'Lowest Price'
      - 'Trading Volume'
      
    The new DataFrame will have the columns:
      - 'Close', 'Open', 'High', 'Low', 'Volume', 'Next_Close'
      - And forecast columns will be added via the plotting functions.
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression

    # Create a new DataFrame from the original to avoid modifying pricesDF
    df_new = pricesDF.copy()
    
    # Reset the index if needed and assume there's a Date column or the index is Date
    # Here, if the index isn't datetime, we force conversion (you may need to adjust based on your data)
    if not pd.api.types.is_datetime64_any_dtype(df_new.index):
        df_new.index = pd.to_datetime(df_new.index)
    df_new = df_new.sort_index()
    
    # Rename columns for consistency
    df_new = df_new.reset_index(drop=True)
    df_new.rename(columns={
        'Last Closing Price': 'Close',
        'Opening Price': 'Open',
        'Highest Price': 'High',
        'Lowest Price': 'Low',
        'Trading Volume': 'Volume'
    }, inplace=True)
    
    # Create next day's closing price as target column
    df_new['Next_Close'] = df_new['Close'].shift(-1)
    df_new.dropna(inplace=True)  # Remove the last row with no target
    
    # For demonstration purposes, split the data into training and test segments:
    # Here we'll take the last 30 rows as test and the rest as training.
    train = df_new.iloc[:-30]
    test = df_new.iloc[-30:]
    train_plot = df_new.iloc[-60:-30]  # The 30 days before test for plotting historical data

    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    target_col = 'Next_Close'

    ### 🔁 ROLLING ONE-STEP FORECAST
    one_step_preds = []
    ci_lower = []
    ci_upper = []

    for i in range(len(test)):
        # Use all available data up to the current test point
        rolling_train = pd.concat([train, test.iloc[:i]])
        X_train = rolling_train[feature_cols]
        y_train = rolling_train[target_col]

        model = LinearRegression()
        model.fit(X_train, y_train)

        X_input = test.iloc[[i]][feature_cols]
        pred = model.predict(X_input)[0]

        one_step_preds.append(pred)
        ci_lower.append(pred * 0.98)  # Mock a 2% lower bound
        ci_upper.append(pred * 1.02)  # Mock a 2% upper bound

    one_step_forecasts = pd.Series(one_step_preds, index=test.index)
    ci_lower_series = pd.Series(ci_lower, index=test.index)
    ci_upper_series = pd.Series(ci_upper, index=test.index)

    ### 🔮 STATIC FORECAST FROM LAST DAY OF TRAINING
    model_static = LinearRegression()
    X_train_static = train[feature_cols]
    y_train_static = train[target_col]
    model_static.fit(X_train_static, y_train_static)

    X_forecast = test[feature_cols]
    static_preds = model_static.predict(X_forecast)
    static_forecast = pd.Series(static_preds, index=test.index)

    static_ci_lower = static_forecast * 0.98
    static_ci_upper = static_forecast * 1.02

    # 📈 Plotting the forecasts side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # LEFT: Rolling One-Step Forecast
    axes[0].plot(train_plot.index, train_plot['Close'], label='Historical Data', marker='o')
    axes[0].plot(test.index, test[target_col], label='Actual', color='black', marker='o')
    axes[0].plot(one_step_forecasts.index, one_step_forecasts, label='One-step Forecast', color='red', marker='o')
    axes[0].fill_between(test.index, ci_lower_series, ci_upper_series, color='red', alpha=0.3)
    axes[0].set_title("Rolling One-Step Ahead Forecast")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Price")
    axes[0].legend()

    # RIGHT: Static Forecast from End of Training
    axes[1].plot(train_plot.index, train_plot['Close'], label='Historical Data', marker='o')
    axes[1].plot(test.index, test[target_col], label='Actual', color='black', marker='o')
    axes[1].plot(static_forecast.index, static_forecast, label='Static Forecast', color='green', marker='o')
    axes[1].fill_between(test.index, static_ci_lower, static_ci_upper, color='green', alpha=0.3)
    axes[1].set_title("Static Forecast from End of Training")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Price")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    # Optionally, you can add the forecast results as new columns to df_new
    # For example, for the rolling forecast, align predictions to the test index
    df_new.loc[test.index, 'One_Step_Forecast'] = one_step_forecasts
    df_new.loc[test.index, 'Static_Forecast'] = static_forecast

    return df_new

def run_one_step_forecast_new_df(pricesDF):
    """
    Creates a new DataFrame from the original pricesDF, prepares the data,
    trains a linear regression model to predict the next day's closing price,
    and adds the predicted value to the DataFrame.
    
    Parameters:
      pricesDF: pandas DataFrame with columns:
         - 'Last Closing Price'
         - 'Opening Price'
         - 'Highest Price'
         - 'Lowest Price'
         - 'Trading Volume'
         (and optionally an index or a Date column)
    
    Returns:
      df_new: New DataFrame with columns renamed, a 'Next_Close' target column,
              and the predicted next day's close stored in 'Predicted_Next_Close'
      model: The trained Linear Regression model
    """
    import pandas as pd
    from sklearn.linear_model import LinearRegression

    # Create a new DataFrame to avoid modifying the original
    df_new = pricesDF.copy()
    
    # Reset index if necessary (assuming the index is not needed)
    df_new = df_new.reset_index(drop=True)
    
    # Rename columns for consistency
    df_new.rename(columns={
        'Last Closing Price': 'Close',
        'Opening Price': 'Open',
        'Highest Price': 'High',
        'Lowest Price': 'Low',
        'Trading Volume': 'Volume'
    }, inplace=True)
    
    # Create the target column by shifting 'Close' upward (next day's close)
    df_new['Next_Close'] = df_new['Close'].shift(-1)
    
    # Drop the last row, which now has a missing target value
    df_new = df_new.dropna()
    
    # Define the feature columns and target column
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    target_col = 'Next_Close'
    
    # Use all rows except the last one for training
    train_df = df_new.iloc[:-1]
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    
    # Use the last row as the input for prediction
    latest_data = df_new.iloc[[-1]][feature_cols]
    
    # Initialize and train the Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict the next day's closing price using the latest data
    next_day_prediction = model.predict(latest_data)[0]
    
    # Add the prediction to the new DataFrame (in the last row)
    df_new.loc[df_new.index[-1], 'Predicted_Next_Close'] = next_day_prediction
    
    return df_new, model

# Example usage:
# new_df, trained_model = run_one_step_forecast_new_df(pricesDF)
# new_df.tail()  # This will show the last few rows including the 'Predicted_Next_Close'


def compare_forecasting_models_from_df(df, test_size=0.2, feature_cols=None, target_col=None, plot=True):
    """
    Trains and compares forecasting models using a DataFrame that includes the target.
    
    Parameters:
      df          : pandas DataFrame that must contain the target column (default 'Next_Close')
                    and the feature columns (default ['Open', 'High', 'Low', 'Close', 'Volume']).
      test_size   : Fraction of data to use as test set (default: 0.2).
      feature_cols: List of columns to use as features (default: ['Open', 'High', 'Low', 'Close', 'Volume']).
      target_col  : The name of the target column (default: 'Next_Close').
      plot        : Boolean; if True, plot predictions vs. actual values (default: True).
    
    Returns:
      results_df : DataFrame containing evaluation metrics (MAE, MSE, R²) for each model.
      models     : Dictionary of the trained models.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    # Set default feature and target columns if not provided
    if feature_cols is None:
        feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if target_col is None:
        target_col = 'Next_Close'

    # Define features (X) and target (y)
    X = df[feature_cols]
    y = df[target_col]

    # Split the data (keep time order with shuffle=False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

    # Initialize models
    lr_model = LinearRegression()
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

    # Fit models
    lr_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)

    # Create dictionary of models for easy iteration
    models = {
        'Linear Regression': lr_model,
        'Random Forest': rf_model,
        'XGBoost': xgb_model
    }

    # Evaluate each model
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[name] = {'MAE': mae, 'MSE': mse, 'R²': r2}

    results_df = pd.DataFrame(results).T.sort_values(by='MAE')

    # Plot predictions vs. actual if requested
    if plot:
        plt.figure(figsize=(12, 6))
        for name, model in models.items():
            y_pred = model.predict(X_test)
            plt.plot(y_pred, label=f'{name} Prediction', alpha=0.7)
        plt.plot(y_test.values, label='Actual', linewidth=2, linestyle='--', color='black')
        plt.title('Model Predictions vs. Actual')
        plt.xlabel('Test Samples')
        plt.ylabel('Close Price')
        plt.legend()
        plt.grid(True)
        plt.show()

    return results_df, models