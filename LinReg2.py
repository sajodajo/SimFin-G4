from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    def linreg2(df):
        df['Next Day Closing Price'] = df['Last Closing Price'].shift(-1)
        df['Rolling Mean Closing Price'] = df['Last Closing Price'].rolling(window=30).mean()

        df = df.dropna()
        
        X = df[['Common Shares Outstanding', 'Last Closing Price', 'Adjusted Closing Price', 
                'Highest Price', 'Lowest Price', 'Opening Price', 'Trading Volume','Rolling Mean Closing Price']]
        y = df['Next Day Closing Price']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = model.score(X_test, y_test)
        rmse = np.sqrt(mse)
        percentage_rmse = (rmse / y_test.mean()) * 100
        
        latest_data = df.iloc[[-1]][['Common Shares Outstanding', 'Last Closing Price', 'Adjusted Closing Price', 
                                    'Highest Price', 'Lowest Price', 'Opening Price', 'Trading Volume','Rolling Mean Closing Price']]
        next_day_prediction = model.predict(latest_data)

        plt.figure(figsize=(12,6))
        plt.plot(y_test.values, label='Actual')
        plt.plot(y_pred, label='Predicted', linestyle='--')
        plt.title('Actual vs Predicted Close Prices')
        plt.xlabel('Test Samples')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        return mse, r2, rmse, percentage_rmse, next_day_prediction[0]


    def multiModelTest(df):
        df['Next Day Closing Price'] = df['Last Closing Price'].shift(-1)
        df['Rolling Mean Closing Price'] = df['Last Closing Price'].rolling(window=30).mean()

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