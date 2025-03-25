    def train_linear_model(self,df, test_size=0.2):
        # 1. Copy and rename columns
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


    def run_forecasts_and_plot(self,pricesDF):
        # Create a new DataFrame from the original to avoid modifying pricesDF
        df_new = pricesDF
        
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

        return fig, df_new

    def run_one_step_forecast_new_df(self,df_new):        
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


    def compare_forecasting_models_from_df(self,df, test_size=0.2, feature_cols=None, target_col=None, plot=True):
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

        df = df.rename(columns={
        'Opening Price': 'Open',
        'Highest Price': 'High',
        'Lowest Price': 'Low',
        'Last Closing Price': 'Close',
        'Trading Volume': 'Volume'
    })
        df['Next_Close'] = df['Close'].shift(-1)

        df = df.dropna()
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
    