import simfin as sf
from simfin.names import *
import polars as pl
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
import itertools
from statsmodels.tsa.arima.model import ARIMA

import warnings
warnings.simplefilter("ignore")

class pySimFin:
    def __init__(self):
            self.api_key = '70d5d920-9f9e-4062-9311-1b4df7c98ba4'

    def get_share_prices(self, ticker: str, start: str, end: str):      
         shareprices = sf.load(dataset='shareprices', variant='annual', market='us',start_date=start,end_date=end)
         return shareprices.head(5)
         

    def nextDayPrediction(self,company):
        company = input("Enter a company to analyse: ")

        sf.set_data_dir('~/simfin_data/')
        sf.set_api_key(api_key= self.api_key)

        shareprices_pd = sf.load(dataset='shareprices', variant='daily', market='us')
        companies_pd = sf.load_companies(market='us')

        shareprices = pl.from_pandas(shareprices_pd)
        companies = pl.from_pandas(companies_pd)

        shareprices = shareprices.with_columns(pl.col('Date').str.to_datetime('%Y-%m-%d'))

        ts_prices = shareprices.filter(pl.col("Ticker") == company).select(['Date',"Close"])

        print(f"\n********\n\nShare Price data imported as 'shareprice'\nCompanies data imported as 'companies'\n{company} price data imported as 'ts_prices'")

        ts_prices = ts_prices.with_columns(pl.col("Close").diff().alias("Differenced")).drop_nulls() 

        additive_decomposition = seasonal_decompose(ts_prices.select('Close'), model="additive", period=12)
        multiplicative_decomposition = seasonal_decompose(ts_prices.select('Close'), model="multiplicative", period=12)

        residuals = multiplicative_decomposition.resid
        trend = multiplicative_decomposition.trend
        seasonal = multiplicative_decomposition.seasonal

        residuals = residuals[~np.isnan(residuals)]
        trend = trend[~np.isnan(trend)]
        seasonal = seasonal[~np.isnan(seasonal)]

        # Define range of p, d, q values to test
        p = range(0, 10)
        d = [0] # Because it's an ARMA model, already stationary  
        q = range(0, 10)

        # Store results
        results_list = []

        # Loop through all possible combinations of p, d, q
        for param in itertools.product(p, d, q):
            try:
                model = ARIMA(residuals, order=param)
                model_fit = model.fit()
                results_list.append([param, model_fit.aic, model_fit.bic])
            except:
                continue  # Skip invalid models

        # Convert results to a DataFrame
        results_df = pl.DataFrame(
            {
                "order": [item[0] for item in results_list], 
                "AIC": [item[1] for item in results_list],   
                "BIC": [item[2] for item in results_list]  
            }
        )

        # Sort by AIC (or BIC) to find the best model
        bestOrder = tuple(results_df.filter(pl.col("AIC") == results_df["AIC"].min()).select('order').row(0)[0])

        model = ARIMA(residuals, order=bestOrder)
        arma_results = model.fit()

        # Predict the next day
        next_residual = arma_results.forecast(steps=1)[0]

        next_trend = trend[-1] 
        next_seasonal = seasonal[-1] 

        next_value = next_residual * next_trend * next_seasonal

        print("\n********\n\nNext day price prediction:", next_value)