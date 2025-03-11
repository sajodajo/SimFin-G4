import simfin as sf
from simfin.names import *
import polars as pl
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import itertools
import warnings
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

warnings.simplefilter("ignore")

def train_arima(params):
    """ Train ARIMA model with given parameters """
    try:
        model = ARIMA(residuals, order=params)
        model_fit = model.fit()
        return params, model_fit.aic, model_fit.bic
    except:
        return None

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")  # Fix multiprocessing issue on Windows/macOS

    company = input("Enter a company to analyze: ")

    # Set SimFin API
    sf.set_data_dir('~/simfin_data/')
    sf.set_api_key(api_key='70d5d920-9f9e-4062-9311-1b4df7c98ba4')

    # Load and filter data
    shareprices = pl.from_pandas(sf.load(dataset='shareprices', variant='daily', market='us'))
    shareprices = shareprices.with_columns(pl.col('Date').str.to_datetime('%Y-%m-%d'))
    ts_prices = shareprices.filter(pl.col("Ticker") == company).select(['Date', "Close"])

    if ts_prices.is_empty():
        print(f"No data found for {company}. Exiting.")
        exit()

    print(f"\n********\n{company} price data imported as 'ts_prices'")

    ts_prices = ts_prices.with_columns(pl.col("Close").diff().alias("Differenced")).drop_nulls()

    # Seasonal decomposition
    decomposition = seasonal_decompose(ts_prices["Close"].to_numpy(), model="multiplicative", period=12)
    trend, seasonal, residuals = decomposition.trend, decomposition.seasonal, decomposition.resid

    # Remove NaN values
    mask = ~np.isnan(residuals)
    residuals, trend, seasonal = residuals[mask], trend[mask], seasonal[mask]

    # ARIMA parameter grid
    p, d, q = range(0, 5), [0], range(0, 5)

    # Parallel processing for ARIMA model selection
    with ProcessPoolExecutor() as executor:
        results_list = list(executor.map(train_arima, itertools.product(p, d, q)))

    results_list = [r for r in results_list if r]  # Remove failed models
    results_df = pl.DataFrame({"order": [r[0] for r in results_list], "AIC": [r[1] for r in results_list]})
    best_order = results_df.sort("AIC").select("order").to_numpy()[0, 0]

    # Fit best ARIMA model
    arma_model = ARIMA(residuals, order=best_order).fit()

    # Predict next day price
    next_residual = arma_model.forecast(steps=1)[0]
    next_trend, next_seasonal = trend[-1], seasonal[-1]
    next_price = next_residual * next_trend * next_seasonal

    print(f"\n********\nNext day price prediction for {company}: {next_price:.2f}")