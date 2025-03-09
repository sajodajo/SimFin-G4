import simfin as sf
from simfin.names import *
import polars as pl

company = input("Enter a company to analyse: ")

sf.set_data_dir('~/simfin_data/')
sf.set_api_key(api_key='70d5d920-9f9e-4062-9311-1b4df7c98ba4')

shareprices_pd = sf.load(dataset='shareprices', variant='daily', market='us')
companies_pd = sf.load_companies(market='us')

shareprices = pl.from_pandas(shareprices_pd)
companies = pl.from_pandas(companies_pd)

shareprices = shareprices.with_columns(pl.col('Date').str.to_datetime('%Y-%m-%d'))

ts_prices = shareprices.filter(pl.col("Ticker") == company).select(['Date',"Close"])

print(f"\n********\n\nShare Price data imported as 'shareprice'\nCompanies data imported as 'companies'\n{company} price data imported as 'ts_prices'")