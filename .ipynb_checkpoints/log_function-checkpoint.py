# simfin_api.py
import requests
import pandas as pd
import logging
import os
from dotenv import load_dotenv

# Logging config
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("simfin_api.log", mode='w'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class PySimFin:
    def __init__(self, api_key):
        load_dotenv()
        self.API_KEY = os.getenv('API_KEY')
        
        self.headers = {
            "accept": "application/json",
            "Authorization": self.API_KEY
        }
        if not self.API_KEY:
            raise ValueError("API key is missing. Provide it as an argument or set it in the environment variables.")
        self.base_url = "https://backend.simfin.com/api/v3/"
        logger.info("Initialized PySimFin API wrapper")

    def get_share_prices(self, ticker: str, start: str, end: str):
        url = f"{self.base_url}/companies/{ticker}/shares/prices"
        params = {"start": start, "end": end}
        logger.info(f"Fetching share prices for {ticker} from {start} to {end}")
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            logger.info(f"Data fetched successfully for {ticker}")
            return pd.DataFrame(response.json())
        except Exception as e:
            logger.error(f"Failed to fetch share prices for {ticker}: {e}")
            raise

    def get_financial_statement(self, ticker: str, start: str, end: str):
        url = f"{self.base_url}/companies/{ticker}/financials/statements"
        params = {"start": start, "end": end}
        logger.info(f"Fetching financial statements for {ticker} from {start} to {end}")

        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            logger.info(f"Financial data fetched successfully for {ticker}")
            return pd.DataFrame(response.json())
        except Exception as e:
            logger.error(f"Failed to fetch financial statements for {ticker}: {e}")
            raise
