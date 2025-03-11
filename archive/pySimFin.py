import requests
import pandas as pd

class PySimFin:
    def __init__(self, api_key: str):
        """
        Constructor to initialize the API key and base endpoint.
        """
        self.api_key = api_key
        self.base_url = "https://simfin.com/api/v2/"

    def get_share_prices(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """
        Fetch share prices for a given ticker within a time range.
        """
        endpoint = f"{self.base_url}companies/prices"
        params = {
            "ticker": ticker,
            "start": start,
            "end": end,
            "api-key": self.api_key
        }

        response = requests.get(endpoint, params=params)
        if response.status_code == 200:
            data = response.json()
            return pd.DataFrame(data)
        else:
            raise Exception(f"Error fetching share prices: {response.text}")

    def get_financial_statement(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """
        Fetch financial statements for a given ticker within a time range.
        """
        endpoint = f"{self.base_url}companies/statements"
        params = {
            "ticker": ticker,
            "start": start,
            "end": end,
            "api-key": self.api_key
        }

        response = requests.get(endpoint, params=params)
        if response.status_code == 200:
            data = response.json()
            return pd.DataFrame(data)
        else:
            raise Exception(f"Error fetching financial statements: {response.text}")