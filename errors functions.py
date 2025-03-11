import yfinance as yf
import re
import datetime

def check_date_format(date_string):
    desired_format = re.compile(r'^\d{4}-\d{2}-\d{2}$')
    return desired_format.match(date_string).group() == date_string
        
def end_date_greater_than_start_date(start_date, end_date):
    start_dt = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    return start_dt < end_dt

def get_financial_data(stock_ticker, start_date, end_date, column_to_extract):
    if not isinstance(stock_ticker, str):
        raise TypeError('The stock_ticker should be a string')

    if not isinstance(start_date, str):
        raise TypeError('The start_date should be a string')
    
    if not isinstance(end_date, str):
        raise TypeError('The end_date should be a string')

    if not check_date_format(start_date):
        raise ValueError('The start date is not in the desired format YYYY-MM-DD')
    
    if not check_date_format(end_date):
        raise ValueError('The end date is not in the desired format YYYY-MM-DD')

    if not end_date_greater_than_start_date(start_date, end_date):
        raise ValueError('The start date is greater than the end date')
    
    data = yf.download(stock_ticker, start_date, end_date, column_to_extract)

    return data[[column_to_extract]]