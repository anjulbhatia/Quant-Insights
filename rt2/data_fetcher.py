"""
RT2 - Data Fetcher Module

This module provides functionality to fetch financial data from various sources,
including stock, ETF, crypto, currency, and commodity data. It supports both static
methods for fetching data and instance methods for retrieving and processing data.

GitHub Repository: https://www.github.com/anjulbhatia/rt2
"""

import yfinance as yf  # Financial data fetching
import pandas as pd  # Data analysis and manipulation
from datetime import datetime 

def fetch_historical_data(ticker: str, start_date: str = None, end_date: str = None, period: str = None, interval: str = '1d') -> pd.DataFrame:
    if start_date is not None:
        end_date = end_date or datetime.today().strftime('%Y-%m-%d')
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    elif period is not None:
        data = yf.download(ticker, period=period, interval=interval)
    else:
        raise ValueError("Either 'period' or 'start_date' must be specified.")
    
    return data

def fetch_realtime_data(ticker: str, period: str = '1d', interval: str = '1m') -> pd.DataFrame:
    data = yf.download(ticker, period=period, interval=interval)
    return data

def fetch_company_info(ticker: str) -> dict:
    stock = yf.Ticker(ticker)
    return stock.info