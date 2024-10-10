from typing import Literal

from .utils import *
from .data_fetcher import (fetch_historical_data, fetch_realtime_data, fetch_company_info)
from .indicators import (sma, ema, macd, rsi, bollinger_bands)
from .plotter import Plotter

class TradeModel:
    def __init__(self, data: pd.DataFrame = None, ticker: str = None):
        """
        Initialize TradeModel with either data or ticker.
        If ticker is provided, data can be fetched using get_data().
        """
        self.ticker = ticker
        self.data = data
        self.prices = data['Close'] if data is not None and 'Close' in data.columns else None
    
    def set_ticker(self, ticker: str):
        """
        Set the ticker for the model.
        """
        self.ticker = ticker
        return ticker
    
    def get_data(self, period: str = "5y", start_date: str = None, end_date: str = None, interval: str = "1d") -> pd.DataFrame:
        """
        Fetch historical data for the ticker using the Yfinance API.
        :params:
        """
        if not self.ticker:
            raise ValueError("Ticker must be specified to fetch data.")
        
        data = fetch_historical_data(self.ticker, start_date=start_date, end_date=end_date, period=period, interval=interval)
        self.data = data
        self.prices = data['Close'] if data is not None and 'Close' in data.columns else None
        return self.data
    
    def load_data(self, data: pd.DataFrame):
        """
        Load data directly into the model.
        """
        self.data = data
        self.prices = data['Close'] if data is not None and 'Close' in data.columns else None
   
    def get_intraday(self, interval="1m"):
        """
        Fetch intraday data for the ticker.
        """
        if not self.ticker:
            raise ValueError("Ticker must be specified to fetch intraday data.")
        return fetch_realtime_data(ticker=self.ticker, interval=interval)

    def get_company_info(self):
        """
        Fetch company information for the ticker.
        """
        if self.ticker:
            return fetch_company_info(ticker=self.ticker)
        else:
            raise ValueError("Ticker must be specified to fetch company information.")
        
    def sma(self, period: int = 20):
        """
        Calculate Simple Moving Average (SMA) on the loaded price data.
        """
        if self.prices is None:
            raise ValueError("Unable to load prices (data['Close']); Reload the data or manually set prices by using the self.prices = pricesSeries statement")
        return sma(self.data, period)
    
    def ema(self, period: int = 20):
        """
        Calculate Exponential Moving Average (EMA) on the loaded price data.
        """
        if self.prices is None:
            raise ValueError("Unable to load prices (data['Close']); Reload the data or manually set prices by using the self.prices = pricesSeries statement")
        return ema(self.data, period)
    
    def macd(self, short_period: int = 12, long_period: int = 26, signal_period: int = 9):
        """
        Calculate MACD and Signal line.
        """
        if self.prices is None:
            raise ValueError("Unable to load prices (data['Close']); Reload the data or manually set prices by using the self.prices = pricesSeries statement")
        return macd(self.data, short_period, long_period, signal_period)
    
    def rsi(self, period: int = 14):
        """
        Calculate Relative Strength Index (RSI).
        """
        if self.prices is None:
            raise ValueError("Unable to load prices (data['Close']); Reload the data or manually set prices by using the self.prices = pricesSeries statement")
        return rsi(self.data, period)
    
    def bollinger_bands(self, window: int = 20, std_dev: int = 2):
        """
        Calculate Bollinger Bands.
        """
        if self.prices is None:
            raise ValueError("Unable to load prices (data['Close']); Reload the data or manually set prices by using the self.prices = pricesSeries statement")
        return bollinger_bands(self.data, window)
    

    def plotter(self, ChartType: Literal['plotly','trading_view'] = 'trading_view'):
        return Plotter(ChartType=ChartType)

