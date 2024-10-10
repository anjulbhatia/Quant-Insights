"""
RT2 - Indicator Module

This module provides functions to calculate popular technical indicators such as
Simple Moving Average (SMA), Exponential Moving Average (EMA), MACD, RSI, and 
Bollinger Bands.

GitHub Repository: https://www.github.com/anjulbhatia/rt2
"""

import pandas as pd

# Simple Moving Average (SMA)
def sma(prices: pd.Series, period: int) -> pd.Series: 
    return prices.rolling(window=period).mean()

# Exponential Moving Average (EMA)
def ema(prices: pd.Series, period: int) -> pd.Series:
    return prices.ewm(span=period, adjust=False).mean()

# Moving Average
def moving_average(prices: pd.Series, period: int = 20, method: str = 'SMA') -> pd.Series:
    if method == 'SMA':
        return prices.rolling(window=period).mean()
    elif method == 'EMA':
        return prices.ewm(span=period, adjust=False).mean()
    else:
        raise ValueError("Method must be 'SMA' or 'EMA'")
    
# Relative Strength Index (RSI)
def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Moving Average Convergence Divergence (MACD)
def macd(prices: pd.Series):
    ema12 = prices.ewm(span=12, adjust=False).mean()
    ema26 = prices.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

# Bollinger Bands
def bollinger_bands(prices, window=20):
    sma = prices.rolling(window=window).mean()
    stddev = prices.rolling(window=window).std()
    upper_band = sma + (2 * stddev)
    lower_band = sma - (2 * stddev)
    return sma, upper_band, lower_band