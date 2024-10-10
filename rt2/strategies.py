"""
RT2 - Trade Strategies Module

This module provides implementations of various trade strategies and risk management
techniques to optimize trading decisions. It includes both technical and risk management
strategies, such as the Kelly Criterion for position sizing, which helps traders determine 
the optimal capital allocation based on probabilities and expected returns.

GitHub Repository: https://www.github.com/anjulbhatia/rt2
"""
import pandas as pd
import numpy as np

def mean_reversion_strategy(prices, window=20):
    rolling_mean = prices.rolling(window=window).mean()
    signal = pd.Series(0, index=prices.index)
    signal[prices < rolling_mean] = 1  # Buy when price is below the mean
    signal[prices > rolling_mean] = -1  # Sell when price is above the mean
    return signal


def kelly_criterion(win_probability, win_loss_ratio):
    if win_probability < 0 or win_probability > 1:
        raise ValueError("win_probability must be between 0 and 1.")

    kelly_fraction = win_probability - (1 - win_probability) / win_loss_ratio
    return kelly_fraction
