import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import yfinance as yf
import time
import matplotlib.pyplot as plt
import mplcursors
import sys

# Define strategy and utility functions
def mean_reversion_strategy(df, window=20, std_dev=2):
    """
    Apply mean reversion strategy using Bollinger Bands and Z-score.
    """
    df = df.copy()
    df['mean'] = df['Close'].rolling(window=window).mean()
    df['std'] = df['Close'].rolling(window=window).std()
    df['z_score'] = (df['Close'] - df['mean']) / df['std']
    
    df['upper_band'] = df['mean'] + (std_dev * df['std'])
    df['lower_band'] = df['mean'] - (std_dev * df['std'])

    df['signal'] = 0
    df.loc[df['z_score'] < -std_dev, 'signal'] = 1
    df.loc[df['z_score'] > std_dev, 'signal'] = -1
    
    return df.dropna()

def gbm_simulation(S0, T, mu, sigma, dt=1/252):
    """
    Simulate Geometric Brownian Motion for stock price prediction.
    """
    N = int(T / dt)
    t = np.linspace(0, T, N)
    W = np.random.standard_normal(size=N)
    W = np.cumsum(W) * np.sqrt(dt)
    X = (mu - 0.5 * sigma**2) * t + sigma * W
    S = S0 * np.exp(X)
    return S

def fetch_historical_data(ticker, start_date, end_date, interval):
    """
    Fetch historical stock data from Yahoo Finance.
    """
    return yf.download(ticker, start=start_date, end=end_date, interval=interval)

def fetch_real_time_data(ticker):
    """
    Fetch real-time stock data from Yahoo Finance.
    """
    return yf.download(ticker, period='1d', interval='1m')

def update_data_frame(df, new_data):
    """
    Update DataFrame with new data and remove duplicates.
    """
    return pd.concat([df, new_data]).drop_duplicates().reset_index(drop=True)

def calculate_kelly_criterion(win_prob, win_loss_ratio):
    """
    Calculate Kelly criterion for optimal bet size.
    """
    return win_prob - (1 - win_prob) / win_loss_ratio

def plot_signals(df):
    """
    Plot stock prices with trading signals, Bollinger Bands, and OHLC hover functionality.
    """
    plt.figure(figsize=(12,6))
    plt.plot(df.index, df['Close'], label='Close Price')
    plt.plot(df.index, df['mean'], label='Mean')
    plt.fill_between(df.index, df['lower_band'], df['upper_band'], color='gray', alpha=0.2, label='Bollinger Bands')
    
    buy_signals = df[df['signal'] == 1]
    sell_signals = df[df['signal'] == -1]
    plt.plot(buy_signals.index, buy_signals['Close'], '^', markersize=10, color='g', lw=0, label='Buy Signal')
    plt.plot(sell_signals.index, sell_signals['Close'], 'v', markersize=10, color='r', lw=0, label='Sell Signal')
    
    plt.legend()

    # Add interactive cursor for OHLC data
    cursor = mplcursors.cursor(hover=True)
    @cursor.connect("add")
    def on_add(sel):
        index = sel.target.index
        ohlc_data = df.loc[index, ['Open', 'High', 'Low', 'Close']]
        sel.annotation.set(text=f" Open: {ohlc_data['Open']}\nHigh: {ohlc_data['High']}\nLow: {ohlc_data['Low']}\nClose: {ohlc_data['Close']}")
    
    plt.show()

# Main script for real-time trading simulation
if __name__ == "__main__":
    symbol = "RECLTD.NS"# 'PAYTM.NS'
    start_date = '2024-08-01'
    end_date = '2024-09-10'
    interval = '5m'

    # Fetch historical data and apply mean reversion strategy
    historical_data = fetch_historical_data(symbol, start_date, end_date, interval)
    historical_data = historical_data[historical_data.index.notnull()]  # Ensure no null indices
    real_time_df = mean_reversion_strategy(historical_data)

    # Plot initial signals with hover functionality
    plot_signals(real_time_df)

    # Calculate Kelly criterion
    historical_df = mean_reversion_strategy(historical_data)
    historical_df['returns'] = historical_df['Close'].pct_change()
    historical_df['strategy_returns'] = historical_df['signal'].shift(1) * historical_df['returns']

    win_prob = (historical_df['strategy_returns'] > 0).mean()
    win_loss_ratio = historical_df['strategy_returns'][historical_df['strategy_returns'] > 0].mean() / -historical_df['strategy_returns'][historical_df['strategy_returns'] < 0].mean()

    kelly_fraction = calculate_kelly_criterion(win_prob, win_loss_ratio)
    print(f"Kelly Fraction: {kelly_fraction:.2f}")

    # Initialize capital
    capital = 100000

    # Initialize trade log
    trade_log = []

    # Main loop for real-time trading
    while True:
        try:
            # Fetch real-time data
            new_data = fetch_real_time_data(symbol)
            if not new_data.empty:
                new_data = new_data[new_data.index.notnull()]
                real_time_df = update_data_frame(real_time_df, new_data)
                real_time_df = mean_reversion_strategy(real_time_df)
                signal = real_time_df['signal'].iloc[-1]
                
                # Simulate future prices using GBM
                S0 = real_time_df['Close'].iloc[-1]
                T = 1/252
                mu = 0.001
                sigma = 0.02
                simulated_prices = gbm_simulation(S0, T, mu, sigma)
                expected_price = simulated_prices[-1]
                
                # Add GBM-based signal logic
                if expected_price > real_time_df['upper_band'].iloc[-1]:
                    signal = -1
                elif expected_price < real_time_df['lower_band'].iloc[-1]:
                    signal = 1
                
                # Determine position size based on Kelly criterion
                position_size = int(capital * kelly_fraction / S0)
                
                # Log trade
                trade_log.append({
                    'time': pd.Timestamp.now(),
                    'symbol': symbol,
                    'signal': 'Buy' if signal == 1 else 'Sell',
                    'quantity': position_size,
                    'price': S0
                })
                
                # Update capital based on trade
                if signal == 1:
                    capital -= position_size * S0
                elif signal == -1:
                    capital += position_size * S0

            # Plot updated signals
            plot_signals(real_time_df)
            
            # Wait for 1 minute before the next iteration
            time.sleep(60)
        
        except KeyboardInterrupt:
            print("Trading stopped.")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue
