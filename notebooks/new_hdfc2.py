import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import yfinance as yf
import time
import matplotlib.pyplot as plt
import mplcursors

from lightweight_charts import Chart

# Define utility and strategy functions
def mean_reversion_strategy(df, window=20, std_dev=2):
    """
    Apply mean reversion strategy using Bollinger Bands and Z-score.

    Parameters:
    df (DataFrame): DataFrame containing stock price data
    window (int): Rolling window size for calculating mean and std deviation
    std_dev (int): Standard deviation multiplier for bands

    Returns:
    DataFrame: DataFrame with added mean, std, upper_band, lower_band, z_score, and signal columns
    """
    df = df.copy()
    df['mean'] = df['Close'].rolling(window=window).mean()
    df['std'] = df['Close'].rolling(window=window).std()
    df['z_score'] = (df['Close'] - df['mean']) / df['std']
    
    df['upper_band'] = df['mean'] + (std_dev * df['std'])
    df['lower_band'] = df['mean'] - (std_dev * df['std'])

    df['signal'] = 0
    df.loc[df['z_score'] < -std_dev, 'signal'] = 1  # Buy signal when price is too low
    df.loc[df['z_score'] > std_dev, 'signal'] = -1  # Sell signal when price is too high
    
    return df.dropna()

def execute_trade(signal, symbol, qty):
    """
    Simulate trade execution for equity stocks.

    Parameters:
    signal (int): Trading signal (1 for buy, -1 for sell)
    symbol (str): Stock symbol
    qty (int): Quantity to trade

    Returns:
    None
    """
    if signal == 1:
        print(f"Executing Buy order for {symbol} with quantity {qty}")
    elif signal == -1:
        print(f"Executing Sell order for {symbol} with quantity {qty}")

def fetch_historical_data(ticker, start_date, end_date, interval):
    """
    Fetch historical stock data from Yahoo Finance.

    Parameters:
    ticker (str): Stock ticker symbol
    start_date (str): Start date for fetching data
    end_date (str): End date for fetching data
    interval (str): Data interval (e.g., '1d', '5m')

    Returns:
    DataFrame: Historical stock data
    """
    data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    return data

def fetch_real_time_data(ticker):
    """
    Fetch real-time stock data from Yahoo Finance.

    Parameters:
    ticker (str): Stock ticker symbol

    Returns:
    DataFrame: Real-time stock data
    """
    data = yf.download(ticker, period='1d', interval='1m')
    return data

def update_data_frame(df, new_data):
    """
    Update DataFrame with new data and remove duplicates.

    Parameters:
    df (DataFrame): Original DataFrame
    new_data (DataFrame): New data to be added

    Returns:
    DataFrame: Updated DataFrame
    """
    return pd.concat([df, new_data]).drop_duplicates().reset_index(drop=True)

def calculate_kelly_criterion(win_prob, win_loss_ratio):
    """
    Calculate Kelly criterion for optimal bet size.

    Parameters:
    win_prob (float): Probability of winning
    win_loss_ratio (float): Ratio of average win to average loss

    Returns:
    float: Kelly fraction
    """
    return win_prob - (1 - win_prob) / win_loss_ratio

def plot_pretty(df):
    chart = Chart()

    chart.legend(visible=True, font_size=14)
    line = chart.create_line('SMA 50')
    line.set(df.index,df['Close'])
    chart.show(block=True)

def plot_signals(df):
    """
    Plot stock prices with trading signals and Bollinger Bands.

    Parameters:
    df (DataFrame): DataFrame containing stock price data and signals

    Returns:
    None
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
    plt.show()

# Main script for real-time trading simulation
if __name__ == "__main__":
    symbol = 'PAYTM.NS'
    start_date = '2024-08-01'
    end_date = '2024-09-10'
    interval = '5m'

    # Fetch historical data and apply mean reversion strategy
    historical_data = fetch_historical_data(symbol, start_date, end_date, interval)
    historical_data = historical_data[historical_data.index.notnull()]  # Ensure there are no null indices
    real_time_df = mean_reversion_strategy(historical_data)

    # Plot initial signals
    #plot_signals(real_time_df)

    # Calculate historical performance metrics for Kelly criterion
    historical_df = mean_reversion_strategy(historical_data)
    historical_df['returns'] = historical_df['Close'].pct_change()
    historical_df['strategy_returns'] = historical_df['signal'].shift(1) * historical_df['returns']

    win_prob = (historical_df['strategy_returns'] > 0).mean()
    win_loss_ratio = historical_df['strategy_returns'][historical_df['strategy_returns'] > 0].mean() / -historical_df['strategy_returns'][historical_df['strategy_returns'] < 0].mean()

    kelly_fraction = calculate_kelly_criterion(win_prob, win_loss_ratio)
    print(f"Kelly Fraction: {kelly_fraction:.2f}")

    # Initialize capital
    capital = 100000  # example initial capital

    # Initialize trade log
    trade_log = []

    # Main loop for real-time trading
    while True:
        try:
            # Fetch real-time data
            new_data = fetch_real_time_data(symbol)
            if not new_data.empty:
                new_data = new_data[new_data.index.notnull()]  # Ensure there are no null indices
                real_time_df = update_data_frame(real_time_df, new_data)
                real_time_df = mean_reversion_strategy(real_time_df)
                signal = real_time_df['signal'].iloc[-1]
                
                # Determine position size based on Kelly criterion
                S0 = real_time_df['Close'].iloc[-1]
                position_size = int(capital * kelly_fraction / S0)
                
                # Log trade
                trade_log.append({
                    'time': pd.Timestamp.now(),
                    'symbol': symbol,
                    'signal': 'Buy' if signal == 1 else 'Sell',
                    'quantity': position_size,
                    'price': S0
                })
                
                execute_trade(signal, symbol, qty=position_size)
                
                # Update capital based on the trade
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
