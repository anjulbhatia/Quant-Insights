import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import yfinance as yf
from scipy.stats import norm

# Function to fetch historical data
def fetch_historical_data(ticker, start_date, end_date, interval):
    data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    return data

# Function for mean reversion strategy
def mean_reversion_strategy(df, window=20, std_dev=2):
    df['mean'] = df['Close'].rolling(window=window).mean()
    df['std'] = df['Close'].rolling(window=window).std()
    df['z_score'] = (df['Close'] - df['mean']) / df['std']
    
    df['upper_band'] = df['mean'] + (std_dev * df['std'])
    df['lower_band'] = df['mean'] - (std_dev * df['std'])

    df['signal'] = 0
    df.loc[df['z_score'] < -std_dev, 'signal'] = 1
    df.loc[df['z_score'] > std_dev, 'signal'] = -1
    return df

# Function for GBM simulation
def gbm_simulation(S0, T, mu, sigma, dt=1/252):
    N = int(T / dt)
    t = np.linspace(0, T, N)
    W = np.random.standard_normal(size=N)
    W = np.cumsum(W) * np.sqrt(dt) # standard brownian motion
    X = (mu - 0.5 * sigma**2) * t + sigma * W
    S = S0 * np.exp(X) # GBM
    return S

# Function to fetch real-time data
def fetch_real_time_data(ticker):
    data = yf.download(ticker, period='1d', interval='1m')
    return data

# Function to update DataFrame with new data
def update_data_frame(df, new_data):
    return pd.concat([df, new_data]).drop_duplicates().reset_index(drop=True)

# Function to execute trade based on signal
def execute_trade(signal, symbol, qty):
    if signal == 1:
        return f"Buy order for {symbol} with quantity {qty}"
    elif signal == -1:
        return f"Sell order for {symbol} with quantity {qty}"
    else:
        return "No trade signal"

# Streamlit app
def main():
    st.title('Real-Time Trading Signal Predictor')
    
    # Sidebar inputs
    st.sidebar.header('Input Parameters')
    ticker = st.sidebar.text_input('Enter Ticker Symbol', 'TATAMOTORS.NS')
    
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=59)).strftime('%Y-%m-%d')
    start_date = st.sidebar.date_input('Start Date', value=datetime.strptime(start_date, '%Y-%m-%d'))
    end_date = st.sidebar.date_input('End Date', value=datetime.strptime(end_date, '%Y-%m-%d'))
    
    interval = st.sidebar.selectbox('Interval', ['1m', '5m', '15m'], index=1)
    
    if st.sidebar.button('Predict'):
        st.subheader('Trading Signals')
        st.write(f"Fetching data for {ticker} from {start_date} to {end_date} with interval {interval}...")
        
        # Fetch historical data
        historical_data = fetch_historical_data(ticker, start_date, end_date, interval)
        historical_data = mean_reversion_strategy(historical_data)
        
        # Simulate future prices using GBM
        S0 = historical_data['Close'].iloc[-1]
        T = (end_date - start_date).days / 252
        mu = 0.001  # Example expected return
        sigma = 0.02  # Example volatility
        simulated_prices = gbm_simulation(S0, T, mu, sigma)
        expected_price = simulated_prices[-1]
        
        # Add GBM-based signal logic (example)
        if expected_price > historical_data['upper_band'].iloc[-1]:
            signal = -1
        elif expected_price < historical_data['lower_band'].iloc[-1]:
            signal = 1
        else:
            signal = 0
        
        # Execute trade based on signal
        trade_result = execute_trade(signal, ticker, qty=25)
        st.write(trade_result)
    
    if st.sidebar.button('Clear'):
        st.subheader('Cleared Screen')
        st.write('')

if __name__ == '__main__':
    main()
