import pandas as pd
import numpy as np
import yfinance as yf
import ta
import plotly.graph_objects as go
import streamlit as st

# Fetch and resample data
# TODO: use Close or Adj Close?
# TODO: migrate to use a DB at some point
@st.cache_data
def get_ticker_data(ticker, start_date, end_date):
    daily_data = yf.download(ticker, start=start_date, end=end_date, interval='1d')
    daily_data.dropna(inplace=True)

    weekly_data = daily_data.resample('W').agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }
    )

    monthly_data = daily_data.resample('M').agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }
    )

    return daily_data, weekly_data, monthly_data

def calculate_indicators(df, period=14):
    # 20w SMA w/21w EMA = bull market support band
    # Calculate SMA
    df['sma_20'] = ta.trend.sma_indicator(df['Close'], 20)

    # Calculate EMA
    df['ema_9'] = ta.trend.ema_indicator(df['Close'], 9)
    df['ema_21'] = ta.trend.ema_indicator(df['Close'], 21)
    df['ema_55'] = ta.trend.ema_indicator(df['Close'], 55)
    df['ema_200'] = ta.trend.ema_indicator(df['Close'], 200)
        
    # Calculate Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df['Close'], period)
    df['bb_high'] = bollinger.bollinger_hband()
    df['bb_avg'] = bollinger.bollinger_mavg()
    df['bb_low'] = bollinger.bollinger_lband()

    # Calculate RSI
    df['rsi'] = ta.momentum.rsi(df['Close'], period)

    # Calculate Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'], period)
    df['stoch'] = stoch.stoch()

    # Create two new columns for the EMA > SMA and EMA < SMA periods
    df['ema_greater_sma'] = df['ema_21'] > df['sma_20']
    df['ema_less_sma'] = df['ema_21'] < df['sma_20']
        
    df['macd_line'] = ta.trend.macd(df['Close'])
    df['signal_line'] = ta.trend.macd_signal(df['Close'])

    return df

def calculate_signals(df):
    # Initialize signals to neutral
    df['sma_signal'] = 'Neutral'
    df['ema_signal'] = 'Neutral'
    df['rsi_signal'] = 'Neutral'
    df['stoch_signal'] = 'Neutral'
    df['macd_signal'] = 'Neutral'
    df['bb_signal'] = 'Neutral'

    # SMA and EMA bullish if price above SMA or EMA, bearish if below
    df.loc[df['Close'] > df['sma_20'], 'sma_signal'] = 'Bullish'
    df.loc[df['Close'] < df['sma_20'], 'sma_signal'] = 'Bearish'
    df.loc[df['Close'] > df['ema_55'], 'ema_signal'] = 'Bullish'
    df.loc[df['Close'] < df['ema_55'], 'ema_signal'] = 'Bearish'

    # RSI bullish if value below 30 (oversold), bearish if above 70 (overbought)
    df.loc[df['rsi'] < 30, 'rsi_signal'] = 'Bullish'
    df.loc[df['rsi'] > 70, 'rsi_signal'] = 'Bearish'

    # Stochastic bullish if value below 20 (oversold), bearish if above 80 (overbought)
    df.loc[df['stoch'] < 20, 'stoch_signal'] = 'Bullish'
    df.loc[df['stoch'] > 80, 'stoch_signal'] = 'Bearish'

    # MACD bullish if MACD line crosses above signal line, bearish if it crosses below
    df.loc[df['macd_line'] > df['signal_line'], 'macd_signal'] = 'Bullish'
    df.loc[df['macd_line'] < df['signal_line'], 'macd_signal'] = 'Bearish'

    # Bollinger Bands bullish if price crosses above average, bearish if it crosses below
    df.loc[df['Close'] > df['bb_avg'], 'bb_signal'] = 'Bullish'
    df.loc[df['Close'] < df['bb_avg'], 'bb_signal'] = 'Bearish'

    return df