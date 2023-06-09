import pandas as pd
import numpy as np
import yfinance as yf
import ta
import plotly.graph_objects as go
import streamlit as st

import logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s (%(levelname)s):  %(module)s.%(funcName)s - %(message)s')

# Set up logger for a specific module to a different level
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import src.utils as utils

"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Create SQLite engine and session
engine = create_engine('sqlite:///stock_data.db')
Session = sessionmaker(bind=engine)

# In fetch_resample_data function, before returning data
session = Session()
daily_data.to_sql(ticker + '_daily', con=engine, if_exists='replace')
weekly_data.to_sql(ticker + '_weekly', con=engine, if_exists='replace')
monthly_data.to_sql(ticker + '_monthly', con=engine, if_exists='replace')
session.commit()

# In analyze_ticker function, before fetching data
session = Session()
daily_data = pd.read_sql_table(ticker + '_daily', con=engine)
weekly_data = pd.read_sql_table(ticker + '_weekly', con=engine)
monthly_data = pd.read_sql_table(ticker + '_monthly', con=engine)
session.close()
"""

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

# Calculate signals
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