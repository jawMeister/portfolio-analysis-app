import streamlit as st

import src.technical.calculate as calculate
import src.technical.plot as plot

def display_technical_analysis(portfolio_summary):
    # only pull technicals for the stocks that are in the portfolio
    ticker_weights_sorted = portfolio_summary['weights'][portfolio_summary['weights'] > 0].sort_values(ascending=False)
    
    for ticker in ticker_weights_sorted.index:
        with st.container():
            with st.expander(f"{ticker} Technical Analysis", expanded=False):
                with st.spinner(f"Loading {ticker} data..."):
                    daily_data, weekly_data, monthly_data = calculate.get_ticker_data(ticker, portfolio_summary['start_date'], portfolio_summary['end_date'])

                daily_data = calculate.calculate_indicators(daily_data)
                weekly_data = calculate.calculate_indicators(weekly_data)
                monthly_data = calculate.calculate_indicators(monthly_data)

                daily_signals = calculate.calculate_signals(daily_data)
                weekly_signals = calculate.calculate_signals(weekly_data)
                monthly_signals = calculate.calculate_signals(monthly_data)
                
                col1, col2, col3 = st.columns(3, gap="large")
                with col1:
                    upper, lower = plot.plot_indicators(monthly_data, ticker, "Monthly")
                    st.plotly_chart(upper, use_container_width=True)
                    display_technical_signals(monthly_signals)
                    st.plotly_chart(lower, use_container_width=True)
                with col2:
                    upper, lower = plot.plot_indicators(weekly_data, ticker, "Weekly")
                    st.plotly_chart(upper, use_container_width=True)
                    display_technical_signals(weekly_signals)
                    st.plotly_chart(lower, use_container_width=True)
                with col3:
                    upper, lower = plot.plot_indicators(daily_data, ticker, "Daily")
                    st.plotly_chart(upper, use_container_width=True)
                    display_technical_signals(daily_signals)
                    st.plotly_chart(lower, use_container_width=True)
                
                st.markdown("<hr style='color:#FF4B4B;'>", unsafe_allow_html=True)
            
    return daily_data, weekly_data, monthly_data

def display_technical_signals(df):
    # Calculate deltas
    df['sma_delta'] = df['sma_20'].diff()
    df['ema_delta'] = df['ema_21'].diff()
    df['rsi_delta'] = df['rsi'].diff()
    df['stoch_delta'] = df['stoch'].diff()
    df['bb_delta'] = df['bb_avg'].diff()
    df['macd_delta'] = df['macd_line'].diff() - df['signal_line'].diff()

    # Get the latest values for each indicator
    latest_sma_signal = df['sma_signal'].iloc[-1]
    latest_ema_signal = df['ema_signal'].iloc[-1]
    latest_rsi_signal = df['rsi_signal'].iloc[-1]
    latest_stoch_signal = df['stoch_signal'].iloc[-1]
    latest_bb_signal = df['bb_signal'].iloc[-1]
    latest_macd_signal = df['macd_signal'].iloc[-1]

    # Get the latest deltas for each indicator
    latest_sma_delta = df['sma_delta'].iloc[-1]
    latest_ema_delta = df['ema_delta'].iloc[-1]
    latest_rsi_delta = df['rsi_delta'].iloc[-1]
    latest_stoch_delta = df['stoch_delta'].iloc[-1]
    latest_bb_delta = df['bb_delta'].iloc[-1]
    latest_macd_delta = df['macd_delta'].iloc[-1]

    # Display the metrics using Streamlit's st.metric element
    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="SMA Signal > 20", value=latest_sma_signal, delta=latest_sma_delta)
            st.metric(label="EMA Signal > 55", value=latest_ema_signal, delta=latest_ema_delta)
        with col2:
            st.metric(label="RSI Signal", value=latest_rsi_signal, delta=latest_rsi_delta)
            st.metric(label="Stochastic Signal", value=latest_stoch_signal, delta=latest_stoch_delta)
        with col3:
            st.metric(label="Bollinger Bands Signal", value=latest_bb_signal, delta=latest_bb_delta)
            st.metric(label="MACD Signal", value=latest_macd_signal, delta=latest_macd_delta)