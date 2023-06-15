import streamlit as st

import logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s (%(levelname)s):  %(module)s.%(funcName)s - %(message)s')

# Set up logger for a specific module to a different level
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import src.utils as utils
import src.technical.calculate as calculate
import src.technical.plot as plot

def initialize_technical_analysis_inputs(portfolio_summary):
    # Let's use ticker_weights_sorted as a key for our session state
    cur_ticker_weights_sorted = portfolio_summary['weights'][portfolio_summary['weights'] > 0].sort_values(ascending=False)

    st.session_state.setdefault('technical_start_date', None)
    st.session_state.setdefault('technical_end_date', None)
    st.session_state.setdefault('technical_weighted_ticker_list', cur_ticker_weights_sorted.index.tolist())

    for ticker in cur_ticker_weights_sorted.index:
        st.session_state.setdefault(f'tech_{ticker}_daily_data', None)
        st.session_state.setdefault(f'tech_{ticker}_weekly_data', None)
        st.session_state.setdefault(f'tech_{ticker}_monthly_data', None)

        st.session_state.setdefault(f'tech_{ticker}_daily_signals', None)
        st.session_state.setdefault(f'tech_{ticker}_weekly_signals', None)
        st.session_state.setdefault(f'tech_{ticker}_monthly_signals', None)

def update_technical_calculations(ticker):
    logger.info(f"Updating technical calculations for {ticker}")
    with st.spinner(f"Loading {ticker} data..."):
        logger.debug(f"updating calculations for {ticker} data...")
        st.session_state[f'tech_{ticker}_daily_data'], st.session_state[f'tech_{ticker}_weekly_data'], st.session_state[f'tech_{ticker}_monthly_data'] = utils.get_ticker_data(ticker, st.session_state['start_date'], st.session_state['end_date'])
        logger.debug(f'updated {ticker} data...daily: {st.session_state[f"tech_{ticker}_daily_data"].shape}, weekly: {st.session_state[f"tech_{ticker}_weekly_data"].shape}, monthly: {st.session_state[f"tech_{ticker}_monthly_data"].shape}')

    st.session_state[f'tech_{ticker}_daily_data'] = calculate.calculate_indicators(st.session_state[f'tech_{ticker}_daily_data'])
    st.session_state[f'tech_{ticker}_weekly_data'] = calculate.calculate_indicators(st.session_state[f'tech_{ticker}_weekly_data'])
    st.session_state[f'tech_{ticker}_monthly_data'] = calculate.calculate_indicators(st.session_state[f'tech_{ticker}_monthly_data'])

    st.session_state[f'tech_{ticker}_daily_signals'] = calculate.calculate_signals(st.session_state[f'tech_{ticker}_daily_data'])
    st.session_state[f'tech_{ticker}_weekly_signals'] = calculate.calculate_signals(st.session_state[f'tech_{ticker}_weekly_data'])
    st.session_state[f'tech_{ticker}_monthly_signals'] = calculate.calculate_signals(st.session_state[f'tech_{ticker}_monthly_data'])

def display_technical_analysis(portfolio_summary):
    initialize_technical_analysis_inputs(portfolio_summary)

    # get the current ticker weights sorted - which may be different than what is in session state for this tab
    cur_ticker_weights_sorted = portfolio_summary['weights'][portfolio_summary['weights'] > 0].sort_values(ascending=False)

    for ticker in cur_ticker_weights_sorted.index:
        with st.container():
            with st.expander(f"{ticker} Technical Analysis", expanded=False):
                
                # Check if ticker data needs to be updated based on date or ticker changes
                if (st.session_state['technical_start_date'] != st.session_state['start_date']) or \
                    (st.session_state['technical_end_date'] != st.session_state['end_date']) or \
                    (ticker not in st.session_state['technical_weighted_ticker_list']):
                    
                    update_technical_calculations(ticker)

                daily_data = st.session_state[f'tech_{ticker}_daily_data']
                weekly_data = st.session_state[f'tech_{ticker}_weekly_data']
                monthly_data = st.session_state[f'tech_{ticker}_monthly_data']

                daily_signals = st.session_state[f'tech_{ticker}_daily_signals']
                weekly_signals = st.session_state[f'tech_{ticker}_weekly_signals']
                monthly_signals = st.session_state[f'tech_{ticker}_monthly_signals']

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
            
    # after every run, update the technical tab session state to the latest values
    st.session_state['technical_start_date'] = st.session_state['start_date']
    st.session_state['technical_end_date'] = st.session_state['end_date']
    st.session_state['technical_weighted_ticker_list'] = cur_ticker_weights_sorted.index.tolist()
            
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