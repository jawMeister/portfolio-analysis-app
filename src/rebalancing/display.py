import streamlit as st

import src.rebalancing.calculate as calculate
import src.rebalancing.plot as plot
from src.portfolio.calculate import calculate_portfolio_performance

import logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s (%(levelname)s):  %(module)s.%(funcName)s - %(message)s')

# Set up logger for a specific module to a different level
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def initialize_inputs():
    if 'rebalancing_period' not in st.session_state:
        st.session_state.rebalancing_period = 12
        
    if 'trailing_period' not in st.session_state:
        st.session_state.trailing_period = 3
        
    if 'original_v_rebalanced_plot' not in st.session_state:
        st.session_state.original_v_rebalanced_plot = None
        
    if 'optimal_portfolios' not in st.session_state:
        st.session_state.optimal_portfolios_plot = None
        
    if 'optimal_ticker_weights' not in st.session_state:
        st.session_state.ticker_weights_plot = None

def display_rebalancing_analysis(portfolio_summary):
    with st.container():
        col1, col2 = st.columns([1, 3])
        with col1:
            with st.form("rebalancing_analysis_form"):
                st.slider('Rebalancing Period (months)', 1, 60, 12, 1, key='rebalancing_period')
                st.slider('Trailing Period (months)', 1, 60, 3, 1, key='trailing_period')
                rebalance = st.form_submit_button('Calculate')        
            st.caption("Calculates affects of selected rebalancing period based on optimal portfolio calculation leveraging trailing months data. " + \
                        "Other rebalancing approaches (e.g., macro, technical inflection points) along with simulations to optimize rebalancing " + \
                        "period and trailing period in progress.")
            
        if rebalance:
            rebalance_segments = calculate.calculate_rebalance_segments(portfolio_summary["stock_data"], 
                                                                        st.session_state.rebalancing_period)
            rebalanced_returns, optimal_portfolios = calculate.optimize_and_apply_weights(rebalance_segments, st.session_state.trailing_period)
            df_dict = calculate_portfolio_performance(portfolio_summary["stock_data"], portfolio_summary['dividend_data'],
                                                        portfolio_summary['weights'], portfolio_summary['start_date'],
                                                        portfolio_summary['end_date'])
            original_returns = df_dict['df_weighted_portfolio_returns']
            
            original_v_rebalanced_plot = plot.plot_cumulative_returns(original_returns, rebalanced_returns)
            st.session_state.original_v_rebalanced_plot = original_v_rebalanced_plot
            
            optimal_ticker_weights_plot, optimal_portfolios_plot = plot.plot_optimal_portfolios_over_time(optimal_portfolios, portfolio_summary['tickers'])
            st.session_state.optimal_portfolios_plot = optimal_portfolios_plot
            st.session_state.optimal_ticker_weights_plot = optimal_ticker_weights_plot

    with st.container():
        col1, col2, col3 = st.columns([1, 1, 1])
        if 'original_v_rebalanced_plot' in st.session_state and 'optimal_portfolios_plot' in st.session_state and 'optimal_ticker_weights_plot' in st.session_state:
            with col1:
                st.plotly_chart(st.session_state.original_v_rebalanced_plot, use_container_width=True)
            with col2:
                st.plotly_chart(st.session_state.optimal_portfolios_plot, use_container_width=True)
            with col3:
                st.plotly_chart(st.session_state.optimal_ticker_weights_plot, use_container_width=True)
