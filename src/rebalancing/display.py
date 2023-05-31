import streamlit as st
from datetime import datetime, timedelta
import time

import src.utils as utils
import src.rebalancing.calculate as calculate
import src.rebalancing.plot as plot
from src.portfolio.calculate import calculate_portfolio_performance

import logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s (%(levelname)s):  %(module)s.%(funcName)s - %(message)s')

# Set up logger for a specific module to a different level
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# TODO: recalculate these if the user changes the start or end date
def initialize_inputs():
    logger.debug("Initializing inputs")
    if 'rebalancing_tab_initialized' not in st.session_state:
        st.session_state.rebalancing_tab_initialized = False
        
    if not st.session_state.rebalancing_tab_initialized:
        if "rebalance_time_span" not in st.session_state:
            time_span = st.session_state.end_date - st.session_state.start_date
            # number of months between start and end date from the sidebar
            time_span = time_span / timedelta(weeks=4)
            st.session_state.rebalance_time_span = time_span
            logger.debug(f"Time span: {time_span}")
            
        if 'rebalancing_period' not in st.session_state:
            st.session_state.rebalancing_period = max(int(time_span / 10), 1) 
            
        if 'trailing_period' not in st.session_state:
            st.session_state.trailing_period = max(int(time_span / 5), 1) 
            
        if 'original_v_rebalanced_plot' not in st.session_state:
            st.session_state.original_v_rebalanced_plot = None
            
        if 'rebalanced_portfolios_plot' not in st.session_state:
            st.session_state.rebalanced_portfolios_plot = None
            
        if 'rebalanced_ticker_weights_plot' not in st.session_state:
            st.session_state.ticker_weights_plot = None
            
        if 'rebalance_portfolio_type' not in st.session_state:
            st.session_state.rebalance_type = 'Selected Portfolio'
        
        logger.debug(f"init: rebalance portfolio type in session: {'rebalance_returns_model_type' in st.session_state}")
        if 'rebalance_returns_model_type' not in st.session_state:
            st.session_state.rebalance_returns_model_type = st.session_state.mean_returns_model
            logger.debug(f"not in session, set to: rebalance returns model type: {st.session_state.rebalance_returns_model_type}")
        logger.debug(f"init: rebalance returns model type: {st.session_state.rebalance_returns_model_type}")
        
        if 'rebalance_risk_level' not in st.session_state:
            st.session_state.rebalance_risk_level = st.session_state.risk_level
            
        if 'rebalance_min_risk' not in st.session_state:
            st.session_state.rebalance_min_risk = st.session_state.min_risk
            
        if 'rebalance_max_risk' not in st.session_state:
            st.session_state.rebalance_max_risk = st.session_state.max_risk
        
        st.session_state.rebalancing_tab_initialized = True
        logger.debug("Inputs initialized")
        
def recalculate_risk_extents():
    logger.debug("Recalculating risk extents")
    mu = utils.calculate_mean_returns(st.session_state.stock_data, st.session_state.rebalance_returns_model_type, st.session_state.risk_free_rate)
    S = utils.calculate_covariance_matrix(st.session_state.stock_data)
    
    min_risk, max_risk = utils.calculate_risk_extents(mu, S, st.session_state.risk_free_rate)
    
    # if the risk level is outside the new extents, set it to the min or max
    if st.session_state.rebalance_risk_level < min_risk:
        logger.info(f"Risk level {st.session_state.rebalance_risk_level} is below the minimum risk {min_risk}, setting to min risk")
        st.session_state.rebalance_risk_level = float(min_risk)
    elif st.session_state.rebalance_risk_level > max_risk:
        logger.info(f"Risk level {st.session_state.rebalance_risk_level} is above the maximum risk {max_risk}, setting to max risk")
        st.session_state.rebalance_risk_level = float(max_risk)
        
            
    st.session_state.rebalance_min_risk = float(min_risk)
    st.session_state.rebalance_max_risk = float(max_risk)

def display_rebalancing_analysis(portfolio_summary):
    initialize_inputs()

    with st.container():
        col1, col2 = st.columns([1, 3])
        with col1:
            # TODO: style this like a form
            with st.form("Rebalance Variables"):  # wish could use form, altho no calbacks within form to update risk extents
                st.slider('Rebalancing Period (months)', min_value=1, max_value=int(st.session_state.rebalance_time_span/2)-1, step=1, key='rebalancing_period')
                st.slider('Trailing Period (months)', min_value=1, max_value=int(st.session_state.rebalance_time_span/2)-1, step=1, key='trailing_period')
                
                subcol1, subcol2 = st.columns([1, 1])
                with subcol1:
                    st.radio('Rebalance Strategy (from Efficient Frontier)', utils.get_efficient_frontier_models(), key='rebalance_portfolio_type')
                with subcol2:
                    st.radio('Rebalance Returns Model', utils.get_mean_returns_models(), key='rebalance_returns_model_type')
                    
                # unfortunately, can't use callbacks on the radio button from within a form, so have to calculate the risk extents every time
                recalculate_risk_extents()

                st.slider('Rebalance Risk Level', min_value=st.session_state.rebalance_min_risk, max_value=st.session_state.rebalance_max_risk, step=0.01, key='rebalance_risk_level', format="%.2f")
                rebalance = st.form_submit_button('Calculate', use_container_width=True)
                       
            st.caption("Calculates affects of selected rebalancing period based on optimal portfolio calculation leveraging trailing months data. " + \
                        "Other rebalancing approaches (e.g., macro, technical inflection points) along with simulations to optimize rebalancing " + \
                        "period and trailing period in progress.")
            
        if rebalance:
            # split the historical data into rebalancing periods
            start_time = time.time()
            
            # unfortunately, can't use callbacks on the mean returns radio button from within a form, so have to calculate the risk extents every time
            # to ensure the risk level is within the extents as calculated by the mean returns model. if we don't use a form, the main app page will
            # refresh every time we change the radio button, which is annoying
            recalculate_risk_extents()
                
            logger.debug("Calculating rebalancing segments at: " + str(start_time))
            rebalance_segments = calculate.calculate_rebalance_segments(st.session_state.stock_data, 
                                                                        st.session_state.rebalancing_period)
            # do the math!
            logger.debug("Rebalanced segments calculated in " + str(time.time() - start_time) + " seconds")
            rebalanced_returns, optimal_portfolios = calculate.optimize_and_apply_weights(rebalance_segments, 
                                                                                          st.session_state.trailing_period,
                                                                                          st.session_state.rebalance_portfolio_type,
                                                                                          st.session_state.rebalance_returns_model_type,
                                                                                          st.session_state.rebalance_risk_level)
            logger.debug("Calculating portfolio performance took " + str(time.time() - start_time) + " seconds")
            df_dict = calculate_portfolio_performance(portfolio_summary["stock_data"], portfolio_summary['dividend_data'],
                                                        portfolio_summary['weights'], portfolio_summary['start_date'],
                                                        portfolio_summary['end_date'])
            
            original_returns = df_dict['df_weighted_portfolio_returns']
            logger.debug("Plotting portfolio performance took " + str(time.time() - start_time) + " seconds")
            original_v_rebalanced_plot = plot.plot_cumulative_returns(original_returns, rebalanced_returns)
            st.session_state.original_v_rebalanced_plot = original_v_rebalanced_plot
            
            rebalanced_ticker_weights_plot, rebalanced_portfolios_plot = plot.plot_optimal_portfolios_over_time(optimal_portfolios, portfolio_summary['tickers'])
            st.session_state.rebalanced_portfolios_plot = rebalanced_portfolios_plot
            st.session_state.rebalanced_ticker_weights_plot = rebalanced_ticker_weights_plot
            logger.debug("Displaying portfolio performance took " + str(time.time() - start_time) + " seconds")

    with st.container():
        col1, col2, col3 = st.columns([1, 1, 1])
        if 'original_v_rebalanced_plot' in st.session_state and 'rebalanced_portfolios_plot' in st.session_state and 'rebalanced_ticker_weights_plot' in st.session_state:
            with col1:
                if st.session_state.original_v_rebalanced_plot is not None:
                    st.plotly_chart(st.session_state.original_v_rebalanced_plot, use_container_width=True)
            with col2:
                if st.session_state.rebalanced_portfolios_plot is not None:
                    st.plotly_chart(st.session_state.rebalanced_portfolios_plot, use_container_width=True)
            with col3:
                if st.session_state.rebalanced_ticker_weights_plot is not None:
                    st.plotly_chart(st.session_state.rebalanced_ticker_weights_plot, use_container_width=True)
