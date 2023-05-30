import pandas as pd
import streamlit as st
from pypfopt import expected_returns, risk_models, EfficientFrontier
from empyrical import sharpe_ratio, sortino_ratio
from stqdm import stqdm

import multiprocessing
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor

import logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s (%(levelname)s):  %(module)s.%(funcName)s - %(message)s')

# Set up logger for a specific module to a different level
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

import src.utils as utils

def calculate_rebalance_segments(data, rebalance_period):
    # Group the data into segments, each covering the rebalance_period
    data_grouped = data.groupby(pd.Grouper(freq=f'{rebalance_period}M'))

    # Split the grouped data into a list of dataframes
    data_segments = [group for name, group in data_grouped]

    return data_segments

def calculate_segment_return(i, data_segments, trailing_period):
    if i >= trailing_period:
        trailing_data = pd.concat(data_segments[i-trailing_period:i])
    else:
        trailing_data = data_segments[i]

    # Calculate the daily returns of the trailing data
    trailing_returns = trailing_data.pct_change().dropna()

    # Calculate the expected returns and the annualized sample covariance matrix of the trailing returns
    mu = utils.calculate_mean_returns(trailing_data, st.session_state.mean_returns_model, st.session_state.risk_free_rate)
    S = utils.calculate_covariance_matrix(trailing_data)

    # Calculate the efficient portfolios
    efficient_portfolios = utils.calculate_efficient_portfolios(mu, S, st.session_state.risk_free_rate)

    # Find the optimal portfolio
    optimal_portfolio = utils.calculate_optimal_portfolio(efficient_portfolios)

    # Apply the optimal weights to the current segment and calculate the portfolio return for the segment
    segment_return = (data_segments[i].pct_change() * optimal_portfolio['weights']).sum(axis=1)
    
    return segment_return, optimal_portfolio['weights']
                
def optimize_and_apply_weights(data_segments, trailing_period):

    n_cores = multiprocessing.cpu_count()
    optimal_portfolios = []    
    with st.spinner("Calculating rebalanced portfolio..."):
        with concurrent.futures.ProcessPoolExecutor(max_workers=min(len(data_segments), n_cores)) as executor:
            segment_returns = []

            futures = []
            for i in stqdm(range(len(data_segments))):
                # Extract the trailing data
                future = executor.submit(calculate_segment_return, i, data_segments, trailing_period)
                futures.append(future)
                
            for future in futures:
                segment_return, optimal_weights = future.result()
                segment_returns.append(segment_return)
                optimal_portfolios.append(optimal_weights)
                
            # Concatenate the segment returns to get the overall portfolio return
            portfolio_return = pd.concat(segment_returns)

    return portfolio_return, optimal_portfolios
