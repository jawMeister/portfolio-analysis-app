import pandas as pd
import streamlit as st
from pypfopt import expected_returns, risk_models, EfficientFrontier
from empyrical import sharpe_ratio, sortino_ratio
from stqdm import stqdm
import time

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

def calculate_segment_return(i, data_segments, trailing_period, rebalance_portfolio_type, rebalance_returns_model_type, risk_level):
    if i >= trailing_period:
        trailing_data = pd.concat(data_segments[i-trailing_period:i])
    else:
        trailing_data = data_segments[i]

    logger.debug(f"Calculating segment return for segment {i} with head:\n{trailing_data.head()}\nand tail:\n{trailing_data.tail()}")
                 
    # Calculate the expected returns and the annualized sample covariance matrix of the trailing returns
    mu = utils.calculate_mean_returns(trailing_data, rebalance_returns_model_type, st.session_state.risk_free_rate)
    S = utils.calculate_covariance_matrix(trailing_data)

    # consider tuning this to calculate a less dense set of efficient portfolios
    start_time = time.time()
    logger.debug(f"Calculating efficient portfolios for segment {i} with mu:\n{mu}\nand S:\n{S}")
    efficient_portfolios = utils.calculate_efficient_portfolios(mu, S, st.session_state.risk_free_rate)
    logger.debug(f"The calculating efficient portfolios takes {time.time() - start_time} seconds to run")

    # Find the optimal portfolio
    rebalanced_portfolio = get_portfolio_weights_for_rebalance(efficient_portfolios, rebalance_portfolio_type, mu, S, risk_level)

    # Apply the optimal weights to the current segment and calculate the portfolio return for the segment
    segment_return = (data_segments[i].pct_change() * rebalanced_portfolio['weights']).sum(axis=1)
    
    return segment_return, rebalanced_portfolio['weights']
                
def optimize_and_apply_weights(data_segments, trailing_period, rebalance_portfolio_type, rebalance_returns_model_type, risk_level):
    n_cores = multiprocessing.cpu_count()
    optimal_portfolios = []    
    with st.spinner("Calculating rebalanced portfolio..."):
        with concurrent.futures.ProcessPoolExecutor(max_workers=min(len(data_segments), n_cores)) as executor:
            segment_returns = []

            futures = []
            for i in stqdm(range(len(data_segments))):
                # Extract the trailing data
                future = executor.submit(calculate_segment_return, i, data_segments, trailing_period, rebalance_portfolio_type, rebalance_returns_model_type, risk_level)
                futures.append(future)
                
            for future in futures:
                segment_return, optimal_weights = future.result()
                segment_returns.append(segment_return)
                optimal_portfolios.append(optimal_weights)
                
            # Concatenate the segment returns to get the overall portfolio return
            portfolio_return = pd.concat(segment_returns)

    return portfolio_return, optimal_portfolios

def get_portfolio_weights_for_rebalance(efficient_portfolios, rebalance_portfolio_type, mu, S, risk_level):
    
    if rebalance_portfolio_type == 'Minimum Volatility':
        rebalanced_portfolio = efficient_portfolios[0]
    elif rebalance_portfolio_type == 'Maximum Sharpe Ratio':
        rebalanced_portfolio = efficient_portfolios[-1]
    elif rebalance_portfolio_type == 'Optimal Portfolio (Sharpe Ratio = 1)':
        rebalanced_portfolio = utils.calculate_optimal_portfolio(efficient_portfolios)
    else:
        # could be that risk_level selected doesn't map to the risk_level of the efficient portfolios
        # for this data segment, so need to calculate risk extents each time and if the risk_level
        # is less than min or greater than max, then need to adjust the risk_level to be within the
        
        min_risk, max_risk = efficient_portfolios[0]['volatility'], efficient_portfolios[-1]['volatility']
        
        if risk_level < min_risk:
            risk_level = min_risk
        elif risk_level > max_risk:
            risk_level = max_risk
        
        start_time = time.time()
        rebalanced_portfolio = find_exact_portfolio(mu, S, risk_level)
        logger.debug(f"The calculating exact portfolio takes {time.time() - start_time} seconds to run")    
        
        #start_time = time.time()
        #rebalanced_portfolio = find_closest_portfolio(efficient_portfolios, rebalanced_portfolio['volatility'])
        #logger.info(f"The calculating closest portfolio takes {time.time() - start_time} seconds to run")
        
    return rebalanced_portfolio

def find_closest_portfolio(efficient_portfolios, target_volatility):
    low, high = 0, len(efficient_portfolios) - 1

    while low < high:
        mid = (low + high) // 2
        if efficient_portfolios[mid]['volatility'] < target_volatility:
            low = mid + 1
        else:
            high = mid

    # After the loop, low should be the first portfolio whose volatility is >= target_volatility.
    # The closest portfolio is either this portfolio or the one before it.
    if low == 0:
        return 0
    elif abs(efficient_portfolios[low - 1]['volatility'] - target_volatility) < abs(efficient_portfolios[low]['volatility'] - target_volatility):
        return low - 1
    else:
        return low
    
def find_exact_portfolio(mu, S, risk_level):
    ef = EfficientFrontier(mu, S)
    ef.efficient_risk(risk_level)
    weights = ef.clean_weights()
    ret, vol, sharpe = ef.portfolio_performance()
    # TODO: make this dict a class / utility function
    portfolio = {'weights': weights, 'portfolio_return': ret, 'volatility': vol, 'sharpe_ratio': sharpe}
    
    return portfolio