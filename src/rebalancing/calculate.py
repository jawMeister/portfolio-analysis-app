import pandas as pd
import numpy as np
import streamlit as st
from pypfopt import expected_returns, risk_models, EfficientFrontier
from stqdm import stqdm
import time

import warnings
# Ignore all FutureWarnings from pypfopt library
warnings.filterwarnings("ignore", category=FutureWarning, module="pypfopt")
warnings.filterwarnings("ignore", category=FutureWarning, module="cvxy")

import multiprocessing
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor

import logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s (%(levelname)s):  %(module)s.%(funcName)s - %(message)s')

# Set up logger for a specific module to a different level
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import src.utils as utils

def calculate_rebalance_segments(data, rebalance_period):
    # Group the data into segments, each covering the rebalance_period
    data_grouped = data.groupby(pd.Grouper(freq=f'{rebalance_period}ME'))

    # Split the grouped data into a list of dataframes
    data_segments = [group for name, group in data_grouped]

    return data_segments

def calculate_segment_return(i, data_segments, trailing_period, rebalance_portfolio_type, rebalance_returns_model_type, risk_level):
    if i >= trailing_period:
        trailing_data = pd.concat(data_segments[i-trailing_period:i])
    else:
        trailing_data = data_segments[i]

    logger.debug(f"Calculating segment return for segment {i} with head:\n{trailing_data.head()}\nand tail:\n{trailing_data.tail()}")


    mu = utils.calculate_mean_returns(trailing_data, rebalance_returns_model_type, st.session_state.risk_free_rate)
    S = utils.calculate_covariance_matrix(trailing_data)
    
    logger.debug(f"Calculating segment return for segment {i} with mean returns:\n{mu}\nand covariance matrix:\n{S}")
    
    ef_min_v = EfficientFrontier(mu, S)
    min_volatility_portfolio_weights = ef_min_v.min_volatility()
    min_volatility_portfolio = get_portfolio_performance(ef_min_v)
    
    ef_max_sharpe = EfficientFrontier(mu, S)
    max_sharpe_portfolio_weights = ef_max_sharpe.max_sharpe(risk_free_rate=st.session_state.risk_free_rate)
    max_sharpe_portfolio = get_portfolio_performance(ef_max_sharpe)
    
    if rebalance_portfolio_type == 'Maximum Sharpe Ratio':
        rebalanced_portfolio = max_sharpe_portfolio
    elif rebalance_portfolio_type == 'Minimum Volatility':
        rebalanced_portfolio = min_volatility_portfolio
    elif rebalance_portfolio_type == 'Selected Risk Level':
        # calculate risk extents for this data segment an ensure the risk level is within the extents
        logger.debug(f"Selected Risk Level: Risk extents for segment {i} are {min_volatility_portfolio['volatility']} and {max_sharpe_portfolio['volatility']}")
        
        # adjust the risk level if it is outside the extents
        if risk_level < min_volatility_portfolio['volatility']:
            rebalanced_portfolio = min_volatility_portfolio
        elif risk_level > max_sharpe_portfolio['volatility']:
            rebalanced_portfolio = max_sharpe_portfolio
        else:
            try:
                logger.debug(f"Risk level for segment {i} is {risk_level} which is between {min_volatility_portfolio['volatility']} and {max_sharpe_portfolio['volatility']}")
                ef_exact = EfficientFrontier(mu, S)
                ef_exact_weight = ef_exact.efficient_risk(risk_level)
    
                rebalanced_portfolio = get_portfolio_performance(ef_exact)  
                logger.debug(f"Rebalanced portfolio for segment {i} is {rebalanced_portfolio}")
            except Exception as e:
                logger.error(f"***Error finding exact portfolio for segment {i} with risk level {risk_level} between {min_volatility_portfolio['volatility']} and {max_sharpe_portfolio['volatility']}: {e}")
                logger.error(f"***Segment {i} mu {type(mu)}:\n{mu}\nS {type(S)}:\n{S}")
                
                # just use the max? or min? or sharpe around 1?
                rebalanced_portfolio = max_sharpe_portfolio
                logger.error(f"***Using max sharpe portfolio instead: {rebalanced_portfolio}")
    else: 
        #rebalance_portfolio_type == 'Balanced Portfolio (Sharpe Ratio = 1)':
        #solve for a range of portfolios and select the one with the sharpe ratio closest to 1
        efficient_portfolios = calculate_efficient_portfolios(mu, S, st.session_state.risk_free_rate)
        rebalanced_portfolio = utils.calculate_optimal_portfolio(efficient_portfolios)

    data_segments[i].ffill(inplace=True).dropna()    
    # Apply the optimal weights to the current segment and calculate the portfolio return for the segment
    segment_return = (data_segments[i].pct_change() * rebalanced_portfolio['weights']).sum(axis=1)
    
    return segment_return, rebalanced_portfolio['weights']
                
def optimize_and_apply_weights(data_segments, trailing_period, rebalance_portfolio_type, rebalance_returns_model_type, risk_level):
    n_cores = multiprocessing.cpu_count()
    rebalanced_portfolios = []    
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
                rebalanced_portfolios.append(optimal_weights)
                
            # Concatenate the segment returns to get the overall portfolio return
            portfolio_return = pd.concat(segment_returns)

    return portfolio_return, rebalanced_portfolios


def get_portfolio_performance(efficient_frontier):
    weights = efficient_frontier.clean_weights()
    ret, vol, sharpe = efficient_frontier.portfolio_performance(verbose=False, risk_free_rate=st.session_state.risk_free_rate)

    return {'weights': weights, 'portfolio_return': ret, 'volatility': vol, 'sharpe_ratio': sharpe}


# TODO: copied from utils to avoid blowing cache on the first page
def calculate_efficient_portfolios(mu, S, risk_free_rate, num_portfolios=500):
    min_risk, max_sharpe_ratio = utils.calculate_risk_extents(mu, S, risk_free_rate)
    
    if min_risk < 0 or max_sharpe_ratio < 0:
        logger.error(f"***Error calculating risk extents: min_risk {min_risk} and max_sharpe_ratio {max_sharpe_ratio}")
        logger.error(f"***mu {type(mu)}:\n{mu}\nS {type(S)}:\n{S}")
        raise Exception(f"***Error calculating risk extents: min_risk {min_risk} and max_sharpe_ratio {max_sharpe_ratio}")
    
    # TODO: see if 500 is necessary and/or make it a parameter
    risk_range = np.linspace(min_risk, max_sharpe_ratio, num_portfolios)
    
    efficient_portfolios = []
    for risk in risk_range:
        ef = EfficientFrontier(mu, S)
        ef.efficient_risk(risk)
        weights = ef.clean_weights()
        ret, vol, sharpe = ef.portfolio_performance()
        # TODO: make this efficient_portfolio dict a class / utility function
        # TODO: sometimes solver has issues, need to check that the weights add up to 1
        efficient_portfolios.append({
            'portfolio_return': ret,
            'volatility': vol,
            'sharpe_ratio': sharpe,
            'weights': weights
        })
    return efficient_portfolios