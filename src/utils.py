import os
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from pypfopt import expected_returns, risk_models, EfficientFrontier
from empyrical import sharpe_ratio, sortino_ratio
from datetime import datetime, date, timedelta
import streamlit as st
import yfinance as yf
from fredapi import Fred

import traceback
import logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s (%(levelname)s):  %(module)s.%(funcName)s - %(message)s')

# Set up logger for a specific module to a different level
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

import src.session as session

@st.cache_data(persist=True)
def get_dividend_data(tickers, start_date, end_date):
    dividend_data = {}

    logger.info(f"Getting dividend data for {tickers} from {start_date} to {end_date}")
    logger.debug("\n".join(traceback.format_stack()))   
    
    # Get yesterday's date - streamlit cache is initializing the function with today's date
    # which is resulting in errors in downloading stock data as dividends are not available
    yesterday = datetime.now() - timedelta(days=1)
    yesterday = yesterday.date()  # Convert datetime to date
    
    two_days_ago = datetime.now() - timedelta(days=2)
    two_days_ago = two_days_ago.date()  # Convert datetime to date

    # Convert start_date and end_date to date objects if they are string
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d').date()

    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
    
    logger.debug(f'start_date: {start_date}, end_date: {end_date}, yesterday: {yesterday}')
    # If start_date or end_date are past yesterday, set them to yesterday
    if start_date > yesterday:
        logger.debug(f'start_date ({start_date}) is greater than yesterday ({yesterday}), setting start_date to two days ago')
        start_date = two_days_ago
    if end_date > yesterday:
        logger.debug(f'end_date ({end_date}) is greater than yesterday ({yesterday}), setting end_date to yesterday')
        end_date = yesterday

    logger.debug(f'Pulling {tickers} stock history from start_date: {start_date} to end_date: {end_date}')
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        stock_data = stock.history(start=start_date, end=end_date)
        if 'Dividends' not in stock_data.columns:
            dividend_data[ticker] = pd.Series(np.zeros(len(stock_data)), index=stock_data.index)
        else:
            dividend_data[ticker] = stock_data['Dividends']

    return pd.DataFrame(dividend_data)

@st.cache_data(persist=True)
def get_stock_and_dividend_data(tickers, start_date, end_date):
    # Get yesterday's date
    yesterday = datetime.now() - timedelta(days=1)
    yesterday = yesterday.date()  # Convert datetime to date
    
    two_days_ago = datetime.now() - timedelta(days=2)
    two_days_ago = two_days_ago.date()  # Convert datetime to date

    # Convert start_date and end_date to date objects if they are string or datetime
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
    elif isinstance(start_date, datetime):
        start_date = start_date.date()

    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
    elif isinstance(end_date, datetime):
        end_date = end_date.date()
    
    logger.debug(f'start_date: {start_date}, end_date: {end_date}, yesterday: {yesterday}')
    # If start_date or end_date are past yesterday, set them to yesterday
    if start_date > yesterday:
        logger.debug(f'start_date ({start_date}) is greater than yesterday ({yesterday}), setting start_date to two days ago')
        start_date = two_days_ago
    if end_date > yesterday:
        logger.debug(f'end_date ({end_date}) is greater than yesterday ({yesterday}), setting end_date to yesterday')
        end_date = yesterday
        
    logger.info(f"Getting stock and dividend data for {tickers} from {start_date} to {end_date}")
    logger.debug("\n".join(traceback.format_stack()[-2:]))    
    
    data = yf.download(tickers, start=start_date, end=end_date)
    
    stock_data = data["Adj Close"]
    dividend_data = get_dividend_data(tickers, start_date, end_date)

    return stock_data, dividend_data.fillna(0)

@st.cache_data(persist=True)
def get_stock_data(tickers, start_date, end_date):
    if isinstance(start_date, (datetime, date)):
        start_date = start_date.strftime('%Y-%m-%d')
    if isinstance(end_date, (datetime, date)):
        end_date = end_date.strftime('%Y-%m-%d')
    if not isinstance(start_date, str) or not isinstance(end_date, str):
        raise TypeError(f"start_date ({type(start_date)}) and end_date ({type(end_date)}) must be either strings or datetime objects.")
    
    logger.info(f"Getting stock data for {tickers} from {start_date} to {end_date}")
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    return data

def calculate_dividend_yield(stock_data, dividend_data):
    # Calculate the annual dividends and stock prices
    annual_dividends = dividend_data.resample('Y').sum()
    annual_prices = stock_data.resample('Y').mean()

    # Convert to tz-naive
    annual_dividends.index = annual_dividends.index.tz_localize(None)
    annual_prices.index = annual_prices.index.tz_localize(None)

    # Calculate the dividend yield
    dividend_yield = annual_dividends / annual_prices

    return dividend_yield.mean()

def calculate_weighted_dividend_yield(stock_data, dividend_data, span=3):
    # Calculate the annual dividends and stock prices
    annual_dividends = dividend_data.resample('Y').sum()
    annual_prices = stock_data.resample('Y').mean()

    # Convert to tz-naive
    annual_dividends.index = annual_dividends.index.tz_localize(None)
    annual_prices.index = annual_prices.index.tz_localize(None)

    # Calculate the dividend yield
    dividend_yield = annual_dividends / annual_prices

    # Use EMA for average to weight more recent data - eg, last 3 years
    dividend_yield_ema = dividend_yield.ewm(span=span).mean()

    return dividend_yield_ema.iloc[-1]

@st.cache_data(persist=True)
def get_sp500_daily_returns(start_date, end_date):
    sp500 = yf.download('^GSPC', start=start_date, end=end_date)['Adj Close'].pct_change()
    return sp500

@st.cache_data(persist=True)
def get_ticker_data(ticker, start_date, end_date):
    logger.info(f"Fetching data for {ticker} from {start_date} to {end_date}")
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


def calculate_mean_returns(stock_data, mean_returns_model, risk_free_rate):
    
    if mean_returns_model == "Historical Returns (Geometric Mean)":
        mu = expected_returns.mean_historical_return(stock_data)
    elif mean_returns_model == "Historical Weighted w/Recent Data":
        mu = expected_returns.ema_historical_return(stock_data)
    elif mean_returns_model == "Capital Asset Pricing Model (CAPM)":
        mu = expected_returns.capm_return(stock_data, risk_free_rate=risk_free_rate)
        
    return mu
            
def update_forecast_info(forecasted_stock_info, ticker, current_asset_price, dividend_yield, yearly_dividend, current_asset_shares, current_asset_value, year):
    forecasted_stock_info.loc[year, 'Year'] = year
    forecasted_stock_info.loc[year, 'Stock'] = ticker
    forecasted_stock_info.loc[year, 'Stock Price'] = current_asset_price
    forecasted_stock_info.loc[year, 'Dividend Yield'] = dividend_yield
    forecasted_stock_info.loc[year, 'Dividend'] = yearly_dividend
    forecasted_stock_info.loc[year, 'Shares'] = current_asset_shares
    forecasted_stock_info.loc[year, 'Total Value'] = current_asset_value
    
    return forecasted_stock_info

def calculate_weight_error_factor(weights):
    # small floating point error with the weights, so we need to adjust the weights to make sure it sums to 1
    return (1 - weights.sum()) / np.count_nonzero(weights)

def get_current_price(stock_data, ticker):
    return stock_data.loc[stock_data[ticker].last_valid_index(), ticker]

def calculate_future_asset_holdings(portfolio_summary):                
    # dictionary of ticker with dataframe detail of the holdings
    future_asset_holdings_detail = {}
            
    # setup a blank dataframe to store the summary detail by ticker, by year    
    yearly_columns = ['Year'] + portfolio_summary["tickers"] + ['Total Dividends', 'Total Asset Value']
    yearly_data_df = pd.DataFrame(index=range(portfolio_summary["years"] + 1), columns=yearly_columns) # +1 for the initial investment year
    yearly_data_df = yearly_data_df.fillna(0)
    
    # small floating point error with the weights, so we need to adjust the weights to make sure it sums to 1
    weight_error_factor = calculate_weight_error_factor(portfolio_summary["weights"])

    for ticker in portfolio_summary["tickers"]:
        # only do calculations for tickers with an investment allocation
        if (portfolio_summary["weights"][ticker] > 0):
            # store the forecasted detail data for each asset in a detail dataframe
            asset_forecast_details_df = pd.DataFrame(index=range(portfolio_summary["years"] + 1), columns=['Year', 'Value', 'Shares', 'Price per Share', 'Dividend Yield (%)', 'Dividends per Share', 'Dividends'])
            
            weight = portfolio_summary["weights"][ticker] + weight_error_factor
            yearly_return = (1 + portfolio_summary["mu"][ticker])
            future_value = portfolio_summary["initial_investment"] * weight # initialize to the initial investment x weight
            starting_price = get_current_price(portfolio_summary["stock_data"], ticker) # initialize to most recent price from yfinance
            shares = future_value / starting_price # initialize investment amount / most recent price
            dividend_yield = calculate_weighted_dividend_yield(portfolio_summary["stock_data"][ticker], portfolio_summary["dividend_data"][ticker])

            # initial price to the most recent price
            future_price = starting_price
            row_name =  "Initial"
            future_dividends_per_share = 0
            # for every year in the forecast, calculate the future value of the asset including reinvestment of dividends
            for year in range(portfolio_summary["years"] + 1):
                # start buying shares and calculating dividends after the first year
                if year > 0:
                    row_name = f"Year {year}"
                    shares += portfolio_summary['yearly_contribution'] * weight / future_price
                    future_price *= yearly_return  # estimate future price based on calcuated returns
                    future_dividends_per_share = future_price * dividend_yield  

                # calculate dividends, then reinvest in same stock purchase
                reinvestment = future_dividends_per_share * shares
                shares += reinvestment / future_price
                
                # calculate future value of this asset based on the new shares and price
                future_value = shares * future_price

                asset_forecast_details_df.loc[year, 'Value'] = future_value
                asset_forecast_details_df.loc[year, 'Shares'] = shares
                asset_forecast_details_df.loc[year, 'Price per Share'] = future_price
                asset_forecast_details_df.loc[year, 'Dividend Yield (%)'] = dividend_yield
                asset_forecast_details_df.loc[year, 'Dividends per Share'] = future_dividends_per_share
                asset_forecast_details_df.loc[year, 'Dividends'] = reinvestment
                future_asset_holdings_detail[ticker] = asset_forecast_details_df
                
                # update the summary dataframe with the calculated values
                yearly_data_df.loc[year, 'Year'] = row_name
                yearly_data_df.loc[year, ticker] = future_value
                yearly_data_df.loc[year, 'Total Dividends'] += reinvestment
                yearly_data_df.loc[year, 'Total Asset Value'] += future_value
                       
    # add summary columns
    yearly_data_df['YoY Asset Growth (%)'] = yearly_data_df['Total Asset Value'].pct_change()
    yearly_data_df['Total Growth (%)'] = yearly_data_df['Total Asset Value'] / portfolio_summary["initial_investment"] - 1
        
    yearly_data_df = yearly_data_df.fillna(0)
    
    """ uncomment to export the data to excel
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f'portfolio_projection_{timestamp}.xlsx'
    

    full_path = os.path.join('reports', file_name)
    with pd.ExcelWriter(full_path) as writer:
        yearly_data_df.to_excel(writer, sheet_name='Summary', index=False)
        for ticker in future_asset_holdings_detail.keys():
            future_asset_holdings_detail[ticker].to_excel(writer, sheet_name=ticker, index=False)
        # add more lines here if want to export other tables
    """
    
    return yearly_data_df, future_asset_holdings_detail


def calculate_portfolio_df(stock_data, dividend_data, mu, S, start_date, end_date, risk_level, initial_investment, yearly_contribution, years, risk_free_rate):
    
    # Solve for the efficient portfolio at the desired risk level
    # reset the EfficientFrontier instance to use the cleaned weights
    ef = EfficientFrontier(mu, S)
    ef.efficient_risk(risk_level)
        
    simple_returns = stock_data.pct_change()
        
    # cumulative returns
    cumulative_daily_returns = (1 + stock_data.pct_change()).cumprod() - 1
    cumulative_daily_returns.reset_index(inplace=True)
    


    # Obtain weights, return, volatility, and Sharpe ratio
    weights = ef.clean_weights()
    weights = pd.Series(weights).reindex(simple_returns.columns)
    portfolio_expected_return, volatility, sharpe_ratio = ef.portfolio_performance()
    
    weighted_portfolio_returns = stock_data.pct_change().multiply(weights, axis=1).sum(axis=1)
    cumulative_weighted_portfolio_returns = (1 + weighted_portfolio_returns).cumprod() - 1
   
    treynor_ratio = calculate_treynor_ratio(simple_returns, weights, start_date, end_date, risk_free_rate)
    
    # TODO: this sortino may be incorrect as the dot likely nulling out assets with NaN (eg, BTC-USD prior to 2015)
    weighted_returns = simple_returns.dot(weights)
    sortino_ratio_val = sortino_ratio(weighted_returns)
    alpha = 0.05
    cvar = -np.nanpercentile(weighted_returns[weighted_returns < 0], alpha * 100)
    
    portfolio_summary = {"stock_data": stock_data, 
                         "dividend_data": dividend_data, 
                         "mu": mu, 
                         "S": S, 
                         "simple_returns_by_ticker": simple_returns,
                         "start_date": start_date, 
                         "end_date": end_date, 
                         "risk_level": risk_level, 
                         "initial_investment": initial_investment, 
                         "yearly_contribution": yearly_contribution, 
                         "years": years, 
                         "risk_free_rate": risk_free_rate, 
                         "weights": weights, 
#                         "portfolio_expected_return": portfolio_expected_return, 
                         "volatility": volatility, 
                         "sharpe_ratio": sharpe_ratio, 
                         "portfolio_return": portfolio_expected_return, 
                         "sortino_ratio": sortino_ratio_val, 
                         "cvar": cvar,
                         "treynor_ratio": treynor_ratio,
                         "tickers": weights.index.tolist(),
                         "cumulative_weighted_portfolio_returns": cumulative_weighted_portfolio_returns}
        
    portfolio_df = pd.DataFrame({'Stock': stock_data.columns,'Weight': weights})
    portfolio_df['Cumulative Return'] = cumulative_daily_returns.iloc[-1]
    portfolio_df['Initial Allocation'] = portfolio_df['Weight'] * initial_investment
    portfolio_df['Expected Return (%)'] = mu
    portfolio_df['Expected Dividend Yield (%)'] = calculate_dividend_yield(stock_data, dividend_data)
    portfolio_df['Expected 1 Year Return ($)'] = mu * portfolio_df['Initial Allocation']
     
    """ debug, 20 year value should be close to calcuated, difference is reinvestment of dividends
    # compound interest formula with yearly contributions
    # https://www.thecalculatorsite.com/articles/finance/compound-interest-formula.php
    #P = initial_investment
    #PMT = yearly_contribution
    #r = annual interest rate / mu
    #n = 1 # number of times interest is compounded per year, assume once/year for this calculation
    #t = years
    #A = P * (1 + r/n)**(n*t) + PMT * (((1 + r/n)**(n*t) - 1) / (r/n))
    #r = portfolio_df['Expected Return (%)'] + portfolio_df['Expected Dividend Yield (%)']
    #future_principal = portfolio_df['Initial Allocation'] * (1 + r/1)**(1*years)
    #contributions = (yearly_contribution*portfolio_df['Weight']) * (((1 + r/1)**(1*years) - 1) / (r/1))
    #portfolio_df[f'{years} Year Potential ($)'] = future_principal + contributions
    """
    return portfolio_df, portfolio_summary

# TODO: fix the math
def calculate_treynor_ratio(stock_data, weights, start_date, end_date, risk_free_rate):
    """
    Calculate the Treynor Ratio for a given portfolio.

    Parameters:
    stock_data (pd.DataFrame): DataFrame with stock data for each ticker
    weights (pd.Series): Investment weights for each ticker
    start_date (str): Start date for data in 'YYYY-MM-DD' format
    end_date (str): End date for data in 'YYYY-MM-DD' format
    risk_free_rate (float): Risk-free rate

    Returns:
    float: Treynor Ratio of the portfolio
    """

    # Calculate portfolio return
    returns = stock_data.pct_change().dropna()
    weighted_returns = returns.mul(weights, axis=1)
    portfolio_return = weighted_returns.sum(axis=1).mean()

    # Calculate portfolio beta
    market_returns = get_stock_data('^GSPC', start_date, end_date).pct_change().dropna()
    portfolio_returns = weighted_returns.sum(axis=1)
    portfolio_beta = portfolio_returns.cov(market_returns) / market_returns.var()

    # Calculate Treynor Ratio
    treynor_ratio = (portfolio_return - risk_free_rate) / portfolio_beta

    return treynor_ratio

def calculate_covariance_matrix(stock_data):
    return risk_models.CovarianceShrinkage(stock_data).ledoit_wolf()

# TODO: build a custom cache for this as streamlit cache gets a lot of misses
# due to volume of calls and this call is expensive - ~6s on a 5950x
@st.cache_data(persist=True)
def calculate_efficient_portfolios(mu, S, risk_free_rate, num_portfolios=500):
    min_risk, max_sharpe_ratio = calculate_risk_extents(mu, S, risk_free_rate)
    
    # TODO: see if 500 is necessary and/or make it a parameter
    risk_range = np.linspace(min_risk, max_sharpe_ratio, num_portfolios)
    
    efficient_portfolios = []
    for risk in risk_range:
        ef = EfficientFrontier(mu, S)
        ef.efficient_risk(risk)
        weights = ef.clean_weights()
        ret, vol, sharpe = ef.portfolio_performance()
        # TODO: make this efficient_portfolio dict a class / utility function
        efficient_portfolios.append({
            'portfolio_return': ret,
            'volatility': vol,
            'sharpe_ratio': sharpe,
            'weights': weights
        })
    return efficient_portfolios

""" OLD ... just stores result of efficient frontier portfolio performance
def calculate_portfolio_performance(risk, weights, ret, vol, sharpe_ratio_val):
    selected_portfolio = {
        'portfolio_return': ret,
        'risks': vol,
        'sharpe_ratio': sharpe_ratio_val,
        'weights': weights
    }
    return selected_portfolio
"""

def calculate_total_return(initial_investment, annual_return, yearly_contribution, years):
    total_return = initial_investment
    for _ in range(years):
        total_return = total_return * (1 + annual_return) + yearly_contribution
    return total_return

def calculate_optimal_portfolio(efficient_portfolios):
    # Calculate the difference between each portfolio's Sharpe ratio and 1
    sharpe_diffs = [abs(portfolio['sharpe_ratio'] - 1) for portfolio in efficient_portfolios]

    # Find the index of the portfolio with the smallest difference
    optimal_index = sharpe_diffs.index(min(sharpe_diffs))

    return efficient_portfolios[optimal_index]

# TODO: build a custom cache for this as streamlit cache gets a lot of misses
# due to volume of calls
def calculate_risk_extents(mu, S, risk_free_rate):
    # Calculate the minimum volatility portfolio
    ef_min_v = EfficientFrontier(mu, S)
    min_volatility_portfolio = ef_min_v.min_volatility()

    # calculate the efficient frontier for max_sharpe
    ef_max_sharpe = EfficientFrontier(mu, S)
    # Calculate the maximum Sharpe ratio portfolio
    max_sharpe_portfolio = ef_max_sharpe.max_sharpe(risk_free_rate=risk_free_rate)
    
    # Calculate the min / max risk - with some slop factor
    min_risk = calculate_risk(min_volatility_portfolio, S) + 0.005
    max_risk = calculate_risk(max_sharpe_portfolio, S) - 0.005
                    
    return min_risk, max_risk        
    
def calculate_risk(portfolio, S):
    # Extract the portfolio weights
    weights = np.array(list(portfolio.values()))

    # Calculate the risk of the portfolio
    risk = np.sqrt(np.dot(weights.T, np.dot(S, weights)))
    return risk
    
def calculate_portfolio_value(asset_data, weights):
    # Multiply the asset prices by the weights
    weighted_asset_data = asset_data * weights

    # Sum across the columns to get the portfolio value at each time point
    portfolio_value = weighted_asset_data.sum(axis=1)

    # Convert the Series to a DataFrame and reset the index
    portfolio_value = portfolio_value.to_frame().reset_index()
    portfolio_value.columns = ['Date', 'Value']

    return portfolio_value


def calculate_monthly_returns(data):
    # Convert Series to DataFrame
    data = pd.DataFrame(data)

    # Calculate daily returns
    daily_returns = data.pct_change()

    # Resample to the monthly level
    monthly_returns = daily_returns.resample('M').mean()
    #logger.debug(f"Monthly returns:\n{monthly_returns.head()}")

    return monthly_returns

@st.cache_data(persist=True)
def retrieve_historical_data(ticker, start_date, end_date):
    logger.debug(f"Retrieving historical data for {ticker} from {start_date} to {end_date}")
    data = yf.download(ticker, start=start_date, end=end_date)['Adj Close']
    return data

@st.cache_data(persist=True)
def retrieve_risk_free_rate(start_date, end_date):
    risk_free_rate_data = None
    if session.get_fred_api_key() is not None:
        logger.debug(f"FRED API key found {session.get_fred_api_key()}")
        fred = Fred(api_key=session.get_fred_api_key())
        logger.info(f"Retrieving risk-free rate from FRED from {start_date} to {end_date}")
        risk_free_rate_data = fred.get_series('TB3MS', start_date, end_date).rename("risk_free_rate").to_frame() / 100 
        risk_free_rate_data = risk_free_rate_data.resample('D').ffill() / 252 # resample to daily and forward-fill missing data
    return risk_free_rate_data

def get_mean_returns_models():
    return ["Historical Returns (Geometric Mean)", "Historical Weighted w/Recent Data", "Capital Asset Pricing Model (CAPM)"]

def get_efficient_frontier_models():
    return ["Minimum Volatility",  "Balanced Portfolio (Sharpe Ratio = 1)", "Maximum Sharpe Ratio", "Selected Risk Level"]

# calculates cumulative daily returns, each row contains the cumulative return up until that day
def calculate_cumulative_daily_returns(stock_data):
    cumulative_returns = (1 + stock_data.pct_change()).cumprod() - 1
    cumulative_returns.reset_index(inplace=True)
    
    return cumulative_returns
