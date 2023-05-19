import os
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from pypfopt import expected_returns, risk_models, EfficientFrontier
from empyrical import sharpe_ratio, sortino_ratio
from datetime import datetime, date
import streamlit as st
import yfinance as yf
from fredapi import Fred

@st.cache_data
def get_dividend_data(tickers, start_date, end_date):
    dividend_data = {}

    for ticker in tickers:
        stock = yf.Ticker(ticker)
        dividends = stock.history(start=start_date, end=end_date).Dividends
        dividend_data[ticker] = dividends

    return pd.DataFrame(dividend_data)

@st.cache_data
def get_stock_data(tickers, start_date, end_date):
    if isinstance(start_date, (datetime, date)):
        start_date = start_date.strftime('%Y-%m-%d')
    if isinstance(end_date, (datetime, date)):
        end_date = end_date.strftime('%Y-%m-%d')
    if not isinstance(start_date, str) or not isinstance(end_date, str):
        raise TypeError(f"start_date ({type(start_date)}) and end_date ({type(end_date)}) must be either strings or datetime objects.")
    
    #TODO: can this be optimized to not call yf.download twice?
    dividend_data = get_dividend_data(tickers, start_date, end_date)

    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    return data, dividend_data

@st.cache_data
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

@st.cache_data
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
        
    returns = stock_data.pct_change().dropna()
        
    # Obtain weights, return, volatility, and Sharpe ratio
    weights = ef.clean_weights()
    weights = pd.Series(weights).reindex(returns.columns)
    portfolio_expected_return, volatility, sharpe_ratio = ef.portfolio_performance()
   
    portfolio_return = np.sum([np.mean(returns[ticker]) * weights[ticker] for ticker in weights.index])

    # Calculate portfolio beta using weights
    #market_returns = yf.download('SPY', start_date, end_date)['Adj Close'].pct_change().dropna()
    #market_returns = market_returns.reindex(returns.index).fillna(0)  # align the index and fill missing values
    #market_covar = np.sum([np.cov(returns[ticker], market_returns)[0][1] * weights[ticker] for ticker in weights.index])
    #market_var = np.var(market_returns)
    #portfolio_beta = market_covar / market_var

    # Calculate Treynor Ratio
    #treynor_ratio = (portfolio_return - risk_free_rate) / portfolio_beta
    treynor_ratio = 0
    
    weighted_returns = returns.dot(weights)
    sortino_ratio_val = sortino_ratio(weighted_returns)
    alpha = 0.05
    cvar = -np.nanpercentile(weighted_returns[weighted_returns < 0], alpha * 100)
    
    portfolio_summary = {"stock_data": stock_data, 
                         "dividend_data": dividend_data, 
                         "mu": mu, 
                         "S": S, 
                         "individual_returns": returns,
                         "start_date": start_date, 
                         "end_date": end_date, 
                         "risk_level": risk_level, 
                         "initial_investment": initial_investment, 
                         "yearly_contribution": yearly_contribution, 
                         "years": years, 
                         "risk_free_rate": risk_free_rate, 
                         "weights": weights, 
                         "portfolio_expected_return": portfolio_expected_return, 
                         "volatility": volatility, 
                         "sharpe_ratio": sharpe_ratio, 
                         "portfolio_return": portfolio_return, 
                         "sortino_ratio": sortino_ratio_val, 
                         "cvar": cvar,
                         "treynor_ratio": treynor_ratio,
                         "tickers": weights.index.tolist()}
        
    portfolio_df = pd.DataFrame({'Stock': stock_data.columns, 'Weight': weights})
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

@st.cache_data
def calculate_efficient_portfolios(mu, S, risk_free_rate):
    min_risk, max_sharpe_ratio = calculate_risk_extents(mu, S, risk_free_rate)
    
    risk_range = np.linspace(min_risk, max_sharpe_ratio, 500)
    
    efficient_portfolios = []
    for risk in risk_range:
        ef = EfficientFrontier(mu, S)
        ef.efficient_risk(risk)
        weights = ef.clean_weights()
        ret, vol, sharpe = ef.portfolio_performance()
        efficient_portfolios.append({
            'returns': ret,
            'risks': vol,
            'sharpe_ratio': sharpe,
            'weights': weights
        })
    return efficient_portfolios

def calculate_portfolio_performance(risk, weights, ret, vol, sharpe_ratio_val):
    selected_portfolio = {
        'returns': ret,
        'risks': vol,
        'sharpe_ratio': sharpe_ratio_val,
        'weights': weights
    }
    return selected_portfolio

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

def calculate_risk_extents(mu, S, risk_free_rate):
    # Calculate the minimum volatility portfolio
    ef_min_v = EfficientFrontier(mu, S)
    min_volatility_portfolio = ef_min_v.min_volatility()

    # calculate the efficient frontier for max_sharpe
    ef_max_sharpe = EfficientFrontier(mu, S)
    # Calculate the maximum Sharpe ratio portfolio
    max_sharpe_portfolio = ef_max_sharpe.max_sharpe(risk_free_rate=risk_free_rate)
        
    def calculate_risk(portfolio, S):
        # Extract the portfolio weights
        weights = np.array(list(portfolio.values()))
    
        # Calculate the risk of the portfolio
        risk = np.sqrt(np.dot(weights.T, np.dot(S, weights)))
        return risk
    
    # Calculate the min / max risk - with some slop factor
    min_risk = calculate_risk(min_volatility_portfolio, S) + 0.005
    max_risk = calculate_risk(max_sharpe_portfolio, S) - 0.005
        
    return min_risk, max_risk        

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
    #print(f"calculate_monthly_returns: data: {data.head()}****************")

    # Convert Series to DataFrame
    data = pd.DataFrame(data)

    # Calculate daily returns
    daily_returns = data.pct_change()

    # Resample to the monthly level
    #monthly_returns = daily_returns.resample('M').apply(lambda x: (x + 1).prod() - 1)
    monthly_returns = daily_returns.resample('M').mean()

    return monthly_returns

def retrieve_historical_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)['Adj Close']
    return data

def retrieve_risk_free_rate(start_date, end_date):
    fred = Fred(api_key='XXX')
    risk_free_rate_data = fred.get_series('TB3MS', start_date, end_date) / 100 / 252
    return risk_free_rate_data