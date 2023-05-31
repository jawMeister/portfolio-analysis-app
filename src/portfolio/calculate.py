import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st

from src import utils

import logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s (%(levelname)s):  %(module)s.%(funcName)s - %(message)s')

# Set up logger for a specific module to a different level
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

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

# this assumes constant growth at the same rate of return - may be useful for modeling if simulating random asset growth
# TODO: add in the ability to use a random rate of return for simulations as would be useful to see dividend growth too
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
            starting_price = utils.get_current_price(portfolio_summary["stock_data"], ticker) # initialize to most recent price from yfinance
            shares = future_value / starting_price # initialize investment amount / most recent price
            dividend_yield = utils.calculate_weighted_dividend_yield(portfolio_summary["stock_data"][ticker], portfolio_summary["dividend_data"][ticker])

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

def calculate_total_return(initial_investment, annual_return, yearly_contribution, years):
    total_return = initial_investment
    for _ in range(years):
        total_return = total_return * (1 + annual_return) + yearly_contribution
    return total_return

def calculate_weighted_portfolio_index_value(asset_data, weights):
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


def calculate_test_ratio(portfolio_summary):
    years_of_historical_data = portfolio_summary["stock_data"].resample("Y").last().shape[0]
    test_ratio = 1 - portfolio_summary["years"] / (years_of_historical_data + portfolio_summary["years"])
    
    years_of_historical_data = portfolio_summary["end_date"].year - portfolio_summary["start_date"].year
    #print(f"years_of_historical_data: {years_of_historical_data}")
    
    return test_ratio

def split_data(data, train_size=0.8):
    data.index = data.index.tz_convert(None)
    split_index = int(len(data) * train_size)
    split_date = data.index[split_index].date()

    # If the data at the split_index is not the first data of the day,
    # adjust the split_index to the start of the next day.
    if not (data.loc[str(split_date)].index[0] == data.index[split_index]):
        split_date += pd.Timedelta(days=1)
        split_index = data.index.get_indexer([split_date], method='nearest')[0]

    train_data = data.iloc[:split_index].copy()
    test_data = data.iloc[split_index:].copy()

    return train_data, test_data

def calculate_portfolio_performance(stock_data, dividend_data, weights, start_date, end_date):
    
    stock_data.index = stock_data.index.tz_localize(None)
    dividend_data.index = dividend_data.index.tz_localize(None)
    
    # Calculate daily returns
    daily_returns = stock_data.pct_change()

    # Calculate daily dividend returns
    daily_dividend_returns = dividend_data / stock_data.shift()

    # Calculate total returns
    daily_total_returns = daily_returns + daily_dividend_returns

    # TODO: Calculate portfolio returns and adjust for weights for any stock/asset that did not exist for the full time series
    # eg., BTC-USD or a stock that had an IPO in the middle of the time series
    adjusted_weights = adjust_weights(weights, stock_data)
    #daily_portfolio_returns = (daily_total_returns.mul(adjusted_weights, axis=1)).sum(axis=1)
    daily_portfolio_returns = (daily_total_returns.mul(weights, axis=1)).sum(axis=1)

    # Download S&P 500 data for benchmarking
    sp500 = utils.get_sp500_daily_returns(start_date, end_date)
    # Align sp500 to daily_portfolio_returns
    sp500 = sp500.reindex(daily_portfolio_returns.index).ffill()

    # Calculate relative returns
    portfolio_returns_relative_to_sp500 = daily_portfolio_returns - sp500

    # Download risk-free rate data
    rf_rate = utils.retrieve_risk_free_rate(start_date, end_date)
    daily_rf_rate = rf_rate.reindex(daily_portfolio_returns.index, method='ffill')['risk_free_rate']

    portfolio_returns_relative_to_rf = daily_portfolio_returns - daily_rf_rate
    
    df_dict = {"df_returns_by_ticker": daily_returns,
                "df_dividend_returns_by_ticker": daily_dividend_returns,
                "df_total_returns_by_ticker": daily_total_returns,
                "df_weighted_portfolio_returns": daily_portfolio_returns,
                "df_sp500_returns": sp500,
                "df_portfolio_returns_relative_to_sp500": portfolio_returns_relative_to_sp500,
                "df_portfolio_returns_relative_to_rf": portfolio_returns_relative_to_rf
                }

    return df_dict 

def adjust_weights(weights, stock_data):
    # Create a DataFrame of the current trading status of each stock
    is_trading = ~stock_data.isna()

    # Convert weights into a DataFrame with the same index and columns as is_trading
    weights_df = pd.DataFrame(index=is_trading.index, columns=is_trading.columns)
    for ticker in weights.index:
        weights_df[ticker] = weights[ticker]
    
    # Adjust the weights for non-zero tickers
    adjusted_weights_df = weights_df.mul(is_trading).div(is_trading.mul(weights_df).sum(axis=1), axis='index')
    
    # Convert adjusted weights DataFrame back to Series
    adjusted_weights = adjusted_weights_df.sum()

    return adjusted_weights

