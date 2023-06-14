import os, sys
# enable absolute paths transversal (from notebooks folder to src folder)
parent_dir = os.path.abspath('..')
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import multiprocessing
# use fork to avoid issues with prophet and multiprocessing

import concurrent.futures
from multiprocessing import cpu_count

from dotenv import load_dotenv
    
from datetime import datetime, timedelta, date
import time

from fredapi import Fred
import yfinance as yf

import itertools
import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from pypfopt import expected_returns, risk_models, EfficientFrontier
from empyrical import sharpe_ratio, sortino_ratio
import json
import glob

# disable prophet logging
import logging
logging.getLogger('prophet').setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").disabled=True

logging.basicConfig(level=logging.WARNING, format='%(asctime)s (%(levelname)s):  %(module)s.%(funcName)s - %(message)s')

# Set up logger for a specific module to a different level
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
import traceback

import quandl

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

def calculate_covariance_matrix(stock_data):
    return risk_models.CovarianceShrinkage(stock_data).ledoit_wolf()

def calculate_risk(portfolio, S):
    # Extract the portfolio weights
    weights = np.array(list(portfolio.values()))

    # Calculate the risk of the portfolio
    risk = np.sqrt(np.dot(weights.T, np.dot(S, weights)))
    return risk

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

def resample_from_monthly_factor_series(series):
    monthly_mean = series.resample('MS').mean()  
    monthly_change = monthly_mean.pct_change()
    quarterly_mean = series.resample('QS').mean()  
    quarterly_change = quarterly_mean.pct_change()
    yearly_mean = series.resample('YS').mean()  
    yearly_change = yearly_mean.pct_change()
    
    return {
        'Monthly': monthly_mean, 
        'Monthly Change': monthly_change, 
        'Quarterly': quarterly_mean, 
        'Quarterly Change': quarterly_change, 
        'Yearly': yearly_mean, 
        'Yearly Change': yearly_change
    }

def resample_from_daily_stock_returns_series(daily_returns_series):    
    portfolio_daily_returns = daily_returns_series.sum(axis=1)
    portfolio_monthly_returns = portfolio_daily_returns.resample('MS').apply(lambda x: (1 + x).prod() - 1).dropna()
    portfolio_quarterly_returns = portfolio_daily_returns.resample('QS').apply(lambda x: (1 + x).prod() - 1).dropna()
    portfolio_annual_returns = portfolio_daily_returns.resample('YS').apply(lambda x: (1 + x).prod() - 1).dropna()

    portfolio_cumulative_daily_returns = ((1 + portfolio_daily_returns).cumprod() - 1).dropna()
    portfolio_cumulative_monthly_returns = ((1 + portfolio_monthly_returns).cumprod() - 1).dropna()
    portfolio_cumulative_quarterly_returns = ((1 + portfolio_quarterly_returns).cumprod() - 1).dropna()
    portfolio_cumulative_annual_returns = ((1 + portfolio_annual_returns).cumprod() - 1).dropna()
    
    # establish "Change" to imply daily, monthly, etc. returns
    # establish w/o "Change" to imply cumulative returns
    return {
        'Daily': portfolio_cumulative_daily_returns, 
        'Daily Change': portfolio_daily_returns,
        'Monthly': portfolio_cumulative_monthly_returns, 
        'Monthly Change': portfolio_monthly_returns, 
        'Quarterly': portfolio_cumulative_quarterly_returns, 
        'Quarterly Change': portfolio_quarterly_returns, 
        'Yearly': portfolio_cumulative_annual_returns, 
        'Yearly Change': portfolio_annual_returns
    }

def resample_from_quarterly_factor_series(series):
    quarterly_mean = series.resample('QS').mean()
    quarterly_change = quarterly_mean.pct_change()
    yearly_mean = series.resample('YS').mean()
    yearly_change = yearly_mean.pct_change()
    
    return {
        'Quarterly': quarterly_mean, 
        'Quarterly Change': quarterly_change, 
        'Yearly': yearly_mean, 
        'Yearly Change': yearly_change
    }

def get_historical_macro_data(start_date, end_date):
    rate_series = ['FEDFUNDS', 'T5YIFR']
    monthly_series_names = {
        'FEDFUNDS': 'Federal Funds Rate',
        'UNRATE': 'Unemployment Rate',
        'CPIAUCSL': 'CPI',
        'PCE': 'PCE',
        'RSAFS': 'Retail Sales',
        'ICSA': 'Initial Claims',
        'HOUST': 'Housing Starts',
        'T5YIFR': '5-Year Forward Inflation Expectation Rate',
        'USEPUINDXD': 'Economic Policy Uncertainty Index for United States',
        'GS10': '10-Year Treasury Constant Maturity Rate'
    }

    quarterly_series_names = {
        'GDPC1': 'GDP',
    }

    fred = Fred(api_key=os.environ.get('FRED_API_KEY'))
    
    macro_data_dict = {}
    for series_code, series_name in monthly_series_names.items():
        series = fred.get_series(series_code, start_date, end_date)
        
        if series_code in rate_series:
            series = series / 100  # convert from decimal to percentage

        macro_data_dict[series_name] = resample_from_monthly_factor_series(series)

    for series_code, series_name in quarterly_series_names.items():
        series = fred.get_series(series_code, start_date, end_date)

        macro_data_dict[series_name] = resample_from_quarterly_factor_series(series)

    us_m2_money_supply_base = quandl.get("FED/M2_N_M", authtoken=os.environ.get('NASDAQ_API_KEY'), start_date=start_date, end_date=end_date)
    us_m2_money_supply_base = us_m2_money_supply_base.shift(1, freq='D')  # shift the index by 1 day to align with FRED data reporting

    macro_data_dict['US M2 Money Supply'] = resample_from_monthly_factor_series(us_m2_money_supply_base['Value'])

    return macro_data_dict

def get_historical_portfolio_weighted_returns_data(tickers, start_date, end_date):
    # just grabbing the 'Adj Close' column
    stock_data = get_stock_data(tickers, start_date, end_date)

    """similar to portfolio/display.py, set the portfolio weights and calculate the performance
    """
    risk_free_rate = 0.04
    mu = expected_returns.mean_historical_return(stock_data)
    S = calculate_covariance_matrix(stock_data)

    # Calculating the cumulative monthly returns of a portfolio of stocks with given weights:
    min_risk, max_risk = calculate_risk_extents(mu, S, risk_free_rate)
    risk = (max_risk + min_risk) / 2 # set to mid point risk? or max risk?

    ef = EfficientFrontier(mu, S)
    ef.efficient_risk(risk)
    weights = ef.clean_weights()
    weights = pd.Series(weights).reindex(stock_data.columns)
    ef_returns, ef_volatility, ef_sharpe = ef.portfolio_performance(risk_free_rate)
    print(f'ef weights:\n{weights}')
    print(f'ef performance: {ef_returns, ef_volatility, ef_sharpe}')

    daily_returns = stock_data.pct_change()
    weighted_daily_returns = daily_returns.mul(weights, axis=1)
    
    # return ticker list with non-zero weights
    portfolio_tickers = [ticker for ticker, weight in weights.items() if weight > 0]
    
    portfolio_returns_dict = resample_from_daily_stock_returns_series(weighted_daily_returns)
        
    return portfolio_returns_dict, portfolio_tickers

        
def get_combined_returns_data(tickers, start_date, end_date):
   
    sp500_data = get_stock_data(['^GSPC'], start_date, end_date)
    sp500_daily_returns = sp500_data['Adj Close'].pct_change()
    sp500_returns_dict = resample_from_daily_stock_returns_series(sp500_daily_returns)

    portfolio_returns_dict, portfolio_tickers = get_historical_portfolio_weighted_returns_data(tickers, start_date, end_date)
    macro_data_dict = get_historical_macro_data(start_date, end_date)

    monthly_macro_data = {}
    mom_change_in_macro_data = {}

    quarterly_macro_data = {}
    qoq_change_in_macro_data = {}

    yearly_macro_data = {}
    yoy_change_in_macro_data = {}
       
    for factor, time_bases in macro_data_dict.items():
        for time_basis, time_series in time_bases.items():
            if time_basis == 'Monthly':
                monthly_macro_data[factor] = time_series.dropna()
                
            if time_basis == 'Monthly Change': 
                mom_change_in_macro_data[factor] = time_series.dropna()
                
            if time_basis == 'Quarterly':
                quarterly_macro_data[factor] = time_series.dropna()
                
            if time_basis == 'Quarterly Change':
                qoq_change_in_macro_data[factor] = time_series.dropna()
                
            if time_basis == 'Yearly':
                yearly_macro_data[factor] = time_series.dropna()
            
            if time_basis == 'Yearly Change':
                yoy_change_in_macro_data[factor] = time_series.dropna()
                
            print(f'{factor} {time_basis}:\n{time_series.head()}\n')
            
    # what to do with the daily returns? maybe return the portfolio and benchmark daily returns separately?
    monthly_returns = {'Portfolio': portfolio_returns_dict['Monthly Change'], 'S&P500': sp500_returns_dict['Monthly Change'], 'Macro': mom_change_in_macro_data} # compare macro with monthly returns
    quarterly_returns = {'Portfolio': portfolio_returns_dict['Quarterly Change'], 'S&P500': sp500_returns_dict['Quarterly Change'], 'Macro': qoq_change_in_macro_data} # compare macro with quarterly returns
    annual_returns = {'Portfolio': portfolio_returns_dict['Yearly Change'], 'S&P500': sp500_returns_dict['Yearly Change'], 'Macro': yoy_change_in_macro_data} # compare macro with annual returns

    cumulative_monthly_returns = {'Portfolio': portfolio_returns_dict['Monthly'], 'S&P500': sp500_returns_dict['Monthly'], 'Macro': monthly_macro_data} # compare macro with cumulative monthly returns
    cumulative_quarterly_returns = {'Portfolio': portfolio_returns_dict['Quarterly'], 'S&P500': sp500_returns_dict['Quarterly'], 'Macro': quarterly_macro_data} # compare macro with cumulative quarterly returns
    cumulative_annual_returns = {'Portfolio': portfolio_returns_dict['Yearly'], 'S&P500': sp500_returns_dict['Yearly'], 'Macro': yoy_change_in_macro_data} # compare macro with cumulative annual returns

    combined_returns_data = {
        'Monthly': monthly_returns,
        'Quarterly': quarterly_returns,
        'Yearly': annual_returns,
    }

    combined_cumulative_returns_data = {
        'Monthly': cumulative_monthly_returns,
        'Quarterly': cumulative_quarterly_returns,
        'Yearly': cumulative_annual_returns,
    }

    return combined_returns_data, combined_cumulative_returns_data, portfolio_returns_dict, sp500_returns_dict, macro_data_dict

def save_tuned_hyperparameters(hyper_parameter_dict, suffix=None):
    # Save hyperparameters to a json file
    date = datetime.now().strftime("%Y%m%d")
    path = os.getenv('OUTPUT_PATH')
    
    os.makedirs(path, exist_ok=True)
    
    # Include the desired path in the file name
    file_name = f'{path}/tuned_macro_hyperparameters_{suffix}_{date}.json'
    #file_name = f'tuned_macro_hyperparameters_{suffix}_{date}.json'
    
    file_name = file_name.replace(" ", "_")
    
    logger.info(f'Saving tuned hyperparameters to {file_name}')
    
    with open(file_name, 'w') as f:
        json.dump(hyper_parameter_dict, f)
        
    logger.info(f'Saved tuned hyperparameters to {file_name}')
    return

def check_for_files(suffix):
    path = os.getenv('OUTPUT_PATH')
    file_name = f'{path}/tuned_macro_hyperparameters_{suffix}*.json'
    file_name = file_name.replace(" ", "_")
    
    files = glob.glob(file_name)

    # If no files found, return None
    if not files:
        # Log the identified files
        logger.info(f'Did not find {file_name} files with suffix "{suffix}": {files}')
        return False
    else:
        # Log the identified files
        logger.info(f'Found {file_name} files with suffix "{suffix}": {files}')
        return True
    
def tune_hyperparameters(df, initial, period, horizon, factor, time_basis):
    logger.debug(f'initial_window_in_days: {initial}, period_in_days: {period}, horizon_in_days: {horizon}')
    logger.debug(f'df.shape: {df.shape}\ndf.head():\n{df.head()}')
    
    param_grid = {  
        'changepoint_prior_scale': [0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5],
        'seasonality_prior_scale': [0.01, 0.1, 1.0, 5.0, 10.0],
        'seasonality_mode': ['additive', 'multiplicative'],
    }
    # Generate all combinations of parameters
    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    
    # Generate all combinations of parameters
    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    rmses = []  # Store the RMSEs for each params here
    smapes = []  # Store the sMAPEs for each params here
    coverages = []  # Store the Coverages for each params here
    
    # Use cross validation to evaluate all parameters
    for params in all_params:
        try:
            if factor == 'Unemployment Rate' and time_basis == 'Monthly Change' and params['seasonality_mode'] == 'multiplicative' and params['seasonality_prior_scale'] == 0.01 and params['changepoint_prior_scale'] == 0.001:
                logger.info(f'*********** skipping fit and cross validation for {factor} on {time_basis} w/ params: {params} with initial: {initial}, period: {period}, horizon: {horizon}')
                continue
            else:
                logger.info(f'starting fit and cross validation for {factor} on {time_basis} w/ params: {params} with initial: {initial}, period: {period}, horizon: {horizon}')
                m = Prophet(**params).fit(df)  # Fit model with given params
                df_cv = cross_validation(m, initial=f'{initial} days', period=f'{period} days', horizon=f'{horizon} days', parallel="processes")
                df_p = performance_metrics(df_cv, rolling_window=1)
                rmses.append(df_p['rmse'].values[0])
                smapes.append(df_p['smape'].values[0])
                coverages.append(df_p['coverage'].values[0])
                logger.info(f'finished cross validation for {factor} on {time_basis} w/ params: {params} with initial: {initial}, period: {period}, horizon: {horizon}')
        except Exception as e:
            tb_str = traceback.format_exception(type(e), e, e.__traceback__)
            message = "".join(tb_str)  # convert traceback to string
            logger.error(f'Error in tuning hyperparameters for {factor} on {time_basis} w/ params: {params}, initial: {initial}, period: {period}, horizon: {horizon}\n{message}', exc_info=True)

    # Find the best parameters
    best_params_rmse = all_params[np.argmin(rmses)]
    best_params_smape = all_params[np.argmin(smapes)]
    best_params_coverage = all_params[np.argmax(coverages)]

    return best_params_rmse, best_params_smape, best_params_coverage

def tune_hyperparameters_for_macro_factor(factor, initial_window_in_days, start_date, end_date):
#    load_dotenv()

    # given only tuning one factor at a time, maybe just pass in the factor name
    # or... instead of pulling every time, pull once from web and store in a file
    macro_data_dict = get_historical_macro_data(start_date, end_date)
    hyper_parameter_dict = {}
    
    for horizon_in_days in [732, 366, 184, 92, 62]:
        for period_in_days in [horizon_in_days*0.5]:

                time_bases = macro_data_dict[factor]

                suffix = f'{factor}_w_init_{initial_window_in_days}_period_{period_in_days}_horizon_{horizon_in_days}'
                if check_for_files(suffix):
                    logger.info(f'Already have tuned hyperparameters for {factor} with initial_window_in_days: {initial_window_in_days}, period_in_days: {period_in_days}, horizon_in_days: {horizon_in_days}, skipping')
                else:
                    hyper_parameter_dict[factor] = {}

                    for time_basis, time_series in time_bases.items():
                        try: 
                            hyper_parameter_dict[factor][time_basis] = {}
                            
                            df = time_series.to_frame()
                            df.columns = ['y']
                            df['ds'] = df.index
                            df.dropna(inplace=True)
                            df.reset_index(drop=True, inplace=True)
                            
                            logger.info(f'Tuning hyperparameters for {factor} on {time_basis} basis with initial_window_in_days: {initial_window_in_days}, period_in_days: {period_in_days}, horizon_in_days: {horizon_in_days}')
                            best_rmse, best_mape, best_coverage = tune_hyperparameters(df, initial_window_in_days, period_in_days, horizon_in_days, factor, time_basis)
                        
                            # Store the tuned hyperparameters
                            hyper_parameter_dict[factor][time_basis]['rmse'] = {'rating': 'n/a', **best_rmse}
                            hyper_parameter_dict[factor][time_basis]['smape'] = {'rating': 'n/a', **best_mape}
                            hyper_parameter_dict[factor][time_basis]['coverage'] = {'rating': 'n/a', **best_coverage}
                        except Exception as e:
                            tb_str = traceback.format_exception(type(e), e, e.__traceback__)
                            message = "".join(tb_str)  # convert traceback to string
                            logger.error(f'Error in tuning hyperparameter time_basis for {factor} on {time_basis} initial: {initial_window_in_days}, period: {period_in_days}, horizon: {horizon_in_days}\n{message}', exc_info=True)
                           
                    # save any time series we were able to tune hyperparameters for         
                    save_tuned_hyperparameters(hyper_parameter_dict[factor], suffix)

    
    return hyper_parameter_dict

def tune_hyperparameters_for_portfolio(tickers, initial_window_in_days, start_date, end_date):
#    load_dotenv()

    portfolio_data_dict, portfolio_tickers = get_historical_portfolio_weighted_returns_data(tickers, start_date, end_date)
    # map tickers to a single string
    tickers_str = '_'.join(portfolio_tickers)
    
    hyper_parameter_dict = {}
    
    for horizon_in_days in [732, 366, 184, 92, 62]:
        for period_in_days in [horizon_in_days*0.5]:

                time_bases = portfolio_data_dict

                suffix = f'{tickers_str}_w_init_{initial_window_in_days}_period_{period_in_days}_horizon_{horizon_in_days}'
                if check_for_files(suffix):
                    logger.info(f'Already have tuned hyperparameters for {tickers_str} with initial_window_in_days: {initial_window_in_days}, period_in_days: {period_in_days}, horizon_in_days: {horizon_in_days}, skipping')
                else:
                    hyper_parameter_dict[tickers_str] = {}

                    for time_basis, time_series in time_bases.items():
                        try: 
                            hyper_parameter_dict[tickers_str][time_basis] = {}
                            
                            df = time_series.to_frame()
                            df.columns = ['y']
                            df['ds'] = df.index
                            df.dropna(inplace=True)
                            df.reset_index(drop=True, inplace=True)
                            
                            logger.info(f'Tuning hyperparameters for {tickers_str} on {time_basis} basis with initial_window_in_days: {initial_window_in_days}, period_in_days: {period_in_days}, horizon_in_days: {horizon_in_days}')
                            best_rmse, best_mape, best_coverage = tune_hyperparameters(df, initial_window_in_days, period_in_days, horizon_in_days, 'portfolio', time_basis)
                        
                            # Store the tuned hyperparameters
                            hyper_parameter_dict[tickers_str][time_basis]['rmse'] = {'rating': 'n/a', **best_rmse}
                            hyper_parameter_dict[tickers_str][time_basis]['smape'] = {'rating': 'n/a', **best_mape}
                            hyper_parameter_dict[tickers_str][time_basis]['coverage'] = {'rating': 'n/a', **best_coverage}
                        except Exception as e:
                            tb_str = traceback.format_exception(type(e), e, e.__traceback__)
                            message = "".join(tb_str)  # convert traceback to string
                            logger.error(f'Error in tuning hyperparameter time_basis for portfolio on {time_basis} initial: {initial_window_in_days}, period: {period_in_days}, horizon: {horizon_in_days}\n{message}', exc_info=True)
                           
                    # save
                    save_tuned_hyperparameters(hyper_parameter_dict[tickers_str], suffix)

print(__name__, type(__name__))
if __name__ == '__main__':
    load_dotenv()
    
    initial_window_in_days = int(os.getenv('INITIAL_WINDOW_IN_DAYS'))
    start_date = os.getenv('START_DATE')
    end_date = os.getenv('END_DATE')
    path = os.getenv('OUTPUT_PATH')
    factor = os.getenv('FACTOR')
    fred_api_key = os.getenv('FRED_API_KEY')
    nasdaq_api_key = os.getenv('NASDAQ_API_KEY')
    
    logger.info(f"Factor from .env is {factor}, tuning from start_date {start_date} to end_date {end_date} with initial_window_in_days {initial_window_in_days}, output to {path}")

    if factor == 'Portfolio':
        tickers = os.getenv('TICKERS')
        if tickers:
            tickers = tickers.split('_')
        logger.info(f"Tuning hyperparameters for portfolio of: {tickers}")
        tune_hyperparameters_for_portfolio(tickers, initial_window_in_days, start_date, end_date)
    else:
        logger.info(f"Tuning hyperparameters for factor of: {factor}")
        tune_hyperparameters_for_macro_factor(factor, initial_window_in_days, start_date, end_date)
