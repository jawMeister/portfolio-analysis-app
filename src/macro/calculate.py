import os
import glob
from datetime import datetime, timedelta

import streamlit as st

import itertools
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from statsmodels.api import OLS
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import ccf
from scipy.stats import pearsonr

import plotly.graph_objects as go
import plotly.subplots as sp
import colorsys

from fredapi import Fred
import quandl

import src.macro.plot as plot

import logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s (%(levelname)s):  %(module)s.%(funcName)s - %(message)s')

# Set up logger for a specific module to a different level
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

import config as config
import src.utils as utils

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

@st.cache_data
def get_historical_macro_data(start_date, end_date):
    rate_series = ['FEDFUNDS', 'T5YIFR', 'UNRATE', 'GS10']
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

    fred = Fred(api_key=config.get_api_key('fred'))
    
    macro_data_dict = {}
    for series_code, series_name in monthly_series_names.items():
        series = fred.get_series(series_code, start_date, end_date)
        
        if series_code in rate_series:
            series = series / 100  # convert from decimal to percentage

        macro_data_dict[series_name] = resample_from_monthly_factor_series(series)

    for series_code, series_name in quarterly_series_names.items():
        series = fred.get_series(series_code, start_date, end_date)

        macro_data_dict[series_name] = resample_from_quarterly_factor_series(series)

    # 01/06/23 - FED/M2_N_M is no longer available on Quandl, replaced with FRED/M2SL
    #us_m2_money_supply_base = quandl.get("FED/M2_N_M", authtoken=config.get_api_key('nasdaq'), start_date=start_date, end_date=end_date)
    #us_m2_money_supply_base = us_m2_money_supply_base.shift(1, freq='D')  # shift the index by 1 day to align with FRED data reporting
    us_m2_money_supply_base = quandl.get("FRED/M2SL", authtoken=config.get_api_key('nasdaq'), start_date=start_date, end_date=end_date)
    logger.debug(f'us_m2_money_supply_base:\n{us_m2_money_supply_base.head()}')

    macro_data_dict['US M2 Money Supply'] = resample_from_monthly_factor_series(us_m2_money_supply_base['Value'])

    return macro_data_dict

def resample_from_daily_stock_returns_series(daily_returns_series):    
    monthly_returns = daily_returns_series.resample('MS').apply(lambda x: (1 + x).prod() - 1).dropna()
    quarterly_returns = daily_returns_series.resample('QS').apply(lambda x: (1 + x).prod() - 1).dropna()
    annual_returns = daily_returns_series.resample('YS').apply(lambda x: (1 + x).prod() - 1).dropna()

    cumulative_daily_returns = ((1 + daily_returns_series).cumprod() - 1).dropna()
    cumulative_monthly_returns = ((1 + monthly_returns).cumprod() - 1).dropna()
    cumulative_quarterly_returns = ((1 + quarterly_returns).cumprod() - 1).dropna()
    cumulative_annual_returns = ((1 + annual_returns).cumprod() - 1).dropna()
    
    # establish "Change" to imply daily, monthly, etc. returns
    # establish w/o "Change" to imply cumulative returns
    return {
        'Daily': cumulative_daily_returns, 
        'Daily Change': daily_returns_series,
        'Monthly': cumulative_monthly_returns, 
        'Monthly Change': monthly_returns, 
        'Quarterly': cumulative_quarterly_returns, 
        'Quarterly Change': quarterly_returns, 
        'Yearly': cumulative_annual_returns, 
        'Yearly Change': annual_returns
    }
    
def get_historical_portfolio_weighted_returns_data(simple_daily_returns, weights):

    weighted_daily_returns = simple_daily_returns.mul(weights, axis=1)
    portfolio_daily_returns = weighted_daily_returns.sum(axis=1)
        
    # return ticker list with non-zero weights
    portfolio_tickers = [ticker for ticker, weight in weights.items() if weight > 0]
    
    portfolio_returns_dict = resample_from_daily_stock_returns_series(portfolio_daily_returns)
        
    return portfolio_returns_dict, portfolio_tickers

@st.cache_data        
def get_combined_returns_data(simple_daily_returns, weights, start_date, end_date):
   
    sp500_data = utils.get_stock_data(['^GSPC'], start_date, end_date)
    sp500_daily_returns = sp500_data.pct_change()
    sp500_returns_dict = resample_from_daily_stock_returns_series(sp500_daily_returns)

    portfolio_returns_dict, portfolio_tickers = get_historical_portfolio_weighted_returns_data(simple_daily_returns, weights)
    macro_data_dict = get_historical_macro_data(start_date, end_date)
    logger.debug(f'retrieved historical macro data for: {macro_data_dict.keys()}')

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
                
            logger.debug(f'{factor} {time_basis}:\n{time_series.head()}\n')
            
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

    return combined_returns_data, combined_cumulative_returns_data, portfolio_returns_dict, sp500_returns_dict, macro_data_dict, portfolio_tickers

# Aligning the dataframes
def align_dataframes(dfs):
    # Get the common datetime index
    common_index = dfs[0].index
    for df in dfs[1:]:
        common_index = common_index.intersection(df.index)
    
    # Align all dataframes to the common index
    aligned_dfs = [df.loc[common_index] for df in dfs]
    return aligned_dfs

def prepare_data(returns_data, time_basis='Monthly'):
    portfolio_returns = returns_data[time_basis]['Portfolio'].to_frame(name='Portfolio')
    sp500_returns = returns_data[time_basis]['S&P500'].to_frame(name='S&P500')

    # concat sp500 with portfolio returns
    portfolio_returns = pd.concat([portfolio_returns, sp500_returns], axis=1)

    # Loop through all macro factors and add to DataFrame
    for factor_name, factor_df in returns_data[time_basis]['Macro'].items():
        factor_df = factor_df.rename(factor_name)  # Rename the series
        portfolio_returns = pd.concat([portfolio_returns, factor_df], axis=1)

    # Use align_dataframes to ensure the data is properly aligned
    aligned_data = align_dataframes([portfolio_returns])

    aligned_data[0].index = pd.to_datetime(aligned_data[0].index)
    aligned_data[0] = aligned_data[0].dropna()
    
    return aligned_data[0]  # align_dataframes returns a list, so we take the first element

"""
The check_multicollinearity() function calculates the Variance Inflation Factor (VIF) for each macro factor. 
VIF measures the correlation between each factor and all other factors. If a factor's VIF exceeds a certain 
threshold (in this case, 5), it indicates high multicollinearity.
"""
def check_multicollinearity(df, vif_threshold=5):
    df = df.copy()
    df.drop('Portfolio',axis=1,inplace=True)
    
    variables = df.columns
    vif_df = pd.DataFrame()
    vif_df["VIF"] = [variance_inflation_factor(df[variables].values, df.columns.get_loc(var)) for var in df.columns]
    vif_df["Features"] = variables

    columns_to_drop = []
    while vif_df['VIF'].max() > vif_threshold:
        remove = vif_df.sort_values('VIF',ascending=False)['Features'][:1]
        columns_to_drop.append(remove.values[0])
        df.drop(remove,axis=1,inplace=True)
        variables = df.columns
        vif_df = pd.DataFrame()
        vif_df["VIF"] = [variance_inflation_factor(df[variables].values, df.columns.get_loc(var)) for var in df.columns]
        vif_df["Features"] = variables
    return columns_to_drop

def get_factor_list(df):
    factor_list = list(df.columns)
    factor_list.remove('Portfolio')
    factor_list.remove('S&P500')
    return factor_list

def calculate_linear_regression_model_for_factor(input_df, factor):
    df = input_df.copy()
    df = df.dropna(how='any').reset_index(drop=True)
    X = df[[factor]]
    y = df['Portfolio']
    X2 = add_constant(X)
    model = OLS(y, X2).fit()
    coefficient = model.params[factor]
    p_value = model.pvalues[factor]
    return model, coefficient, p_value

def calculate_multivariate_analysis(input_df):
    df = input_df.copy()
    df = df.dropna(how='any').reset_index(drop=True)
    X = df[get_factor_list(df)]  
    y = df['Portfolio']
    
    model = OLS(y, add_constant(X)).fit()
    
    # Extract the significant features based on p-values
    significant_features = model.pvalues[model.pvalues < 0.05].index.tolist()
    
    if 'const' in significant_features:
        significant_features.remove('const')
        
    return model, significant_features


def check_stationarity(series):
    """
    Perform Dickey-Fuller test and check for stationarity of a given series.
    """
    result = adfuller(series)
    return result[1] <= 0.05  # If p-value is less than 0.05, the series is stationary

def create_var_model(data, maxlag=6, vif_threshold=5):
    # Ensure data is stationary
    non_stationary_columns = [column for column in data.columns if not check_stationarity(data[column])]
    if non_stationary_columns:
        # Difference non-stationary series
        data[non_stationary_columns] = data[non_stationary_columns].diff()
        # Drop the first row in all series
        data = data.dropna()

    threshold = 0.001 # if any columns with less than a 0.1% standard deviation, just drop it as more or less constant
    near_constant_columns = data.columns[data.std() < threshold]
    logger.debug(f'Near constant columns: {near_constant_columns}')
        
    data = data.drop(columns=near_constant_columns)

    # Get the list of columns to drop based on high correlation to each other
    columns_to_drop = check_multicollinearity(data, vif_threshold)
    if columns_to_drop is None:
        columns_to_drop = []
    logger.debug(f'After check_multicollinearity(), columns to drop: {columns_to_drop}')

    # Add columns with high correlation to the columns to drop list
    corr_matrix = data.corr()
    for i in range(len(data.columns)):
        for j in range(i+1, len(data.columns)):
            if abs(corr_matrix.iloc[i,j]) > 0.8:
                col_i = data.columns[i]
                col_j = data.columns[j]
                if col_i not in columns_to_drop and col_j not in columns_to_drop:
                    logger.debug(f'Adding columns {col_i} and {col_j} to columns_to_drop')
                    columns_to_drop.append(col_i)
                    columns_to_drop.append(col_j)

    # Create the VAR model with all columns
    best_model = None
    best_aic = np.inf
    try:
        # Create the VAR model
        model = VAR(data)

        # Fit the model with the optimal lag order
        results = model.fit(ic='aic')
        #logger.debug(f'VAR model results:\n{results.summary()}')

        # Update the best model if the AIC is lower
        if results.aic < best_aic:
            logger.debug(f'New best model found with AIC: {results.aic}')
            best_model = results
            best_aic = results.aic
    except Exception as e:
        logger.error(f'Failed to create VAR model with all columns. Error: {e}')

    # if 'Portfolio' in columns_to_drop:, remove it
    if 'Portfolio' in columns_to_drop:
        columns_to_drop.remove('Portfolio')
        logger.debug(f'Removing Portfolio from columns_to_drop, leaving these: {columns_to_drop}')

    # Iterate through the columns to drop and create a VAR model with the optimal lag order
    for i in range(len(columns_to_drop) + 1):
        try:
            columns = columns_to_drop[:i]
            logger.debug(f'Creating VAR model with columns dropped: {columns}')
            # Drop the specified columns
            data_dropped = data.drop(columns=columns)

            #logger.debug(f'After dropping columns, data.keys(): {data_dropped.keys()}\ndata.head():\n{data_dropped.head()}\ndata.tail():\n{data_dropped.tail()}')
            # Create the VAR model
            model = VAR(data_dropped)

            # Fit the model with the optimal lag order
            results = model.fit(ic='aic')
            #logger.debug(f'VAR model results:\n{results.summary()}')

            # Update the best model if the AIC is lower
            if results.aic < best_aic:
                logger.debug(f'New best model found with AIC: {results.aic}')
                best_model = results
                best_aic = results.aic
        except Exception as e:
            logger.error(f'Failed to create VAR model with columns dropped: {columns}. Error: {e}')

    if best_model is not None:
        logger.debug(f'Best model AIC: {best_aic}')
        logger.debug(f'Columns dropped: {columns_to_drop[:i]}')
        coefficients = best_model.params
        lag_order = best_model.k_ar
        return best_model, coefficients, lag_order
    else:
        logger.debug('Failed to create any VAR model.')
        return None, None, None
    

def calculate_optimal_lag(input_df, factor, maxlag=12):
    """Compute cross-correlation between portfolio returns and several lagged versions of the factor, and return the lag with the highest absolute cross-correlation"""
    df = input_df.copy()
    df = df.dropna(how='any').reset_index(drop=True)
    
    optimal_lag = 0
    max_cross_correlation = 0
    for lag in range(-maxlag, maxlag + 1):
        cross_correlation = abs(np.correlate(df['Portfolio'], df[factor].shift(lag), mode='valid').mean())
        if cross_correlation > max_cross_correlation:
            optimal_lag = lag
            max_cross_correlation = cross_correlation
    return optimal_lag


def create_regression_models(monthly_input_data_df, time_basis, cumulative_performance=False):
#    monthly_input_data_df = prepare_data(returns_data)
    
    factor_list = get_factor_list(monthly_input_data_df)

    # Initialize dataframes to store models
    regression_models_df = pd.DataFrame(columns=['Model Type', 'Factor', 'Model', 'Coefficient', 'P-value', 'R-squared', 'Correlation', 'Optimal Lag'])
    multivariate_models_df = pd.DataFrame(columns=['Model Type', 'Model', 'Significant Features'])
    var_models_df = pd.DataFrame(columns=['Model Type', 'Model', 'Coefficients', 'Lag Order'])

    # Loop through all macro factors and build a regression model for each
    for factor in factor_list:
        # Create a dataframe for the current factor and the portfolio returns
        df_factor = monthly_input_data_df[['Portfolio', factor]].dropna()
        
        # Calculate optimal lag for the current factor - a change in macro factor may take some time to affect the portfolio returns
        # The optimal lag is the lag with the highest absolute cross-correlation between the factor and the portfolio returns
        optimal_lag = calculate_optimal_lag(df_factor, factor)
        
        if optimal_lag > 0:
            df_factor[factor] = df_factor[factor].shift(optimal_lag).dropna()
            
        # Calculate linear regression models for each factor
        regression_model, coefficient, p_value = calculate_linear_regression_model_for_factor(df_factor, factor)

        # Calculate R-squared and correlation
        r_squared = regression_model.rsquared
        correlation, _ = pearsonr(df_factor['Portfolio'], df_factor[factor])

        regression_models_df.loc[len(regression_models_df)] = {
            'Model Type': 'linear',
            'Factor': factor,
            'Model': regression_model, 
            'Coefficient': coefficient, 
            'P-value': p_value,
            'R-squared': r_squared,
            'Correlation': correlation, 
            'Optimal Lag': optimal_lag}
        
    # Use the multivariate model to identify significant factors
    multivariate_model, significant_features = calculate_multivariate_analysis(monthly_input_data_df)

    multivariate_models_df.loc[len(multivariate_models_df)] = {
        'Model Type': 'multivariate',
        'Model': multivariate_model, 
        'Significant Features': significant_features
    }

    # Create and store the VAR model separately - only works with stationary data
    if not cumulative_performance:
        try:         
            var_model, coefficients, lag_order = create_var_model(monthly_input_data_df)
            if var_model is not None:
                var_models_df.loc[len(var_models_df)] = {
                    'Model Type': 'VAR',
                    'Model': var_model, 
                    'Coefficients': coefficients, 
                    'Lag Order': lag_order}
                #logger.debug(f'VAR model coefficients:\n{coefficients}')
                #logger.debug(f'VAR model lag order: {lag_order}')
                # TODO: move this to display
#                forecast_var_model(var_model, steps=5)
        except Exception as e:
            logger.error(f'VAR model failed to run\n{e}')
    
    return regression_models_df, multivariate_models_df, var_models_df