
import pandas as pd
import numpy as np
from fredapi import Fred
from sklearn.linear_model import LinearRegression
import streamlit as st

import logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s (%(levelname)s):  %(module)s.%(funcName)s - %(message)s')

# Set up logger for a specific module to a different level
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import src.session as session

def get_macro_factor_list():
    return ["US Interest Rate", "US Inflation Rate", "US M2 Money Supply", "China M2 Money Supply"]

# Let's assume that we believe that in the future:
# - US interest rate will increase by 1%
# - US inflation will increase by 2%
# - US M2 money supply will increase by 10%
# - China M2 money supply will increase by 15%
def get_macro_factor_defaults():
    return {"US Interest Rate": 0.01, "US Inflation Rate": 0.02, "US M2 Money Supply": 0.1, "China M2 Money Supply": 0.15}

@st.cache_data
def get_historical_macro_data(start_date, end_date):
    fred = Fred(api_key=session.get_fred_api_key())
    logger.info(f"Fetching macroeconomic data from FRED from {start_date} to {end_date}")
    # Get macroeconomic data
    us_interest_rate = fred.get_series('GS10', start_date, end_date)  # 10-Year Treasury Constant Maturity Rate
    us_inflation = fred.get_series('T10YIE', start_date, end_date)  # 10-Year Breakeven Inflation Rate
    us_m2_money_supply = fred.get_series('M2', start_date, end_date)  # M2 Money Stock
    china_m2_money_supply = fred.get_series('MYAGM2CNM189N', start_date, end_date)  # China M2 Money Supply
    
    # Combine into a single dataframe
    macroeconomic_data = pd.concat([us_interest_rate, us_inflation, us_m2_money_supply, china_m2_money_supply], axis=1)
    logger.debug("macroeconomic_data.head():\n{}".format(macroeconomic_data.head()))
    macroeconomic_data.columns = get_macro_factor_list()
    
    return macroeconomic_data

def clean_and_combine_macro_data(portfolio_summary, macroeconomic_data):
    # Calculate daily returns for each stock
    stock_returns = portfolio_summary['stock_data'].pct_change()

    # Remove the timezone information from the index to enable the concat
    if stock_returns.index.tz is not None:
        stock_returns.index = stock_returns.index.tz_convert(None)
        
    if macroeconomic_data.index.tz is not None:
        macroeconomic_data.index = macroeconomic_data.index.tz_convert(None)
    
    # Resample stock returns and macroeconomic data separately
    # taking the mean monthly return here, need to revisit this
    stock_returns_monthly = stock_returns.sort_index().resample('M').mean()
    
    # Sort by index
    macroeconomic_data = macroeconomic_data.sort_index()

    # Define aggregation dictionary
    agg_dict = {factor: 'last' for factor in get_macro_factor_list()}
    
    # Resample and aggregate
    macroeconomic_data_monthly = macroeconomic_data.resample('M').agg(agg_dict)

    # Calculate monthly change in macroeconomic data
    macroeconomic_data_monthly_change = macroeconomic_data_monthly.pct_change()

    # Calculate monthly change in macroeconomic data
    macroeconomic_data_monthly_change = macroeconomic_data_monthly_change.add_suffix(' Change')
    
    # Combine resampled stock returns and macroeconomic data
    combined_data = pd.concat([stock_returns_monthly, macroeconomic_data_monthly, macroeconomic_data_monthly_change], axis=1)
    
    # The dependent variable (y) is the portfolio returns
    portfolio_returns_absolute = (combined_data[portfolio_summary['tickers']].mul(portfolio_summary['weights'], axis=1)).sum(axis=1)
    combined_data['portfolio_returns'] = portfolio_returns_absolute

    # Calculate cumulative returns
    cumulative_returns_absolute = (1 + portfolio_returns_absolute).cumprod() - 1
    combined_data['cumulative_returns'] = cumulative_returns_absolute

    # Calculate cumulative inflation (if desired)
    combined_data['cumulative_inflation'] = combined_data['US Inflation Rate'].cumsum()

    return combined_data


def calculate_new_X(X, new_macro_vars):
    # Calculate the new X matrix for the new macroeconomic data
    new_X = pd.DataFrame({'US Interest Rate': [X['US Interest Rate'].iloc[-1] + new_macro_vars['US Interest Rate']],
                            'US Inflation Rate': [X['US Inflation Rate'].iloc[-1] + new_macro_vars['US Inflation Rate']],
                            'US M2 Money Supply': [X['US M2 Money Supply'].iloc[-1] * (1 + new_macro_vars['US M2 Money Supply'])],
                            'China M2 Money Supply': [X['China M2 Money Supply'].iloc[-1] * (1 + new_macro_vars['China M2 Money Supply'])]})

    logger.debug("new_X: {}".format(new_X))
    
    return new_X

def predict_change_in_returns(model, new_X):
    # Predict the change in portfolio returns for the new macroeconomic data
    predicted_change_in_returns = model.predict(new_X)
    logger.debug("predicted_change_in_returns: {}".format(predicted_change_in_returns))
    
    return predicted_change_in_returns

# model how changes in the interest rate, inflation, etc have historically impacted the returns of our portfolio.

def calculate_linear_regression_models_from_macro_data_per_factor(combined_data, factors, y_label):
    logger.debug(f"combined column names:\n{combined_data.columns}")
    logger.debug(f"factors:\n{factors}")
    logger.debug(f"all factors in column names? : {set(factors + [y_label]).issubset(set(combined_data.columns))}")

    models = {}

    # The dependent variable (y) is the portfolio returns
    y_data = combined_data[y_label].copy()
    y_data = y_data.dropna()

    for factor in factors:
        # The independent variables (X) are the macroeconomic indicators
        X_data = combined_data[[factor]].copy()
        X_data = X_data.dropna()

        # We only keep the intersection of valid data points
        common_index = y_data.index.intersection(X_data.index)
        X = X_data.loc[common_index]
        y = y_data.loc[common_index]

        model = LinearRegression().fit(X, y)

        models[factor] = model

    # multivariate model for all factors
    model_data = combined_data[factors + [y_label]].copy()
    model_data = model_data.dropna()
    X = model_data[factors]
    y = model_data[y_label]
    model_all = LinearRegression().fit(X, y)
    models["all_factors"] = model_all
    
    logger.debug(f"models created: {models.keys()} to predict {y_label}")

    return models, model_data

"""
This function can take as input the user's future estimates of the macroeconomic factors and the models, and output the models' predictions 
for each factor and the combined effect of all factors on portfolio returns.
"""
def predict_portfolio_returns_from_user_macro_input(user_macro_input, models):
    predictions = {}

    for factor, model in models.items():
        logger.debug(f"predicting for factor: {factor}")
        if factor in user_macro_input:
            if factor == 'all_factors':
                future_factor_values = [user_macro_input[f] for f in models.keys() if f != 'all_factors']
                future_factor_values = pd.DataFrame([future_factor_values], columns=models.keys())
            else:
                future_factor_values = pd.DataFrame([user_macro_input[factor]], columns=[factor])
        
            prediction = model.predict(future_factor_values)
            predictions[factor] = prediction
        else:
            prediction = None

    return predictions
