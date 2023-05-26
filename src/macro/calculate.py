
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
    return ["US Interest Rate", "US Inflation Rate", "US M2 Money Supply Rate", "China M2 Money Supply Rate"]

# Let's assume that we believe that in the future:
# - US interest rate will increase by 1%
# - US inflation will increase by 2%
# - US M2 money supply will increase by 10%
# - China M2 money supply will increase by 15%
def get_macro_factor_defaults():
    return {"US Interest Rate": 0.01, "US Inflation Rate": 0.02, "US M2 Money Supply Rate": 1.1, "China M2 Money Supply Rate": 1.15}

@st.cache_data
def get_historical_macro_data(start_date, end_date):
    fred = Fred(api_key=session.get_fred_api_key())
    
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
    logger.debug("portfolio_summary.keys():\/{}".format(portfolio_summary.keys()))

    logger.debug("macroeconomic_data.describe():\n{}".format(macroeconomic_data.describe()))
    logger.debug("macroeconomic_data.head():\n{}".format(macroeconomic_data.head()))
    
    # Calculate daily returns for each stock
    stock_returns = portfolio_summary['stock_data'].pct_change()
    logger.debug("stock_returns.head():\n{}".format(stock_returns.head()))
    logger.debug("stock_returns.tail():\n{}".format(stock_returns.tail())) 
    logger.debug(f"stock_returns describe:\n{stock_returns.describe()}")
    
    
    # Remove the timezone information from the index to enable the concat
    if stock_returns.index.tz is not None:
        stock_returns.index = stock_returns.index.tz_convert(None)
        
    if macroeconomic_data.index.tz is not None:
        macroeconomic_data.index = macroeconomic_data.index.tz_convert(None)
    
    # Combine stock returns and macroeconomic data
    combined_data = pd.concat([stock_returns, macroeconomic_data], axis=1)
    
    # Resample to monthly frequency
    combined_data = combined_data.sort_index().resample('M').agg({**{ticker: 'mean' for ticker in portfolio_summary["tickers"]}, **{factor: 'last' for factor in get_macro_factor_list()}})
    
    # Drop any rows with missing values
    #combined_data = combined_data.dropna()
    
    # The dependent variable (y) is the portfolio returns
    portfolio_returns_absolute = (combined_data[portfolio_summary['tickers']].mul(portfolio_summary['weights'], axis=1)).sum(axis=1)
    combined_data['portfolio_returns'] = portfolio_returns_absolute

    # Calculate cumulative returns
    cumulative_returns_absolute = (1 + portfolio_returns_absolute).cumprod() - 1
    combined_data['cumulative_returns'] = cumulative_returns_absolute
    
    logger.debug("combined_data.head():\n{}".format(combined_data.head()))
    logger.debug("combined_data.tail():\n{}".format(combined_data.tail()))
    logger.debug("combined_data.describe():\n{}".format(combined_data.describe()))
    
    # The dependent variable (y) is the portfolio returns
    combined_data['portfolio_returns'] = combined_data[portfolio_summary['tickers']].dot(portfolio_summary['weights'])

    return combined_data

# model how changes in the interest rate and inflation have historically impacted the returns of our portfolio.
# we can do this by calculating the correlation between the daily returns of our portfolio and the daily changes in the interest rate and inflation.
# If the correlation is positive, it means that when the interest rate or inflation increases, our portfolio returns also tend to increase.
# Past performance is not indicative of future results, and this analysis assumes that the relationships between these variables and portfolio 
# returns will remain constant in the future, which may not be the case.
#
# The model also assumes a linear relationship between the predictors and the response variable. There could be a non-linear relationship 
# between them which cannot be captured by this model
def calculate_linear_regression_model_from_macro_data(combined_data):
    logger.debug("combined_data.head():\n{}".format(combined_data.head()))
    logger.debug("combined_data.tail():\n{}".format(combined_data.tail()))
    logger.debug("combined_data.describe():\n{}".format(combined_data.describe()))

    model_data = combined_data.copy()
    model_data = model_data.dropna()
    
    # The independent variables (X) are the macroeconomic indicators
    X = model_data[get_macro_factor_list()]
    y = model_data['portfolio_returns']
    logger.debug("X.head():\n{}".format(X.head()))
    logger.debug("y.head():\n{}".format(y.head()))

    # This will give you the estimated intercept and coefficients of the regression model. 
    # These coefficients represent the estimated change in portfolio returns for a one-unit change 
    # in the corresponding macroeconomic indicator, holding all other indicators constant.
    model = LinearRegression().fit(X, y)
    
    logger.debug("model.intercept_: {}".format(model.intercept_))
    logger.debug("model.coef_: {}".format(model.coef_))
    
    return model, model_data, X, y

def calculate_new_X(X, new_macro_vars):
    # Calculate the new X matrix for the new macroeconomic data
    new_X = pd.DataFrame({'US Interest Rate': [X['US Interest Rate'].iloc[-1] + new_macro_vars['US Interest Rate']],
                            'US Inflation Rate': [X['US Inflation Rate'].iloc[-1] + new_macro_vars['US Inflation Rate']],
                            'US M2 Money Supply Rate': [X['US M2 Money Supply Rate'].iloc[-1] * new_macro_vars['US M2 Money Supply Rate']],
                            'China M2 Money Supply Rate': [X['China M2 Money Supply Rate'].iloc[-1] * new_macro_vars['China M2 Money Supply Rate']]})

    logger.debug("new_X: {}".format(new_X))
    
    return new_X

def predict_change_in_returns(model, new_X):
    # Predict the change in portfolio returns for the new macroeconomic data
    predicted_change_in_returns = model.predict(new_X)
    logger.debug("predicted_change_in_returns: {}".format(predicted_change_in_returns))
    
    return predicted_change_in_returns
