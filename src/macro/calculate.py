
import pandas as pd
from fredapi import Fred

import src.session as session

def get_historical_macro_data(start_date, end_date):
    fred = Fred(api_key=session.get_fred_api_key())
    
    # Get macroeconomic data
    us_interest_rate = fred.get_series('GS10', start_date, end_date)  # 10-Year Treasury Constant Maturity Rate
    us_inflation = fred.get_series('T10YIE', start_date, end_date)  # 10-Year Breakeven Inflation Rate
    us_m2_money_supply = fred.get_series('M2', start_date, end_date)  # M2 Money Stock
    china_m2_money_supply = fred.get_series('MYAGM2CNM189N', start_date, end_date)  # China M2 Money Supply
    
    # Combine into a single dataframe
    macroeconomic_data = pd.concat([us_interest_rate, us_inflation, us_m2_money_supply, china_m2_money_supply], axis=1)
    macroeconomic_data.columns = ['us_interest_rate', 'us_inflation', 'us_m2_money_supply', 'china_m2_money_supply']
