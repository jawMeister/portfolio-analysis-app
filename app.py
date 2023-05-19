import streamlit as st
from datetime import datetime
from pypfopt import expected_returns, risk_models

import warnings
warnings.filterwarnings("ignore", message="Module \"zipline.assets\" not found")

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
import utils
import display
import plots

st.set_page_config(page_title="stock portfolio optimization", layout="wide")

with st.sidebar:
    #tickers = st.multiselect("Select ticker symbols", ["AAPL", "AMZN", "NVDA", "MMC", "GOOG", "MSFT", "BTC-USD","XOM","BAC","V","MMC","GOLD"])
    tickers = st.text_area("Enter ticker symbols (comma separated)", "AAPL,AMZN,NVDA,MMC,GOOG,MSFT,BTC-USD,ETH-USD,XOM,BAC,V,GOLD")
    tickers = tickers.split(",")
    tickers = list(set(tickers))

    start_date = st.date_input("Start date (for historical stock data)", datetime(2014, 1, 1))
    end_date = st.date_input("End date (for historical stock data)", datetime(2023, 5, 11))

    risk_free_rate = st.number_input("Risk free rate % (t-bills rate for safe returns)", min_value=0.0, max_value=7.5, step=0.1, value=5.0, format="%.1f") / 100

    initial_investment = st.slider("Initial investment", min_value=0, max_value=1000000, step=5000, value=50000, format="$%d")
    yearly_contribution = st.slider("Yearly contribution", min_value=0, max_value=250000, step=5000, value=25000, format="$%d")

    years = st.slider("Years to invest", min_value=1, max_value=50, step=1, value=20)

    #TODO: if different mean return model is selected or tickers updated, need to reset risk level and re-calculate
    if tickers and start_date and end_date and risk_free_rate:
        stock_data, dividend_data = utils.get_stock_data(tickers, start_date, end_date)
        
        # radio button for risk model to leverage - put into session state?
        mean_returns_model = st.radio("Mean returns model", ("Historical Returns (Geometric Mean)", "Historical Weighted w/Recent Data", "Capital Asset Pricing Model (CAPM)"))
        
        if mean_returns_model == "Historical Returns (Geometric Mean)":
            mu = expected_returns.mean_historical_return(stock_data)
        elif mean_returns_model == "Historical Weighted w/Recent Data":
            mu = expected_returns.ema_historical_return(stock_data)
        elif mean_returns_model == "Capital Asset Pricing Model (CAPM)":
            mu = expected_returns.capm_return(stock_data, risk_free_rate=risk_free_rate)

        S = risk_models.CovarianceShrinkage(stock_data).ledoit_wolf()
        
        min_risk, max_risk = utils.calculate_risk_extents(mu, S, risk_free_rate)
        if "risk_level" not in st.session_state:
            r0 = min_risk + ((max_risk - min_risk) /2)
            st.session_state.risk_level = float(r0)
            
        #print(f"min_risk: {min_risk}, max_risk: {max_risk}, r: {st.session_state.risk}, type: {type(st.session_state.risk)}")
        st.slider("Risk Level", min_value=min_risk, max_value=max_risk, step=0.01, key="risk_level", format="%.2f")
        #st.slider("Risk Level", min_value=min_risk, max_value=max_risk, step=0.01, key="risk_level", format="%.2f", on_change=utils.calculate_risk_extents(mu, S, risk_free_rate))
        
        """
        Harry Markowitz, William Sharpe, and Merton Miller were jointly awarded the Nobel Memorial Prize in Economic Sciences in 1990 for their pioneering work 
        in the theory of financial economics.

        Harry Markowitz was awarded for his development of the theory of portfolio choice. His work, first published in 1952, demonstrated that portfolio diversification 
        could reduce risk, and showed how to optimally select a portfolio of risky assets. This work led to the development of what is now known as Modern Portfolio Theory.

        William Sharpe was awarded for his development of the Capital Asset Pricing Model (CAPM). The CAPM, first published by Sharpe in 1964, provides a way to calculate 
        the expected return on an investment given its systematic risk (or "beta"). This model also led to the concept of the "security market line," a graphical 
        representation of the CAPM that shows the expected return of a security as a function of its systematic risk.
        
        Merton Miller was recognized for his fundamental contributions to the theory of corporate finance. His work, in collaboration with Franco Modigliani (who won the 
        Nobel Prize in 1985), developed what is now known as the Modigliani-Miller theorem. This theorem states, in its most basic form, that under certain market 
        conditions (such as no taxes, bankruptcy costs, or asymmetric information), the value of a firm is unaffected by how it is financed.

        Their collective work forms much of the foundational theory in financial economics, influencing both academic research and practical investment and corporate 
        finance strategies.
        """
        st.markdown("Source: [Nobel Prize](https://www.nobelprize.org/prizes/economic-sciences/1990/summary/)")
        
if tickers and start_date and end_date and initial_investment and years:
    tab1, tab2 = st.tabs(["Portfolio Analysis","Returns Analysis"])
    
    with tab1:
        asset_values, detailed_asset_holdings = display.display_portfolio(stock_data, dividend_data, mu, S, start_date, end_date, st.session_state.risk_level, initial_investment, yearly_contribution, years, risk_free_rate)

    with tab2:
        st.write("TODO: Returns Analysis leveraging various prediction models to include volatility")
        st.write("Assuming constant growth rate (NOT reality), forecast below")
        
        display.display_asset_values(asset_values)
        plots.plot_asset_values(asset_values)
