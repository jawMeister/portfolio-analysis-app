import streamlit as st
from datetime import datetime, timedelta
from pypfopt import expected_returns, risk_models

import warnings
warnings.filterwarnings("ignore", message="Module \"zipline.assets\" not found")

import traceback
import logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s (%(levelname)s):  %(module)s.%(funcName)s - %(message)s')

# Set up logger for a specific module to a different level
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from src import utils as utils
from src.portfolio import display as portfolio
from src.returns import display as returns
from src.macro import display as macro
from src.optimization import display as optimization
from src.rebalancing import display as rebalancing
from src.financials import display as financials
from src.technical import display as technical

st.set_page_config(page_title="stock portfolio optimization", layout="wide")

# TODO: add an on change event to the sidebar to reinitialize the session state vars
def initalize_inputs():
    if 'app_initialized' not in st.session_state:
        st.session_state.app_initialized = False
        
    if "risk_free_rate" not in st.session_state:
        st.session_state.risk_free_rate = 0.04
        
    if "start_date" not in st.session_state:
        st.session_state.start_date = datetime(2014, 1, 1)
        logger.debug(f"app init: start_date = {st.session_state.start_date}")
        
    if "end_date" not in st.session_state:
        st.session_state.end_date = datetime.now() - timedelta(1)
        logger.debug(f"app init: end_date = {st.session_state.end_date}")
        
    if "mean_returns_model" not in st.session_state:
        st.session_state.mean_returns_model = "Historical Returns (Geometric Mean)"
        
    if "stock_data" not in st.session_state:
        st.session_state.stock_data = None
        
    if "openai_portfolio_response" not in st.session_state:
        st.session_state.openai_portfolio_response = None
        
    if "portfolio_tab_initialized" not in st.session_state:
        st.session_state.portfolio_tab_initialized = False
    
    st.session_state.app_initialized = True
        
# some of the tabs have settings dependent on the sidebar, so just have them each reinit on sidebar changes
def reinitalize_tabs():
    st.session_state.portfolio_tab_initialized = False
    st.session_state.returns_tab_initialized = False
    st.session_state.macro_tab_initialized = False
    st.session_state.optimization_tab_initialized = False
    st.session_state.rebalancing_tab_initialized = False
    st.session_state.financials_tab_initialized = False
    st.session_state.technical_tab_initialized = False
    
with st.sidebar:
    initalize_inputs()
    
    st.write("This app is pre-beta, still iterating over calculations which may not be correct although should be directionally accurate.  Please report any issues or suggestions to the [github repo](https://github.com/jawMeister/portfolio-analysis-app)")
    
    #tickers = st.multiselect("Select ticker symbols", ["AAPL", "AMZN", "NVDA", "MMC", "GOOG", "MSFT", "BTC-USD","XOM","BAC","V","MMC","GOLD"])
    tickers = st.text_area("Enter ticker symbols (comma separated)", "AAPL,AMZN,NVDA,MMC,GOOG,MSFT,BTC-USD,ETH-USD,XOM,BAC,V,GOLD")
    tickers = tickers.split(",")
    tickers = list(set(tickers))
    
    st.date_input("Start date (for historical stock data)", key="start_date", value=datetime(2014, 1, 1), on_change=reinitalize_tabs)
    st.date_input("End date (for historical stock data)", key="end_date", value=(datetime.now() - timedelta(1)), on_change=reinitalize_tabs)
    logger.debug(f"start_date input set to start_date: {st.session_state.start_date}, end_date: {st.session_state.end_date}, session start_date: {st.session_state.start_date}")

    # TODO: make this a different kind of input as clicking +/- a bunch of time causes many repaints
    rfr = st.slider("Risk free rate % (t-bills rate for safe returns)", min_value=0.0, max_value=7.5, step=0.1, value=4.0, format="%.1f", on_change=reinitalize_tabs)
    st.session_state.risk_free_rate = rfr / 100.0

    st.slider("Initial investment", min_value=0, max_value=1000000, step=5000, value=50000, key='initial_investment', format="$%d", on_change=reinitalize_tabs)
    st.slider("Yearly contribution", min_value=0, max_value=250000, step=5000, value=25000, key='yearly_contribution', format="$%d", on_change=reinitalize_tabs)

    st.slider("Years to invest", min_value=1, max_value=50, step=1, value=20, key='years', on_change=reinitalize_tabs)

    logger.debug(f"tickers: {tickers}, start_date: {st.session_state.start_date}, end_date: {st.session_state.end_date}")
    stock_data, dividend_data = utils.get_stock_and_dividend_data(tickers, st.session_state.start_date, st.session_state.end_date)
    st.session_state.stock_data = stock_data
    st.session_state.dividend_data = dividend_data
                
    # radio button for risk model to leverage - put into session state?
    st.radio("Mean returns model", utils.get_mean_returns_models(), key="mean_returns_model", on_change=reinitalize_tabs)
    
    try:
        #logging.debug(f"mean_returns_model: {st.session_state.mean_returns_model}, risk_free_rate: {st.session_state.risk_free_rate}")
        mu = utils.calculate_mean_returns(st.session_state.stock_data, st.session_state.mean_returns_model, st.session_state.risk_free_rate)
        S = utils.calculate_covariance_matrix(st.session_state.stock_data)
        
        min_risk, max_risk = utils.calculate_risk_extents(mu, S, st.session_state.risk_free_rate)
        if "risk_level" not in st.session_state:
            r0 = min_risk + ((max_risk - min_risk) /2)
            st.session_state.risk_level = float(r0)
            
        st.session_state.min_risk = min_risk
        st.session_state.max_risk = max_risk
        st.session_state.mu = mu
        st.session_state.S = S
            
        st.slider("Risk Level", min_value=st.session_state.min_risk, max_value=st.session_state.max_risk, step=0.01, key="risk_level", format="%.2f", on_change=reinitalize_tabs)
    except Exception as e:
        logger.error(f"Error calculating mean returns and covariance matrix: {e}")
        logger.error(traceback.format_exc())
        logger.error(f"session state keys: {st.session_state.keys()}")
        for k,v in st.session_state.items():
            logger.error(f"{k}: {v}")
        

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
        
if tickers and st.session_state.end_date > st.session_state.start_date:
    logger.debug(f"tickers: {tickers}, initial_investment: {st.session_state.initial_investment}, years: {st.session_state.years}, start_date: {st.session_state.start_date}, end_date: {st.session_state.end_date}")
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Portfolio Analysis","Returns Analysis", "Macro Economic Analysis", 
                                            "Rebalancing Analysis", "Portfolio Optimization","Financials Analysis","Technical Analysis"])
    
    # Calculate portfolio statistics
    portfolio_df, portfolio_summary = \
        utils.calculate_portfolio_df(st.session_state.stock_data, st.session_state.dividend_data, 
                                    st.session_state.mu, st.session_state.S, st.session_state.start_date, st.session_state.end_date, st.session_state.risk_level, 
                                    st.session_state.initial_investment, st.session_state.yearly_contribution, st.session_state.years, st.session_state.risk_free_rate)
        
    with tab1:
        portfolio.display_selected_portfolio(portfolio_summary, portfolio_df)
        st.session_state.portfolio_tab_initialized = True
       
    with tab2:
        returns.display_portfolio_returns_analysis(portfolio_summary)
        
    with tab3:
        macro.display_macro_analysis(portfolio_summary)
        
    with tab4:
        rebalancing.display_rebalancing_analysis(portfolio_summary)
 
    with tab5:
        optimization.display_portfolio_optimization(portfolio_summary)
        
    with tab6:
        financials.display_financials_analysis(portfolio_summary)
        
    with tab7:
        technical.display_technical_analysis(portfolio_summary)