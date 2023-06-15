import os
import streamlit as st
from datetime import datetime, timedelta

import warnings
warnings.filterwarnings("ignore", message="Module \"zipline.assets\" not found")

import traceback
import logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s (%(levelname)s):  %(module)s.%(funcName)s - %(message)s')

# Set up logger for a specific module to a different level
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from src import utils as utils
from src.portfolio import display as portfolio
from src.returns import display as returns
from src.macro import display as macro
from src.optimization import display as optimization
from src.rebalancing import display as rebalancing
from src.financials import display as financials
from src.technical import display as technical
from src.forecasting import display as forecasting
from src.news import display as news

   
st.set_page_config(page_title="stock portfolio optimization", layout="wide")

def initialize_inputs():
    st.session_state.setdefault('app_initialized', False)
    st.session_state.setdefault('risk_free_rate', 0.04)
    st.session_state.setdefault('start_date', datetime(2014, 1, 1).date())
    st.session_state.setdefault('end_date', (datetime.now() - timedelta(1)).date())
    st.session_state.setdefault('mean_returns_model', "Historical Returns (Geometric Mean)")
    st.session_state.setdefault('stock_data', None)
    st.session_state.setdefault('dividend_data', None)
    st.session_state.setdefault('openai_portfolio_response', None)
    st.session_state.setdefault('risk_level', None)
    st.session_state.setdefault('min_risk', None)
    st.session_state.setdefault('max_risk', None)
    st.session_state.setdefault('mu', None)
    st.session_state.setdefault('S', None)
    st.session_state.setdefault('initial_investment', 50000)
    st.session_state.setdefault('yearly_contribution', 25000)
    st.session_state.setdefault('years', 20)

    st.session_state.setdefault('portfolio_tab_initialized', False)
    st.session_state.setdefault('returns_tab_initialized', False)
    st.session_state.setdefault('macro_tab_initialized', False)
    st.session_state.setdefault('optimization_tab_initialized', False)
    st.session_state.setdefault('rebalancing_tab_initialized', False)
    st.session_state.setdefault('financials_tab_initialized', False)
    st.session_state.setdefault('technical_tab_initialized', False)
    
    if not st.session_state.app_initialized:
        tickers = "AAPL,AMZN,NVDA,MMC,GOOGL,MSFT,BTC-USD,ETH-USD,XOM,BAC,V,GOLD"
        st.session_state.tickers = tickers.split(",")
        update_calculations('app initialization')
        st.session_state.app_initialized = True

def reinitialize_tabs(**kwargs):
    widget_source = kwargs.get('widget_source')
    logger.debug(f"reinitializing tabs from {widget_source}")
    update_calculations(widget_source)
    
    #st.session_state.portfolio_tab_initialized = False
    st.session_state.returns_tab_initialized = False
    st.session_state.macro_tab_initialized = False
    st.session_state.optimization_tab_initialized = False
    st.session_state.rebalancing_tab_initialized = False
    st.session_state.financials_tab_initialized = False
    st.session_state.technical_tab_initialized = False
    
def update_calculations(widget_source=None):
    logger.info(f"updating calculations from {widget_source}")
    # if the start date is beyond two days ago, set it to two days ago
    if st.session_state.start_date > (datetime.now() - timedelta(2)).date():
        st.session_state.start_date = (datetime.now() - timedelta(2)).date()
        
    # if the end date is before the start date, set it to the start date plus one day
    if st.session_state.end_date < st.session_state.start_date:
        st.session_state.end_date = st.session_state.start_date + timedelta(1)
        
    st.session_state.stock_data, st.session_state.dividend_data = utils.get_stock_and_dividend_data(st.session_state.tickers, st.session_state.start_date, st.session_state.end_date)
    
    try:
        st.session_state.mu = utils.calculate_mean_returns(st.session_state.stock_data, st.session_state.mean_returns_model, st.session_state.risk_free_rate)
        st.session_state.S = utils.calculate_covariance_matrix(st.session_state.stock_data)
        st.session_state.min_risk, st.session_state.max_risk = utils.calculate_risk_extents(st.session_state.mu, st.session_state.S, st.session_state.risk_free_rate)
        
        # set default risk level to halfway between min and max - this should only happen once to set a default value, otherwise it will be overwritten by the slider
        if 'risk_level' not in st.session_state or st.session_state.risk_level is None:
            st.session_state.risk_level = float(st.session_state.min_risk + ((st.session_state.max_risk - st.session_state.min_risk) /2))
            
                        # Calculate portfolio statistics
        portfolio_df, portfolio_summary = \
            utils.calculate_portfolio_df(st.session_state.stock_data, st.session_state.dividend_data, 
                                        st.session_state.mu, st.session_state.S, st.session_state.start_date, st.session_state.end_date, st.session_state.risk_level, 
                                        st.session_state.initial_investment, st.session_state.yearly_contribution, st.session_state.years, st.session_state.risk_free_rate)
        
        # TODO: likely refactor these into a single object, also repetitive with above, so streamline
        st.session_state.portfolio_df = portfolio_df
        st.session_state.portfolio_summary = portfolio_summary
    except Exception as e:
        logger.error(f"Error calculating mean returns and covariance matrix: {e}")
        logger.error(traceback.format_exc())
        logger.error(f"session state keys: {st.session_state.keys()}")
        for k,v in st.session_state.items():
            logger.error(f"{k}: {v}")

def app():
    initialize_inputs()

    with st.sidebar:
        st.write("This app is pre-beta, still iterating over calculations which may not be correct although should be directionally accurate.  Please report any issues or suggestions to the [github repo](https://github.com/jawMeister/portfolio-analysis-app)")
        
        tickers = st.text_area("Enter ticker symbols (comma separated)", ','.join(st.session_state.tickers))
        tickers = [ticker.strip() for ticker in tickers.split(",")]
        if sorted(tickers) != sorted(st.session_state.tickers):
            st.session_state.tickers = tickers
            reinitialize_tabs('tickers')
            
        start_date = st.date_input("Start date (for historical stock data)", value=st.session_state.start_date)
        if start_date != st.session_state.start_date:
            st.session_state.start_date = start_date
            reinitialize_tabs(widget_source='start_date')

        end_date = st.date_input("End date (for historical stock data)", value=st.session_state.end_date)
        if end_date != st.session_state.end_date:
            st.session_state.end_date = end_date
            reinitialize_tabs(widget_source='end_date')
            
        rfr = st.slider("Risk free rate % (t-bills rate for safe returns)", min_value=0.0, max_value=7.5, step=0.1, value=4.0, format="%.1f", )
        # since we have to convert to a float, we need to check if the value has changed and then reinitialize the tabs if necessary
        if rfr != st.session_state.risk_free_rate * 100.0:
            st.session_state.risk_free_rate = rfr / 100.0
            reinitialize_tabs('risk_free_rate')
    
        st.slider("Initial investment", min_value=0, max_value=1000000, step=5000, key='initial_investment', format="$%d", on_change=reinitialize_tabs, kwargs={'widget_source':'initial_investment'})
        st.slider("Yearly contribution", min_value=0, max_value=250000, step=5000, key='yearly_contribution', format="$%d", on_change=reinitialize_tabs, kwargs={'widget_source':'yearly_contribution'})
        st.slider("Years to invest", min_value=1, max_value=50, step=1, key='years', on_change=reinitialize_tabs, kwargs={'widget_source':'years'})
        st.radio("Mean returns model", utils.get_mean_returns_models(), key="mean_returns_model", on_change=reinitialize_tabs, kwargs={'widget_source':'mean_returns_model'})
    
        st.slider("Risk Level", min_value=st.session_state.min_risk, max_value=st.session_state.max_risk, step=0.01, key="risk_level", format="%.2f", on_change=reinitialize_tabs, kwargs={'widget_source':'risk_level'})


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
        
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(["Portfolio Analysis","Returns Analysis", "Macro Economic Analysis", 
                                                                        "Rebalancing Analysis", "Portfolio Optimization","Financials Analysis",
                                                                        "Technical Analysis","Forecasting Analysis","News Analysis"])

        
        with tab1:
            portfolio.display_selected_portfolio(st.session_state.portfolio_summary, st.session_state.portfolio_df)
        
        with tab2:
            returns.display_portfolio_returns_analysis(st.session_state.portfolio_summary)
            
        with tab3:
            macro.display_macro_analysis(st.session_state.portfolio_summary)
            
        with tab4:
            rebalancing.display_rebalancing_analysis(st.session_state.portfolio_summary)
    
        with tab5:
            optimization.display_portfolio_optimization(st.session_state.portfolio_summary)
            
        with tab6:
            financials.display_financials_analysis(st.session_state.portfolio_summary)
            
        with tab7:
            technical.display_technical_analysis(st.session_state.portfolio_summary)
            
        with tab8:
            forecasting.display_forecasting_analysis(st.session_state.portfolio_summary)
            
        with tab9:
            news.display_news_analysis(st.session_state.portfolio_summary)
    else:
        st.write("Please enter valid tickers and be sure end date is after start date")
        
if __name__ == "__main__":
    app()        