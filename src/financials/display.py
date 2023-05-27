import streamlit as st
import pandas as pd
print(pd.__version__)

from stqdm import stqdm

import logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s (%(levelname)s):  %(module)s.%(funcName)s - %(message)s')

# Set up logger for a specific module to a different level
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

import src.session as session
import src.financials.interpret as interpret
import src.financials.calculate as calculate

def display_financials_analysis(portfolio_summary):
    input_container = st.container()
    output_container = st.container()
    
    with input_container:
        st.caption("[Data provided by Financial Modeling Prep](https://financialmodelingprep.com/developer/docs/)")

        col1, col2, col3, col4, col5, col6 = st.columns(6, gap="large")
        with col1:
            # TODO: just remove the financial statement type altogether and pull all three
            if 'statement_type' not in st.session_state:
                st.session_state['statement_type'] = ['Income Statement', 'Balance Sheet', 'Cash Flow Statement']
                
            period = st.selectbox("Select period:", ["Annual", "Quarterly"]).lower()
            n_periods = st.number_input("Number of past financial statements to analyze:", min_value=2, max_value=4, value=4)

            if 'tickers_for_financials' not in st.session_state:
                tickers_default = ""
                for i, ticker in enumerate(portfolio_summary['tickers']):
                    # strip out the cryptos and anything without a weight
                    if '-USD' not in ticker and portfolio_summary['weights'][ticker] > 0:
                        tickers_default += f"{ticker},"
                        
                logger.debug(f"tickers_default: {tickers_default}")
                # remove the last comma
                tickers_default = tickers_default[:-1]
                            
                st.session_state['tickers_for_financials'] = tickers_default
                logger.debug(f"Setting tickers_for_financials to {st.session_state['tickers_for_financials']}")
            
            #TODO: get this from the portfolio summary
            tickers = st.text_area("Enter ticker symbols (comma separated)", key="tickers_for_financials")
            tickers = tickers.split(",")
            tickers = list(set(tickers))
  
    if st.button('Retrieve & Analyze Income Statement, Balance Sheet and Cash Flow Statements'):
        # TODO: make this a callback
        for ticker in stqdm(tickers):
            with st.container():
                col1, col2, col3 = st.columns(3)
                
                st.markdown("<hr style='color:#FF4B4B;'>", unsafe_allow_html=True)
                st.write(f"View Financial Statements and Analysis for {ticker}")
                ticker = ticker.upper()
                
                with col1:
                    logger.debug(f"************Getting {n_periods} {period} financial statements for {ticker}***********")
                    # TODO: multi-thread / process this
                    financial_statements = {}
                    for financial_statement_type in st.session_state['statement_type']:
                        logger.info(f"Getting {financial_statement_type} for {ticker}")
                        financial_statements[financial_statement_type] = calculate.get_financial_statement(financial_statement_type, ticker, period, n_periods)

                    financial_summary = calculate.create_financial_summary_dict(financial_statements, ticker, period, n_periods)
                    with st.expander(f"View Financial Statement Summaries for {ticker}"):
                        for financial_summary_type in financial_summary.keys():
                            st.write(f"Summary of {period} {financial_summary_type} key metrics for {ticker} across {n_periods}:")
                            st.write(financial_summary[financial_summary_type]) 
                    
                    if session.check_for_openai_api_key():
                        if f"openai_financials_analysis_response_{ticker}" not in st.session_state:
                            st.session_state[f'openai_financials_analysis_response_{ticker}'] = None
                            
                        with st.spinner(f"Waiting for OpenAI API to Analyze Financial Statements for {ticker}..."):
                            response = interpret.openai_analyze_financial_statements_dict(financial_summary, ticker, period, n_periods)
                            st.session_state[f'openai_financials_analysis_response_{ticker}'] = response
                                
                        if st.session_state[f'openai_financials_analysis_response_{ticker}']:
                            logger.debug(f"{st.session_state[f'openai_financials_analysis_response_{ticker}']}")
                            st.write(f"{st.session_state[f'openai_financials_analysis_response_{ticker}']}")
                            
                with col2:
                    income_df = financial_summary['Income Statement']
                    balance_df = financial_summary['Balance Sheet']
                    cash_flow_df = financial_summary['Cash Flow Statement']
                    cross_df = financial_summary['Cross Statement Metrics']
                    
                    st.write("Trend of Revenues, Net Income, and Cash Flow Over Time:")
                    st.line_chart(income_df[['revenue', 'netIncome']])
                    st.line_chart(cash_flow_df['netCashProvidedByOperatingActivities'])
                    
                    st.write("Trend of Assets, Liabilities, and Equity Over Time:")
                    st.line_chart(balance_df[['totalAssets', 'totalLiabilities', 'totalStockholdersEquity']])
                    
                with col3:
                    st.write("Trend of Key Ratios Over Time:")
                    st.line_chart(balance_df['debt_to_equity'])
                    st.line_chart(balance_df['current_ratio'])
                    st.line_chart(income_df[['net_profit_margin', 'operating_margin']])
                    st.line_chart(cross_df['cash_flow_margin'])
                    
#                with st.expander("View Raw Financial Statements (JSON)"):
#                    st.write(financial_statements)


    
