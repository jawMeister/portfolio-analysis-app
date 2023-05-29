import streamlit as st
import pandas as pd
print(pd.__version__)

import logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s (%(levelname)s):  %(module)s.%(funcName)s - %(message)s')

# Set up logger for a specific module to a different level
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

import src.session as session
import src.financials.interpret as interpret
import src.financials.calculate as calculate
import src.financials.plot as plot

def initialize_input_variables(portfolio_summary):
    if 'statement_types' not in st.session_state:
        st.session_state['statement_types'] = ['Income Statement', 'Balance Sheet', 'Cash Flow Statement']
    
    if 'period' not in st.session_state:
        st.session_state['period'] = 'Annual'
    
    # TODO: periods max based on token limit going to open ai for financials processing
    # as of 5/28/2023, 4 periods is the max as it is 1,000 tokens
    # gets initialized in the radio button below
    #if 'n_periods' not in st.session_state:
    #    st.session_state['n_periods'] = 4
    if 'financial_summary_and_analysis_by_ticker' not in st.session_state:
        st.session_state['financial_summary_and_analysis_by_ticker'] = None
        
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
        

def display_financials_analysis_for_tickers(tickers):
    financial_summaries = st.session_state['financial_summary_and_analysis_by_ticker']
    for ticker in tickers:
        with st.container():
            st.markdown("<hr style='color:#FF4B4B;'>", unsafe_allow_html=True)
            st.subheader(f"View Financial Statements and Analysis for {ticker}")

            financial_summary = financial_summaries[ticker]['financial_summary']
            analysis = financial_summaries[ticker]['analysis']
            
            st.write(analysis)    

            income_df = financial_summary['Income Statement']
            balance_df = financial_summary['Balance Sheet']
            cash_flow_df = financial_summary['Cash Flow Statement']
            cross_df = financial_summary['Cross Statement Metrics']
                    
            with st.container():
                col1, col2, col3 = st.columns(3, gap="large")
                with col1:
                    # Trend of Revenues, Net Income, and Cash Flow Over Time
                    st.plotly_chart(plot.plot_line_chart(income_df, ['revenue', 'netIncome'], 'Trend of Revenues and Net Income Over Time'), use_container_width=True)
                    st.plotly_chart(plot.plot_line_chart(cash_flow_df, ['netCashProvidedByOperatingActivities'], 'Trend of Net Cash Provided by Operating Activities Over Time'), use_container_width=True)
                    
                with col2:
                    # Trend of Assets, Liabilities, and Equity Over Time
                    st.plotly_chart(plot.plot_line_chart(balance_df, ['totalAssets', 'totalLiabilities', 'totalStockholdersEquity'], 'Trend of Assets, Liabilities, and Equity Over Time'), use_container_width=True)
                    
                    # Trend of Key Ratios Over Time
                    st.plotly_chart(plot.plot_line_chart(balance_df, ['debt_to_equity', 'current_ratio'], 'Trend of Debt to Equity and Current Ratio Over Time'), use_container_width=True)
                    
                with col3:
                    # Trend of Margins Over Time
                    st.plotly_chart(plot.plot_line_chart(income_df, ['net_profit_margin', 'operating_margin'], 'Trend of Net Profit Margin and Operating Margin Over Time'), use_container_width=True)
                    st.plotly_chart(plot.plot_line_chart(cross_df, ['cash_flow_margin'], 'Trend of Cash Flow Margin Over Time'), use_container_width=True)
                    
            with st.expander(f"View Income Statements for {ticker}"):
                st.dataframe(income_df)
            with st.expander(f"View Balance Sheets for {ticker}"):
                st.dataframe(balance_df)
            with st.expander(f"View Cash Flow Statements for {ticker}"):
                st.dataframe(cash_flow_df)
            with st.expander(f"View Cross Statement Metrics for {ticker}"):
                st.dataframe(cross_df)

def display_financials_analysis(portfolio_summary):
    input_container = st.container()
    output_container = st.container()
    
    with input_container:
        col1, col2, col3 = st.columns([2,2,2], gap="large")
        with col1:
            st.markdown("<b>Financials Analysis Inputs</b>", unsafe_allow_html=True)
            with st.form(key='financials_form'):
                subcol1, subcol2 = st.columns(2)
                with subcol1:
                    initialize_input_variables(portfolio_summary)
                            
                    st.radio("Select period:", ["Annual", "Quarterly"], key="period")
                    st.radio("Number of past financial statements to analyze:", [2,3,4], index=2, key="n_periods")
                    
                    if not session.check_for_fmp_api_key():
                        label = "Enter [FMP API Key](https://financialmodelingprep.com/developer/docs/) to retrieve financial statements"
                        temp_key = st.text_input(label, value=session.get_fmp_api_key())
                        if temp_key:
                            session.set_fmp_api_key(temp_key)
                        
                with subcol2:
                    tickers = st.text_area("Enter ticker symbols (comma separated)", key="tickers_for_financials")
                    tickers = tickers.split(",")
                    tickers = list(set(tickers))
                    
                    submitted = st.form_submit_button("Retrieve & Analyze Income Statement, Balance Sheet and Cash Flow Statements")
            st.caption("OpenAI token limit currently limiting to 4 periods of financial statements. Financial statement data provided by [Financial Modeling Prep](https://financialmodelingprep.com/developer/docs/).")
        with col2:
            display_trend_revenue_description()
            display_trend_assets_and_liabilities_description()
        with col3:
            display_trend_ratios_description()
            
    with output_container:
        if submitted:
            st.session_state['ticker_list'] = tickers
            with st.spinner("Retrieving financial statements and analyzing..."):
                financial_summary_and_analysis_by_ticker = calculate.retrieve_financial_summary_and_analysis_for_tickers(st.session_state['ticker_list'], 
                                                                                                                        st.session_state['period'], 
                                                                                                                        st.session_state['n_periods'], 
                                                                                                                        st.session_state['statement_types'])
            st.session_state['financial_summary_and_analysis_by_ticker'] = financial_summary_and_analysis_by_ticker
            
        if 'financial_summary_and_analysis_by_ticker' in st.session_state and st.session_state['financial_summary_and_analysis_by_ticker']:
            display_financials_analysis_for_tickers(st.session_state['ticker_list'])

            
def display_trend_revenue_description():
    st.markdown("<ul>", unsafe_allow_html=True)
    st.markdown("<b>Trend of Revenues, Net Income, and Cash Flow Over Time:</b>", unsafe_allow_html=True)
    
    st.markdown("<li><span style='color:#FF4B4B;'>Revenue:</span> It's the total income of a business from its operations. It's important because it gives an idea of how much money a company is bringing in. An increasing trend is generally seen as positive.", unsafe_allow_html=True)
    st.markdown("<li><span style='color:#FF4B4B;'>Net Income:</span> It's the profit of a company after subtracting all expenses, including taxes and costs. Increasing net income over time can indicate a growing business.", unsafe_allow_html=True)
    st.markdown("<li><span style='color:#FF4B4B;'>Net Cash Provided by Operating Activities:</span> This is the cash flow from the company's core business operations. It reflects how much cash is generated from a company's products or services.", unsafe_allow_html=True)
    st.markdown("</ul>", unsafe_allow_html=True)

def display_trend_assets_and_liabilities_description():
    st.markdown("<ul>", unsafe_allow_html=True)
    st.markdown("<b>Trend of Assets, Liabilities, and Equity Over Time:</b>", unsafe_allow_html=True)

    st.markdown("<li><span style='color:#FF4B4B;'>Total Assets:</span> They are everything a company owns that has monetary value. This includes both tangible and intangible assets.", unsafe_allow_html=True)
    st.markdown("<li><span style='color:#FF4B4B;'>Total Liabilities:</span> They are the company's debts or obligations.", unsafe_allow_html=True)
    st.markdown("<li><span style='color:#FF4B4B;'>Total Stockholders Equity:</span> It represents the net value of a company, i.e., the amount that shareholders would receive if all the company's assets were sold and all its debts repaid.", unsafe_allow_html=True)
    st.markdown("</ul>", unsafe_allow_html=True)
            
def display_trend_ratios_description():
    st.markdown("<ul>", unsafe_allow_html=True)
    st.markdown("<b>Trend of Key Ratios Over Time:</b>", unsafe_allow_html=True)

    st.markdown("<li><span style='color:#FF4B4B;'>Debt to Equity Ratio (D/E):</span> This ratio is used to evaluate a company's financial leverage and is calculated by dividing total liabilities by shareholder's equity. A high D/E ratio generally means that a company has been aggressive in financing its growth with debt.", unsafe_allow_html=True)
    st.markdown("<li><span style='color:#FF4B4B;'>Current Ratio:</span> This is a liquidity ratio that measures a company's ability to pay short-term obligations.", unsafe_allow_html=True)
    st.markdown("<li><span style='color:#FF4B4B;'>Net Profit Margin:</span> It is a key profitability metric for a company. It is calculated as Net Income divided by Total Revenue.", unsafe_allow_html=True)
    st.markdown("<li><span style='color:#FF4B4B;'>Operating Margin:</span> It measures the proportion of revenue left after deducting the cost of goods sold (COGS) and operational expenses. It is a measure of operational efficiency and profitability.", unsafe_allow_html=True)
    st.markdown("<li><span style='color:#FF4B4B;'>Cash Flow Margin:</span> This ratio reflects the ability of a company to turn sales into cash. It's an efficiency ratio that measures how effectively a company uses its sales and is calculated as Cash Flow from Operations divided by Net Sales.", unsafe_allow_html=True)
    st.markdown("</ul>", unsafe_allow_html=True)