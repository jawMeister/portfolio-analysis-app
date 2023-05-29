import pandas as pd
import streamlit as st
import requests
import json

import multiprocessing
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
import time

from stqdm import stqdm

import logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s (%(levelname)s):  %(module)s.%(funcName)s - %(message)s')

# Set up logger for a specific module to a different level
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

import src.session as session
import src.financials.interpret as interpret

def create_financial_summary_dict(financial_statements, ticker, period, n_periods):
    # Initialize dictionary to hold all dataframes
    dfs_dict = {'Income Statement': None, 'Balance Sheet': None, 'Cash Flow Statement': None}

    for statement_type in financial_statements:
        logger.debug(f"Summarizing {statement_type} for ticker {ticker}, expecting {n_periods}, have {len(financial_statements[statement_type])}\n\n")    
        statements = financial_statements[statement_type]
        
        if len(financial_statements[statement_type]) != n_periods:
            #st.write(f"Expected {n_periods} {period} {statement_type} for {ticker}, found {len(financial_statements[statement_type])}")
            logger.error(f"Expected {n_periods} {period} {statement_type} for {ticker}, found {len(financial_statements[statement_type])}")
            
        all_statements = []
        for i, statement in enumerate(statements):
            logger.debug(f"Summarizing {statement_type} {i+1} for {ticker} from\n{statement}\n\n")
            # Convert statement dictionary to DataFrame
            statement_df = pd.json_normalize(statement)
            # Append statement_df to list of all statements of this type
            all_statements.append(statement_df)
        
        # Concatenate all statements of this type into one dataframe
        dfs_dict[statement_type] = pd.concat(all_statements, ignore_index=True)
        # Set 'date' as index for the dataframe
        dfs_dict[statement_type].set_index('date', inplace=True)

        # Sort by date
        dfs_dict[statement_type].sort_index(inplace=True)
        logger.debug(f"Concatenated {statement_type} for {ticker}:\n{dfs_dict[statement_type]}\n\n")

        # Calculate rolling averages for financial ratios specific to this type of statement
        if statement_type == "Income Statement":
            dfs_dict[statement_type]['net_profit_margin'] = dfs_dict[statement_type]['netIncome'] / dfs_dict[statement_type]['revenue']
            dfs_dict[statement_type]['operating_margin'] = dfs_dict[statement_type]['operatingIncome'] / dfs_dict[statement_type]['revenue']
            dfs_dict[statement_type]['net_profit_margin_rolling'] = dfs_dict[statement_type]['net_profit_margin'].rolling(n_periods).mean()
            dfs_dict[statement_type]['operating_margin_rolling'] = dfs_dict[statement_type]['operating_margin'].rolling(n_periods).mean()
        elif statement_type == "Balance Sheet":
            dfs_dict[statement_type]['current_ratio'] = dfs_dict[statement_type]['totalCurrentAssets'] / dfs_dict[statement_type]['totalCurrentLiabilities']
            dfs_dict[statement_type]['debt_to_equity'] = dfs_dict[statement_type]['totalDebt'] / dfs_dict[statement_type]['totalStockholdersEquity']
            dfs_dict[statement_type]['current_ratio_rolling'] = dfs_dict[statement_type]['current_ratio'].rolling(n_periods).mean()
            dfs_dict[statement_type]['debt_to_equity_rolling'] = dfs_dict[statement_type]['debt_to_equity'].rolling(n_periods).mean()

    logger.debug(f"Calculating cross statement metrics for {ticker}\n\n")
    # Calculate cross-statement metrics
    cross_statement_metrics = dfs_dict['Income Statement'][['revenue']].copy()  # start with a DataFrame that includes only the 'revenue' column
    logger.debug(f"cross_statement_metrics:\n{cross_statement_metrics}\n\n")
    logger.debug(f"dfs_dict['Cash Flow Statement']:\n{dfs_dict['Cash Flow Statement']}\n\n")
    logger.debug(f"cross_statement_metrics['revenue']:\n{cross_statement_metrics['revenue']}\n\n")
    cross_statement_metrics['cash_flow_margin'] = dfs_dict['Cash Flow Statement']['netCashProvidedByOperatingActivities'] / cross_statement_metrics['revenue']
    cross_statement_metrics['cash_flow_margin_rolling'] = cross_statement_metrics['cash_flow_margin'].rolling(n_periods).mean()

    # Add cross-statement metrics to dictionary
    dfs_dict['Cross Statement Metrics'] = cross_statement_metrics

    # Return dictionary of DataFrames
    return dfs_dict

def create_financial_summary_df(financial_statements, ticker, n_periods, period):
    # Create empty lists for each financial statement type
    income_statements = []
    balance_sheets = []
    cash_flow_statements = []

    for statement_type in financial_statements:
        logger.debug(f"Summarizing {statement_type} for ticker {ticker}, expecting {n_periods}, have {len(financial_statements[statement_type])}\n\n")        
        statements = financial_statements[statement_type]
        
        if len(financial_statements[statement_type]) != n_periods:
            #st.write(f"Expected {n_periods} {period} {statement_type} for {ticker}, found {len(financial_statements[statement_type])}")
            logger.error(f"Expected {n_periods} {period} {statement_type} for {ticker}, found {len(financial_statements[statement_type])}")
            
        for i, statement in enumerate(statements):
            logger.debug(f"Summarizing {statement_type} {i+1} for {ticker} from\n{statement}\n\n")
            # Convert statement dictionary to DataFrame
            statement_df = pd.json_normalize(statement)
            # Append statement_df to the corresponding list
            if statement_type == "Income Statement":
                income_statements.append(statement_df)
            elif statement_type == "Balance Sheet":
                balance_sheets.append(statement_df)
            elif statement_type == "Cash Flow Statement":
                cash_flow_statements.append(statement_df)

    logger.debug(f"Concatenating {len(income_statements)} income statements, {len(balance_sheets)} balance sheets, and {len(cash_flow_statements)} cash flow statements for {ticker}")
    # Use pd.concat to concatenate all DataFrames in each list
    income_df = pd.concat(income_statements, ignore_index=True)
    logger.debug(f"Concatenated income statements:\n{income_df}\n\n")
    balance_df = pd.concat(balance_sheets, ignore_index=True)
    logger.debug(f"Concatenated balance sheets:\n{balance_df}\n\n")
    cash_flow_df = pd.concat(cash_flow_statements, ignore_index=True)
    logger.debug(f"Concatenated cash flow statements:\n{cash_flow_df}\n\n")
    
    # Set 'date' as index for each DataFrame
    income_df.set_index('date', inplace=True)
    balance_df.set_index('date', inplace=True)
    cash_flow_df.set_index('date', inplace=True)
    
    # Join all the DataFrames together
    financial_summary = pd.concat([income_df, balance_df, cash_flow_df], axis=1)
    logger.debug(f"Joined financial summary:\n{financial_summary}\n\n")
    
    # Calculate ratios
    financial_summary['net_profit_margin'] = financial_summary['netIncome'] / financial_summary['revenue']
    financial_summary['operating_margin'] = financial_summary['operatingIncome'] / financial_summary['revenue']
    financial_summary['current_ratio'] = financial_summary['totalCurrentAssets'] / financial_summary['totalCurrentLiabilities']
    financial_summary['debt_to_equity'] = financial_summary['totalDebt'] / financial_summary['totalStockholdersEquity']
    financial_summary['cash_flow_margin'] = financial_summary['netCashProvidedByOperatingActivities'] / financial_summary['revenue']
    logger.debug(f"Final Financial summary with ratios:\n{financial_summary}\n\n")

    # Return financial summary DataFrame
    return financial_summary

@st.cache_data
def get_financial_statement(statement_type, ticker, period, n_periods):
    if session.check_for_fmp_api_key():
        fmp_api_key = session.get_fmp_api_key()
        base_url = "https://financialmodelingprep.com/api/v3"

        if statement_type == "Income Statement":
            url = f"{base_url}/income-statement/{ticker}"
        elif statement_type == "Balance Sheet":
            url = f"{base_url}/balance-sheet-statement/{ticker}"
        elif statement_type == "Cash Flow Statement":
            url = f"{base_url}/cash-flow-statement/{ticker}"
        else:
            raise ValueError("Invalid statement_type. Must be one of ['Income Statement', 'Balance Sheet', 'Cash Flow Statement']")

        params = {
            "period": period.lower(),
            "limit": n_periods,
            "apikey": fmp_api_key
        }

        logger.debug(f"Querying {url} with params {params}")
        response = requests.get(url, params=params)

        if response.status_code == 200:
            data = response.json()
            return data
        else:
            raise Exception(f"Query failed to run by returning code of {response.status_code}. {response.json()}")
        
def retrieve_financial_summary_and_analysis_for_tickers(tickers, period, n_periods, statement_types):
    start_time = time.time()
    n_cores = multiprocessing.cpu_count()
    max_workers = min(n_cores, len(tickers))
    
    results = {}
    with stqdm(total=len(tickers)) as pbar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for ticker in tickers:
                results[ticker] = {}
                results[ticker]['financial_summary'] = None
                results[ticker]['analysis'] = None
                logger.debug(f"Submitting {ticker} to executor")
                future = executor.submit(retrieve_financial_summary_and_analysis, ticker, period, n_periods, statement_types)
                futures.append(future)
                
            for future in concurrent.futures.as_completed(futures):
                ticker, financial_summary, analysis = future.result()
                results[ticker]['financial_summary'] = financial_summary
                results[ticker]['analysis'] = analysis
                logger.debug(f"finished processing {ticker}")
                pbar.update(1)

        
    end_time = time.time()
    logger.debug(f"The loop took {end_time - start_time} seconds to run")
    
    return results

def retrieve_financial_summary_and_analysis(ticker, period, n_periods, statement_types):
    
    financial_statements = retrieve_financial_statements(ticker, period, n_periods, statement_types)
    financial_summary = create_financial_summary_dict(financial_statements, ticker, period, n_periods)
    analysis = analyze_financial_statements(financial_summary, ticker, period, n_periods)
    
    logger.debug(f"Finished processing {ticker}, analysis:\n{analysis}\n\n")
    
    return ticker, financial_summary, analysis

def retrieve_financial_statements(ticker, period, n_periods, statement_types):
    ticker = ticker.upper()

    logger.debug(f"Getting {n_periods} {period} financial statements of {statement_types} for {ticker}")
    # TODO: multi-thread / process this
    financial_statements = {}
    for financial_statement_type in statement_types:
        logger.info(f"Getting {financial_statement_type} for {ticker}")
        financial_statements[financial_statement_type] = get_financial_statement(financial_statement_type, ticker, st.session_state['period'], st.session_state['n_periods'])
        
    return financial_statements

def analyze_financial_statements(financial_summary, ticker, period, n_periods):
    escaped_text = None
    if session.check_for_openai_api_key():            
        with st.spinner(f"Waiting for OpenAI API to Analyze Financial Statements for {ticker}..."):
            response = interpret.openai_analyze_financial_statements_dict(financial_summary, ticker, period, n_periods)
            logger.debug(f"Analysis for {ticker}:\n{response}")
            escaped_text = response.replace("$", "\\$")
            
    return escaped_text
    
    """    with st.expander(f"View Financial Statement Summaries for {ticker}"):
        for financial_summary_type in financial_summary.keys():
            st.write(f"Summary of {st.session_state['n_periods']} {financial_summary_type} key metrics for {ticker} across {st.session_state['n_periods']}:")
            st.write(financial_summary[financial_summary_type]) _summary_
    """