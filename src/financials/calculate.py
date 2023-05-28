import pandas as pd
import streamlit as st
import requests
import json

import logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s (%(levelname)s):  %(module)s.%(funcName)s - %(message)s')

# Set up logger for a specific module to a different level
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

import src.session as session

def create_financial_summary_dict(financial_statements, ticker, period, n_periods=2):
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
    
def generate_financial_summary(financial_statements, ticker, period, n_periods):

    logger.debug(f"Summarizing {n_periods} {period} {financial_statements.keys()} for {ticker}")
    # Create a summary of key financial metrics for all four periods
    all_summaries = ""
    summaries = []

    # loop over each statement type, eg, Income Statement, Balance Sheet, Cash Flow
    for statement_type in financial_statements:
        logger.debug(f"Summarizing {statement_type} for ticker {ticker}, expecting {n_periods}, have {len(financial_statements[statement_type])}\n\n")
        if len(financial_statements[statement_type]) != n_periods:
            #st.write(f"Expected {n_periods} {period} {statement_type} for {ticker}, found {len(financial_statements[statement_type])}")
            logger.error(f"Expected {n_periods} {period} {statement_type} for {ticker}, found {len(financial_statements[statement_type])}")
            #raise ValueError(f"Expected {n_periods} {period} {statement_type} for {ticker}, found {len(financial_statements[statement_type])}")
        
        # loop over each statement, eg, 2020-12-31, 2020-09-30, 2020-06-30, 2020-03-31 depending on limit set
        statements = financial_statements[statement_type]
        for i, statement in enumerate(statements):
            logger.debug(f"Summarizing {statement_type} {i+1} for {ticker} from\n{statement}\n\n")
            if statement_type == "Income Statement":
                """Net Profit Margin: This is a profitability ratio calculated as net income divided by revenue. 
                   It represents the percentage of revenue that is net profit.

                   Operating Margin: This is a profitability ratio calculated as operating income divided by revenue. 
                   It measures what proportion of a company's revenue is left over after paying for variable costs of 
                   production such as wages, raw materials, etc.
                """
                # TODO: store revenue, gross profit, operating income, net income, ebitda, research and development expenses, interest expense
                net_profit_margin = statement['netIncome'] / statement['revenue'] if statement['revenue'] != 0 else None
                operating_margin = statement['operatingIncome'] / statement['revenue'] if statement['revenue'] != 0 else None

                summary = f"""
                    For the period ending {statement['date']}, the company reported the following:
                    - Revenue: ${statement['revenue']}
                    - Gross Profit: ${statement['grossProfit']}
                    - Operating Income: ${statement['operatingIncome']}
                    - Net Income: ${statement['netIncome']}
                    - EBITDA: ${statement['ebitda']}
                    - Research and Development Expenses: ${statement['researchAndDevelopmentExpenses']}
                    - Interest Expense: ${statement['interestExpense']}
                    - Net Profit Margin: {net_profit_margin if net_profit_margin else 'N/A'}
                    - Operating Margin: {operating_margin if operating_margin else 'N/A'}
            """
            elif statement_type == "Balance Sheet":
                """Current Ratio: This is a liquidity ratio that measures a company's ability to 
                   pay short-term and long-term obligations. A higher current ratio indicates the 
                   higher capability of clearing its debts over the next 12 months. It's desirable 
                   to have a higher ratio as it represents good financial health.

                   Debt to Equity Ratio: This is a financial ratio indicating the relative proportion 
                   of shareholders' equity and debt used to finance a company's assets. The debt to equity 
                   ratio provides a good understanding of a company's financial leverage.
                """
                current_ratio = statement['totalCurrentAssets'] / statement['totalCurrentLiabilities'] if statement['totalCurrentLiabilities'] != 0 else None
                debt_to_equity = statement['totalDebt'] / statement['totalStockholdersEquity'] if statement['totalStockholdersEquity'] != 0 else None

                summary = f"""
                    For the period ending {statement['date']}, the company reported the following:
                    - Total Assets: ${statement['totalAssets']}
                    - Total Liabilities: ${statement['totalLiabilities']}
                    - Total Stockholder's Equity: ${statement['totalStockholdersEquity']}
                    - Cash and Short-term Investments: ${statement['cashAndShortTermInvestments']}
                    - Long-term Investments: ${statement['longTermInvestments']}
                    - Total Current Assets: ${statement['totalCurrentAssets']}
                    - Total Current Liabilities: ${statement['totalCurrentLiabilities']}
                    - Long-term Debt: ${statement['longTermDebt']}
                    - Total Debt: ${statement['totalDebt']}
                    - Net Debt: ${statement['netDebt']}
                    - Current Ratio: {current_ratio if current_ratio else 'N/A'}
                    - Debt to Equity Ratio: {debt_to_equity if debt_to_equity else 'N/A'}
                    """
            elif statement_type == "Cash Flow Statement":
                """Cash Flow Margin: This is a profitability ratio calculated as net cash provided by operating activities divided by revenue. 
                   It represents the percentage of each dollar of revenue that is generated as cash flow from operations.
                """
                #cash_flow_margin = statement['netCashProvidedByOperatingActivities'] / statement['revenue'] if statement['revenue'] != 0 else None

                # Calculate financial ratios that might be important
                # Note: Cash Flow Margin has been removed because 'revenue' is typically not in cash flow statements

                summary = f"""
                    For the period ending {statement['date']}, the company reported the following:
                    - Net Income: ${statement['netIncome']}
                    - Depreciation and Amortization: ${statement['depreciationAndAmortization']}
                    - Stock Based Compensation: ${statement['stockBasedCompensation']}
                    - Operating Cash Flow: ${statement['operatingCashFlow']}
                    - Capital Expenditure: ${statement['capitalExpenditure']}
                    - Free Cash Flow: ${statement['freeCashFlow']}
                    - Net Cash used in Investing Activities: ${statement['netCashUsedForInvestingActivites']}
                    - Net Cash used in Financing Activities: ${statement['netCashUsedProvidedByFinancingActivities']}
                    """
            summaries.append(summary)

    # Combine all summaries into a single string
    all_summaries = "\n\n".join(summaries)

    return all_summaries

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