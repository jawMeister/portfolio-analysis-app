import streamlit as st
import traceback
import numpy as np
import plotly.graph_objects as go
from dateutil.relativedelta import relativedelta
import logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s (%(levelname)s):  %(module)s.%(funcName)s - %(message)s')

# Set up logger for a specific module to a different level
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# refactored project in a structure and had issues with relative imports, so using absolute imports
import src.utils as utils
import config as config

import src.portfolio.plot as plot
import src.portfolio.interpret as interpret
import src.portfolio.calculate as calculate


def display_selected_portfolio_table(portfolio_df, portfolio_summary):
    expected_return = portfolio_summary["portfolio_return"]
    initial_investment = portfolio_summary["initial_investment"]
    logger.debug(f"expected_return: {expected_return}, initial_investment: {initial_investment}")
    
    cumulative_weighted_portfolio_returns = portfolio_summary["cumulative_weighted_portfolio_returns"].iloc[-1]
    
    st.write(f"\nPortfolio Performance: Cumulative returns in time period are {cumulative_weighted_portfolio_returns*100:.0f}% with projected annual return **{expected_return*100:.1f}%** or \
                **\\${initial_investment*expected_return:,.0f}** based on initial investment of \\${initial_investment:,.0f}")
    
    displayed_portfolio = portfolio_df.copy()
    displayed_portfolio = displayed_portfolio.sort_values(by=["Weight"], ascending=False)
    
    # drop the first column as default display is the index which are the tickers, so redundant
    displayed_portfolio = displayed_portfolio.drop(displayed_portfolio.columns[0], axis=1)
    
    # Formatting
    displayed_portfolio['Cumulative Return'] = (displayed_portfolio['Cumulative Return'] * 100).map("{:.0f}%".format)
    displayed_portfolio = displayed_portfolio.rename(columns={"Cumulative Return (%)": "Cumulative Return"})
    
    displayed_portfolio['Weight'] = (displayed_portfolio['Weight'] * 100).map("{:.1f}%".format)
    displayed_portfolio['Initial Allocation'] = displayed_portfolio['Initial Allocation'].map("${:,.0f}".format)
    displayed_portfolio['Expected Return (%)'] = (displayed_portfolio['Expected Return (%)'] * 100).map("{:.1f}%".format)
    displayed_portfolio['Expected Dividend Yield (%)'] = (displayed_portfolio['Expected Dividend Yield (%)'] * 100).map("{:.1f}%".format)
    displayed_portfolio['Expected 1 Year Return ($)'] = displayed_portfolio['Expected 1 Year Return ($)'].map("${:,.0f}".format)
    
    st.dataframe(displayed_portfolio, use_container_width=True)

def display_portfolio_results(initial_investment, ret, sharpe_ratio_val, sortino_ratio_val, cvar, treynor_ratio, total_return, years):
    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.write(f"**Sharpe Ratio:** {sharpe_ratio_val:.2f}")
        with col2:
            st.write(f"**Sortino Ratio:** {sortino_ratio_val:.2f}")
        with col3:
            st.write(f"**CVaR:** {cvar:.2f}")
        with col4:
            st.write(f"**Treynor Ratio:** {treynor_ratio:.2f}")
            
        logger.info(f"initial_investment: {initial_investment}, ret: {ret}, years: {years}, total_return: {total_return}")
        st.write(f"Assuming that yearly contributions are made one time at the end of each year (after the annual return has \
                    been applied for that year), not including any taxes, fees or dividends and not accounting for individual \
                    appreciation rates by asset, a portfolio with a {ret*100:.1f}% annual return over {years} years could be \
                    worth ${total_return:,.0f}. Currently working on simulating future returns based on selected portfolio, \
                    see the returns tab for progress so far.")
            

def display_asset_values(asset_values):
    st.write(f"\n**Projected return over {len(asset_values)-1} years based on portfolio weights against initial and yearly contribution with reinvested dividends:**")
    
    formatted_asset_values = asset_values.copy()
    
    # drop any column that has 0 values for all rows
    formatted_asset_values = formatted_asset_values.loc[:, (formatted_asset_values != 0).any(axis=0)]
    
    for col in formatted_asset_values.columns:
        # if it's not the year column
        if col.find('Year') == -1:
            # if it's a percentage column
            if col.find('(%)') > 0:
                # format all rows to %
                formatted_asset_values.loc[:, col] = (formatted_asset_values.loc[:, col] * 100).map("{:.0f}%".format)
            else:
                # format all rows to $
                formatted_asset_values.loc[:, col] = formatted_asset_values.loc[:, col].map("${:,.0f}".format)
        
    st.dataframe(formatted_asset_values, use_container_width=True)

# main display function
def display_selected_portfolio(portfolio_summary, portfolio_df):
        
    try:
        # Calculate efficient portfolios for plotting
        # TODO: reconsider caching this as it's not used anywhere else and button clicks on other tabs trigger this
        # maybe have a utils lib that is not cached and a wrapper per tab that is cached?
        efficient_portfolios = utils.calculate_efficient_portfolios(portfolio_summary["mu"], 
                                                                    portfolio_summary["S"], 
                                                                    st.session_state.risk_free_rate)

                        
        optimal_portfolio = utils.calculate_optimal_portfolio(efficient_portfolios)
        
        logger.info(f"optimal portfolio:\n{optimal_portfolio}")
        asset_values, detailed_asset_holdings = calculate.calculate_future_asset_holdings(portfolio_summary)
    
        # estimate projected returns
        total_return = calculate.calculate_total_return(portfolio_summary["initial_investment"], 
                                                        portfolio_summary["portfolio_return"], 
                                                        portfolio_summary["yearly_contribution"], 
                                                        portfolio_summary["years"])
        logger.debug(f"initial_investment: {portfolio_summary['initial_investment']}, \
                        sharpes: {portfolio_summary['sharpe_ratio']}, sortino: {portfolio_summary['sortino_ratio']}, \
                        cvar: {portfolio_summary['cvar']}, treynor: {portfolio_summary['treynor_ratio']}, \
                        total_return: {total_return}, years: {portfolio_summary['years']}")
                
        with st.container():
            col1, col2, col3 = st.columns([1, 1, 1], gap="large")
    
            with col1:
                display_selected_portfolio_table(portfolio_df, portfolio_summary)

                
                display_portfolio_results(portfolio_summary["initial_investment"], 
                                          portfolio_summary["portfolio_return"], 
                                          portfolio_summary["sharpe_ratio"], 
                                          portfolio_summary["sortino_ratio"], 
                                          portfolio_summary["cvar"],
                                          portfolio_summary["treynor_ratio"], 
                                          total_return, 
                                          portfolio_summary["years"])
                
            with col2:
                #plot.plot_historical_performance(portfolio_summary["stock_data"], 
                #                                  portfolio_summary["dividend_data"], 
                #                                  portfolio_summary["start_date"], 
                #                                  portfolio_summary["end_date"], 
                #                                  portfolio_summary["weights"])
                            
                df_dict = calculate.calculate_portfolio_performance(portfolio_summary["stock_data"], 
                                                                    portfolio_summary["dividend_data"], 
                                                                    portfolio_summary["weights"], 
                                                                    portfolio_summary["start_date"], 
                                                                    portfolio_summary["end_date"])    
                    
                performance_by_ticker = plot.plot_historical_performance_by_ticker(df_dict)
                portfolio_performance_by_benchmark = plot.plot_portfolio_performance_by_benchmark(df_dict)
                month_to_month_portfolio_performance = plot.plot_month_to_month_portfolio_performance(df_dict)
                    
                st.plotly_chart(performance_by_ticker, use_container_width=True)
                st.plotly_chart(portfolio_performance_by_benchmark, use_container_width=True)
                st.plotly_chart(month_to_month_portfolio_performance, use_container_width=True) 

            with col3:
                # Display portfolio details
                plot.plot_efficient_frontier(efficient_portfolios, portfolio_summary, optimal_portfolio)
                plot.plot_efficient_frontier_bar_chart(efficient_portfolios, portfolio_summary, optimal_portfolio)  
                
            with col1:
                # put up an input box if the api key not avaliable
                # TODO: what if they enter a bad key?
                if not config.check_for_api_key('openai'):
                    label = "Enter [OpenAI API Key](https://platform.openai.com/account/api-keys) to interpret portfolio results"
                    temp_key = st.text_input(label, value=config.get_api_key('openai'))
                    if temp_key:
                        config.set_api_key('openai', temp_key)
                            
                with st.form(key="Interpret Portfolio Results Form"):
                    if st.form_submit_button("Ask OpenAI to Interpret Results"):
                        # Display a message indicating the application is waiting for the API to respond
                        if config.check_for_api_key('openai'):
                            with st.spinner("Waiting for OpenAI API to respond..."):
                                response = interpret.openai_interpret_portfolio_summary(portfolio_summary)
                                st.session_state.openai_portfolio_response = response
                                logger.info(f"openAI response to portfolio question: {response}")
                        else:
                            st.error("Please enter an OpenAI API key to interpret portfolio results")
                                
                if st.session_state.openai_portfolio_response:
                    st.success(st.session_state.openai_portfolio_response)

                st.write("Calculations based on the [PyPortfolioOpt](https://pyportfolioopt.readthedocs.io/en/latest/index.html) library, additional references for education and chosen calculations:")
                st.markdown("- https://reasonabledeviations.com/2018/09/27/lessons-portfolio-opt/\n- https://www.investopedia.com/terms/c/capm.asp\n- https://reasonabledeviations.com/notes/papers/ledoit_wolf_covariance/\n")
    
    except Exception as e:
        st.write("An error occurred during the calculation. Please check your inputs.")
        st.write(str(e))
        
        # send to stdout
        traceback.print_exc()
        
        # send to web screen
        stack_trace = traceback.format_exc()
        st.write(stack_trace)
        
