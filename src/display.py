import streamlit as st
import traceback

import utils
import plots

def display_selected_portfolio(portfolio_df, portfolio_summary):
    expected_return = portfolio_summary["portfolio_expected_return"]
    initial_investment = portfolio_summary["initial_investment"]
    st.write(f"\nPortfolio Performance: projected annual return **{expected_return*100:.1f}%** or \
                **\\${initial_investment*expected_return:,.0f}** based on initial investment of \\${initial_investment:,.0f}")
    
    displayed_portfolio = portfolio_df.copy()
    displayed_portfolio = displayed_portfolio.sort_values(by=["Weight"], ascending=False)
    
    # drop the first column as default display is the index which are the tickers, so redundant
    displayed_portfolio = displayed_portfolio.drop(displayed_portfolio.columns[0], axis=1)
    
    # Formatting
    displayed_portfolio['Weight'] = (displayed_portfolio['Weight'] * 100).map("{:.1f}%".format)
    displayed_portfolio['Initial Allocation'] = displayed_portfolio['Initial Allocation'].map("${:,.0f}".format)
    displayed_portfolio['Expected Return (%)'] = (displayed_portfolio['Expected Return (%)'] * 100).map("{:.1f}%".format)
    displayed_portfolio['Expected Dividend Yield (%)'] = (displayed_portfolio['Expected Dividend Yield (%)'] * 100).map("{:.1f}%".format)
    displayed_portfolio['Expected 1 Year Return ($)'] = displayed_portfolio['Expected 1 Year Return ($)'].map("${:,.0f}".format)
    
    st.dataframe(displayed_portfolio, use_container_width=True)

def display_portfolio_results(initial_investment, ret, sharpe_ratio_val, sortino_ratio_val, cvar, total_return, years):
    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.write(f"**Sharpe Ratio:** {sharpe_ratio_val:.2f}")
        with col2:
            st.write(f"**Sortino Ratio:** {sortino_ratio_val:.2f}")
        with col3:
            st.write(f"**CVaR:** {cvar:.2f}")
        with col4:
            st.write(f"**Treynor Ratio (TBD):** {0:.2f}")
            
        st.write(f"Assuming that yearly contributions are made one time at the end of each year (after the annual return has \
                    been applied for that year), not including any taxes, fees or dividends and not accounting for individual \
                    appreciation rates by asset, a portfolio with a {ret*100:.1f}% annual return over {years} years could be \
                    worth ${total_return:,.0f}. Currently working on simulating future returns based on selected portfolio.")
            
    st.write("Calculations based on the [PyPortfolioOpt](https://pyportfolioopt.readthedocs.io/en/latest/index.html) library, additional references for education and chosen calculations:")
    st.markdown("- https://reasonabledeviations.com/2018/09/27/lessons-portfolio-opt/\n- https://www.investopedia.com/terms/c/capm.asp\n- https://reasonabledeviations.com/notes/papers/ledoit_wolf_covariance/\n")

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
def display_portfolio(stock_data, dividend_data, mu, S, start_date, end_date, risk_level, initial_investment, yearly_contribution, years, risk_free_rate):
    try:
        # Calculate portfolio statistics
        portfolio_df, portfolio_summary = \
            utils.calculate_portfolio_df(stock_data, dividend_data, 
                                         mu, S, start_date, end_date, risk_level, initial_investment, yearly_contribution, years, risk_free_rate)
        
        # Calculate efficient portfolios for plotting
        efficient_portfolios = utils.calculate_efficient_portfolios(mu, S, risk_free_rate)
        
        # Get the selected and optimal portfolios
        selected_portfolio = utils.calculate_portfolio_performance(portfolio_summary["risk_level"], 
                                                                   portfolio_summary["weights"], 
                                                                   portfolio_summary["portfolio_expected_return"], 
                                                                   portfolio_summary["volatility"], 
                                                                   portfolio_summary["sharpe_ratio"])
        
        optimal_portfolio = utils.calculate_optimal_portfolio(efficient_portfolios)
        
        # Calculate projected returns
        total_return = utils.calculate_total_return(initial_investment, portfolio_summary["portfolio_expected_return"], yearly_contribution, years)
        #asset_values, detailed_asset_holdings = utils.calculate_asset_values(stock_data, dividend_data, portfolio_summary["weights"], portfolio_summary["individual_returns"], portfolio_summary["mu"], initial_investment, yearly_contribution, years)
        asset_values, detailed_asset_holdings = utils.calculate_future_asset_holdings(portfolio_summary)
        
        with st.container():
            col1, col2, col3 = st.columns([1, 1, 1])
    
            with col1:
                display_selected_portfolio(portfolio_df, portfolio_summary)
                display_portfolio_results(initial_investment, portfolio_summary["portfolio_expected_return"], portfolio_summary["sharpe_ratio"], 
                                          portfolio_summary["sortino_ratio"], portfolio_summary["cvar"], total_return, years)
                st.write("Sharpe Ratio of 1.08: A Sharpe ratio of 1.08 indicates that the investment or portfolio generated a positive risk-adjusted return. It suggests that, on average, the investment or portfolio earned 1.08 units of excess return over the risk-free rate per unit of standard deviation. A higher Sharpe ratio is generally considered favorable, indicating better risk-adjusted performance.")
                st.write("Sortino Ratio of 1.86: A Sortino ratio of 1.86 suggests that the investment or portfolio achieved a favorable risk-adjusted return relative to its downside risk. It means that the investment's or portfolio's return was 1.86 times greater than its downside deviation. The Sortino ratio emphasizes the protection against downside risk, and a higher value is generally desirable.")
                st.write("CVaR (Conditional Value at Risk) of 0.03: A CVaR of 0.03 indicates that there is a 3% probability that the investment or portfolio may experience a loss beyond the specified confidence level. A lower CVaR indicates a lower expected loss, which can be seen as a more favorable risk characteristic.")
                st.write("\nTaken together, these metrics (1.08, 1.86, 0.03) suggest that the investment or portfolio has generated positive risk-adjusted returns, exhibited favorable performance relative to both overall and downside risk, and has a relatively low expected loss during extreme events. However, it's essential to consider these metrics in conjunction with other factors such as investment objectives, time horizon, and risk tolerance to make informed investment decisions.")
                # st.write(f"\n**Total Return** based on \\${initial_investment:,.0f} initial investment and compounded with \\${yearly_contribution:,.0f} yearly contributions: \\${total_return:,.2f}")  
            
            with col2:
                # Display portfolio details
                plots.plot_historical_performance(stock_data, dividend_data, start_date, end_date, selected_portfolio)
                            
            with col3:
                # Display portfolio details
                plots.plot_efficient_frontier(efficient_portfolios, selected_portfolio, optimal_portfolio)
                plots.plot_efficient_frontier_bar_chart(efficient_portfolios, selected_portfolio, optimal_portfolio)  
        
        return asset_values, detailed_asset_holdings
        
    except Exception as e:
        st.write("An error occurred during the calculation. Please check your inputs.")
        st.write(str(e))
        
        # send to stdout
        traceback.print_exc()
        
        # send to web screen
        stack_trace = traceback.format_exc()
        st.write(stack_trace)