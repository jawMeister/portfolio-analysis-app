import streamlit as st
import traceback

import utils
import plots
import interpret
import analysis

from config import OPENAI_API_KEY, FRED_API_KEY

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
def display_portfolio(portfolio_summary, portfolio_df, selected_portfolio, optimal_portfolio, efficient_portfolios):
    try:

        # estimate projected returns
        total_return = utils.calculate_total_return(portfolio_summary["initial_investment"], portfolio_summary["portfolio_expected_return"], portfolio_summary["yearly_contribution"], portfolio_summary["years"])
        
        with st.container():
            col1, col2, col3 = st.columns([1, 1, 1])
    
            with col1:
                display_selected_portfolio(portfolio_df, portfolio_summary)
                display_portfolio_results(portfolio_summary["initial_investment"], 
                                          portfolio_summary["portfolio_expected_return"], 
                                          portfolio_summary["sharpe_ratio"], 
                                          portfolio_summary["sortino_ratio"], 
                                          portfolio_summary["cvar"], 
                                          total_return, 
                                          portfolio_summary["years"])
                
            with col2:
                # Display portfolio details
                plots.plot_historical_performance(portfolio_summary["stock_data"], 
                                                  portfolio_summary["dividend_data"], 
                                                  portfolio_summary["start_date"], 
                                                  portfolio_summary["end_date"], 
                                                  selected_portfolio)
                            
            with col3:
                # Display portfolio details
                plots.plot_efficient_frontier(efficient_portfolios, selected_portfolio, optimal_portfolio)
                plots.plot_efficient_frontier_bar_chart(efficient_portfolios, selected_portfolio, optimal_portfolio)  
                
            with col1:
                # Initialize the API key and the flag in the session state if they are not already present
                if 'openai_api_key' not in st.session_state:
                    # Import OPENAI_API_KEY only if it has a non-empty value and is not "None"
                    if OPENAI_API_KEY and OPENAI_API_KEY.strip() and OPENAI_API_KEY != "None":
                        st.session_state.openai_api_key = OPENAI_API_KEY
                        st.session_state.key_provided = True
                    else:
                        st.session_state.openai_api_key = ""
                        st.session_state.key_provided = False

                if not st.session_state.key_provided:
                    label = "Enter [OpenAI API Key](https://platform.openai.com/account/api-keys) to interpret portfolio results"
                    temp_key = st.text_input(label, value=st.session_state.openai_api_key)
                    if temp_key:
                        st.session_state.openai_api_key = temp_key
                        st.session_state.key_provided = True

                # Create a placeholder for API response message
                placeholder = st.empty()

                if st.session_state.key_provided and st.session_state.openai_api_key != "None":
                    #print(f"OpenAI API Key: {st.session_state.openai_api_key} of type {type(st.session_state.openai_api_key)}")
                    if st.button("Ask OpenAI to Interpret Results"):
                        # Display a message indicating the application is waiting for the API to respond
                        placeholder.markdown('<p style="color:red;">Waiting for OpenAI API to respond...</p>', unsafe_allow_html=True)
                        
                        interpret.openai_interpret_portfolio_summary(portfolio_summary, st.session_state.openai_api_key)

                        # Clear the placeholder
                        placeholder.empty()

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
        
def display_portfolio_returns_analysis(portfolio_summary, asset_values):
    st.write("Analysis leveraging Monte Carlo simulations on historical volatility to estimate future returns")
    simulation_results = None
    
    with st.container():
        col1, col2, col3, col4 = st.columns([2,2,2,2])
    
        with col1:
            subcol1, subcol2 = st.columns([1,1])
            with subcol1:
                if "n_simulations" not in st.session_state:
                    st.session_state.n_simulations = 5000
                
                #TODO: add a button to execute the simulations
                st.slider("Portfolio Simulations (higher more accurate, takes longer)", min_value=500, max_value=25000, step=500, key="n_simulations")
        
#            with subcol2:
                distribution = st.radio("Returns Distribution for Simulations", ("T-Distribution", "Cauchy", "Normal"), 
                                        key="volatility_distribution", horizontal=True)
                


    with st.container():
        if st.button("Run Simulations"):
            simulation_results = analysis.run_portfolio_simulations(portfolio_summary, st.session_state.n_simulations, distribution)
            analysis.plot_simulation_results(simulation_results)
            
    with st.container():
        if simulation_results:
            analysis.plot_density_plots(simulation_results)
            
    st.write("TODO: add more analysis here")

    with st.expander("NOT reality, assumes constant growth rate"):
        display_asset_values(asset_values)
        plots.plot_asset_values(asset_values)