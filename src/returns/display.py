import streamlit as st

import logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s (%(levelname)s):  %(module)s.%(funcName)s - %(message)s')

# Set up logger for a specific module to a different level
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import src.utils as utils
import src.session as session

import src.returns.calculate as calculate
import src.returns.plot as plot
import src.returns.interpret as interpret

# TODO: perhaps make these common? or create a class for portfolio and add the methods
from src.portfolio.plot import plot_historical_performance, plot_efficient_frontier
from src.portfolio.display import display_selected_portfolio_table

from config import OPENAI_API_KEY, FRED_API_KEY


def display_portfolio_returns_analysis(portfolio_summary):
    simulation_results = None
    input_container = st.container()
    forecast_output_container = st.container()
    backtest_output_container = st.container()
    
    with input_container:
        col1, col2, col3 = st.columns([2,2,2])
    
        with col1:
            subcol1, subcol2 = st.columns([1,1])
            with subcol1:
                if "volatility_distribution" not in st.session_state:
                    st.session_state.volatility_distribution = "T-Distribution"
                    
                st.radio("Returns Distribution for Simulations", ("T-Distribution", "Cauchy", "Normal"), key="volatility_distribution")
                                
                if "n_simulations" not in st.session_state:
                    st.session_state.n_simulations = 5000
                    
            with subcol2:
                if "simulation_mode" not in st.session_state:
                    st.session_state.simulation_mode = "Backtest and Forecast"
                    
                st.radio("Simulation Mode", ("Backtest and Forecast", "Forecast only", "Backtest only"), key="simulation_mode")
                
            #TODO: add a button to execute the simulations
            st.slider("Monte Carlo based Portfolio Simulations (higher for smoother curves, takes longer)", min_value=500, max_value=25000, step=500, key="n_simulations")
            run_simulation = st.button("Run Simulations", use_container_width=True)
            
        with col2:
            if session.check_for_openai_api_key():
                if "openai_returns_response" not in st.session_state:
                    st.session_state.openai_returns_response = None
                    
                if st.button("Ask OpenAI about the validity of this simulation"):
                    with st.spinner("Waiting for OpenAI API to respond..."):
                        response = interpret.openai_interpret_montecarlo_simulation(portfolio_summary, st.session_state.n_simulations, st.session_state.volatility_distribution)
                        st.session_state.openai_returns_response = response
                if st.session_state.openai_returns_response:
                    st.write(st.session_state.openai_returns_response)
            
    with forecast_output_container:
        st.markdown("""---""")
        st.markdown('<p style="color:red;">Forecast Simulation Results</p>',unsafe_allow_html=True)
        if run_simulation:
            if st.session_state.simulation_mode == "Forecast only" or st.session_state.simulation_mode == "Backtest and Forecast":
                simulation_results = calculate.run_portfolio_simulations(portfolio_summary, st.session_state.n_simulations, st.session_state.volatility_distribution)

                with st.container():
                    col1, col2, col3 = st.columns([1,1,1])
                    df_hist, df_scatter, df_box = calculate.summarize_simulation_results(simulation_results)
                    
                    with col1:
                        st.plotly_chart(plot.plot_histogram_data(df_hist), use_container_width=True)
                    with col2:
                        st.plotly_chart(plot.plot_scatter_data(df_scatter), use_container_width=True)
                    with col3:
                        st.plotly_chart(plot.plot_box_data(df_box), use_container_width=True)

                with st.container():
                    display_simulation_probability_density_plots(simulation_results, portfolio_summary)
    

    #with st.expander("NOT reality, assumes constant growth rate"):
    #    display_asset_values(asset_values)
    #    plots.plot_asset_values(asset_values)
    
    #TODO: code is a hot mess, refactor this
    with backtest_output_container:
        st.markdown("""---""")
        final_result_plot_placeholder = st.empty()
        st.markdown('<p style="color:red;">Backtest intermediate results, final result will be above</p>',unsafe_allow_html=True)
        
        if run_simulation:
            if st.session_state.simulation_mode == "Backtest only" or st.session_state.simulation_mode == "Backtest and Forecast":                    
                sim_portfolio_df, sim_portfolio_summary, actuals_data, simulation_data, actuals_start_date, actuals_end_date, years_to_simulate = display_setup_simulation_portfolio(portfolio_summary)
                
                with st.container():
                    col1, col2, col3 = st.columns([1,1,1])
                    with col1:
                        display_selected_portfolio_table(sim_portfolio_df, sim_portfolio_summary)
                    with col2:
                        st.write("Simulation data")
                        st.write(simulation_data)
                    with col3:
                        st.write("Actuals data")
                        st.write(actuals_data)  
    
                # this simulates performance of every asset for years_to_simulate (eg, if 7 years, will have index 0-6)
                simulation_results = calculate.run_portfolio_simulations(sim_portfolio_summary, st.session_state.n_simulations, st.session_state.volatility_distribution)

                with st.container():
                    col1, col2, col3 = st.columns([1,1,1])
                    df_hist, df_scatter, df_box = calculate.summarize_simulation_results(simulation_results)
                    
                    with col1:
                        st.plotly_chart(plot.plot_histogram_data(df_hist))
                    with col2:
                        st.plotly_chart(plot.plot_scatter_data(df_scatter))
                    with col3:
                        st.plotly_chart(plot.plot_box_data(df_box))
                
                # create the pdf plots for every year in the simulation so we can plot the sigma levels by year in combination with the actuals chart
                specific_years_to_calculate = list(range(1, years_to_simulate+1)) # 1 to years_to_simulate inclusive, so year 1, 2, 3, 4, etc.
                #print(f"specific_years_to_calculate: {specific_years_to_calculate}")
                sigma_levels_by_year, plots_by_year, returns_probability_by_year = calculate.calculate_probability_density_for_returns(simulation_results, 
                                                                                                                                        portfolio_summary["initial_investment"], 
                                                                                                                                        portfolio_summary["yearly_contribution"], 
                                                                                                                                        specific_years_to_calculate)
                # similar to above forecast simulation, plot year 1, midpoint and end year
                col1, col2, col3 = st.columns([1,1,1], gap="large")
                columns = [col1, col2, col3]                 
                # if 7 years to sim, this will be year 1, 3 and 6         
                specific_years_to_plot = [1, years_to_simulate//2, years_to_simulate]          
                for i, year in enumerate(specific_years_to_plot):
                    #print(f"plotting col {i} for year: {year}")
                    columns[i].plotly_chart(plots_by_year[year], use_container_width=True)
                    columns[i].write(returns_probability_by_year[year])
                
                for ticker, weight in sim_portfolio_summary["weights"].items():
                    if weight == 0:
                        actuals_data.drop(ticker, axis=1, inplace=True)
                        sim_portfolio_summary["dividend_data"].drop(ticker, axis=1, inplace=True)
                        
                # given the actual stock closing data, calculate what the portfolio value would have been
                actuals_portfolio_values = calculate.calculate_portfolio_value(actuals_data, 
                                                                                sim_portfolio_summary["weights"], 
                                                                                sim_portfolio_summary["initial_investment"], 
                                                                                sim_portfolio_summary["yearly_contribution"])
                
                with st.container():
                    col1, col2, col3 = st.columns([1,1,1])
                    with col1:
                        st.write("Actuals weighted performance by day")
                        st.write(actuals_portfolio_values)
                        # table of mean value by month
                        st.write("Actuals value by asset by month")
                        st.write(actuals_portfolio_values.resample('M').mean())
                        st.write("Actuals value by asset by year")
                        st.write(actuals_portfolio_values.resample('Y').mean())
                    with col2:
                        mean_cumulative_returns_plot = calculate.calculate_mean_cumulative_returns(actuals_portfolio_values)
                        st.plotly_chart(mean_cumulative_returns_plot, theme="streamlit", use_container_width=True)
                        asset_performance_plot, annual_weighted_value_plot, annual_stacked_bar_plot = calculate.calculate_plots_for_portfolio_value(actuals_portfolio_values)
                        st.plotly_chart(annual_stacked_bar_plot, theme="streamlit", use_container_width=True)
                        st.plotly_chart(asset_performance_plot, theme="streamlit", use_container_width=True)
                        st.plotly_chart(annual_weighted_value_plot, theme="streamlit", use_container_width=True)


                    with col3:
                        plot_historical_performance(actuals_data, 
                                                        sim_portfolio_summary["dividend_data"], 
                                                        actuals_start_date, 
                                                        actuals_end_date, 
                                                        sim_portfolio_summary["weights"])
                        

                    final_results_plot = plot.plot_backtest_simulation_w_sigma_levels(sigma_levels_by_year, actuals_portfolio_values, actuals_start_date)
                    
                    with final_result_plot_placeholder.container():
                        st.markdown('<p style="color:red;">Backtest Final Results: Comparing Forecast Model Probability Ranges with Actual Historical Results</p>',unsafe_allow_html=True)
                        st.write("The charts below shows the backtest simulation of the selected portfolios (blue line) compared to the forecast model probability ranges (orange lines).")
                        st.write("TODO: add absolute performance metrics to the charts below")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("Backtest simulation of selected portfolio")
                            st.plotly_chart(final_results_plot, theme="streamlit", use_container_width=True)
                            st.write("Backtest simulation of min volatility portfolio (PLACEHOLDER)")
                            st.plotly_chart(annual_weighted_value_plot, theme="streamlit", use_container_width=True)
                        with col2:
                            st.write("Backtest simulation of optimal portfolio (PLACEHOLDER)")
                            st.plotly_chart(annual_weighted_value_plot, theme="streamlit", use_container_width=True)
                            st.write("Backtest simulation of max sharpe portfolio (PLACEHOLDER)")
                            st.plotly_chart(annual_weighted_value_plot, theme="streamlit", use_container_width=True)

def display_simulation_future_returns_results(simulation_results):
    col1, col2, col3 = st.columns(3, gap="large")
        
    df_hist, df_scatter, df_box = calculate.calculate_simultion_future_returns(simulation_results)
    
    with col1:
        with st.spinner('Calculating Histogram of Final Portfolio Values...'):
            hist_plot = plot.plot_histogram(df_hist)
            st.plotly_chart(hist_plot, use_container_width=True)
    with col2:  
        with st.spinner('Calculating Scatter Plot of Simulated Portfolio Values...'):
            scat_plot = plot.plot_scatter(df_scatter)
            st.plotly_chart(scat_plot, use_container_width=True)    
    with col3:
        with st.spinner('Calculating Scatter Box Plot of Simulated Portfolio Values...'):
            box_plot = plot.plot_box(df_box)
            st.plotly_chart(box_plot, use_container_width=True) 

def display_simulation_probability_density_plots(simulation_results, portfolio_summary):
    years_in_results = len(simulation_results[0])
    # create density plots for year 1, midpoint and the last simulated year
    specific_years_to_plot = [1, years_in_results//2, years_in_results-1] 
    sigma_levels_by_year, plots_by_year, returns_probability_by_year = \
                calculate.calculate_probability_density_for_returns(simulation_results, 
                                                                    portfolio_summary["initial_investment"], 
                                                                    portfolio_summary["yearly_contribution"], 
                                                                    specific_years_to_plot)
                
    col1, col2, col3 = st.columns([1,1,1], gap="large")
    columns = [col1, col2, col3]                                 
    for i, year in enumerate(specific_years_to_plot):
        columns[i].plotly_chart(plots_by_year[year], use_container_width=True)
        columns[i].write(returns_probability_by_year[year])
        
def display_setup_simulation_portfolio(portfolio_summary):
    
    test_ratio = calculate.calculate_test_ratio(portfolio_summary)
    #print(f"test_ratio: {test_ratio}")
    simulation_data, actuals_data = calculate.split_data(portfolio_summary["stock_data"], test_ratio)
    simulation_dividends, _ = calculate.split_data(portfolio_summary["dividend_data"], test_ratio)
    #print(f"simulation_data: {simulation_data.tail()}")
    #print(f"actuals_data: {actuals_data.tail()}")
    
    sim_start_date = simulation_data.index[0]
    sim_end_date = simulation_data.index[-1]

    actuals_start_date = actuals_data.index[0]
    actuals_end_date = actuals_data.index[-1]
    
    # since validating simulation vs actuals, end the simulation at the end of the actuals data
    years_to_simulate = int((actuals_end_date - actuals_start_date).days / 365 + 1)
    
    st.write(f"Simulation of Historical Data from {sim_start_date.strftime('%Y/%m/%d')} to {sim_end_date.strftime('%Y/%m/%d')}, \
                against Actuals Data from {actuals_start_date.strftime('%Y/%m/%d')} to {actuals_end_date.strftime('%Y/%m/%d')}, \
                Forecast {years_to_simulate} years from {sim_end_date.strftime('%Y/%m/%d')}")
    """                    
    weights = portfolio_summary["weights"]
    initial_investment = portfolio_summary["initial_investment"]
    annual_contribution = portfolio_summary["yearly_contribution"]
    mean_returns = portfolio_summary["mu"]
    cov_returns = portfolio_summary["S"]
    years = portfolio_summary["years"]
    """
    
    # TODO: run the sim for min volatility portfolio, selected portfolio, optimal portfolio and max sharpes portfolio to compare
    mu = utils.calculate_mean_returns(simulation_data, st.session_state.mean_returns_model, st.session_state.risk_free_rate)
    S = utils.calculate_covariance_matrix(simulation_data)
    sim_portfolio_df, sim_portfolio_summary = utils.calculate_portfolio_df(simulation_data, simulation_dividends,
                                                                mu, S, sim_start_date, sim_end_date, 
                                                                st.session_state.risk_level, 
                                                                portfolio_summary["initial_investment"], 
                                                                portfolio_summary["yearly_contribution"], 
                                                                years_to_simulate, 
                                                                st.session_state.risk_free_rate)
    with st.container():
        col1, col2, col3 = st.columns([1,1,1])
        with col1:
            display_selected_portfolio_table(sim_portfolio_df, sim_portfolio_summary)
        with col2:
            st.write("Simulation data")
            st.write(simulation_data)
        with col3:
            st.write("Actuals data")
            st.write(actuals_data)  
            
    return sim_portfolio_df, sim_portfolio_summary, actuals_data, simulation_data, actuals_start_date, actuals_end_date, years_to_simulate