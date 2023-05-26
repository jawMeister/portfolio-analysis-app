import streamlit as st
import pandas as pd

import logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s (%(levelname)s):  %(module)s.%(funcName)s - %(message)s')

# Set up logger for a specific module to a different level
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

import src.session as session
import src.macro.interpret as interpret
import src.macro.calculate as calculate
import src.macro.plot as plot

def display_collect_future_macro_estimates(user_macro_input):
    
    logger.debug("user_macro_input: {}".format(user_macro_input))
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        user_macro_input['US Interest Rate'] = \
            st.number_input("US Interest Rate (%)", min_value=-2.0, max_value=10.0, step=0.1, value=(user_macro_input['US Interest Rate']*100), format="%.1f") / 100
        
        user_macro_input['US Inflation Rate'] = \
            st.number_input("US Inflation Rate (%)", min_value=-10.0, max_value=20.0, step=0.5, value=(user_macro_input['US Inflation Rate']*100), format="%.1f") / 100

        user_macro_input['US M2 Money Supply Rate'] = \
            st.number_input("US M2 Money Supply Rate (%)", min_value=90.0, max_value=150.0, step=2.5, value=(user_macro_input['US M2 Money Supply Rate']*100), format="%.1f") / 100

        user_macro_input['China M2 Money Supply Rate'] = \
            st.number_input("China M2 Money Supply Rate (%)", min_value=90.0, max_value=150.0, step=2.5, value=(user_macro_input['China M2 Money Supply Rate']*100), format="%.1f") / 100

#    user_macro_input['US GDP Growth Rate'] = \
#        st.number_input("US GDP Growth Rate", min_value=-2.0, max_value=10.0, step=0.1, value=user_macro_input['US GDP Growth Rate']*100, format="%.1f") / 100

#    user_macro_input['US Unemployment Rate'] = \
#        st.number_input("US Unemployment Rate", min_value=-10.0, max_value=20.0, step=0.5, value=user_macro_input['US Unemployment Rate']*100, format="%.1f") / 100


    return user_macro_input

def display_macro_analysis(portfolio_summary):

    col1, col2, col3 = st.columns(3)
    with col1:
        user_macro_input = calculate.get_macro_factor_defaults()
        user_macro_input = display_collect_future_macro_estimates(user_macro_input)
        
        historical_macro_data = calculate.get_historical_macro_data(portfolio_summary["start_date"], portfolio_summary["end_date"])
        combined_data = calculate.clean_and_combine_macro_data(portfolio_summary, historical_macro_data)
        
        historical_macro_plots = plot.plot_historical_macro_data(historical_macro_data)
        macro_vs_portfolio_returns_plots = plot.plot_macro_vs_portfolio_performance(portfolio_summary, combined_data)
        
        # calculate the correlation between the daily returns of our portfolio and the daily changes
        linear_macro_model, historical_macro_and_stock_data, X, y = calculate.calculate_linear_regression_model_from_macro_data(combined_data)
        
        # calculate the predicted change in returns for our portfolio based on the macroeconomic factors we input
        macro_estimate_df = calculate.calculate_new_X(X, user_macro_input)
        logger.debug("macro_estimate_df: {}".format(macro_estimate_df))
        predicted_change_in_returns = linear_macro_model.predict(macro_estimate_df)
        
        linear_regression_plots = plot.plot_linear_regression(linear_macro_model, historical_macro_and_stock_data, X, macro_estimate_df, predicted_change_in_returns)
    
        # calculate macro heatmap and/or importance factor to the portfolio?
    
    with col2:
        st.write("Deciding which macroeconomic factors to explore depends on the nature of your portfolio and your investment horizon. For a long-term equity portfolio, interest rates, inflation, and GDP growth could be more critical. For a short-term or bond portfolio, interest rates and liquidity might be more important.")
        st.write("The relevance of each factor also depends on the specific holdings in your portfolio. For instance, if your portfolio consists largely of technology companies, interest rates and GDP growth might have a significant impact due to these companies' reliance on economic growth and cheap financing. Conversely, if you hold many commodity companies, inflation and geopolitical events might be more relevant.")
        st.write("For these simulations, we prioritize interest rates, inflation, and liquidity as the most critical factors to explore.")
        
        if session.check_for_openai_api_key():
            if "openai_macro_response" not in st.session_state:
                st.session_state.openai_macro_response = None
                
            if st.button("Ask OpenAI about Macro Economic Factors that may impact this portfolio"):
                with st.spinner("Waiting for OpenAI API to respond..."):
                    response = interpret.openai_ask_about_macro_economic_factors(portfolio_summary)
                    st.session_state.openai_macro_response = response
            if st.session_state.openai_macro_response:
                st.write(st.session_state.openai_macro_response)
                        
    with col3:
        st.write("Interest rates: Higher interest rates can reduce the attractiveness of stocks, as bonds and other interest-bearing assets become more appealing. Lower interest rates can boost stock prices as companies benefit from lower borrowing costs.")
        st.write("Inflation: Higher inflation can erode purchasing power and negatively impact the real returns of a portfolio. Some assets, like certain commodities or inflation-protected securities, may perform better under higher inflation.")
        st.write("Liquidity: During times of low liquidity, it can be harder to buy or sell assets without impacting the market price. This can affect the ability to efficiently manage a portfolio.")
        st.write("GDP Growth: Higher GDP growth can indicate a strong economy and be positive for stocks, while lower GDP growth or a recession can be negative.")
        st.write("Unemployment rate: High unemployment can be a sign of a weak economy and could negatively impact stocks, while low unemployment could indicate a strong economy and be positive for stocks.")
        st.write("Political Stability: Political instability can increase market volatility and affect portfolio returns.")
        st.write("Geopolitical events: Wars, trade disputes, and other geopolitical events can cause market volatility and affect portfolio returns.")
        st.write("Regulatory changes: Changes in regulations can affect specific sectors positively or negatively.")

    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            for historical_macro_plot in historical_macro_plots:
                st.plotly_chart(historical_macro_plot)
                
        with col2:
            for macro_vs_portfolio_returns_plot in macro_vs_portfolio_returns_plots:
                st.plotly_chart(macro_vs_portfolio_returns_plot)
        
        with col3:
            # Show the plots
            for linear_regression_plot in linear_regression_plots:
                st.plotly_chart(linear_regression_plot)

        st.plotly_chart(plot.plot_portfolio_performance(historical_macro_and_stock_data))