import streamlit as st
import pandas as pd

import logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s (%(levelname)s):  %(module)s.%(funcName)s - %(message)s')

# Set up logger for a specific module to a different level
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import config as config
import src.macro.interpret as interpret
import src.macro.calculate as calculate
import src.macro.plot as plot

def display_collect_future_macro_estimates(user_macro_input):
    st.markdown('<p style="color:red;">WORK IN PROGRESS: Analyzing Macro Factor Impact on Portfolio Performance</p>',unsafe_allow_html=True)
    logger.debug("user_macro_input: {}".format(user_macro_input))
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        user_macro_input['US Interest Rate'] = \
            st.number_input("US Interest Rate (%)", min_value=-2.0, max_value=10.0, step=0.1, value=(user_macro_input['US Interest Rate']*100), format="%.1f") / 100
        
        user_macro_input['US Inflation Rate'] = \
            st.number_input("US Inflation Rate (%)", min_value=-10.0, max_value=20.0, step=0.5, value=(user_macro_input['US Inflation Rate']*100), format="%.1f") / 100

        user_macro_input['US M2 Money Supply'] = \
            st.number_input("US M2 Money Supply Rate (%)", min_value=7.5, max_value=50.0, step=2.5, value=(user_macro_input['US M2 Money Supply']*100), format="%.1f") / 100

        user_macro_input['China M2 Money Supply'] = \
            st.number_input("China M2 Money Supply Rate (%)", min_value=7.5, max_value=50.0, step=2.5, value=(user_macro_input['China M2 Money Supply']*100), format="%.1f") / 100

#    user_macro_input['US GDP Growth Rate'] = \
#        st.number_input("US GDP Growth Rate", min_value=-2.0, max_value=10.0, step=0.1, value=user_macro_input['US GDP Growth Rate']*100, format="%.1f") / 100

#    user_macro_input['US Unemployment Rate'] = \
#        st.number_input("US Unemployment Rate", min_value=-10.0, max_value=20.0, step=0.5, value=user_macro_input['US Unemployment Rate']*100, format="%.1f") / 100

    with col2:
        if not config.check_for_api_key('fred'):
            label = "Enter [FRED API Key](https://fred.stlouisfed.org/docs/api/api_key.html) for macro indicators"
            temp_key = st.text_input(label, value=config.get_api_key('fred'))
            if temp_key:
                config.set_api_key('fred', temp_key)
        
        if not config.check_for_api_key('fmp'):
            label = "Enter [FMP API Key](https://financialmodelingprep.com/developer/docs/) to retrieve varios macro indicators"
            temp_key = st.text_input(label, value=config.get_api_key('fmp'))
            if temp_key:
                config.set_api_key('fmp', temp_key)
                    
    return user_macro_input

def display_macro_analysis(portfolio_summary):

    user_input_container = st.container()
    plotting_container = st.container()
    
    with user_input_container:
        col1, col2, col3 = st.columns(3)
        with col1:

            user_macro_input = calculate.get_macro_factor_defaults()
            user_macro_input = display_collect_future_macro_estimates(user_macro_input)
            
            if config.check_for_api_key('fred') and config.check_for_api_key('fmp'):
                # get historical macro data, unfiltered
                historical_macro_data = calculate.get_historical_macro_data(portfolio_summary["start_date"], portfolio_summary["end_date"])
                # bring the macro data into the same format as the portfolio data as a new df (monthly basis), clean it and do some summary calcs
                combined_data = calculate.clean_and_combine_macro_data(portfolio_summary['stock_data'], portfolio_summary['weights'], historical_macro_data)
                
                # calculate the correlation between the macro factors and the portfolio returns (monthly basis)
                # model data will have less data points than combined_data because we need to drop any rows with NaNs for the linear regression
                # If the correlation is positive, it means that when the interest rate or inflation increases, our portfolio returns also tend to increase.
                #
                # Past performance is not indicative of future results, and this analysis assumes that the relationships between these variables and portfolio 
                # returns will remain constant in the future, which may not be the case.
                #
                # The model also assumes a linear relationship between the predictors and the response variable. There could be a non-linear relationship 
                # between them which cannot be captured by this model
                
                # hypothesis that these factors may impact cumulative performance
                cumulative_macro_factors = ['cumulative_inflation', 'US M2 Money Supply', 'China M2 Money Supply']
                y_cumulative = 'weighted_portfolio_cumulative_returns'
                cumulative_models, cumulative_model_data = calculate.calculate_linear_regression_models_from_macro_data_per_factor(combined_data, cumulative_macro_factors, y_cumulative)
                cumulative_performance_predictions = calculate.predict_portfolio_returns_from_user_macro_input(user_macro_input, cumulative_models)

                # hypothesis that these factors may impact month to month performance
                rate_macro_factors = ['US Interest Rate', 'US Inflation Rate', 'US M2 Money Supply', 'China M2 Money Supply']
                y_rate = 'weighted_portfolio_returns_monthly'
                rate_models, rate_model_data = calculate.calculate_linear_regression_models_from_macro_data_per_factor(combined_data, rate_macro_factors, y_rate)
                rate_performance_predictions = calculate.predict_portfolio_returns_from_user_macro_input(user_macro_input, rate_models)

        with col2:
            display_ask_open_ai_about_macro(portfolio_summary)
                            
        with col3:
            display_macro_factors_descriptions()

    with plotting_container:
        st.markdown("""---""")
        col1, col2, col3 = st.columns(3, gap="large")
        
        if config.check_for_api_key('fred') and config.check_for_api_key('fmp'):
            with col1:
                # plot the unfiltered macro data
                historical_macro_plots = plot.plot_historical_macro_data(historical_macro_data)
                for historical_macro_plot in historical_macro_plots:
                    st.plotly_chart(historical_macro_plot, use_container_width=True)
                
                absolute_portfolio_plot, cumulative_portfolio_plot = plot.plot_historical_portfolio_performance(combined_data)
                st.plotly_chart(absolute_portfolio_plot, use_container_width=True)
                st.plotly_chart(cumulative_portfolio_plot, use_container_width=True)
                st.write(combined_data)
                
            with col2:
                # plot the combined macro and portfolio data
                macro_vs_portfolio_returns_plots = plot.plot_macro_vs_portfolio_performance(combined_data)
            
                for macro_vs_portfolio_returns_plot in macro_vs_portfolio_returns_plots:
                    st.plotly_chart(macro_vs_portfolio_returns_plot, use_container_width=True)
            
            with col3:
                # plot the linear regression model
                for factor, model in cumulative_models.items():
                    st.write(display_regression_formula(model, factor, "Cumulative Returns"))
                    prediction = None
                    if factor in cumulative_performance_predictions:
                        prediction = {factor: user_macro_input[factor], 'prediction': cumulative_performance_predictions[factor][0]}
                    st.plotly_chart(plot.plot_linear_regression_v_single_model(model, cumulative_model_data, factor, "weighted_portfolio_cumulative_returns", prediction=prediction))

                    
                for factor, model in rate_models.items():
                    # ... (plot the regression here)
                    st.write(display_regression_formula(model, factor, "Monthly Returns"))
                    prediction = None
                    if factor in rate_performance_predictions:
                        prediction = {factor: user_macro_input[factor], 'prediction': rate_performance_predictions[factor][0]}
                    st.plotly_chart(plot.plot_linear_regression_v_single_model(model, rate_model_data, factor, "weighted_portfolio_returns_monthly", prediction=prediction))
                    
                                    
                #linear_regression_plots = plot.plot_linear_regression(linear_macro_model, model_data, X, macro_estimate_df, predicted_change_in_returns)
                #for linear_regression_plot in linear_regression_plots:
                #    st.plotly_chart(linear_regression_plot, use_container_width=True)
            

def display_ask_open_ai_about_macro(portfolio_summary):
    st.write("Deciding which macroeconomic factors to explore depends on the nature of your portfolio and your investment horizon. For a long-term equity portfolio, interest rates, inflation, and GDP growth could be more critical. For a short-term or bond portfolio, interest rates and liquidity might be more important.")
    st.write("The relevance of each factor also depends on the specific holdings in your portfolio. For instance, if your portfolio consists largely of technology companies, interest rates and GDP growth might have a significant impact due to these companies' reliance on economic growth and cheap financing. Conversely, if you hold many commodity companies, inflation and geopolitical events might be more relevant.")
    st.write("For these simulations, we prioritize interest rates, inflation, and liquidity as the most critical factors to explore.")
    
    if config.check_for_api_key('openai'):
        if "openai_macro_response" not in st.session_state:
            st.session_state.openai_macro_response = None
            
        if st.button("Ask OpenAI about Macro Economic Factors that may impact this portfolio"):
            with st.spinner("Waiting for OpenAI API to respond..."):
                response = interpret.openai_ask_about_macro_economic_factors(portfolio_summary)
                st.session_state.openai_macro_response = response
        if st.session_state.openai_macro_response:
            st.write(st.session_state.openai_macro_response)
            
def display_macro_factors_descriptions():
    st.markdown("<span style='color:#FF4B4B;'>Interest rates:</span> Higher interest rates can reduce the attractiveness of stocks, as bonds and other interest-bearing assets become more appealing. Lower interest rates can boost stock prices as companies benefit from lower borrowing costs.", unsafe_allow_html=True)
    st.markdown("<span style='color:#FF4B4B;'>Inflation:</span> Higher inflation can erode purchasing power and negatively impact the real returns of a portfolio. Some assets, like certain commodities or inflation-protected securities, may perform better under higher inflation.", unsafe_allow_html=True)
    st.markdown("<span style='color:#FF4B4B;'>Liquidity:</span> During times of low liquidity, it can be harder to buy or sell assets without impacting the market price. This can affect the ability to efficiently manage a portfolio.", unsafe_allow_html=True)
    st.markdown("<span style='color:#FF4B4B;'>GDP Growth:</span> Higher GDP growth can indicate a strong economy and be positive for stocks, while lower GDP growth or a recession can be negative.", unsafe_allow_html=True)
    st.markdown("<span style='color:#FF4B4B;'>Unemployment rate:</span> High unemployment can be a sign of a weak economy and could negatively impact stocks, while low unemployment could indicate a strong economy and be positive for stocks.", unsafe_allow_html=True)
    st.markdown("<span style='color:#FF4B4B;'>Political Stability:</span> Political instability can increase market volatility and affect portfolio returns.", unsafe_allow_html=True)
    st.markdown("<span style='color:#FF4B4B;'>Geopolitical events:</span> Wars, trade disputes, and other geopolitical events can cause market volatility and affect portfolio returns.", unsafe_allow_html=True)
    st.markdown("<span style='color:#FF4B4B;'>Regulatory changes:</span> Changes in regulations can affect specific sectors positively or negatively.", unsafe_allow_html=True)

def display_regression_formula(model, factor_name, y):
    # Get the intercept and coefficient
    intercept = model.intercept_
    coef = model.coef_[0]
    
    # Format the formula string
    formula = f"{y} = {intercept:.4f} + ({coef:.4f} * {factor_name})"
    
    # Display the formula
    st.write(formula)