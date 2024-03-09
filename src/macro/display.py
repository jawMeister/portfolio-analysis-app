import streamlit as st
import pandas as pd

import logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s (%(levelname)s):  %(module)s.%(funcName)s - %(message)s')

# Set up logger for a specific module to a different level
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

import config as config
import src.utils as utils
import src.macro.interpret as interpret
import src.macro.calculate as calculate
import src.macro.plot as plot

def initialize_macro_analysis_inputs(portfolio_summary):
    # We use ticker_weights_sorted as a key for our session state
    ticker_weights_sorted = portfolio_summary['weights'][portfolio_summary['weights'] > 0].sort_values(ascending=False)

    st.session_state.setdefault('macro_start_date',None)
    st.session_state.setdefault('macro_end_date', None)
    st.session_state.setdefault('macro_weighted_ticker_list', ticker_weights_sorted.index.tolist())

    st.session_state.setdefault('returns_data', None)
    st.session_state.setdefault('cumulative_returns_data', None)
    st.session_state.setdefault('portfolio_returns_dict', None)
    st.session_state.setdefault('sp500_returns_dict', None)
    st.session_state.setdefault('macro_data_dict', None)
    st.session_state.setdefault('portfolio_tickers', None)

    st.session_state.setdefault('cum_quarterly_input_data_df', None)
    st.session_state.setdefault('cum_quarterly_regression_models_df', None)
    st.session_state.setdefault('cum_quarterly_multivariate_models_df', None)
    st.session_state.setdefault('cum_quarterly_var_models_df', None)

    st.session_state.setdefault('cum_monthly_input_data_df', None)
    st.session_state.setdefault('cum_monthly_regression_models_df', None)
    st.session_state.setdefault('cum_monthly_multivariate_models_df', None)
    st.session_state.setdefault('cum_monthly_var_models_df', None)

    st.session_state.setdefault('monthly_input_data_df', None)
    st.session_state.setdefault('monthly_regression_models_df', None)
    st.session_state.setdefault('monthly_multivariate_models_df', None)
    st.session_state.setdefault('monthly_var_models_df', None)

def display_get_api_keys():
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
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

def calculate_macro_session_vars(portfolio_summary):
    ticker_weights_sorted = portfolio_summary['weights'][portfolio_summary['weights'] > 0].sort_values(ascending=False)
    logger.debug(f"ticker_weights_sorted:\n{ticker_weights_sorted}, start_date: {portfolio_summary['start_date']}, end_date: {portfolio_summary['end_date']}")
        
    first_row, last_row = utils.get_first_last_row_dictionaries(portfolio_summary['daily_returns_by_ticker'])
    logger.debug(f'portfolio_summary["daily_returns_by_ticker"] {portfolio_summary["daily_returns_by_ticker"].shape}:\n{first_row}\n{last_row}')
    
    # Store results directly into the session state
    st.session_state['returns_data'], st.session_state['cumulative_returns_data'], st.session_state['portfolio_returns_dict'], st.session_state['sp500_returns_dict'], st.session_state['macro_data_dict'], st.session_state['portfolio_tickers'] = \
        calculate.get_combined_returns_data(portfolio_summary['daily_returns_by_ticker'], portfolio_summary['weights'], portfolio_summary['start_date'], portfolio_summary['end_date'])

    st.session_state['cum_quarterly_input_data_df'] = calculate.prepare_data(st.session_state['cumulative_returns_data'], 'Quarterly')
    first_row, last_row = utils.get_first_last_row_dictionaries(st.session_state['cum_quarterly_input_data_df'])
    logger.debug(f'cum_quarterly_input_data_df {st.session_state["cum_quarterly_input_data_df"].shape}:\n{first_row}\n{last_row}')
    st.session_state['cum_quarterly_regression_models_df'], st.session_state['cum_quarterly_multivariate_models_df'], st.session_state['cum_quarterly_var_models_df'] = calculate.create_regression_models(st.session_state['cum_quarterly_input_data_df'], 'Quarterly', True)

    st.session_state['cum_monthly_input_data_df'] = calculate.prepare_data(st.session_state['cumulative_returns_data'])
    first_row, last_row = utils.get_first_last_row_dictionaries(st.session_state['cum_monthly_input_data_df'])
    logger.debug(f'cum_monthly_input_data_df {st.session_state["cum_monthly_input_data_df"].shape}:\n{first_row}\n{last_row}')
    st.session_state['cum_monthly_regression_models_df'], st.session_state['cum_monthly_multivariate_models_df'], st.session_state['cum_monthly_var_models_df'] = calculate.create_regression_models(st.session_state['cum_monthly_input_data_df'], 'Monthly', True)

    st.session_state['monthly_input_data_df'] = calculate.prepare_data(st.session_state['returns_data'])
    first_row, last_row = utils.get_first_last_row_dictionaries(st.session_state['monthly_input_data_df'])
    logger.debug(f'monthly_input_data_df {st.session_state["monthly_input_data_df"].shape}:\n{first_row}\n{last_row}')
    st.session_state['monthly_regression_models_df'], st.session_state['monthly_multivariate_models_df'], st.session_state['monthly_var_models_df'] = calculate.create_regression_models(st.session_state['monthly_input_data_df'], 'Monthly', False)

    # After all calculations for a session are done, update session state variables and leverage as indicators to recalculate macro analysis
    st.session_state['macro_start_date'] = st.session_state['start_date']
    st.session_state['macro_end_date'] = st.session_state['end_date']
    st.session_state['macro_weighted_ticker_list'] = ticker_weights_sorted.index.tolist()
            
def display_macro_analysis(portfolio_summary):
    initialize_macro_analysis_inputs(portfolio_summary)
    macro_factor_description_container = st.container()
    user_input_container = st.container()
    plotting_container = st.container()

    logger.debug('Displaying macro analysis')
    if config.check_for_api_key('fred') and config.check_for_api_key('fmp'):
       
        # if the dates nor the weights have not changed, no need to recalculate - use existing session state
        #if ('macro_start_date' not in st.session_state or st.session_state['macro_start_date'] != portfolio_summary['start_date'] or 
        #    'macro_end_date' not in st.session_state or st.session_state['macro_end_date'] != portfolio_summary['end_date'] or 
        #    'macro_weighted_ticker_list' not in st.session_state or st.session_state['macro_weighted_ticker_list'] != ticker_weights_sorted.index.tolist()):

        #    calculate_macro_session_vars(portfolio_summary)
        #else:
        #    logger.debug(f"Using existing session state for macro analysis")
            
        calculate_macro_session_vars(portfolio_summary)
    else:
        display_get_api_keys()
                

    with user_input_container:
        display_ask_open_ai_about_macro(portfolio_summary, st.session_state.cum_monthly_regression_models_df, st.session_state.cum_monthly_multivariate_models_df, st.session_state.monthly_var_models_df)
        
        if 'openai_macro_synthesis_response' in st.session_state and st.session_state.openai_macro_synthesis_response: 
            st.success(st.session_state.openai_macro_synthesis_response)
                
    with macro_factor_description_container:
        with st.expander("Macro Factor Descriptions", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                display_general_macro_factors_descriptions()
            with col2:
                display_macro_factors_descriptions_1()
            with col3:
                display_macro_factors_descriptions_2()

    with plotting_container:
        st.markdown("""---""")

        if st.session_state.cum_quarterly_regression_models_df is not None and st.session_state.cum_monthly_regression_models_df is not None and st.session_state.monthly_regression_models_df is not None:
            with st.expander("Cumlative Monthly Returns vs. Macro Factors (simple linear regression one macro factor at a time)", expanded=True):
                logger.debug('Displaying summary and individual regression results')                    
                display_summary_of_linear_regression_results(st.session_state.cum_monthly_regression_models_df, 'Monthly')
                display_individual_linear_regression_results(st.session_state.cum_monthly_input_data_df, st.session_state.cum_monthly_regression_models_df, 'Monthly', cumulative_performance=True)
                    
            with st.expander("Cumulative Monthly Returns vs. Macro Factors (multivariate regression of portfolio performance showing most significant factors)"):
                with st.container():
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write("The p-value in a multivariate regression is used to determine the statistical significance of each macro factor in a model of all the factors together against portfolio performance. It is the probability that the effect of a factor in the model is due to chance, under the null hypothesis. The null hypothesis usually posits that the true coefficient of the variable in the regression is zero, meaning that the factor has no effect on the portfolio returns.")
                    with col2:
                        st.write("A small p-value (typically ≤ 0.05) indicates strong evidence that the factor is meaningful or 'significant' in the relationship to the outcome variable (portfolio returns). It suggests that changes in the predictor's value (the factor) are related to changes in the response variable (portfolio returns).")
                        st.write("A larger p-value suggests that changes in the predictor (the macro factor) are not associated with changes in the response. If you see a large p-value, you would fail to reject the null hypothesis because there isn't enough evidence to conclude that a meaningful relationship exists.")
                    with col3:
                        st.write("In the context of this regression model, for each macroeconomic factor, the corresponding p-value can tell you whether changes in this factor are statistically significantly associated with changes in the portfolio return, given the presence of all the other factors in the model.")
                        st.write("It's important to note that while the p-value can inform the statistical significance of an effect, it does not provide information about the size or practical significance of the effect. Furthermore, p-values are also dependent on the sample size, and very large samples can result in small p-values even for trivially small effects. Therefore, it's always a good idea to look at the size of the effect and not just the p-value.")

                with st.container():
                    col1, col2 = st.columns(2)
                    logger.debug('Displaying multivariate regression results')
                    fig_coef, fig_pval, sig_feature_plots = plot.plot_multivariate_results(st.session_state.cum_monthly_input_data_df, st.session_state.cum_monthly_multivariate_models_df.loc[0, 'Model'], st.session_state.cum_monthly_multivariate_models_df.loc[0, 'Significant Features'])
                    with col1:
                        st.plotly_chart(fig_pval, use_container_width=True)
                    with col2:
                        st.plotly_chart(fig_coef, use_container_width=True)
                        
                    last_column_used = 2
                    for feature_plot in sig_feature_plots:
                        # Decide the column based on the last one used
                        if last_column_used == 1:
                            col = col2
                            last_column_used = 2
                        else:
                            col = col1
                            last_column_used = 1
                        # Add the plot to the chosen column
                        with col:
                            st.plotly_chart(feature_plot, use_container_width=True)
                    
            with st.expander("Predicted Impact on Portfolio Returns with Macro Factor Shock (vector autogression - testing impact of macro factor shock on portfolio returns)"):
                logger.debug('Displaying VAR model results')
                logger.debug(f'cum_var_models_df.columns:\n{st.session_state.monthly_var_models_df.columns}, var_models_df.shape: {st.session_state.monthly_var_models_df.shape}')
                fig = plot.plot_irf(st.session_state.monthly_var_models_df.loc[0, 'Model'])
                st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("Cumlative Quarterly Returns vs. Macro Factors (simple linear regression one macro factor at a time)", expanded=False):
                logger.debug('Displaying summary and individual regression results for quarterly data')
                display_summary_of_linear_regression_results(st.session_state.cum_quarterly_regression_models_df, 'Quarterly')
                display_individual_linear_regression_results(st.session_state.cum_quarterly_input_data_df, st.session_state.cum_quarterly_regression_models_df, 'Quarterly', cumulative_performance=True)
                    
            logger.debug(f'regression_models_df.columns:\n{st.session_state.monthly_regression_models_df.columns}')
            with st.expander("Monthly Returns vs. Macro Factors (simple linear regression one macro factor at a time)"):
                logger.debug('Displaying summary and individual regression results for Monthly Returns vs Macro Factors (simple linear regression one macro factor at a time)')
                display_summary_of_linear_regression_results(st.session_state.monthly_regression_models_df, 'Monthly')
                display_individual_linear_regression_results(st.session_state.monthly_input_data_df, st.session_state.monthly_regression_models_df, 'Monthly', cumulative_performance=False)    
    

def display_ask_open_ai_about_macro(portfolio_summary, cum_monthly_regression_models_df, cum_monthly_multivariate_models_df, monthly_var_models_df):
    if config.check_for_api_key('openai'):
        if "openai_macro_response" not in st.session_state:
            st.session_state.openai_macro_response = None
            
        if st.button("Ask OpenAI about Macro Economic Factors that may impact this portfolio", use_container_width=True):
            with st.spinner("Waiting for OpenAI API to respond..."):
                synthesis_response, portfolio_response, linear_regression_response, multivariate_regression_response, var_model_response = interpret.openai_ask_about_macro_economic_factors(portfolio_summary, cum_monthly_regression_models_df, cum_monthly_multivariate_models_df, monthly_var_models_df)
                st.session_state.openai_macro_synthesis_response = synthesis_response
                st.session_state.openai_macro_portfolio_response = portfolio_response
                st.session_state.openai_macro_linear_regression_response = linear_regression_response
                st.session_state.openai_macro_multivariate_regression_response = multivariate_regression_response
                st.session_state.openai_macro_var_model_response = var_model_response
                
                logger.info(f"openai_macro_synthesis_response: {st.session_state.openai_macro_synthesis_response}")
            
def display_general_macro_factors_descriptions():
    st.markdown("<span style='color:#FF4B4B;'>General Macro Economic Conditions that may affect Portfolio Performance</span>", unsafe_allow_html=True)
    st.markdown("<span style='color:#FF4B4B;'>Interest rates:</span> This is the interest rate at which depository institutions trade federal funds (balances held at Federal Reserve Banks) with each other overnight. When the Federal Reserve (the Fed) changes the target federal funds rate, it aims to influence short-term interest rates, inflation, and the overall economy including the stock market. Higher interest rates can make borrowing costlier for companies, potentially leading to a decrease in corporate profits and a corresponding decline in stock prices.", unsafe_allow_html=True)
    st.markdown("<span style='color:#FF4B4B;'>Inflation:</span> Higher inflation can erode purchasing power and negatively impact the real returns of a portfolio. Some assets, like certain commodities or inflation-protected securities, may perform better under higher inflation.", unsafe_allow_html=True)
    st.markdown("<span style='color:#FF4B4B;'>Liquidity:</span> During times of low liquidity, it can be harder to buy or sell assets without impacting the market price. This can affect the ability to efficiently manage a portfolio.", unsafe_allow_html=True)
    st.markdown("<span style='color:#FF4B4B;'>GDP Growth:</span> Higher GDP growth can indicate a strong economy and be positive for stocks, while lower GDP growth or a recession can be negative.", unsafe_allow_html=True)
    st.markdown("<span style='color:#FF4B4B;'>Unemployment rate:</span> High unemployment can be a sign of a weak economy and could negatively impact stocks, while low unemployment could indicate a strong economy and be positive for stocks.", unsafe_allow_html=True)
    st.markdown("<span style='color:#FF4B4B;'>Political Stability:</span> Political instability can increase market volatility and affect portfolio returns.", unsafe_allow_html=True)
    
    st.write("Deciding which macroeconomic factors to explore depends on the nature of your portfolio and your investment horizon. For a long-term equity portfolio, interest rates, inflation, and GDP growth could be more critical. For a short-term or bond portfolio, interest rates and liquidity might be more important.")
    st.write("The relevance of each factor also depends on the specific holdings in your portfolio. For instance, if your portfolio consists largely of technology companies, interest rates and GDP growth might have a significant impact due to these companies' reliance on economic growth and cheap financing. Conversely, if you hold many commodity companies, inflation and geopolitical events might be more relevant.")
    
def display_macro_factors_descriptions_1():
    st.markdown("<span style='color:#FF4B4B;'>Specific Macro Economic Factors Analyzed against Portfolio Performance</span>", unsafe_allow_html=True)
    st.markdown("<span style='color:#FF4B4B;'>Federal Funds Rate (FEDFUNDS):</span> This is the interest rate at which depository institutions trade federal funds (balances held at Federal Reserve Banks) with each other overnight. When the Federal Reserve (the Fed) changes the target federal funds rate, it aims to influence short-term interest rates, inflation, and the overall economy including the stock market. Higher interest rates can make borrowing costlier for companies, potentially leading to a decrease in corporate profits and a corresponding decline in stock prices.", unsafe_allow_html=True)
    st.markdown("<span style='color:#FF4B4B;'>Unemployment Rate (UNRATE):</span> This is the percentage of the total labor force that is jobless and actively seeking employment. A high unemployment rate typically corresponds to a slow economy, which can negatively affect corporate profits and stock prices. Conversely, a low unemployment rate often signals a strong economy, which can be positive for stocks.", unsafe_allow_html=True)
    st.markdown("<span style='color:#FF4B4B;'>Consumer Price Index (CPIAUCSL):</span> This is a measure of the average change over time in the prices paid by urban consumers for a market basket of consumer goods and services. An increase in the CPI is a signal that inflation is present, which can erode purchasing power and potentially lead to higher interest rates, negatively affecting stock prices.", unsafe_allow_html=True)
    st.markdown("<span style='color:#FF4B4B;'>Personal Consumption Expenditures (PCE):</span> This measures the value of the goods and services purchased by, or on the behalf of, U.S. residents. High PCE indicates strong consumer spending and could be positive for stocks, especially those in the consumer discretionary sector.", unsafe_allow_html=True)
    st.markdown("<span style='color:#FF4B4B;'>Retail Sales (RSAFS):</span> This measures the total receipts at stores that sell merchandise and related services to final consumers. Rising retail sales can indicate a strong economy and higher corporate profits, which could be positive for stocks.", unsafe_allow_html=True)

def display_macro_factors_descriptions_2():
    st.markdown("<span style='color:#FF4B4B;'>Initial Claims (ICSA):</span> This measures the number of people filing first-time claims for state unemployment insurance. An increase in initial claims can indicate weakening economic conditions and can be a negative signal for the stock market.", unsafe_allow_html=True)
    st.markdown("<span style='color:#FF4B4B;'>Housing Starts (HOUST):</span> This measures the number of new residential construction projects that have begun during any particular month. Housing starts can provide a signal about the health of the economy, with increasing housing starts potentially indicating a strong economy and being positive for stocks, especially those in the housing and related sectors.", unsafe_allow_html=True)
    st.markdown("<span style='color:#FF4B4B;'>5-Year Forward Inflation Expectation Rate (T5YIFR):</span> This is a measure of expected inflation (on average) over the five-year period that begins five years from today. If investors expect high inflation in the future, they may demand higher yields on investments leading to a potential decline in stock prices.", unsafe_allow_html=True)
    st.markdown("<span style='color:#FF4B4B;'>Economic Policy Uncertainty Index for United States (USEPUINDXD):</span> This index quantifies newspaper coverage of policy-related economic uncertainty. A higher index level indicates greater uncertainty, which can increase market volatility and potentially lead to lower stock prices.", unsafe_allow_html=True)
    st.markdown("<span style='color:#FF4B4B;'>10-Year Treasury Constant Maturity Rate (GS10):</span> This is the yield on U.S. Treasury securities at a constant maturity of 10 years. Treasury yields serve as a benchmark for interest rates and can influence the prices of stocks. When yields rise, stocks can become less appealing to investors as bonds offer higher guaranteed returns.", unsafe_allow_html=True)
    st.markdown("<span style='color:#FF4B4B;'>Gross Domestic Product (GDPC1):</span> This is the broadest measure of economic activity. A growing GDP indicates a healthy, expanding economy, which is generally good for corporate profits and equity markets. On the other hand, a contracting GDP may signal a recession, which can lead to declining stock prices.", unsafe_allow_html=True)
    
def display_summary_of_linear_regression_results(regression_models_df, time_basis):
    col1, col2, col3 = st.columns(3, gap="small")
    fig_pval, fig_r2, fig_corr = plot.create_linear_regression_summary_charts(regression_models_df)
    with col1:
        st.markdown("<span style='color:#FF4B4B;'>p-value:</span> This value is related to the significance of the macro factor vs. portfolio returns. A lower p-value indicates that a factor is statistically significant. Sorting by p-value in ascending order will give you the factors in order of their statistical significance.", unsafe_allow_html=True)
        st.plotly_chart(fig_pval, use_container_width=True)
    with col2:
        st.markdown("<span style='color:#FF4B4B;'>Correlation:</span> Measures the strength and direction of a linear relationship between two variables. Sorting by absolute correlation value in descending order will give you the variables in order of their linear relationship strength.", unsafe_allow_html=True)
        st.plotly_chart(fig_corr, use_container_width=True)
    with col3:
        st.markdown("<span style='color:#FF4B4B;'>R-squared value:</span>  Measures how much of the variance in the dependent variable is explained by the independent variable(s) in a regression model. A higher R-squared value indicates a higher explanatory power of the model regarding the dependent variable's variability.", unsafe_allow_html=True)
        st.plotly_chart(fig_r2, use_container_width=True)
        
def display_individual_linear_regression_results(input_data_df, regression_models_df, time_basis, cumulative_performance=False):
    st.markdown("""---""")
    col1, col2, col3, col4 = st.columns(4, gap="small")
    with col1:
        st.write("<span style='color:#FF4B4B;'>Portfolio Performance vs Macro Factor over Time</span>", unsafe_allow_html=True)
        st.write("This plot shows the portfolio returns and the given factor (e.g., S&P 500, unemployment rate, GDP growth, etc.) over time. You want to see if there are any visible patterns or correlations. Do they seem to move together (i.e., both go up or down at the same time)? Do they move in opposite directions? If there's a visible relationship, that's a good sign that your factor may be a useful predictor. Of course, you'd have to validate this with more rigorous statistical tests.")
    with col2:
        st.write("<span style='color:#FF4B4B;'>Scatter of Portfolio Returns vs Macro Factor Values</span>", unsafe_allow_html=True)
        st.write("This plot shows each observation of your portfolio returns against the factor. You're looking for any kind of relationship - linear, non-linear, or no relationship at all. A linear relationship where the points cluster around a straight line (either increasing or decreasing) is a good sign that linear regression may be a good model. If you see a different pattern, like a curve or clusters, a different model might be more appropriate.")
    with col3:
        st.write("<span style='color:#FF4B4B;'>Predicted Portfolio Returns vs Macro Factor Values</span>", unsafe_allow_html=True)
        st.write("This plot shows the predicted portfolio returns given the actual values of the macro factor. The model's predictions for different factor levels are visualized here. If the model has captured the relationship accurately, you would expect to see a trend in the predicted returns corresponding to the actual values of the macro factor.")
    with col4:
        st.write("<span style='color:#FF4B4B;'>Residuals (error) vs Macro Factor Values</span>", unsafe_allow_html=True)
        st.write("This plot shows the residuals, i.e., the differences between the actual portfolio returns and the model's predictions, plotted against the macro factor's values. In a well-performing model, you'd expect to see the residuals scattered randomly around the zero line across the range of the factor values. If you see patterns in the residuals, such as a curve or a funnel shape, it's a sign that the model isn't capturing some aspect of the data, and the assumptions of the linear regression may not be fully satisfied.")
   
    # Sort regression models by p-value
    sorted_regression_models_df = regression_models_df.sort_values(by='P-value', ascending=True)
    for index, row in sorted_regression_models_df.iterrows():
        fig = plot.create_linear_regression_plots(input_data_df, row['Factor'], sorted_regression_models_df, time_basis, cumulative_performance=cumulative_performance)
        st.plotly_chart(fig, use_container_width=True)