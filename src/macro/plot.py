import pandas as pd
import numpy as np
from statsmodels.tools.tools import add_constant

import streamlit as st
import plotly.graph_objs as go
import plotly.subplots as sp
import colorsys

import logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s (%(levelname)s):  %(module)s.%(funcName)s - %(message)s')

# Set up logger for a specific module to a different level
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import src.macro.calculate as calculate

def plot_linear_regression(model, historical_macro_data, X, new_X, predicted_change_in_returns):
    # Initialize an empty list to store the plots
    plots = []

    # Loop over each macroeconomic indicator
    for indicator in X.columns:
        logger.debug(f"Portfolio Returns = {model.intercept_:.4f} + ({model.coef_[0]:.4f} * {indicator})")
    
        # Create a scatter plot of the indicator versus portfolio returns
        scatter = go.Scatter(x=historical_macro_data[indicator], y=historical_macro_data['portfolio_returns'], mode='markers', name='Data')

        # Create a line plot of the fitted values from the regression
        fitted_values = model.intercept_ + model.coef_[X.columns.tolist().index(indicator)] * historical_macro_data[indicator]
        line = go.Scatter(x=historical_macro_data[indicator], y=fitted_values, mode='lines', name='Fitted line')

        # Create a layout
        layout = go.Layout(title=f'Portfolio Returns vs {indicator}',
                        xaxis=dict(title=indicator),
                        yaxis=dict(title='Portfolio Returns'))

        # Add the scatter plot and line plot to the list of plots
        plots.append(go.Figure(data=[scatter, line], layout=layout))

    # Add the prediction to the plots
    for indicator, plot in zip(X.columns, plots):
        prediction = go.Scatter(x=[new_X[indicator].values[0]], y=[predicted_change_in_returns[0]], mode='markers', name='Prediction', marker=dict(color='red'))
        plot.add_trace(prediction)
        
    return plots

def plot_linear_regression_v_single_model(model, model_data, factor, y, title, prediction=None):
    
    logger.debug(f"f(x) = {model.intercept_:.4f} + ({model.coef_[0]:.4f} * {factor}), prediction = {prediction['prediction'] if prediction else None}")
    logger.debug(f"model data keys: {model_data.keys()}")
    
    if factor == 'all_factors':
        x = model_data.drop(columns=[y])
    else:
        x = model_data[factor]
        
    # Create a scatter plot of the indicator versus portfolio returns
    scatter = go.Scatter(x=x, y=model_data[y], mode='markers', name='Data')

    # Create a line plot of the fitted values from the regression
    fitted_values = model.intercept_ + model.coef_[0] * x
    line = go.Scatter(x=x, y=fitted_values, mode='lines', name='Fitted line')

    # Create a layout
    layout = go.Layout(title=f'{title}',
                    xaxis=dict(title=factor),
                    yaxis=dict(title='Portfolio Returns'))

    # Add the scatter plot and line plot to a Plotly Figure object
    fig = go.Figure(data=[scatter, line], layout=layout)

    # Add the prediction to the plot if provided
    if prediction is not None:
        pred_point = go.Scatter(x=[prediction[factor]], y=[prediction['prediction']], mode='markers', name='Prediction', marker=dict(color='red'))
        fig.add_trace(pred_point)
        
    # Display the plot
    return fig


# need a plotting function that takes in the portfolio, benchmark, and macroeconomic data and plots them all together
# need to be able to specify the time period (monthly, quarterly, annual) and loop over the factors
def plot_portfolio_vs_benchmark_vs_macro(portfolio_returns, benchmark_returns, macro_time_series, time_period, title_prefix=''):
    """Plot the portfolio returns vs the benchmark returns vs the macroeconomic data
    
    Parameters
    ----------
    portfolio_returns : pandas.Series
        The portfolio returns
    benchmark_returns : pandas.Series
        The benchmark returns
    macro_data_dict : dict
        A dictionary containing the macroeconomic data
    macro_time_series_name : str
        The name of the macroeconomic time series
    time_period : str
        The time period to plot the data for (monthly, quarterly, or annual)
    """
    
    
    # Create subplot for each macro factor
    for factor, macro_factor_series in macro_time_series.items():

        # Create a subplot with two y-axes
        fig = sp.make_subplots(specs=[[{"secondary_y": True}]])

        # Add S&P500 return trace
        fig.add_trace(
            go.Scatter(x=benchmark_returns.index, y=benchmark_returns, mode='lines', name='S&P500 Returns'),
            secondary_y=False,
        )
        
        # Add portfolio return trace
        fig.add_trace(
            go.Scatter(x=portfolio_returns.index, y=portfolio_returns, mode='lines', name='Weighted Portfolio Returns', line=dict(color='royalblue')),
            secondary_y=False,
        )

        # Add macroeconomic factor trace
        fig.add_trace(
            go.Scatter(x=macro_factor_series.index, y=macro_factor_series, mode='lines', name=factor, line=dict(dash='dot')),
            secondary_y=True
        )

        # Set axis titles
        #fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Returns %", secondary_y=False, tickformat=".1%")
        
        if 'Change' in factor or 'Rate' in factor:
            fig.update_yaxes(title_text=f'{factor} %', secondary_y=True, tickformat=".1%")
        else:
            fig.update_yaxes(title_text=factor, secondary_y=True)

        # Set title
        fig.update_layout(title_text=f'{title_prefix}Weighted Portfolio vs {title_prefix}S&P500 Returns vs {factor}<br><sup>Time Period: {time_period}</sup>')
        
        fig.show()

def plot_historical_macro_data(factor, time_basis, time_series):
    plots = []

    fig = go.Figure()
    # Add macroeconomic factor trace
    fig.add_trace(
        go.Scatter(x=time_series.index, y=time_series, mode='lines', name=factor))

    if 'Change' in factor or 'Rate' in factor:            
        fig.update_layout(
            title=f'{factor} Historical Data<br><sup>({time_basis})</sup>',
            yaxis_tickformat='.1%'
        )
    else:
        # TODO: need a lookup by factor for what the units are on the y-axis
        fig.update_layout(
            title=f'{factor} Historical Data<br><sup>({time_basis})</sup>',
        )

    return fig
    
def plot_multivariate_results(input_df, multivariate_model, significant_features):
    # Figure for coefficients
    fig_coef = go.Figure(data=[go.Bar(name='Coefficient', x=multivariate_model.params.index, y=multivariate_model.params.values)])
    fig_coef.update_layout(title_text='Multivariate Regression Coefficients')

    # Figure for p-values
    fig_pval = go.Figure(data=[go.Bar(name='p-value', x=multivariate_model.pvalues.index, y=multivariate_model.pvalues.values)])
    x_values = list(multivariate_model.pvalues.index)  # List of categories
    y_value = [0.05] * len(x_values)  # p-value threshold

    fig_pval.add_trace(go.Scatter(x=x_values, y=y_value, mode='lines', name='p-value threshold', line=dict(color='red', dash='dash')))
    fig_pval.update_layout(title_text='Multivariate Regression p-values')

    # Scatter plots for each significant feature
    sig_feature_plots = []
    for feature in significant_features:
        # Calculate the OLS trendline
        m, b = np.polyfit(input_df[feature], input_df['Portfolio'], 1)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=input_df[feature], y=input_df['Portfolio'], mode='markers', name='observations'))
        fig2.add_trace(go.Scatter(x=input_df[feature], y=m*input_df[feature] + b, mode='lines', name='OLS trendline'))
        fig2.update_layout(title=f'Portfolio Returns vs {feature}', xaxis_title=feature, yaxis_title='Portfolio Returns')
        fig2.update_yaxes(tickformat=".1%")
        if 'Rate' in feature:
            fig2.update_xaxes(tickformat=".1%")
            
        sig_feature_plots.append(fig2)
        
    return fig_coef, fig_pval, sig_feature_plots
        
def create_linear_regression_summary_charts(linear_regression_models_df, pval_threshold=0.05, r2_threshold=0.5, corr_threshold=0.3):
    df = linear_regression_models_df.copy()
    # Sort the dataframe for each chart
    df_pval = df.sort_values('P-value', ascending=True)
    df_r2 = df.sort_values('R-squared', ascending=False)
    df_corr = df.sort_values('Correlation', ascending=False)

    # Create a bar chart for p-values
    fig_pval = go.Figure(data=[go.Bar(x=df_pval['Factor'], y=df_pval['P-value'])])
    fig_pval.add_trace(go.Scatter(x=df_pval['Factor'], y=[pval_threshold] * len(df_pval), mode='lines', name='Significance Threshold', line=dict(color='red', dash='dash')))
    fig_pval.update_layout(title='P-values', yaxis_title='P-value', showlegend=False)

    # Create a bar chart for R-squared values
    fig_r2 = go.Figure(data=[go.Bar(x=df_r2['Factor'], y=df_r2['R-squared'])])
    fig_r2.add_trace(go.Scatter(x=df_r2['Factor'], y=[r2_threshold] * len(df_r2), mode='lines', name='Significance Threshold', line=dict(color='red', dash='dash')))
    fig_r2.update_layout(title='R-squared values', yaxis_title='R-squared', showlegend=False)

    # Create a bar chart for Correlations
    fig_corr = go.Figure(data=[go.Bar(x=df_corr['Factor'], y=df_corr['Correlation'])])
    fig_corr.add_trace(go.Scatter(x=df_corr['Factor'], y=[corr_threshold] * len(df_corr), mode='lines', name='Significance Threshold', line=dict(color='red', dash='dash')))
    fig_corr.update_layout(title='Correlations', yaxis_title='Correlation', showlegend=False)

    return fig_pval, fig_r2, fig_corr
    

def create_linear_regression_plots(input_df, factor, regression_models_df, time_basis, cumulative_performance):
    logger.debug(f'Creating linear regression plots for {factor}')
    linear_regression_model = regression_models_df.loc[regression_models_df['Factor'] == factor, 'Model'].values[0]
    df = input_df.copy()
    df = df.dropna(how='any')
    
    X = df[[factor]]
    y = df['Portfolio']
    X2 = add_constant(X)
    predictions = linear_regression_model.predict(X2)
    residuals = y - predictions

    # Intercept and slope for the annotation
    intercept = linear_regression_model.params['const']
    slope = linear_regression_model.params[factor]

    # Correlation between the factor and the portfolio returns
    correlation = df["Portfolio"].corr(df[factor])

    if cumulative_performance:
        # Initialize subplots
        fig = sp.make_subplots(rows=1, cols=4, subplot_titles=(f"Portfolio Returns vs<br>{factor} over Time", f"Scatter of Portfolio Returns vs<br>{factor}", f"Model Prediction of Returns vs<br>{factor}", f"Regression Model Residuals (error) vs<br>{factor}"),  specs=[[{'secondary_y': True}, {'secondary_y': True}, {'secondary_y': True}, {}]], horizontal_spacing=0.075)
    else:
        fig = sp.make_subplots(rows=1, cols=4, subplot_titles=(f"Portfolio Returns vs<br>{time_basis} Change of {factor}", f"Scatter of Portfolio Returns vs<br>{time_basis} Change of {factor}", f"Model Prediction of Returns vs<br>{time_basis} Change of {factor}", f"Regression Model Residuals (error) vs<br>{time_basis} Change of {factor}"),  specs=[[{'secondary_y': True}, {'secondary_y': True}, {'secondary_y': True}, {}]], horizontal_spacing=0.075)

    # Subplot 1: Portfolio vs Factor over time
    fig.add_trace(go.Scatter(x=df.index, y=df["Portfolio"], mode='lines', name='Portfolio'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df[factor], mode='lines', name=factor, line=dict(dash='dot')), secondary_y=True, row=1, col=1)
   
    # Set range based on respective variable - if not cumulative performance we're comparing month to month percetange change, center the y-axis at 0
    if cumulative_performance:
        # we're comparing cumulative returns which cannot be negative, so let the range be what it is
        fig.update_yaxes(title_text="Actual Returns", tickformat=".1%", row=1, col=1)
        if 'Rate' in factor:
            fig.update_yaxes(title_text=factor, tickformat=".1%", secondary_y=True, row=1, col=1)
        else:
            fig.update_yaxes(title_text=factor, tickformat="", secondary_y=True, row=1, col=1)        
    else:
        # we're comparing month to month percetange change, center the y-axis at 0
        y_range = max(abs(df["Portfolio"])) + 0.1*max(abs(df["Portfolio"])) # max absolute value of portfolio returns + 5% buffer for visuals
        secondary_y_range = max(abs(df[factor])) + 0.1*max(abs(df[factor]))
        
        fig.update_yaxes(title_text="Actual Returns", range=[-y_range, y_range], tickformat=".1%", row=1, col=1)        
        fig.update_yaxes(title_text=f'{time_basis} Change of<br>{factor}', range=[-secondary_y_range, secondary_y_range], tickformat=".1%", secondary_y=True, row=1, col=1)


    # Subplot 2: Scatter of Portfolio Returns vs Factor
    fig.add_trace(go.Scatter(x=df[factor], y=df["Portfolio"], mode='markers', name='Scatter of Factor vs Portfolio', marker=dict(size=10, color='rgba(0, 0, 152, .8)')), row=1, col=2)
    fig.add_trace(go.Scatter(x=df[factor], y=intercept + slope * df[factor], mode='lines', name='Model Fitted line', line=dict(color='red', width=2)), row=1, col=2)
    
    # Set range based on respective variable
    if cumulative_performance:
        # cumulative returns cannot be negative, so let the range be what it is
        fig.update_yaxes(title_text="Actual Returns", tickformat=".1%", row=1, col=2)   
        if 'Rate' in factor: # add % sign to tick labels if the factor is a rate, eg, unemployment rate
            fig.update_xaxes(title_text=factor, tickformat=".1%", row=1, col=2)    
        else:
            fig.update_xaxes(title_text=factor, row=1, col=2)
    else:
        # we're comparing month to month percetange change, center the y-axis at 0
        x_range = max(abs(df[factor])) + 0.1*max(abs(df[factor])) # max absolute value of factor + 5% buffer for visuals
        y_range = max(abs(df["Portfolio"])) + 0.1*max(abs(df["Portfolio"])) # max absolute value of portfolio returns + 5% buffer for visuals
        
        fig.update_xaxes(title_text=f'{time_basis} Change of<br>{factor}', range=[-x_range, x_range], tickformat=".1%", row=1, col=2)
        fig.update_yaxes(title_text=f"Actual {time_basis} Returns", range=[-y_range, y_range], tickformat=".1%", row=1, col=2)


    # Subplot 3: Predicted Return vs Actual Factor Values 
    fig.add_trace(go.Scatter(x=df[factor], y=predictions, mode='markers', name='Model Predicted vs Actual Factor', marker=dict(size=10, color='rgba(0, 152, 0, .8)')), row=1, col=3)
    fig.add_trace(go.Scatter(x=df[factor], y=df['Portfolio'], mode='markers', name='Actual Returns vs Actual Factor', marker=dict(size=10, color='rgba(0, 0, 152, .8)')), row=1, col=3)
    if cumulative_performance:
        # cumulative returns cannot be negative, so let the range be what it is
        fig.update_yaxes(title_text=f"Predicted Cumulative {time_basis} Returns", tickformat=".1%", row=1, col=3)
        if 'Rate' in factor:
            fig.update_xaxes(title_text=f"{factor}", tickformat=".1%", row=1, col=3)
        else:
            fig.update_xaxes(title_text=f"{factor}", row=1, col=3)
    else:
        fig.update_xaxes(title_text=f"{time_basis} Change of<br>{factor}", tickformat=".1%", row=1, col=3)
        fig.update_yaxes(title_text=f"Predicted {time_basis} Returns", tickformat=".1%", row=1, col=3)

    # Subplot 4: Residuals vs Factor Values
    fig.add_trace(go.Scatter(x=df[factor], y=residuals, mode='markers', name='Residuals vs Factor', marker=dict(size=10, color='rgba(152, 0, 0, .8)')), row=1, col=4)
    fig.update_yaxes(title_text="Residuals", tickformat=".1%", row=1, col=4)
    if cumulative_performance:
        if 'Rate' in factor:
            fig.update_xaxes(title_text=f"{factor}", tickformat=".1%", row=1, col=4)
        else:
            fig.update_xaxes(title_text=f"{factor}", row=1, col=4)
    else:
        fig.update_xaxes(title_text=f"{time_basis} Change of<br>{factor}", tickformat=".1%", row=1, col=4)

    # Update subplot titles and axis labels to be a little smaller
    for annotation in fig['layout']['annotations']: 
        annotation['font']['size']=12
    fig.update_xaxes(tickfont=dict(size=12), title_font=dict(size=12))
    fig.for_each_yaxis(lambda axis: axis.update(tickfont=dict(size=12), title_font=dict(size=12)))
    fig.add_annotation(x=0.65, y=0.95, xref="paper", yref="paper", text=f"R-squared = {linear_regression_model.rsquared:.3f}", showarrow=False, font=dict(size=8))
    fig.add_annotation(x=0.65, y=0.90, xref="paper", yref="paper", text=f"f(x) = {slope:.3f} * x + {intercept:.3f}", showarrow=False, font=dict(size=8))
    fig.add_annotation(x=0.65, y=0.85, xref="paper", yref="paper", text=f"Correlation = {correlation:.3f}", showarrow=False, font=dict(size=8))
        
    lag = regression_models_df.loc[regression_models_df['Factor'] == factor, 'Optimal Lag'].values[0]

    if cumulative_performance:
        fig.update_layout(title_text=f"Linear Regression Analysis and Model of {time_basis} Returns vs. {factor} (lag={lag})")
    else:
        
        fig.update_layout(title_text=f"Linear Regression Analysis and Model of {time_basis} Returns vs. {factor} {time_basis} Change (%) (lag={lag})")
        
    fig.update_layout(legend=dict(orientation="h", yanchor="top", xanchor="center", y=-0.25, x=0.5, bordercolor="lightgray", borderwidth=1,))
    return fig


def plot_irf(var_model, periods=10):
    irf = var_model.irf(periods=periods)
    portfolio_index = var_model.names.index('Portfolio')
    data = []
    max_impact = 0
    max_impact_var = ''

    # Find the variable with the maximum cumulative absolute impact on the portfolio
    for i, name in enumerate(var_model.names):
        if i != portfolio_index:
            impact = irf.orth_irfs[:, i, portfolio_index]
            cumulative_impact = sum(abs(imp) for imp in impact)
            if cumulative_impact > max_impact:
                max_impact = cumulative_impact
                max_impact_var = name

    # Create a trace for each variable's impact on the portfolio
    for i, name in enumerate(var_model.names):
        if i != portfolio_index:
            impact = irf.orth_irfs[:, i, portfolio_index]
            color = colorsys.hsv_to_rgb(i / len(var_model.names), 0.5, 0.5)
            
            if name == max_impact_var:
                line_dict = {'width': 3, 'dash': 'dash'}
                mode_dict = 'lines+markers'
            else:
                line_dict = {'width': 1}
                mode_dict = 'lines'
                
            # Set the hover template to show the entire text of the value
            hover_template = f'<b>{name}</b><br>Period: %{{x}}<br>Impact: %{{y:.2f}}'
            trace = go.Scatter(x=list(range(periods+1)), y=impact, mode=mode_dict, line=line_dict, name=f'Impact of {name} on Portfolio', hovertemplate=hover_template)
            data.append(trace)
    
    # Create the layout for the plot
    layout = go.Layout(title='Impulse Response of Portfolio Returns to Shocks in Macro Factors',
                       xaxis=dict(title='Period (portfolio return lag in months due to >1 std deviation change in factor)'),
                       yaxis=dict(title='Change in Portfolio Returns (%)'))

    # Create the figure and plot the traces
    fig = go.Figure(data=data, layout=layout)
    fig.update_yaxes(tickformat=".1%")
    
    return fig
    

def plot_forecast_var_model(var_model, steps=5):
    # Forecast the next steps
    forecast = var_model.forecast(var_model.y, steps)

    # Find the index of the Portfolio variable
    portfolio_index = var_model.names.index('Portfolio')

    # Create the forecast series for the portfolio
    forecast_portfolio = forecast[:, portfolio_index]

    # Create a trace for the forecast
    trace = go.Scatter(x=list(range(len(forecast_portfolio))), y=forecast_portfolio, name='Forecast')

    # Create the layout for the plot
    layout = go.Layout(title='Forecast for Portfolio', xaxis=dict(title='Period'), yaxis=dict(title='Value'))

    # Create the figure and plot the trace
    fig = go.Figure(data=[trace], layout=layout)
    
    return fig

            