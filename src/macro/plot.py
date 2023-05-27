import plotly.graph_objs as go
from plotly.subplots import make_subplots

import streamlit as st

import logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s (%(levelname)s):  %(module)s.%(funcName)s - %(message)s')

# Set up logger for a specific module to a different level
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

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

def plot_linear_regression_v_single_model(model, model_data, factor, y, prediction=None):
    
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
    layout = go.Layout(title=f'Portfolio Returns vs {factor}',
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

def plot_macro_vs_portfolio_performance(portfolio_summary, combined_data):
    logger.debug(f"combined data monthly --- First date: {combined_data.index.min()}, Last date: {combined_data.index.max()}")
    
    plots = []
    # Create subplot for each macro factor
    for factor in calculate.get_macro_factor_list():

        # Resample the macroeconomic data
        macro_factor_data = combined_data[factor]

        # Create a subplot with two y-axes
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add portfolio return trace
        fig.add_trace(
            go.Scatter(x=combined_data.index, y=combined_data['portfolio_returns'], mode='lines', name='Portfolio Returns'),
            secondary_y=False,
        )

        # Add macroeconomic factor trace
        fig.add_trace(
            go.Scatter(x=combined_data.index, y=macro_factor_data, mode='lines', name=factor),
            secondary_y=True,
        )

        # Set axis titles
        #fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Portfolio Returns (MoM % Change)", secondary_y=False)
        fig.update_yaxes(title_text=factor, secondary_y=True)

        # Set title
        fig.update_layout(title_text=f'Portfolio Returns vs {factor}')
        
        plots.append(fig)
        
    return plots

def plot_historical_macro_data(historical_macro_data):
    # some of the macro data has missing values, so we resample to a monthly basis
    resampled_data = historical_macro_data.resample('M').mean()

    plots = []
    # Create subplot for each macro factor
    for factor in calculate.get_macro_factor_list():

        # Resample the macroeconomic data
        resampled_factor_data = resampled_data[factor]
        fig = go.Figure()

        # Add macroeconomic factor trace
        fig.add_trace(
            go.Scatter(x=resampled_data.index, y=resampled_factor_data, mode='lines', name=factor))

        # Set axis titles
        #fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text=factor)
        
        plots.append(fig)
        
    return plots
    
def plot_historical_portfolio_performance(combined_data):

    # Calculate portfolio returns absolute
    portfolio_returns_absolute = combined_data['portfolio_returns']
    cumulative_returns_absolute = combined_data['cumulative_returns']
    
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=portfolio_returns_absolute.index,
        y=portfolio_returns_absolute,
        fill='tozeroy',
        mode='lines',
        name='Portfolio Returns Absolute (Weighted Portfolio)'
    ))

    fig1.update_layout(
        title='Portfolio Returns Absolute (Weighted Portfolio)<br><sup>(not including dividends)</sup>',
        yaxis_title='Portfolio Returns Absolute (Weighted Portfolio)',
        yaxis_tickformat='.1%'
    )
    
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=cumulative_returns_absolute.index,
        y=cumulative_returns_absolute,
        fill='tozeroy',
        mode='lines',
        name='Portfolio Returns Cumulative (Weighted Portfolio)'
    ))

    fig2.update_layout(
        title='Portfolio Returns Cumulative (Weighted Portfolio)<br><sup>(not including dividends)</sup>',
        yaxis_title='Portfolio Returns Cumulative (Weighted Portfolio)',
        yaxis_tickformat='.1%'
    )

    return fig1, fig2