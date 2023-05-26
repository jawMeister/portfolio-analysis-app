import plotly.graph_objs as go
from plotly.subplots import make_subplots

import src.macro.calculate as calculate

def plot_linear_regression(model, historical_macro_data, X, new_X, predicted_change_in_returns):
    # Initialize an empty list to store the plots
    plots = []

    # Loop over each macroeconomic indicator
    for indicator in X.columns:

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


def plot_macro_vs_portfolio_performance(portfolio_summary, historical_macro_data):
    resampled_data = historical_macro_data.resample('M').last()

    # Calculate the month-over-month returns
    resampled_data['portfolio_returns_mom'] = resampled_data['portfolio_returns'].pct_change()

    plots = []
    # Create subplot for each macro factor
    for factor in calculate.get_macro_factor_list():

        # Resample the macroeconomic data
        resampled_factor_data = resampled_data[factor]

        # Create a subplot with two y-axes
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add portfolio return trace
        fig.add_trace(
            go.Scatter(x=resampled_data.index, y=resampled_data['portfolio_returns_mom'], mode='lines', name='Portfolio Returns'),
            secondary_y=False,
        )

        # Add macroeconomic factor trace
        fig.add_trace(
            go.Scatter(x=resampled_data.index, y=resampled_factor_data, mode='lines', name=factor),
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
    resampled_data = historical_macro_data.resample('M').last()

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

def plot_portfolio_performance(historical_returns):
    resampled_data = historical_returns.resample('M').last()

    fig = go.Figure()

    # Add macroeconomic factor trace
    fig.add_trace(
        go.Scatter(x=resampled_data.index, y=resampled_data['portfolio_returns'], mode='lines'))

    # Set axis titles
    #fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text='Historical Portfolio Returns')

    return fig
    