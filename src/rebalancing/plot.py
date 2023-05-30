import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd

def plot_cumulative_returns(original_returns, rebalanced_returns):
    # Calculate cumulative returns
    cumulative_original_returns = (1 + original_returns).cumprod() - 1
    cumulative_rebalanced_returns = (1 + rebalanced_returns).cumprod() - 1

    # Create a Plotly figure
    fig = go.Figure()

    # Add traces for original and rebalanced portfolios
    fig.add_trace(
        go.Scatter(
            x=cumulative_original_returns.index, 
            y=cumulative_original_returns, 
            mode='lines', 
            name='Original Portfolio'
        )
    )

    fig.add_trace(
        go.Scatter(
            x=cumulative_rebalanced_returns.index, 
            y=cumulative_rebalanced_returns, 
            mode='lines', 
            name='Rebalanced Portfolio'
        )
    )

    # Add title and labels
    fig.update_layout(
        title='Cumulative Portfolio Returns Over Time',
        xaxis_title='Date',
        yaxis_title='Cumulative Returns (%)',
        yaxis_tickformat = '.1%',  # Display y-axis ticks as percentages with one decimal place
    )
    
    return fig

def plot_optimal_portfolios_over_time(optimal_portfolios, tickers):
    # Convert the optimal_portfolios to a DataFrame for easier manipulation
    optimal_portfolios_df = pd.DataFrame(optimal_portfolios, columns=tickers)
    
    # Define color sequence
    color_sequence = px.colors.qualitative.Plotly
    
    # Sort columns by their sum
    columns_sorted = optimal_portfolios_df.sum().sort_values(ascending=False).index
    
    # Create subplots for each ticker's weights
    fig1 = make_subplots(rows=len(columns_sorted), cols=1)
    for i, ticker in enumerate(columns_sorted):
        fig1.add_trace(go.Scatter(x=optimal_portfolios_df.index, y=optimal_portfolios_df[ticker], mode='lines', name=ticker,
                                  line=dict(color=color_sequence[i % len(color_sequence)])), row=i+1, col=1)
    
    fig1.update_layout(height=2000, width=800, title_text="Weight change for each Ticker over Rebalance Periods")
    
    # Stacked bar graph of the ticker weights over time
    fig2 = go.Figure(data=[
        go.Bar(name=col, x=optimal_portfolios_df.index, y=optimal_portfolios_df[col], 
               marker=dict(color=color_sequence[i % len(color_sequence)])) for i, col in enumerate(columns_sorted)
    ])
    
    # Change the bar mode
    fig2.update_layout(barmode='stack', title_text="Distribution of weights over Rebalance Periods")
    
    return fig1, fig2