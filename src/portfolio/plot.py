import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import random

import src.portfolio.calculate as calculate
import src.utils as utils

import logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s (%(levelname)s):  %(module)s.%(funcName)s - %(message)s')

# Set up logger for a specific module to a different level
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
    
def plot_efficient_frontier(efficient_portfolios, selected_portfolio, optimal_portfolio):
    # Create a scatter plot for the efficient frontier
    fig = go.Figure()
    returns = [portfolio['portfolio_return'] for portfolio in efficient_portfolios]
    risks = [portfolio['volatility'] for portfolio in efficient_portfolios]
    
    xy_min = min(min(returns), min(risks)) - 0.05
    xy_max = max(max(returns), max(risks)) + 0.05
    
    fig.add_trace(go.Scatter(x=risks, y=returns, mode='lines', name='Efficient Frontier'))

    # Add a point for the selected portfolio
    fig.add_trace(go.Scatter(x=[selected_portfolio['volatility']], y=[selected_portfolio['portfolio_return']], mode='markers', name='Selected Portfolio', marker=dict(size=10, color='red')))

    # Add a point for the optimal portfolio
    opt_label = 'Optimal Portfolio (risk: ' + str(round(optimal_portfolio['volatility'], 4)) + ', return: ' + str(round(optimal_portfolio['portfolio_return'], 4)) + ')'
    fig.add_trace(go.Scatter(x=[optimal_portfolio['volatility']], y=[optimal_portfolio['portfolio_return']], mode='markers', name=opt_label, marker=dict(size=10, color='green')))

    # Add a point for the min volatility portfolio
    fig.add_trace(go.Scatter(x=[efficient_portfolios[0]['volatility']], y=[efficient_portfolios[0]['portfolio_return']], mode='markers', name='Min Volatility Portfolio', marker=dict(size=10, color='blue')))

    # Add a point for the max sharpe portfolio
    fig.add_trace(go.Scatter(x=[efficient_portfolios[-1]['volatility']], y=[efficient_portfolios[-1]['portfolio_return']], mode='markers', name='Max Sharpe Portfolio', marker=dict(size=10, color='yellow')))
    
    fig.add_trace(go.Scatter(x=[xy_min, xy_max], y=[xy_min, xy_max], mode='lines', line=dict(color='grey', dash='dash'), opacity=0.25, showlegend=False))

    fig.update_layout(title='Efficient Frontier', xaxis_title='Risk', yaxis_title='Return', legend_title='Portfolios', xaxis=dict(range=[xy_min, xy_max]), yaxis=dict(range=[xy_min, xy_max]), autosize=True,  legend=dict(x=0,y=1))

    st.plotly_chart(fig, use_container_width=True)
    
# TODO: store color by asset in session state to use across the app
def get_random_color():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return f'rgb({r},{g},{b})'

def plot_efficient_frontier_bar_chart(efficient_portfolios, selected_portfolio, optimal_portfolio):
    
    portfolios = {}
    
    # if the selected portfolio risk is greater than optimal portfolio risk, then add the selected portfolio to the list of portfolios
    if selected_portfolio['volatility'] > optimal_portfolio['volatility']:
        # Define a list of portfolios
        portfolios = {
            'Min Volatility Portfolio': efficient_portfolios[0],
            'Optimal Portfolio': optimal_portfolio,
            'Selected Portfolio': selected_portfolio,
            'Max Sharpe Portfolio': efficient_portfolios[-1]
        }
    else:
        portfolios = {
            'Min Volatility Portfolio': efficient_portfolios[0],
            'Selected Portfolio': selected_portfolio,
            'Optimal Portfolio': optimal_portfolio,
            'Max Sharpe Portfolio': efficient_portfolios[-1]
        }      

    colors = plt.get_cmap('tab10').colors
    #print(f"colors: {colors}")
    color_dict = {}
    
    # Get unique assets across all portfolios
    all_assets = set()
    for portfolio in portfolios.values():
        all_assets.update(portfolio['weights'].keys())
    all_assets = list(all_assets)

    # Create a color dictionary for assets
    colors = plt.get_cmap('tab10').colors
    color_dict = {asset: 'rgb'+str(colors[i % len(colors)]) for i, asset in enumerate(all_assets)}

    
    # Create a DataFrame for the weights
    asset_weights_df = pd.DataFrame()
    
    # Create a stacked bar chart for the weights by portfolio
    fig = go.Figure()
    
    for portfolio_name, portfolio in portfolios.items():
        weights_series = portfolio['weights']  # This is a pandas Series

        weights = []
        for key in weights_series.keys():
            weights.append(weights_series[key])

        # bar chart by asset
        temp_df = pd.DataFrame({
            'Asset': weights_series.keys(),  # Extract ticker symbols from the index of the Series
            'Weight': weights,  # Extract weights from the values of the Series
            'Portfolio': portfolio_name
        })
        asset_weights_df = pd.concat([asset_weights_df, temp_df])
        
        # bar chart by portfolio
        port_weights_df = pd.DataFrame({
            'Asset': weights_series.keys(),  # Extract ticker symbols from the index of the Series
            'Weight': weights,  # Extract weights from the values of the Series
        })
        
        port_weights_df = port_weights_df.sort_values(by='Weight', ascending=False)
        
        # Add each asset as a separate trace within the portfolio
        for asset, weight in zip(port_weights_df['Asset'], port_weights_df['Weight']):
            show_legend = True if portfolio_name == list(portfolios.keys())[0] else False
            fig.add_trace(
                go.Bar(
                    name=asset,
                    x=[portfolio_name],
                    y=[weight],
                    legendgroup=asset,
                    text=asset,
                    hoverinfo='name+y',
                    hovertemplate='%{y:.2f}',
                    marker_color=color_dict[asset],
                    showlegend=show_legend
                )
            )
            
    # Adjust layout
    fig.update_layout(
        barmode='stack',
        title_text='Portfolio Weights Comparison',
        legend=dict(
            yanchor="top",
            y=-0.12,
            xanchor="left",
            x=0.02,
            #bgcolor="LightSteelBlue",
            bordercolor="lightgray",
            borderwidth=1,
            font=dict(
                family="Courier",
                size=12
                #color="white"
            ),
            orientation="h",  # Horizontal orientation
            traceorder="reversed",  # Reverse the order
        )
    )

    # Display the figure
    st.plotly_chart(fig, use_container_width=True)

    # Pivot the DataFrame so that each portfolio is a column
    asset_weights_df = asset_weights_df.pivot(index='Asset', columns='Portfolio', values='Weight')

    # Create a stacked bar chart for the weights
    fig = go.Figure()

    for portfolio_name in portfolios.keys():
        fig.add_trace(go.Bar(
            name=portfolio_name,
            x=asset_weights_df.index,
            y=asset_weights_df[portfolio_name],
        ))

    fig.update_layout(barmode='stack', title_text='Portfolio Weights Comparison', 
                      legend=dict(
                                yanchor="top",
                                y=-0.12,
                                xanchor="left",
                                x=0.02,
                                orientation="h",  # Horizontal orientation

                        )
                      )
    
    st.plotly_chart(fig, use_container_width=True)
    

def plot_historical_performance_v0(stock_data, dividend_data, start_date, end_date, selected_portfolio_weights):

    # Retrieve S&P 500 data, risk-free rate data and calculate portfolio value
    sp500_data = utils.retrieve_historical_data('^GSPC', start_date, end_date)
    #risk_free_rate_data = utils.retrieve_risk_free_rate(start_date, end_date)
    portfolio_value = calculate.calculate_weighted_portfolio_index_value(stock_data, selected_portfolio_weights)
    
    # Calculate absolute portfolio returns and cumulative returns

    # Calculate monthly returns for tickers
    stock_returns = stock_data.pct_change().resample('M').mean()
    sp500_returns = sp500_data.pct_change().resample('M').mean()
    #risk_free_rate_returns = risk_free_rate_data.resample('M').mean()
    monthly_dividend_data = dividend_data.resample('M').sum()
    
    # Calculate relative performance 
    relative_performance_sp500 = stock_returns.sub(sp500_returns, axis=0)
    #relative_performance_risk_free = stock_returns.sub(risk_free_rate_returns, axis=0)
        
    weights = selected_portfolio_weights
    
    # Ensure weights match the columns of ticker_returns
    weights = weights.reindex(stock_returns.columns)
    #risk_free_rate_returns = risk_free_rate_returns.reindex(stock_returns.index)
    
    # Calculate portfolio returns absolute
    portfolio_returns_absolute = (stock_returns.mul(weights, axis=1)).sum(axis=1)
    cumulative_returns_absolute = (1 + portfolio_returns_absolute).cumprod() - 1

    # Calculate portfolio returns relative 
    portfolio_returns_sp500 = (stock_returns * weights).sum(axis=1)
    
    # Before calculating the dividend_yield, ensure that both dataframes have the same timezone
    if stock_data.index.tz is not None and monthly_dividend_data.index.tz is None:
        monthly_dividend_data.index = monthly_dividend_data.index.tz_localize(stock_data.index.tz)
    elif stock_data.index.tz is None and monthly_dividend_data.index.tz is not None:
        stock_data.index = stock_data.index.tz_localize(monthly_dividend_data.index.tz)

    # Now calculate the dividend_yield
    dividend_yield = monthly_dividend_data.div(stock_data.resample('M').last())

    # Ensure that both dataframes have the same timezone
    if sp500_returns.index.tz is not None and dividend_yield.index.tz is None:
        dividend_yield.index = dividend_yield.index.tz_localize(sp500_returns.index.tz)
    elif sp500_returns.index.tz is None and dividend_yield.index.tz is not None:
        sp500_returns.index = sp500_returns.index.tz_localize(dividend_yield.index.tz)

    # Now calculate the relative_dividend_performance_sp500
    relative_dividend_performance_sp500 = dividend_yield.sub(sp500_returns, axis=0)
    portfolio_dividend_yield_sp500 = (dividend_yield * weights).sum(axis=1)

    # Calculate cumulative dividend return
    cumulative_dividend_return = (1 + portfolio_dividend_yield_sp500).cumprod() - 1
    cumulative_returns_sp500 = (1 + portfolio_returns_sp500).cumprod() - 1
    
    # Calculate total returns by adding stock price returns and dividend yield
    total_returns = portfolio_returns_sp500 + dividend_yield

    # Calculate cumulative returns
    cumulative_total_returns = (1 + total_returns).cumprod() - 1

    # Chart 4: Cumulative Performance by Ticker Relative to S&P 500
    fig5 = go.Figure()
    for ticker in stock_returns.columns:
        cumulative_returns_ticker = (1 + relative_performance_sp500[ticker]).cumprod() - 1
        
        fig5.add_trace(go.Scatter(
            x=relative_performance_sp500.index,
            y=cumulative_returns_ticker,
            mode='lines',
            name=ticker
        ))
        
    fig5.update_layout(
        title='Cumulative Performance by Ticker Relative to S&P 500',
        xaxis_title='Date',
        yaxis_title='Cumulative Relative Performance',
        yaxis_tickformat='.1%'
    )
    
    st.plotly_chart(fig5, use_container_width=True)

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=relative_performance_sp500.index,
        y=cumulative_returns_sp500,
        fill='tozeroy',
        mode='lines',
        name='Cumulative Relative to S&P 500 (Weighted Portfolio)'
    ))

    fig3.update_layout(
        title='Cumulative Performance Relative to S&P 500 (Weighted Portfolio)<br><sup>(not including dividends)</sup>',
        xaxis_title='Date',
        yaxis_title='Cumulative Relative Performance',
        yaxis_tickformat='.1%'
    )
    
    st.plotly_chart(fig3, use_container_width=True)

    # Chart 1: Monthly Performance Relative to S&P 500 (Weighted Portfolio)
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        x=relative_performance_sp500.index,
        y=portfolio_returns_sp500,
        name='Relative to S&P 500 (Weighted Portfolio)'
    ))
    fig1.update_layout(
        title='Monthly Performance Relative to S&P 500 (Weighted Portfolio)',
        xaxis_title='Date',
        yaxis_title='Relative Performance',
        yaxis_tickformat='.1%'
    )

    st.plotly_chart(fig1, use_container_width=True)
    
def plot_historical_dividend_performance(portfolio_summary):
    
    # Calculate annual dividends and prices
    annual_dividends = portfolio_summary['dividend_data'].resample('Y').sum()
    annual_prices = portfolio_summary['stock_data'].resample('Y').mean()

    # Calculate the dividend yield
    dividend_yield = annual_dividends / annual_prices

    # Plot the dividend yield for each ticker
    fig = go.Figure()

    for ticker in portfolio_summary['tickers']:
        fig.add_trace(go.Scatter(x=dividend_yield.index, y=dividend_yield[ticker], 
                                 mode='lines', name=ticker))

    fig.update_layout(title='Year Over Year Dividend Yield',
                      xaxis_title='Year',
                      yaxis_title='Dividend Yield',
                      yaxis_tickformat='.2%',
                      autosize=False,
                      width=1000,
                      height=500)
    st.plotly_chart(fig, use_container_width=True)
    
def plot_asset_values(asset_values):

    # grab everything but the 'Year' column
    growth_data = asset_values.drop(columns=['Year'])
    
    # sort growth data by value in last row
    #growth_data = growth_data.sort_values(by=growth_data.index[-1], axis=1, ascending=False)

    # Create a line plot for each ticker
    fig1 = go.Figure()
    for col in growth_data.columns:
        if col != 'Year' and col != 'Total Dividends' and col != 'Total Asset Value' and col != 'YoY Asset Growth (%)' and col != 'Total Growth (%)' and col != 'Total':
            # if the last value in the column is $0, don't plot it
            if growth_data[col].iloc[-1] != 0:
                fig1.add_trace(go.Scatter(
                    x=growth_data.index,
                    y=growth_data[col],
                    mode='lines',
                    name=col
                ))
    
    #TODO:add log scale, clean up the tick marks
    #fig1.update_yaxes(type="log")
    fig1.update_yaxes(tickprefix="$")
        
    # Set plot title and labels
    fig1.update_layout(
        title='Anticipated Individual Asset Growth Over Time',
        xaxis_title='Year',
        yaxis_title='Asset Value'
    )


    growth_data_stacked = asset_values.drop(columns=['Year'])
    # exclued the last 4 columns from the table (total growth)
    growth_data_stacked = growth_data_stacked.iloc[:, :-4]
    # sort the stacked data by the last row (total growth)
    #growth_data_stacked = growth_data_stacked.sort_values(by=growth_data_stacked.index[-1], axis=1, ascending=False)
    
    # Create a stacked bar chart
    fig2 = go.Figure()
    for ticker in growth_data_stacked.columns:
        # if the last value in the column is $0, don't plot it
        if growth_data_stacked[ticker].iloc[-1] != 0:
            fig2.add_trace(go.Bar(
                x=growth_data_stacked.index,
                y=growth_data_stacked[ticker],
                name=ticker
            ))
    fig2.update_yaxes(tickprefix="$")

    # Set plot title and labels
    fig2.update_layout(
        title='Total Projected Growth',
        xaxis_title='Year',
        yaxis_title='Asset Value',
        barmode='stack'
    )

    # Calculate the total value of each ticker at the end of the projection
    end_values = asset_values.iloc[-1, :-5]

    # Create a bar chart for the end values
    fig3 = go.Figure()
    
    fig3.add_trace(go.Bar(
        x=end_values.index,
        y=end_values,
        name='Asset Value'
    ))
    fig3.update_yaxes(tickprefix="$")

    # Set plot title and labels
    fig3.update_layout(
        title='Total Value of Each Ticker at the End of the Projection',
        xaxis_title='Ticker',
        yaxis_title='Asset Value'
    )

    
    with st.container():
        col1, col2, col3 = st.columns([1,1,1])
        with col1:
            st.plotly_chart(fig2, use_container_width=True)
        with col2:
            st.plotly_chart(fig1, use_container_width=True)
        with col3:
            st.plotly_chart(fig3, use_container_width=True)


def plot_historical_and_relative_performance(df_returns_monthly, df_total_returns_monthly, df_portfolio_returns_monthly, df_sp500_returns_monthly, df_relative_to_sp500_monthly, df_relative_to_rf_monthly):
    # relative_performance_sp500 = stock_returns.sub(sp500_returns, axis=0)
    # cumulative_returns_ticker = (1 + relative_performance_sp500[ticker]).cumprod() - 1
   # Compute cumulative returns
    df_cumulative_total_returns_monthly = (1 + df_total_returns_monthly).cumprod() - 1
    cumulative_portfolio_returns_monthly = (1 + df_portfolio_returns_monthly).cumprod() - 1
    cumulative_sp500_returns_monthly = (1 + df_sp500_returns_monthly).cumprod() - 1
    cumulative_relative_to_sp500_monthly = (1 + df_relative_to_sp500_monthly).cumprod() - 1
    cumulative_relative_to_rf_monthly = (1 + df_relative_to_rf_monthly).cumprod() - 1

    # Compute percentage performance relative to S&P 500
    df_cumulative_total_returns_relative_to_sp500 = df_cumulative_total_returns_monthly.div(cumulative_sp500_returns_monthly, axis=0) - 1
    df_cumulative_returns_by_ticker_relative_to_sp500 = df_relative_to_sp500_monthly

    # (1) total returns by ticker as a line chart
    fig1 = go.Figure()
    for col in df_total_returns_monthly.columns:
        fig1.add_trace(go.Scatter(x=df_total_returns_monthly.index, y=df_total_returns_monthly[col], mode='lines', name=col))
    fig1.update_layout(title='Total Returns by Ticker', xaxis_title='Date', yaxis_title='Total Returns')

    # (2) total portfolio returns and total portfolio returns vs s&p and total portfolio returns vs risk free rate as three line charts with the total portfolio returns having a fill
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=cumulative_portfolio_returns_monthly.index, y=cumulative_portfolio_returns_monthly, mode='lines', name='Portfolio', fill='tozeroy'))
    fig2.add_trace(go.Scatter(x=cumulative_relative_to_sp500_monthly.index, y=cumulative_relative_to_sp500_monthly, mode='lines', name='Relative to S&P 500', marker=dict(color='lightseagreen')))
    fig2.add_trace(go.Scatter(x=cumulative_relative_to_rf_monthly.index, y=cumulative_relative_to_rf_monthly, mode='lines', name='Relative to Risk-Free Rate'))
    fig2.update_layout(title='Cumulative Total Portfolio Returns & Benchmarks', xaxis_title='Date', yaxis_title='Total Returns')


    # (3) month to month portfolio performance (positive and negative) on an absolute basis as a column chart with a line chart of the s&p month to month performance in the same time period
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(x=df_portfolio_returns_monthly.index, y=df_portfolio_returns_monthly, name='Portfolio'))
    fig3.add_trace(go.Scatter(x=df_sp500_returns_monthly.index, y=df_sp500_returns_monthly, mode='lines', name='S&P 500', marker=dict(color='lightseagreen')))
    fig3.update_layout(title='Month-to-Month Portfolio & S&P 500 Performance', xaxis_title='Date', yaxis_title='Total Returns')
    
    return fig1, fig2, fig3

# TODO: want this to be total returns vs just ticker returns, although dividend math not working atm
def plot_historical_performance_by_ticker(df_dict):
    # total is returns + dividends
    df_total_monthly_returns_by_ticker = df_dict['df_returns_by_ticker'].resample('M').apply(lambda x: (1 + x).prod() - 1)
    df_sp500_monthly_returns = df_dict['df_sp500_returns'].resample('M').apply(lambda x: (1 + x).prod() - 1)
    
    logger.debug(f'df_total_monthly_returns_by_ticker: {df_total_monthly_returns_by_ticker.keys()}')
    logger.debug(f'df total monthly returns by ticker: {df_total_monthly_returns_by_ticker.tail()}')
    
    total_cumulative_monthly_returns_by_ticker = (1 + df_total_monthly_returns_by_ticker).cumprod() - 1
    sp500_cumulative_monthly_returns = (1 + df_sp500_monthly_returns).cumprod() - 1
    
    total_cumulative_monthly_returns_relative_to_sp500 = total_cumulative_monthly_returns_by_ticker.subtract(sp500_cumulative_monthly_returns, axis=0)
    
    
    # (1) total returns by ticker as a line chart
    fig1 = go.Figure()
    for ticker in total_cumulative_monthly_returns_relative_to_sp500.columns:
        fig1.add_trace(go.Scatter(x=total_cumulative_monthly_returns_relative_to_sp500.index, y=total_cumulative_monthly_returns_relative_to_sp500[ticker], mode='lines', name=ticker))
    fig1.update_layout(title='Cumulative Monthly Returns Relative to S&P 500', yaxis_title='Performance Relative to S&P 500')
    
    fig1.update_yaxes(tickformat='.1%')
    
    return fig1

def plot_portfolio_performance_by_benchmark(df_dict):
    df_portfolio_returns_monthly = df_dict['df_weighted_portfolio_returns'].resample('M').apply(lambda x: (1 + x).prod() - 1)
#    df_relative_to_sp500_monthly = df_dict['df_portfolio_returns_relative_to_sp500'].resample('M').apply(lambda x: (1 + x).prod() - 1)
#    df_relative_to_rf_monthly = df_dict['df_portfolio_returns_relative_to_rf'].resample('M').apply(lambda x: (1 + x).prod() - 1)
    df_sp500_returns_monthly = df_dict['df_sp500_returns'].resample('M').apply(lambda x: (1 + x).prod() - 1)

    cumulative_portfolio_returns_monthly = (1 + df_portfolio_returns_monthly).cumprod() - 1
#    cumulative_relative_to_sp500_monthly = (1 + df_relative_to_sp500_monthly).cumprod() - 1
#    cumulative_relative_to_rf_monthly = (1 + df_relative_to_rf_monthly).cumprod() - 1
    cumulative_sp500_returns_monthly = (1 + df_sp500_returns_monthly).cumprod() - 1

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(x=cumulative_portfolio_returns_monthly.index, y=cumulative_portfolio_returns_monthly, mode='lines', name='Weighted Portfolio', fill='tozeroy'),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=cumulative_sp500_returns_monthly.index, y=cumulative_sp500_returns_monthly, mode='lines', name='Cumulative S&P 500 Returns', marker=dict(color='lightseagreen')),
    )

#    fig.add_trace(
#        go.Scatter(x=cumulative_relative_to_rf_monthly.index, y=cumulative_relative_to_rf_monthly, mode='lines', name='Relative to Risk-Free Rate'),
#        secondary_y=True,
#    )

    fig.update_layout(
        title='Cumulative (Weighted) Portfolio Returns & Benchmarks',
        yaxis_title='Total Returns',
        yaxis2_title='Relative Returns',
        legend=dict(x=0,y=1)
    )
    
    fig.update_yaxes(tickformat='.1%', secondary_y=False)
    fig.update_yaxes(tickformat='.1%', secondary_y=True)

    return fig

def plot_month_to_month_portfolio_performance(df_dict):
    df_portfolio_returns_monthly  = df_dict['df_weighted_portfolio_returns'].resample('M').apply(lambda x: (1 + x).prod() - 1)
    df_sp500_returns_monthly = df_dict['df_sp500_returns'].resample('M').apply(lambda x: (1 + x).prod() - 1)
    
    # (3) month to month portfolio performance (positive and negative) on an absolute basis as a column chart with a line chart of the s&p month to month performance in the same time period
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(x=df_portfolio_returns_monthly.index, y=df_portfolio_returns_monthly, name='Portfolio'))
    fig3.add_trace(go.Scatter(x=df_sp500_returns_monthly.index, y=df_sp500_returns_monthly, mode='lines', name='S&P 500', marker=dict(color='lightseagreen')))
    fig3.update_layout(title='Month-to-Month Portfolio (Weighted) Performance', yaxis_title='Total Returns', yaxis_tickformat='.1%', legend=dict(x=0,y=1))

    return fig3