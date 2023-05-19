import plotly.graph_objects as go
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import random

import utils

def plot_efficient_frontier_v0(efficient_portfolios, selected_portfolio, optimal_portfolio):
    fig, ax = plt.subplots()

    # Plot efficient frontier
    returns = [portfolio['returns'] for portfolio in efficient_portfolios]
    risks = [portfolio['risks'] for portfolio in efficient_portfolios]
    ax.plot(risks, returns, 'y', label='Efficient Frontier')

    # Plot selected portfolio
    ax.plot(selected_portfolio['risks'], selected_portfolio['returns'], 'ro', label='Selected Portfolio')

    # Plot optimal portfolio
    opt_label = 'Optimal Portfolio (risk: ' + str(round(optimal_portfolio['risks'], 4)) + ', return: ' + str(round(optimal_portfolio['returns'], 4)) + ')'
    ax.plot(optimal_portfolio['risks'], optimal_portfolio['returns'], 'go', label=opt_label)
    
    # Plot min volatility portfolio
    ax.plot(efficient_portfolios[0]['risks'], efficient_portfolios[0]['returns'], 'bo', label='Min Volatility Portfolio')
    
    # Plot max sharpe portfolio
    ax.plot(efficient_portfolios[-1]['risks'], efficient_portfolios[-1]['returns'], 'yo', label='Max Sharpe Portfolio')

    ax.legend(loc='best')
    ax.set_xlabel('Risk')
    ax.set_ylabel('Return')
    ax.set_title('Efficient Frontier')

    st.pyplot(fig)
    
def plot_efficient_frontier(efficient_portfolios, selected_portfolio, optimal_portfolio):
    # Create a scatter plot for the efficient frontier
    fig = go.Figure()
    returns = [portfolio['returns'] for portfolio in efficient_portfolios]
    risks = [portfolio['risks'] for portfolio in efficient_portfolios]
    fig.add_trace(go.Scatter(x=risks, y=returns, mode='lines', name='Efficient Frontier'))

    # Add a point for the selected portfolio
    fig.add_trace(go.Scatter(x=[selected_portfolio['risks']], y=[selected_portfolio['returns']], mode='markers', name='Selected Portfolio', marker=dict(size=10, color='red')))

    # Add a point for the optimal portfolio
    opt_label = 'Optimal Portfolio (risk: ' + str(round(optimal_portfolio['risks'], 4)) + ', return: ' + str(round(optimal_portfolio['returns'], 4)) + ')'
    fig.add_trace(go.Scatter(x=[optimal_portfolio['risks']], y=[optimal_portfolio['returns']], mode='markers', name=opt_label, marker=dict(size=10, color='green')))

    # Add a point for the min volatility portfolio
    fig.add_trace(go.Scatter(x=[efficient_portfolios[0]['risks']], y=[efficient_portfolios[0]['returns']], mode='markers', name='Min Volatility Portfolio', marker=dict(size=10, color='blue')))

    # Add a point for the max sharpe portfolio
    fig.add_trace(go.Scatter(x=[efficient_portfolios[-1]['risks']], y=[efficient_portfolios[-1]['returns']], mode='markers', name='Max Sharpe Portfolio', marker=dict(size=10, color='yellow')))

    fig.update_layout(title='Efficient Frontier', xaxis_title='Risk', yaxis_title='Return', legend_title='Portfolios', autosize=True)

    st.plotly_chart(fig)
    
def get_random_color():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return f'rgb({r},{g},{b})'

def plot_efficient_frontier_bar_chart(efficient_portfolios, selected_portfolio, optimal_portfolio):
    
    portfolios = {}
    
    # if the selected portfolio risk is greater than optimal portfolio risk, then add the selected portfolio to the list of portfolios
    if selected_portfolio['risks'] > optimal_portfolio['risks']:
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
    st.plotly_chart(fig)

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

    fig.update_layout(barmode='stack', title_text='Portfolio Weights Comparison')
    st.plotly_chart(fig)
    

def plot_historical_performance(stock_data, dividend_data, start_date, end_date, selected_portfolio):

    # Retrieve S&P 500 data, risk-free rate data and calculate portfolio value
    sp500_data = utils.retrieve_historical_data('^GSPC', start_date, end_date)
    #risk_free_rate_data = utils.retrieve_risk_free_rate(start_date, end_date)
    portfolio_value = utils.calculate_portfolio_value(stock_data, selected_portfolio['weights'])
    
    # Calculate absolute portfolio returns and cumulative returns

    # Calculate monthly returns for tickers
    stock_returns = stock_data.pct_change().resample('M').mean()
    sp500_returns = sp500_data.pct_change().resample('M').mean()
    #risk_free_rate_returns = risk_free_rate_data.resample('M').mean()
    monthly_dividend_data = dividend_data.resample('M').sum()
    
    # Calculate relative performance 
    relative_performance_sp500 = stock_returns.sub(sp500_returns, axis=0)
    #relative_performance_risk_free = stock_returns.sub(risk_free_rate_returns, axis=0)
        
    weights = selected_portfolio['weights']
    
    # Ensure weights match the columns of ticker_returns
    weights = weights.reindex(stock_returns.columns)
    #risk_free_rate_returns = risk_free_rate_returns.reindex(stock_returns.index)
    
    # Calculate portfolio returns absolute
    portfolio_returns_absolute = (stock_returns.mul(weights, axis=1)).sum(axis=1)
    cumulative_returns_absolute = (1 + portfolio_returns_absolute).cumprod() - 1

    # Calculate portfolio returns relative 
    portfolio_returns_sp500 = (stock_returns * weights).sum(axis=1)
    #portfolio_returns_risk_free = (stock_returns.mul(weights, axis=1) - risk_free_rate_returns).sum(axis=1)
    # Calculate relative performance ******************************************************
    #relative_performance_sp500 = stock_returns.sub(sp500_returns, axis='columns')
    #cumulative_relative_performance_sp500 = (1 + relative_performance_sp500).cumprod() - 1
    
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
    
    st.plotly_chart(fig5)
    
    #TODO: fix this chart as the dividend yield is not correct
    """ 
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=cumulative_dividend_return.index,
        y=cumulative_dividend_return,
        fill='tozeroy',
        mode='lines',
        name='Cumulative Dividends Relative to S&P 500 (Weighted Portfolio)'
    ))
    fig3.add_trace(go.Scatter(
        x=cumulative_total_returns.index,
        y=cumulative_total_returns,
        fill='tonexty',
        mode='lines',
        name='Total Cumulative Relative to S&P 500 (Weighted Portfolio)'
    ))
    fig3.update_layout(
        title='Cumulative Performance Relative to S&P 500 (Weighted Portfolio)',
        xaxis_title='Date',
        yaxis_title='Cumulative Relative Performance',
        yaxis_tickformat='.1%'
    )
    """

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=relative_performance_sp500.index,
        y=cumulative_returns_sp500,
        fill='tozeroy',
        mode='lines',
        name='Cumulative Relative to S&P 500 (Weighted Portfolio)'
    ))
    
    ###
    #fig3.add_trace(go.Scatter(
    #    x=relative_performance_sp500.index,
    #    y=cumulative_returns_absolute,
    #    mode='lines',
    #    name='Portfolio Absolute Performance'
    #))
    
    fig3.update_layout(
        title='Cumulative Performance Relative to S&P 500 (Weighted Portfolio)<br><sup>(not including dividends)</sup>',
        xaxis_title='Date',
        yaxis_title='Cumulative Relative Performance',
        yaxis_tickformat='.1%'
    )
    
    st.plotly_chart(fig3)

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

    st.plotly_chart(fig1)


def plot_historical_performance_v0(stock_data, dividend_data, start_date, end_date, selected_portfolio):

    # Retrieve S&P 500 data, risk-free rate data and calculate portfolio value
    #sp500_data = yf.download('^GSPC', start=start_date, end=end_date)['Adj Close']
    sp500_data = utils.retrieve_historical_data('^GSPC', start_date, end_date)
    risk_free_rate_data = utils.retrieve_risk_free_rate(start_date, end_date)
    portfolio_value = utils.calculate_portfolio_value(stock_data, selected_portfolio['weights'])

    
    # Calculate monthly returns for tickers
    stock_returns = stock_data.pct_change().resample('M').mean()
    sp500_returns = sp500_data.pct_change().resample('M').mean()
    risk_free_rate_returns = risk_free_rate_data.resample('M').mean()
    monthly_dividend_data = dividend_data.resample('M').sum()
    
    # Calculate relative performance 
    relative_performance_sp500 = stock_returns.sub(sp500_returns, axis=0)
    relative_performance_risk_free = stock_returns.sub(risk_free_rate_returns, axis=0)
        
    #print(f"selected portfolio weights: {selected_portfolio['weights']}")
    #print(f"selected portfolio weight values: {selected_portfolio['weights'].values}")
    weights = selected_portfolio['weights']
    
    # Ensure weights match the columns of ticker_returns
    weights = weights.reindex(stock_returns.columns)
    risk_free_rate_returns = risk_free_rate_returns.reindex(stock_returns.index)
    
    # Calculate portfolio returns relative 
    portfolio_returns_sp500 = (stock_returns * weights).sum(axis=1)
    portfolio_returns_risk_free = (stock_returns.mul(weights, axis=1) - risk_free_rate_returns).sum(axis=1)
    weighted_dividends = (monthly_dividend_data * weights).sum(axis=1)
    cumulative_dividend_return = (1 + weighted_dividends).cumprod() - 1

    # Chart 4: Cumulative Performance by Ticker Relative to S&P 500
    fig5 = go.Figure()
    for ticker in stock_returns.columns:
        cumulative_returns_ticker = (1 + relative_performance_sp500[ticker]).cumprod() - 1
        fig5.add_trace(go.Scatter(
            x=relative_performance_sp500.index,
            y=cumulative_returns_ticker,
            mode='lines',
            name=f'Cumulative Relative to S&P 500 ({ticker})'
        ))
    fig5.update_layout(
        title='Cumulative Performance by Ticker Relative to S&P 500',
        xaxis_title='Date',
        yaxis_title='Cumulative Relative Performance',
        yaxis_tickformat='.1%'
    )
    
    st.plotly_chart(fig5)
    
    # Chart 3: Cumulative Performance Relative to S&P 500 (Weighted Portfolio)
    cumulative_returns_sp500 = (1 + portfolio_returns_sp500).cumprod() - 1
    
    print(f"cumulative_returns_sp500: {cumulative_returns_sp500.tail()}")
    print(f"cumulative_dividend_return: {cumulative_dividend_return.tail()}")
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=relative_performance_sp500.index,
        y=cumulative_returns_sp500,
        fill='tozeroy',
        mode='lines',
        name='Cumulative Relative to S&P 500 (Weighted Portfolio)<br><sup>(not including dividends)</sup>'
    ))
    fig3.update_layout(
        title='Cumulative Performance Relative to S&P 500 (Weighted Portfolio)',
        xaxis_title='Date',
        yaxis_title='Cumulative Relative Performance',
        yaxis_tickformat='.1%'
    )
    
    st.plotly_chart(fig3)

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

    st.plotly_chart(fig1)
    
    """
    # Chart 2: Monthly Performance Relative to Risk-Free Rate (Weighted Portfolio)
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=relative_performance_risk_free.index,
        y=portfolio_returns_risk_free,
        name='Relative to Risk-Free Rate (Weighted Portfolio)'
    ))
    fig2.update_layout(
        title='Monthly Performance Relative to Risk-Free Rate (Weighted Portfolio)',
        xaxis_title='Date',
        yaxis_title='Relative Performance'
    )
    st.plotly_chart(fig2)
    """
    
    """
    # Chart 4: Individual Ticker Performance Relative to S&P 500
    fig4 = go.Figure()
    for ticker in stock_returns.keys():
        fig4.add_trace(go.Scatter(
            x=relative_performance_sp500.index,
            y=relative_performance_sp500[ticker],
            mode='lines',
            name=f'Relative to S&P 500 ({ticker})'
        ))
    fig4.update_layout(
        title='Individual Ticker Performance Relative to S&P 500',
        xaxis_title='Date',
        yaxis_title='Relative Performance (%)'
    )
    fig4.show()
    """
    
    """   
    # Chart 5: Cumulative Performance Relative to S&P 500 (Weighted Portfolio)
    cumulative_returns_sp500 = (1 + portfolio_returns_sp500).cumprod() - 1
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(
        x=relative_performance_sp500.index,
        y=cumulative_returns_sp500,
        mode='lines',
        name='Cumulative Relative to S&P 500 (Weighted Portfolio)'
    ))
    fig4.update_layout(
        title='Cumulative Performance Relative to S&P 500 (Weighted Portfolio)',
        xaxis_title='Date',
        yaxis_title='Cumulative Relative Performance'
    )
    fig4.show()
    
 
    # Create a figure
    fig1 = go.Figure()
    fig2 = go.Figure()
    
    # Add traces for ticker symbols relative to S&P 500
    for ticker in stock_returns.keys():
        fig1.add_trace(go.Scatter(
            x=relative_performance_sp500.index,
            y=relative_performance_sp500[ticker],
            mode='lines',
            name=f'Relative to S&P 500 ({ticker})'
        ))

    # Add trace for ticker symbols relative to risk-free rate
    fig2.add_trace(go.Scatter(
        x=relative_performance_risk_free.index,
        y=relative_performance_risk_free.mean(axis=1),
        mode='lines',
        name='Relative to Risk-Free Rate (Average)'
    ))

    # Update layout
    fig2.update_layout(
        title='Monthly Performance Relative to Benchmarks',
        xaxis_title='Date',
        yaxis_title='Relative Performance'
    )

    # Show the figure
    fig1.show()
    fig2.show()
    print("********** plot complete ************")
    """
    
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
    st.plotly_chart(fig)
    
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
            st.plotly_chart(fig2)
        with col2:
            st.plotly_chart(fig1)
        with col3:
            st.plotly_chart(fig3)
