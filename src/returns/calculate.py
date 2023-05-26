import streamlit as st
import plotly.graph_objects as go
import altair as alt

import numpy as np
import pandas as pd

import multiprocessing
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor

from datetime import datetime
import time

from scipy.stats import norm, gaussian_kde
from scipy import stats
from scipy.stats import t
from scipy import integrate
from scipy.integrate import quad

from stqdm import stqdm

import logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s (%(levelname)s):  %(module)s.%(funcName)s - %(message)s')

# Set up logger for a specific module to a different level
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def simulate_portfolio(portfolio_summary, distribution="T-Distribution"):
    # initialize asset values with initial investments
    weights = portfolio_summary["weights"]
    initial_investment = portfolio_summary["initial_investment"]
    annual_contribution = portfolio_summary["yearly_contribution"]
    mean_returns = portfolio_summary["mu"]
    cov_returns = portfolio_summary["S"]
    years = portfolio_summary["years"]
    
    asset_dataframes = {}
    for asset, weight in weights.items():
        if weight > 0:
            asset_values = [initial_investment * weight]
            for _ in range(years):
                # add annual contribution and calculate new asset values
                asset_value = asset_values[-1] + annual_contribution * weight
                
                if distribution == 'Normal':
                    asset_return = np.random.normal(mean_returns[asset], cov_returns[asset][asset])
                elif distribution == 'T-Distribution':
                    asset_return = np.random.standard_t(df=5, size=1)[0] * cov_returns[asset][asset] + mean_returns[asset]
                elif distribution == 'Cauchy':
                    asset_return = np.random.standard_cauchy(size=1)[0] * cov_returns[asset][asset] + mean_returns[asset]
                
                if asset_return > -1:
                    asset_value *= (1 + asset_return)
                else:
                    asset_value = 0
                
                asset_value = max(asset_value, annual_contribution * weight)
                asset_values.append(asset_value)
            
            asset_dataframes[asset] = pd.Series(asset_values)

    return pd.DataFrame(asset_dataframes)

# run simulations in parallel using multiprocessing
@st.cache_data
def run_portfolio_simulations(portfolio_summary, n_simulations, distribution="T-Distribution"):
    start_time = time.time()
    n_cores = multiprocessing.cpu_count()
    #print(f"Number of cores: {n_cores}")
    
    results = []
    with stqdm(total=n_simulations) as pbar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_cores) as executor:
            futures = []
            for _ in range(n_simulations):
                future = executor.submit(simulate_portfolio, portfolio_summary, distribution)
                futures.append(future)
                
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                pbar.update(1)
        
    end_time = time.time()
    logger.info(f"The main sim loop took {end_time - start_time} seconds to run")
    
    return results

def calculate_histogram_data(results):
    final_values = [result.sum(axis=1).iloc[-1] for result in results]
    df_hist = pd.DataFrame(final_values, columns=['Final Portfolio Value'])
    return df_hist

def calculate_z_scores(x, mean_per_year, std_devs_per_year):
    epsilon = 1e-8  # very small number to avoid division by zero
    return abs((x - mean_per_year[x.name]) / (std_devs_per_year[x.name] + epsilon))

def calculate_scatter_data(results):
    scatter_data = []
    for result in results:
        yearly_portfolio_values = result.sum(axis=1)
        scatter_data.append(yearly_portfolio_values)

    # Transpose the DataFrame to group data by year
    df_scatter = pd.DataFrame(scatter_data).T

    mean_per_year = df_scatter.mean(axis=1)
    std_devs_per_year = df_scatter.std(axis=1)
    
    z_scores = df_scatter.apply(calculate_z_scores, args=(mean_per_year, std_devs_per_year,), axis=1)
    
    df_scatter = df_scatter.stack().reset_index()
    df_scatter.columns = ['Year', 'Simulation', 'Portfolio Value']
    df_scatter['Z-Score'] = z_scores.stack().reset_index()[0]

    return df_scatter

def calculate_box_data(results):
    aggregated_results = pd.concat([result.sum(axis=1) for result in results], axis=1)
    df_box = aggregated_results.reset_index().melt(id_vars='index', var_name='Simulation', value_name='Portfolio Value')
    df_box['Year'] = df_box['index']
    df_box.drop(columns='index', inplace=True)
    
    return df_box

def summarize_simulation_results(results):        
    start_time = time.time()
    with st.spinner(f'Calculating plots... at time of ' + datetime.now().strftime('%H:%M:%S.%f')[:-3]):
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(calculate_histogram_data, results),
                executor.submit(calculate_scatter_data, results),
                executor.submit(calculate_box_data, results)
            ]

            for future in concurrent.futures.as_completed(futures):
                if future == futures[0]:
                    df_hist = future.result()
                    time_to_run = time.time() - start_time
                    logger.info(f"The histogram took {time_to_run} seconds to run")
                elif future == futures[1]:
                    df_scatter = future.result()
                    time_to_run = future.result()
                    time_to_run = time.time() - start_time
                    logger.info(f"The scatter took {time_to_run} seconds to run")
                else:
                    df_box = future.result()
                    time_to_run = time.time() - start_time
                    logger.info(f"The box plot took {time_to_run} seconds to run")
                    
    return df_hist, df_scatter, df_box
    
def format_currency(value):
    # Store the sign of the value to restore it later
    sign = -1 if value < 0 else 1
    value *= sign

    if value >= 1e12:   # Trillions
        return f'{"-" if sign < 0 else ""}${value/1e12:.1f}T'
    elif value >= 1e9:  # Billions
        return f'{"-" if sign < 0 else ""}${value/1e9:.1f}B'
    elif value >= 1e6:  # Millions
        return f'{"-" if sign < 0 else ""}${value/1e6:.1f}M'
    elif value >= 1e3:  # Thousands
        return f'{"-" if sign < 0 else ""}${value/1e3:.1f}K'
    else:
        return f'{"-" if sign < 0 else ""}${value:.2f}'
        
def gaussian_kde_cdf(kde, x):
    mean_value = np.mean(kde.dataset)  # Note: assuming kde.dataset is the input data
    std_dev = np.std(kde.dataset)
    N = 5
    lower_bound = max(mean_value - N*std_dev, -np.inf)
    return quad(kde.evaluate, lower_bound, x)[0]

def plot_probability_density_for_a_given_year(year, values):
    # Create the KDE
    kde = stats.gaussian_kde(values)

    # Generate the x values
    x = np.linspace(min(values), max(values), 1000)

    # Generate the y values
    y = kde.pdf(x)

    # Create the figure
    fig = go.Figure()

    # Add the density plot
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines'))

    # Calculate the mean and standard deviation
    mean_value = np.mean(values)
    std_dev = np.std(values)

    # Create x-axis ticks and labels based on standard deviations
    xticks = [mean_value + i*std_dev for i in range(-5, 6)]
    xticklabels = [format_currency(v) for v in xticks]  # updated to actual portfolio values
    
    fig.add_shape(type="line", 
                    x0=mean_value, x1=mean_value, 
                    y0=0, y1=max(y),
                    line=dict(color="LightBlue", width=1))

    probability_bands = []
    # Add sigma labels and calculate probability ranges
    for i in range(-5, 6):  # Go up to 6 now, inclusive
        color = calculate_sigma_color(i)

        fig.add_shape(type="line", 
                        x0=mean_value + i*std_dev, x1=mean_value + i*std_dev, 
                        y0=0, y1=max(y),
                        line=dict(color=color, width=1))

        fig.add_annotation(x=mean_value + i*std_dev, y=max(y),
                            text=f'{i} sigma', showarrow=False, 
                            font=dict(color=color),
                            ax=20 if i >= 0 else -20, ay=-40)  # adjusting the angle and position of the annotation

        # Calculate the integral, but only for sigma values between -5 and 4
        if -5 <= i < 5:
            probability_within_band = gaussian_kde_cdf(kde, mean_value + (i+1)*std_dev) - gaussian_kde_cdf(kde, mean_value + i*std_dev)
            probability_bands.append((i, i+1, probability_within_band, xticks[i+5], xticks[i+6]))

    fig.update_layout(title=f'Probability Density for Year {year}',
                        xaxis_title='Portfolio Value',
                        yaxis_title='Density',
                        xaxis=dict(tickmode='array', tickvals=xticks, ticktext=xticklabels),
                        autosize=True,
                        showlegend=False)

    return probability_bands, xticks, fig, year


def plot_probability_density_for_a_given_year_v0(year, values):
    #print(f"Creating probability density plot for year {year}...")
    # Create the KDE
    kde = stats.gaussian_kde(values)

    # Generate the x values
    x = np.linspace(min(values), max(values), 1000)

    # Generate the y values
    y = kde.pdf(x)

    # Create the figure
    fig = go.Figure()

    # Add the density plot
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines'))

    # Calculate the mean and standard deviation
    mean_value = np.mean(values)
    std_dev = np.std(values)

    # Create x-axis ticks and labels based on standard deviations
    xticks = [mean_value + i*std_dev for i in range(-5, 6)]
    xticklabels = [format_currency(v) for v in xticks]  # updated to actual portfolio values
    
    fig.add_shape(type="line", 
                    x0=mean_value, x1=mean_value, 
                    y0=0, y1=max(y),
                    line=dict(color="LightBlue", width=1))

    

    probability_bands = []
    # Add sigma labels and calculate probability ranges
    for i in range(-5, 6):
        color = calculate_sigma_color(i)

        fig.add_shape(type="line", 
                        x0=mean_value + i*std_dev, x1=mean_value + i*std_dev, 
                        y0=0, y1=max(y),
                        line=dict(color=color, width=1))

        fig.add_annotation(x=mean_value + i*std_dev, y=max(y),
                            text=f'{i} sigma', showarrow=False, 
                            font=dict(color=color),
                            ax=20 if i >= 0 else -20, ay=-40)  # adjusting the angle and position of the annotation

        # Calculate the integral
        if -5 < i < 5:
            probability_within_band = gaussian_kde_cdf(kde, mean_value + (i+1)*std_dev) - gaussian_kde_cdf(kde, mean_value + i*std_dev)
            # although xticks create with range(-5, 6), the index is 0-10, so have to offset i by 5 to get the correct xtick for a sigma
            #print(f"Year({year}): Probability within {i} to {i+1} sigma: {probability_within_band}, xticks: {xticks[i+5]} to {xticks[i+6]}")
            probability_bands.append((i, i+1, probability_within_band, xticks[i+5], xticks[i+6]))
        elif i == 5:  # Edge case: Calculate for the upper band μ + 5σ < X
            probability_within_band = 1 - gaussian_kde_cdf(kde, mean_value + i*std_dev)
            #print(f"Year({year}): Probability within {i} to {np.nan} sigma: {probability_within_band}, xticks: {xticks[i+5]} to {np.nan}")
            probability_bands.append((i, np.nan, probability_within_band, xticks[i+5], np.nan))
        elif i == -5:  # Edge case: Calculate for the lower band X < μ - 5σ
            probability_within_band = gaussian_kde_cdf(kde, mean_value + i*std_dev)
            #print(f"Year({year}): Probability within {np.nan} to {i+1} sigma: {probability_within_band}, xticks: {np.nan} to {xticks[i+6]}")
            probability_bands.append((np.nan, i+1, probability_within_band, np.nan, xticks[i+6]))
            
    fig.update_layout(title=f'Probability Density for Year {year}',
                        xaxis_title='Portfolio Value',
                        yaxis_title='Density',
                        xaxis=dict(tickmode='array', tickvals=xticks, ticktext=xticklabels),
                        autosize=True,
                        showlegend=False)

    return probability_bands, xticks, fig, year

# TODO: remove this from here and move to the plot.py
def calculate_sigma_color(sigma):
    colors = ["#FFD700", "#FFA500","#FF8C00","#FF4500","#FF0000"]
    
    if (np.isnan(sigma)):
        sigma = 5
        
    color = colors[abs(sigma)-1] if sigma != 0 else "LightBlue"
    return color
    
def calculate_probability_density_for_returns(results, initial_investment, yearly_contribution, specific_years_to_plot):
    n_cores = multiprocessing.cpu_count()

    sigma_levels_by_year = {}
    plots_by_year = {}
    returns_probability_by_year = {}
    with st.spinner('Calculating Probability Density Plots...'):
        pdf_results = []
        with stqdm(total=len(specific_years_to_plot)) as pbar:
            #print(f"Scheduling PDF calculations...")
            with concurrent.futures.ProcessPoolExecutor(max_workers=min(len(specific_years_to_plot), n_cores)) as executor:
                futures = []
                for year in specific_years_to_plot:
                    #print(f"Submitting calculation of PDF for year {year} time is {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
                    values = [result.iloc[year].sum() for result in results]
                    future = executor.submit(plot_probability_density_for_a_given_year, year, values)
                    futures.append(future)
                    
                for future in concurrent.futures.as_completed(futures):
                    pdf_results.append(future.result())
                    
                    probability_bands, portfolio_values_at_sigma, fig, year = future.result()
                    #print(f"Completed calculation of PDF for year {year} time is {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
                    returns_dict = calculate_returns_for_probability_bands(probability_bands, initial_investment, yearly_contribution, year)
                    #print(f"Completed calculation of probability bands for year {year} time is {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
                    sigma_levels_by_year[year] = probability_bands
                    plots_by_year[year] = fig
                    returns_probability_by_year[year] = returns_dict
            
                    pbar.update(1)
            
    return sigma_levels_by_year, plots_by_year, returns_probability_by_year
    

def calculate_returns_for_probability_bands(probability_bands, initial_investment, annual_contribution, years):
    total_investment = initial_investment + annual_contribution * years
    returns_dict = {}
    for lower_sigma, upper_sigma, probability, lower_edge_value, upper_edge_value in probability_bands:
        if lower_edge_value is not np.nan:
            lower_edge_return = (lower_edge_value - total_investment) / total_investment
        else:
            lower_edge_return = np.nan
        if upper_edge_value is not np.nan:
            upper_edge_return = (upper_edge_value - total_investment) / total_investment
        else:
            upper_edge_return = np.nan
        returns_dict[f'P( μ + {lower_sigma}σ < X < μ + {upper_sigma}σ)'] = f'{probability*100:.2f}%, lower edge return: {lower_edge_return*100:.2f}%, upper edge return: {upper_edge_return*100:.2f}%'
    return returns_dict


def calculate_probability_within_band(probability_bands, lower_sigma, upper_sigma):
    total_probability = 0
    for current_lower_sigma, current_upper_sigma, probability, _, _ in probability_bands:
        if current_lower_sigma >= lower_sigma and current_upper_sigma <= upper_sigma:
            total_probability += probability
    
    return total_probability


def calculate_portfolio_value(test_data, allocations, initial_investment, yearly_contribution):
    # Initialize portfolio DataFrame with same index as test data
    portfolio_value = pd.DataFrame(index=test_data.index)

    for asset, allocation in allocations.items():
        if (allocation > 0):
            # Calculate initial amount of shares bought
            initial_price_index = test_data[asset].first_valid_index()
            initial_price = test_data[asset].loc[initial_price_index]
            initial_shares = (initial_investment * allocation) / initial_price
            portfolio_value[asset] = initial_shares * test_data[asset]

            # Add yearly contributions
            for year in test_data.resample('Y').mean().index.year:
                #print(f"calculating portfolio value for year {year}")
                if year > test_data.index[0].year:
                    yearly_price_index = test_data[asset].loc[str(year)].first_valid_index()
                    yearly_price = test_data[asset].loc[yearly_price_index]
                    yearly_contribution_shares = (yearly_contribution * allocation) / yearly_price
                    #print(f"Year {year} yearly_contribution_shares for {asset}: {yearly_contribution_shares} based on price {yearly_price}")
                    portfolio_value.loc[str(year):, asset] += yearly_contribution_shares * test_data.loc[str(year):, asset]
                    #print(f"yearly_value for {asset} based on {yearly_contribution_shares} shares with price {test_data.loc[str(year):, asset]} is {portfolio_value.loc[str(year):, asset]}")
                
    # Sum across all assets to get total portfolio value
    portfolio_value['Total'] = portfolio_value.sum(axis=1)

    return portfolio_value

def format_tickvals(tickval):
    magnitude = 0
    while abs(tickval) >= 1000:
        magnitude += 1
        tickval /= 1000.0
    return '%.1f%s' % (tickval, ['', 'K', 'M', 'B', 'T'][magnitude])

def calculate_plots_for_portfolio_value(portfolio_value):
    # Extract stock names
    stocks = portfolio_value.columns.drop(['Total'])

    # Resample the data to monthly frequency and calculate the mean value of each stock
    monthly_data = portfolio_value.resample('M').mean()

    # Calculate the average value of each stock over the whole date range
    stock_avgs = monthly_data[stocks].mean()

    # Sort the stocks by their average values
    sorted_stocks = stock_avgs.sort_values(ascending=True).index

    # Calculate returns for each stock
    returns = monthly_data[sorted_stocks].pct_change()
    returns['Date'] = returns.index

    # Plot of absolute performance of each stock month over month
    stock_plot = go.Figure()
    for stock in sorted_stocks:
        stock_plot.add_trace(go.Scattergl(x=returns['Date'], y=returns[stock], mode='lines', name=stock))
    stock_plot.update_layout(title='Monthly Returns by Stock', xaxis_title='Date', yaxis_title='Monthly Return', 
                             yaxis=dict(tickformat=".1%"))

    # Calculate the sum of mean values for the total line
    total_line = monthly_data[sorted_stocks].sum(axis=1)

    # Prepare data for portfolio value plot
    portfolio_data = pd.DataFrame({'Year': total_line.index, 'Portfolio Value': total_line.values})

    # Define the y-axis tickvals and their corresponding labels
    #tickvals = [0, 100000, 1000000, 10000000, 100000000, 1000000000]
    #ticktext = [format_tickvals(val) for val in tickvals]

    # Plot of year over year value of the whole portfolio
    portfolio_plot = go.Figure(data=go.Scattergl(x=portfolio_data['Year'], y=portfolio_data['Portfolio Value'], mode='lines'))
    portfolio_plot.update_layout(title='Yearly Total Value (Weighted Portfolio)', xaxis_title='Year', yaxis_title='Portfolio Value') 
                                 #yaxis=dict(tickvals=tickvals, ticktext=ticktext))

    # Plot a monthly stacked bar chart showing the total value by asset
    stacked_bar_plot = go.Figure()
    for stock in sorted_stocks:
        stacked_bar_plot.add_trace(go.Bar(x=monthly_data.index, y=monthly_data[stock], name=stock))

    # Add total line to the stacked bar chart
    stacked_bar_plot.add_trace(go.Scattergl(x=total_line.index, y=total_line.values, 
                                            mode='lines+markers', name='Total', line=dict(color='lightseagreen', width=3)))

    stacked_bar_plot.update_layout(barmode='stack', title='Monthly Total Value by Asset (Weighted Portfolio)', xaxis_title='Month', yaxis_title='Total Value') 
                                   #yaxis=dict(tickvals=tickvals, ticktext=ticktext))

    return stock_plot, portfolio_plot, stacked_bar_plot

def calculate_mean_cumulative_returns(closing_prices):
    # Extract stock names
    stocks = closing_prices.columns

    # Resample the data to monthly frequency and calculate the mean value of each stock
    monthly_data = closing_prices.resample('M').mean()

    # Calculate the monthly returns for each stock
    monthly_returns = monthly_data.pct_change()

    # Calculate the cumulative returns for each stock
    cumulative_returns = (1 + monthly_returns).cumprod()

    # Sort the stocks by their average cumulative returns
    sorted_stocks = cumulative_returns.mean().sort_values(ascending=True).index

    # Plot of mean cumulative monthly returns for each stock
    cumulative_returns_plot = go.Figure()
    for stock in sorted_stocks:
        if stock != 'Total':
            cumulative_returns_plot.add_trace(go.Scattergl(x=cumulative_returns.index, y=cumulative_returns[stock], mode='lines', name=stock))
    cumulative_returns_plot.update_layout(title='Mean Cumulative Monthly Returns by Stock', xaxis_title='Date', yaxis_title='Cumulative Return',
                                          yaxis=dict(tickformat=".1%"))

    return cumulative_returns_plot

def calculate_test_ratio(portfolio_summary):
    years_of_historical_data = portfolio_summary["stock_data"].resample("Y").last().shape[0]
    test_ratio = 1 - portfolio_summary["years"] / (years_of_historical_data + portfolio_summary["years"])
    
    years_of_historical_data = portfolio_summary["end_date"].year - portfolio_summary["start_date"].year
    #print(f"years_of_historical_data: {years_of_historical_data}")
    
    return test_ratio

def split_data(data, train_size=0.8):
    data.index = data.index.tz_convert(None)
    split_index = int(len(data) * train_size)
    split_date = data.index[split_index].date()

    # If the data at the split_index is not the first data of the day,
    # adjust the split_index to the start of the next day.
    if not (data.loc[str(split_date)].index[0] == data.index[split_index]):
        split_date += pd.Timedelta(days=1)
        split_index = data.index.get_indexer([split_date], method='nearest')[0]

    train_data = data.iloc[:split_index].copy()
    test_data = data.iloc[split_index:].copy()

    return train_data, test_data