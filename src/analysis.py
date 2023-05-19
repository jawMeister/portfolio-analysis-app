
import plotly.graph_objects as go
from plotly import figure_factory as ff

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import time
import streamlit as st
from scipy.stats import norm, gaussian_kde
from scipy import stats

# function to simulate future portfolio values
def simulate_portfolio(portfolio_summary):
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
                asset_return = np.random.normal(mean_returns[asset], cov_returns[asset][asset])
                asset_value *= (1 + asset_return)
                asset_values.append(asset_value)
            
            asset_dataframes[asset] = pd.Series(asset_values)

    return pd.DataFrame(asset_dataframes)

# run simulations in parallel using multiprocessing
@st.cache
def run_portfolio_simulations(portfolio_summary, n_simulations):
    start_time = time.time()
    n_cores = multiprocessing.cpu_count()
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_cores) as executor:
        futures = []
        for _ in range(n_simulations):
            future = executor.submit(simulate_portfolio, portfolio_summary)
            futures.append(future)
            
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
        
    end_time = time.time()
    print("The loop took", end_time - start_time, "seconds to run")
    
    return results
        
def plot_simulation_results(results):
    # Histogram of final portfolio values
    col1, col2, col3 = st.columns(3)
    
    # Histogram of final portfolio values
    # Histogram of final portfolio values
    final_values = [result.sum(axis=1).iloc[-1] for result in results]

    hist_fig = go.Figure(data=[go.Histogram(x=final_values, nbinsx=50, 
                                    marker=dict(
                                        color='blue',
                                        line=dict(width=1)
                                    ))])
    hist_fig.update_layout(title_text='Histogram of Final Portfolio Values',
                           xaxis_title='Portfolio Value',
                           yaxis_title='Frequency',
                           xaxis_tickprefix="$")
    
    with col1:
        st.plotly_chart(hist_fig)

    # Scatter plot of portfolio value over time for all simulations
    scatter_fig = go.Figure()

    scatter_data = []
    for result in results:
        yearly_portfolio_values = result.sum(axis=1)
        scatter_data.append(yearly_portfolio_values)

    x_values = []
    y_values = []
    color_values = []

    # Transpose the DataFrame to group data by year
    scatter_df = pd.DataFrame(scatter_data).T

    mean_per_year = scatter_df.mean(axis=1).tolist()
    std_devs_per_year = scatter_df.std(axis=1).tolist()
    
    epsilon = 1e-8 # very small number to avoid division by zero

    for year, values in scatter_df.iterrows():
        x_values.extend([year] * len(values))
        y_values.extend(values)
        color_values.extend([abs((value - mean_per_year[year]) / (std_devs_per_year[year] + epsilon)) for value in values])  # Absolute Z-score

    scatter_fig.add_trace(go.Scatter(x=x_values, y=y_values, mode='markers', 
                                     marker=dict(color=color_values, size=5, opacity=0.5,
                                                 colorbar=dict(title='Absolute Z-score'),
                                                 colorscale='Viridis', showscale=True)))

    scatter_fig.update_layout(title_text='Portfolio Value Over Time (All Simulations)',
                              xaxis_title='Year',
                              yaxis_title='Portfolio Value',
                              yaxis_type="log",
                              yaxis_tickprefix="$",
                              showlegend=False)
    
    with col2:
        st.plotly_chart(scatter_fig)
    
    # Box plot of portfolio values for each year
    box_fig = go.Figure()

    aggregated_results = pd.concat([result.sum(axis=1) for result in results], axis=1)
    for year in aggregated_results.index:
        box_fig.add_trace(go.Box(y=aggregated_results.loc[year], quartilemethod="inclusive", 
                                 name=f"Year {year+1}", showlegend=False, # No legend, same marker color
                                 marker_color='blue'))

    box_fig.update_layout(title_text='Box plot of Portfolio Values Per Year',
                          yaxis_title='Portfolio Value',
                          yaxis_type="log",
                          yaxis_tickprefix="$")

    with col3:
        st.plotly_chart(box_fig)