
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
                
                asset_value *= (1 + asset_return)
                asset_values.append(asset_value)
            
            asset_dataframes[asset] = pd.Series(asset_values)

    return pd.DataFrame(asset_dataframes)

# run simulations in parallel using multiprocessing
@st.cache_data
def run_portfolio_simulations(portfolio_summary, n_simulations, distribution="T-Distribution"):
    start_time = time.time()
    n_cores = multiprocessing.cpu_count()
    print(f"Number of cores: {n_cores}")
    
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_cores) as executor:
        futures = []
        for _ in range(n_simulations):
            future = executor.submit(simulate_portfolio, portfolio_summary, distribution)
            futures.append(future)
            
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
        
    end_time = time.time()
    print("The main sim loop took", end_time - start_time, "seconds to run")
    
    return results
        
def plot_simulation_results(results):
    # Histogram of final portfolio values
    col1, col2, col3 = st.columns(3)
    

    with col1:
#        placeholder = st.empty()
        with st.spinner('Calculating Histogram of Final Portfolio Values...'):
#            placeholder.markdown("Calculating Histogram of Final Portfolio Values...")
            start_time = time.time()

            # Histogram of final portfolio values
            final_values = [result.sum(axis=1).iloc[-1] for result in results]
            
# running out of memory in the browser when calculating std dev and mean
#            std_dev = np.std(final_values)
#            mean_value = np.mean(final_values)
#            lower_bound = mean_value - 3*std_dev
#            upper_bound = mean_value + 3*std_dev

            hist_fig = go.Figure(data=[go.Histogram(x=final_values, nbinsx=250, 
                                            marker=dict(
                                                color='blue',
                                                line=dict(width=1)
                                            ))])
            

#            for i in range(1, 4):
#                hist_fig.add_shape(type="line", x0=mean_value + i*std_dev, x1=mean_value + i*std_dev, 
#                                line=dict(color="LightSeaGreen", width=2))
#                hist_fig.add_shape(type="line", x0=mean_value - i*std_dev, x1=mean_value - i*std_dev, 
#                                line=dict(color="LightSeaGreen", width=2))
        
            hist_fig.update_layout(title_text='Histogram of Final Portfolio Values',
                                xaxis_title='Portfolio Value',
                                yaxis_title='Frequency',
                                xaxis_tickprefix="$")
            end_time = time.time()
            print("The histogram took", end_time - start_time, "seconds to calculate")
            
            start_time = time.time()
            st.plotly_chart(hist_fig)
#            st.write(f"Mean: ${mean_value:,.2f}, Standard Deviation: ${std_dev:,.2f}, \
#                        3x Standard Deviation Lower/Uppwer: ${lower_bound:,.2f}-${upper_bound:,.2f}")
            
            end_time = time.time()
            print("The histogram took", end_time - start_time, "seconds to render")
  
    with col2:  
        with st.spinner('Calculating Scatter Plot of Simulated Portfolio Values...'):
#        placeholder = st.empty()
#        placeholder.markdown("Calculating Scatter Plot of Simulated Portfolio Values...")
            start_time = time.time()
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
            end_time = time.time()
            print("The scatter took", end_time - start_time, "seconds to calculate")
            

            start_time = time.time()

            st.plotly_chart(scatter_fig)
            end_time = time.time()
            print("The scatter took", end_time - start_time, "seconds to render")
    
    with col3:
#        placeholder = st.empty()
#        placeholder.markdown("Calculating Box Plot of Simulated Portfolio Values...")
        with st.spinner('Calculating Scatter Box Plot of Simulated Portfolio Values...'):
            start_time = time.time()
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
            end_time = time.time()
            print("The box plot took", end_time - start_time, "seconds to calculate")

            start_time = time.time()

            st.plotly_chart(box_fig)
            end_time = time.time()
            print("The box plot took", end_time - start_time, "seconds to render")
#        placeholder = st.empty()