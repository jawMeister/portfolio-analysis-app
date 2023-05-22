
import plotly.graph_objects as go
from plotly import figure_factory as ff
import plotly.io as pio

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
from scipy.stats import t
from scipy import integrate
from scipy.integrate import quad
import altair as alt

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

import base64
from PIL import Image
from io import BytesIO

from stqdm import stqdm

# attempting to set the matplotlib style to the streamlit default
plt.rcParams['text.color'] = '#E0E0E0'  # Setting the text color to a light gray
plt.rcParams['axes.labelcolor'] = '#E0E0E0'  # Setting the color of the axis labels
plt.rcParams['axes.edgecolor'] = '#E0E0E0'  # Setting the color of the axis edges
plt.rcParams['xtick.color'] = '#E0E0E0'  # Setting the color of the tick labels on the x-axis
plt.rcParams['ytick.color'] = '#E0E0E0'  # Setting the color of the tick labels on the y-axis
plt.rcParams['figure.facecolor'] = '#1F1F1F'  # Setting the color of the figure face
plt.rcParams['axes.facecolor'] = '#1F1F1F'  # Setting the color of the axis face
plt.rcParams['figure.autolayout'] = True  # Adjusting the subplot parameters to give the specified padding
plt.rcParams['font.size'] = 6  # Adjusting the font size
mpl.rcParams['axes.linewidth'] = 0.2
mpl.rcParams['lines.linewidth'] = 0.5

# Since IBM Plex Sans may not be installed in your machine, we are using a standard font 'DejaVu Sans'
plt.rcParams['font.family'] = 'DejaVu Sans'

# And finally for the plot color cycle, Streamlit uses a specific color cycle, 
# here we are using a common darker color palette which is the 'Dark2' from matplotlib
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Dark2.colors)

#TODO: work on performance, need to explore other charting libraries, eg, Seaborn, Altair, Bokeh, etc.
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
    print(f"Number of cores: {n_cores}")
    
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
    print("The main sim loop took", end_time - start_time, "seconds to run")
    
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

def plot_simulation_results_altair(results, initial_investment):
    col1, col2, col3 = st.columns(3)
    start_time = time.time()
    with st.spinner('Calculating plots...'):
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(calculate_histogram_data, results),
                executor.submit(calculate_scatter_data, results),
                executor.submit(calculate_box_data, results)
            ]
            
            for future in concurrent.futures.as_completed(futures):
                if future == futures[0]:
                    df_hist = future.result()
                    end_time = time.time()
                    print("The histogram took", end_time - start_time, "seconds to run")
                elif future == futures[1]:
                    df_scatter = future.result()
                    end_time = time.time()
                    print("The scatter took", end_time - start_time, "seconds to run")
                else:
                    df_box = future.result()
                    end_time = time.time()
                    print("The box plot took", end_time - start_time, "seconds to run")
            
    with col1:
        hist_chart = alt.Chart(df_hist).mark_bar().encode(
            alt.X("Final Portfolio Value:Q", bin=alt.Bin(maxbins=100)),
            y='count()',
        )
        st.altair_chart(hist_chart, use_container_width=True)

    with col2:
        scatter_chart = alt.Chart(df_scatter).mark_circle(size=35).encode(
            x=alt.X('Year:Q'),
            y=alt.Y('Portfolio Value:Q', scale=alt.Scale(type='log', domain=[initial_investment, max(df_scatter['Portfolio Value'])])),
            color=alt.Color('Z-Score:Q', scale=alt.Scale(scheme='magma', reverse=True)),
            tooltip=['Year:Q', 'Portfolio Value:Q', 'Z-Score:Q']
        ).properties(title='Portfolio Value Over Time (All Simulations)')
        st.altair_chart(scatter_chart, use_container_width=True) # not working at the moment

    with col3:
        box_chart = alt.Chart(df_box).mark_boxplot(size=25 , extent=0.5).encode(
            x=alt.X('Year:Q', scale=alt.Scale(domain=[min(df_box['Year']), max(df_box['Year'])])),
            y=alt.Y('Portfolio Value:Q', scale=alt.Scale(type='log', domain=[initial_investment, max(df_scatter['Portfolio Value'])])),
            tooltip=['Year:Q', 'Portfolio Value:Q']
        ).properties(
            title='Box plot of Portfolio Values Per Year'
        )
        
        st.altair_chart(box_chart, use_container_width=True) 
        
def plot_simulation_results_seaborn(results, initial_investment):
    col1, col2, col3 = st.columns(3)
    start_time = time.time()
    with st.spinner('Calculating plots...'):
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(calculate_histogram_data, results),
                executor.submit(calculate_scatter_data, results),
                executor.submit(calculate_box_data, results)
            ]
            
            for future in concurrent.futures.as_completed(futures):
                if future == futures[0]:
                    df_hist = future.result()
                    end_time = time.time()
                    print("The histogram took", end_time - start_time, "seconds to run")
                elif future == futures[1]:
                    df_scatter = future.result()
                    end_time = time.time()
                    print("The scatter took", end_time - start_time, "seconds to run")
                else:
                    df_box = future.result()
                    end_time = time.time()
                    print("The box plot took", end_time - start_time, "seconds to run")
    
    with col1:
        fig1, ax1 = plt.subplots(figsize=(6,4))
        sns.histplot(df_hist['Final Portfolio Value'], bins=100, kde=True, ax=ax1, color='#ff2e63')
        ax1.set_xlabel('Final Portfolio Value')
        ax1.set_ylabel('Count')
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots(figsize=(6,4))
        cmap = sns.cubehelix_palette(start=2, rot=0, dark=0.2, light=0.8, reverse=True, as_cmap=True)
        scatter_plot = sns.scatterplot(x='Year', y='Portfolio Value', hue='Z-Score', data=df_scatter, palette='magma', edgecolor=None, alpha=0.5, ax=ax2)
        scatter_plot.set(yscale="log")
        scatter_plot.set_title('Portfolio Value Over Time (All Simulations)')
        st.pyplot(fig2)

    with col3:
        fig3, ax3 = plt.subplots(figsize=(6,4))
        box_plot = sns.boxplot(x='Year', y='Portfolio Value', data=df_box, ax=ax3, color='#ff2e63')
        box_plot.set(yscale="log")
        box_plot.set_title('Box plot of Portfolio Values Per Year')
        st.pyplot(fig3)


def plot_simulation_results_plotly(results, initial_investment):
    col1, col2, col3 = st.columns(3)
        
    start_time = time.time()
    with st.spinner('Calculating plots...'):
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(calculate_histogram_data, results),
                executor.submit(calculate_scatter_data, results),
                executor.submit(calculate_box_data, results)
            ]

            for future in concurrent.futures.as_completed(futures):
                if future == futures[0]:
                    df_hist = future.result()
                    end_time = time.time()
                    print("The histogram took", end_time - start_time, "seconds to run")
                elif future == futures[1]:
                    df_scatter = future.result()
                    end_time = time.time()
                    print("The scatter took", end_time - start_time, "seconds to run")
                else:
                    df_box = future.result()
                    end_time = time.time()
                    print("The box plot took", end_time - start_time, "seconds to run")

    with col1:
        with st.spinner('Calculating Histogram of Final Portfolio Values...'):
            fig1 = go.Figure()
            fig1.add_trace(go.Histogram(x=df_hist["Final Portfolio Value"], nbinsx=100))
            fig1.update_layout(title_text='Histogram of Final Portfolio Values',
                    xaxis_title='Portfolio Value',
                    yaxis_title='Frequency',
                    xaxis_tickprefix="$",
                    bargap=0.1)
            st.plotly_chart(fig1)

    with col2:  
        with st.spinner('Calculating Scatter Plot of Simulated Portfolio Values...'):
            fig2 = go.Figure()

            fig2.add_trace(go.Scattergl(x=df_scatter["Year"], y=df_scatter["Portfolio Value"], mode='markers',
                                        marker=dict(color=df_scatter["Z-Score"], colorscale='inferno', size=5, opacity=0.5, showscale=True, colorbar=dict(title='Absolute Z-score'))))
            fig2.update_layout(title_text='Portfolio Value Over Time (All Simulations)',
                                    xaxis_title='Year',
                                    yaxis_title='Portfolio Value',
                                    yaxis_type="log",
                                    yaxis_tickprefix="$",
                                    showlegend=False)

            st.plotly_chart(fig2)
            
    with col3:
        with st.spinner('Calculating Scatter Box Plot of Simulated Portfolio Values...'):
            fig3 = go.Figure()
            fig3.add_trace(go.Box(y=df_box["Portfolio Value"], x=df_box["Year"]))
            fig3.update_layout(title_text='Box plot of Portfolio Values Per Year',
                    yaxis_title='Portfolio Value',
                    yaxis_type="log",
                    yaxis_tickprefix="$",
                    showlegend=False,
                    xaxis_title='Year')
            st.plotly_chart(fig3)
            
def plot_simulation_results_plotly_static(results, initial_investment):
    start_time = time.time()
    with st.spinner('Calculating plots...'):
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(calculate_histogram_data, results),
                executor.submit(calculate_scatter_data, results),
                executor.submit(calculate_box_data, results)
            ]

            for future in concurrent.futures.as_completed(futures):
                if future == futures[0]:
                    df_hist = future.result()
                    end_time = time.time()
                    print("The histogram took", end_time - start_time, "seconds to run")
                elif future == futures[1]:
                    df_scatter = future.result()
                    end_time = time.time()
                    print("The scatter took", end_time - start_time, "seconds to run")
                else:
                    df_box = future.result()
                    end_time = time.time()
                    print("The box plot took", end_time - start_time, "seconds to run")

    fig1 = go.Figure()
    fig1.add_trace(go.Histogram(x=df_hist["Final Portfolio Value"], nbinsx=100))
    fig1.update_layout(bargap=0.1)
    img1 = pio.to_image(fig1, format="png")
    st.image(img1, use_column_width=True, caption="Histogram")

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df_scatter["Year"], y=df_scatter["Portfolio Value"], mode='markers',
                                marker=dict(color=df_scatter["Z-Score"], colorscale='Magma', size=5, opacity=0.5)))
    fig2.update_yaxes(type="log")
    img2 = pio.to_image(fig2, format="png")
    st.image(img2, use_column_width=True, caption="Scatter Plot")

    fig3 = go.Figure()
    fig3.add_trace(go.Box(y=df_box["Portfolio Value"], x=df_box["Year"]))
    fig3.update_yaxes(type="log")
    img3 = pio.to_image(fig3, format="png")
    st.image(img3, use_column_width=True, caption="Box Plot")
    
def plot_simulation_results_v0(results):
    # Histogram of final portfolio values
    col1, col2, col3 = st.columns(3)
    
    with col1:
        with st.spinner('Calculating Histogram of Final Portfolio Values...'):
            start_time = time.time()

            # Histogram of final portfolio values
            final_values = [result.sum(axis=1).iloc[-1] for result in results]


            hist_fig = go.Figure(data=[go.Histogram(x=final_values, nbinsx=250, 
                                            marker=dict(
                                                color='blue',
                                                line=dict(width=1)
                                            ))])
            
            hist_fig.update_layout(title_text='Histogram of Final Portfolio Values',
                                xaxis_title='Portfolio Value',
                                yaxis_title='Frequency',
                                xaxis_tickprefix="$")
            end_time = time.time()
            print("The histogram took", end_time - start_time, "seconds to calculate")
            
            start_time = time.time()
            st.plotly_chart(hist_fig)

            
            end_time = time.time()
            print("The histogram took", end_time - start_time, "seconds to render")
  
    with col2:  
        with st.spinner('Calculating Scatter Plot of Simulated Portfolio Values...'):
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
                                yaxis_tickprefix="$",
                                xaxis_title='Year')
            end_time = time.time()
            print("The box plot took", end_time - start_time, "seconds to calculate")

            start_time = time.time()

            st.plotly_chart(box_fig)
            end_time = time.time()
            print("The box plot took", end_time - start_time, "seconds to render")


def plot_density_plots(results):
    N = len(results[0])

    # Index of the middle year
    mid_index = N // 2

    with st.spinner('Calculating Probability Density Plots...'):

            # Extract the portfolio values for years 1, N/2, and N            
        col1, col2, col3 = st.columns(3)
        
        with col1:
            _, _, fig1 = plot_probability_density_for_a_given_year_vIntegral("Year 1", [result.iloc[1].sum() for result in results])
            st.plotly_chart(fig1)
            
        with col2:
            _, _, fig2 = plot_probability_density_for_a_given_year_vIntegral(f'Year {N//2}', [result.iloc[mid_index].sum() for result in results])
            st.plotly_chart(fig2)
            
        with col3:
            _, _, fig3 = plot_probability_density_for_a_given_year_vIntegral(f'Year {N}', [result.iloc[-1].sum() for result in results])
            st.plotly_chart(fig3)
            
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

def plot_probability_density_for_a_given_year_vIntegral(year, values):
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

    colors = ["#FFD700", "#FFA500","#FF8C00","#FF4500","#FF0000"]

    results = []
    # Add sigma labels and calculate probability ranges
    for i in range(-5, 6):
        color = colors[abs(i)-1] if i != 0 else "LightBlue"

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
            results.append((i, i+1, probability_within_band, xticks[i+5], xticks[i+6]))
        elif i == 5:  # Edge case: Calculate for the upper band μ + 5σ < X
            probability_within_band = 1 - gaussian_kde_cdf(kde, mean_value + i*std_dev)
            results.append((i, np.nan, probability_within_band, xticks[i+5], np.nan))
        elif i == -5:  # Edge case: Calculate for the lower band X < μ - 5σ
            probability_within_band = gaussian_kde_cdf(kde, mean_value + i*std_dev)
            results.append((np.nan, i+1, probability_within_band, np.nan, xticks[i+6]))
            
    fig.update_layout(title=f'Probability Density for {year}',
                        xaxis_title='Portfolio Value',
                        yaxis_title='Density',
                        xaxis=dict(tickmode='array', tickvals=xticks, ticktext=xticklabels),
                        autosize=True,
                        showlegend=False)

    return results, xticks, fig


def calculate_returns(results, initial_investment, yearly_contribution):
    N = len(results[0])
    mid_index = N // 2
    years_to_plot = [1, mid_index, N-1]
    n_cores = multiprocessing.cpu_count()

    with st.spinner('Calculating Probability Density Plots...'):
        pdf_results = []
        with stqdm(total=len(years_to_plot)) as pbar:
            with concurrent.futures.ProcessPoolExecutor(max_workers=min(len(years_to_plot), n_cores)) as executor:
                futures = []
                for year in years_to_plot:
                    print(f"Calculating PDF for year {year}")
                    values = [result.iloc[year].sum() for result in results]
                    future = executor.submit(plot_probability_density_for_a_given_year_vIntegral, f"Year {year}", values)
                    futures.append(future)
                    
                for future in concurrent.futures.as_completed(futures):
                    pdf_results.append(future.result())
                    pbar.update(1)
    
    pdf_results.sort()  # Sort results by year
    
    with st.spinner('Plotting Probability Density Plots...'):
        col1, col2, col3 = st.columns(3)
        columns = [col1, col2, col3]

        for i, col in enumerate(columns):
            print(f"Plotting PDF for year {years_to_plot[i]}")
            probability_bands, portfolio_values_at_sigma, fig = pdf_results[i]
            col.plotly_chart(fig)
            returns_dict = calculate_returns_for_probability_bands(probability_bands, initial_investment, yearly_contribution, years_to_plot[i])
            col.write(returns_dict)
    

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


