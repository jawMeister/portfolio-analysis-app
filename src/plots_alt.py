
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import base64
from PIL import Image
from io import BytesIO
import plotly.io as pio
import plotly.express as go

import streamlit as st

import time

import multiprocessing
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from plots import calculate_histogram_data, calculate_scatter_data, calculate_box_data

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