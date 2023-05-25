import plotly.graph_objects as go
import numpy as np


def plot_histogram_data(df_hist):
    fig1 = go.Figure()
    fig1.add_trace(go.Histogram(x=df_hist["Final Portfolio Value"], nbinsx=100))
    fig1.update_layout(title_text='Histogram of Final Portfolio Values',
            xaxis_title='Portfolio Value',
            yaxis_title='Frequency',
            xaxis_tickprefix="$",
            bargap=0.1)
    return fig1
            
def plot_scatter_data(df_scatter):
    fig2 = go.Figure()

    fig2.add_trace(go.Scattergl(x=df_scatter["Year"], y=df_scatter["Portfolio Value"], mode='markers',
                                marker=dict(color=df_scatter["Z-Score"], colorscale='inferno', size=5, opacity=0.5, showscale=True, colorbar=dict(title='Absolute Z-score'))))
    fig2.update_layout(title_text='Portfolio Value Over Time (All Simulations)',
                            xaxis_title='Year',
                            yaxis_title='Portfolio Value',
                            yaxis_type="log",
                            yaxis_tickprefix="$",
                            showlegend=False)

    return fig2
    
def plot_box_data(df_box):
    fig3 = go.Figure()
    fig3.add_trace(go.Box(y=df_box["Portfolio Value"], x=df_box["Year"]))
    fig3.update_layout(title_text='Box plot of Portfolio Values Per Year',
            yaxis_title='Portfolio Value',
            yaxis_type="log",
            yaxis_tickprefix="$",
            showlegend=False,
            xaxis_title='Year')
    return fig3