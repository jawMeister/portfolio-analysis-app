import plotly.graph_objects as go
import numpy as np
from dateutil.relativedelta import relativedelta


def calculate_sigma_color(sigma):
    colors = ["#FFD700", "#FFA500","#FF8C00","#FF4500","#FF0000"]
    
    if (np.isnan(sigma)):
        sigma = 5
        
    color = colors[abs(sigma)-1] if sigma != 0 else "LightBlue"
    return color

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
    
# This code plots a box plot of the portfolio value per year
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

def plot_backtest_simulation_w_sigma_levels(sigma_levels_by_year, actuals_portfolio_values, actuals_start_date):
    final_results_plot = go.Figure()
    
    # for every year in the predictions, plot the sigma levels as well as the actual portfolio value on a month over month basis
    for year in sigma_levels_by_year.keys():
        # x values need to have a data point per month, beginning with the first month past the start date
        x0 = actuals_start_date + relativedelta(months = ((year-1) * 12 + 1)) 
        x1 = x0 + relativedelta(months = 12) # each probability is for a 12 month period
        x_mid_point = x0 + relativedelta(months = 6) # for annotations
        
        # for every sigma level in the year (s.b., +/-5), plot the lower and upper sigma levels
        for lower_sigma, upper_sigma, probability, lower_edge_value, upper_edge_value in sigma_levels_by_year[year]:
            
            #print(f"year {year}: lower_sigma: {lower_sigma}, upper_sigma: {upper_sigma}, probability: {probability}, lower_edge_value: {lower_edge_value}, upper_edge_value: {upper_edge_value}")
            #year 7: lower_sigma: -2, upper_sigma: -1, probability: 0.14786165197479317, lower_edge_value: 1040792.7317910342, upper_edge_value: 1339530.4912323258
            #year 7: lower_sigma: -1, upper_sigma: 0, probability: 0.3828323364399347, lower_edge_value: 1339530.4912323258, upper_edge_value: 1638268.2506736175
            #year 7: lower_sigma: 0, upper_sigma: 1, probability: 0.31700283809084806, lower_edge_value: 1638268.2506736175, upper_edge_value: 1937006.0101149091
            
            # plot the lower sigma level which has lower_edge_value as the y value and label it with lower_sigma
            # plot the upper sigma level which has upper_edge_value as the y value and label it with upper_sigma
            
            lower_edge_value = lower_edge_value//1
            upper_edge_value = upper_edge_value//1
            y_mid_point = (lower_edge_value + upper_edge_value) / 2
    
            # lower sigma
            lower_edge_color = calculate_sigma_color(lower_sigma)
            final_results_plot.add_trace(go.Scatter(
                x=[x0, x1],
                y=[lower_edge_value, lower_edge_value],
                mode='lines',
                line=dict(color=lower_edge_color, width=1, dash='dashdot'),
                opacity=0.5,
                name=f'{lower_sigma} sigma, ${lower_edge_value:,.0f}',
                hovertemplate='%{y}',
                hoverinfo='name+y',
                showlegend=False
            ))
            """ good idea to add label, just too busy of a chart
            final_results_plot.add_annotation(x=x0, 
                        y=np.log10(lower_edge_value), 
                        text=f'{lower_sigma} sigma, ${lower_edge_value:,.0f}', 
                        showarrow=False, 
                        font=dict(color=lower_edge_color),
                        bgcolor="#0E1117",
                        xanchor="left",
                        yanchor="middle")
            """
                # add an annotation between the two lines with the probability
            final_results_plot.add_annotation(x=x_mid_point, 
                        y=np.log10(y_mid_point), 
                        text=f'{probability:.0%}', 
                        showarrow=False, 
                        font=dict(color='orange'),
                        bgcolor="#0E1117")
            
            # upper sigma
            upper_edge_color = calculate_sigma_color(upper_sigma)
            final_results_plot.add_trace(go.Scatter(
                x=[x0, x1],
                y=[upper_edge_value, upper_edge_value],
                mode='lines',
                line=dict(color=upper_edge_color, width=1, dash='dashdot'),
                opacity=0.5,
                name=f'{upper_sigma} sigma, ${upper_edge_value:,.0f}',
                hovertemplate='%{y}',
                hoverinfo='name+y',
                showlegend=False
            ))
            """ good idea to add label, just too busy of a chart
            final_results_plot.add_annotation(x=x0, 
                        y=np.log10(upper_edge_value), 
                        text=f'{upper_sigma} sigma, ${upper_edge_value:,.0f}', 
                        showarrow=False, 
                        font=dict(color=lower_edge_color),
                        bgcolor="#0E1117",
                        xanchor="left",
                        yanchor="middle")
            """
            
    # plot actuals_portfolio_values on top of sigma lines
    monthly_data = actuals_portfolio_values.copy()
    # filter out any rows with NaN as crypto trades on weekends, as may through off the mean
    monthly_data.dropna(how='any')
    # Resample the data to monthly frequency and calculate the mean value of each stock
    monthly_data = actuals_portfolio_values.resample('M').mean()
    # filter down to just the total column
    monthly_data = monthly_data[['Total']]
    #print(f"monthly_data:\n{monthly_data.describe()}")
    #print(f"monthly_data:\n{monthly_data.tail()}")

    final_results_plot.add_trace(go.Scatter(x=monthly_data.index, y=monthly_data['Total'], mode='lines'))
    final_results_plot.update_layout(title='Monthly Total Value (Weighted Portfolio)', 
                                        xaxis_title='Year', 
                                        yaxis_title='Portfolio Value',
                                        yaxis=dict(tickformat="$,.0f")) 
    final_results_plot.update_yaxes(type='log', automargin=True)    
    
    return final_results_plot