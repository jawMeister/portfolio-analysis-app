import plotly.graph_objects as go

def plot_line_chart(df, column_list, title):
    fig = go.Figure()

    for column in column_list:
        fig.add_trace(go.Scatter(x=df.index, y=df[column], mode='lines', name=column))

    fig.update_layout(title=title, yaxis_title='Value ($)')
    return fig