import plotly.graph_objects as go
import datetime

def plot_indicators(df, ticker, timeframe):
    fig = go.Figure()

    fig.add_trace(go.Candlestick(x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close']))
                    
    fig.add_trace(go.Scatter(x=df.index, y=df['sma_20'], mode='lines', name='SMA 20', line=dict(color='palegoldenrod', dash='dash')))
    fig.add_trace(go.Scatter(x=df.index, y=df['ema_9'], mode='lines', name='EMA 9', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=df.index, y=df['ema_21'], mode='lines', name='EMA 21', line=dict(color='goldenrod')))
    fig.add_trace(go.Scatter(x=df.index, y=df['ema_55'], mode='lines', name='EMA 55', line=dict(color='lightseagreen')))
    fig.add_trace(go.Scatter(x=df.index, y=df['ema_200'], mode='lines', name='EMA 200', line=dict(color='white')))

    if timeframe == 'Weekly':
        df1 = df[df['ema_greater_sma']]
        df2 = df[df['ema_less_sma']]

        fig.add_trace(go.Scatter(x=df1.index, y=df1['sma_20'], mode='lines', line=dict(width=0), fill=None, showlegend=False))
        fig.add_trace(go.Scatter(x=df1.index, y=df1['ema_21'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='forestgreen', showlegend=False))

        fig.add_trace(go.Scatter(x=df2.index, y=df2['sma_20'], mode='lines', line=dict(width=0), fill=None, showlegend=False))
        fig.add_trace(go.Scatter(x=df2.index, y=df2['ema_21'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='darksalmon', showlegend=False))

    fig.update_layout(
        title=f"{ticker} {timeframe} Technical Indicators",
        yaxis_title="Price",
        legend_title="Indicator"
    )

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df.index, y=df['rsi'], mode='lines', name='RSI'))
    fig2.add_trace(go.Scatter(x=df.index, y=df['stoch'], mode='lines', name='Stochastic Oscillator'))

    fig2.update_layout(
        title=f"{ticker} {timeframe} RSI & Stochastic Oscillator",
        yaxis_title="Value",
        legend_title="Indicator"
    )

    x_axis_start = None
    if timeframe == 'Daily':
        x_axis_start = datetime.datetime.now() - datetime.timedelta(days=30)
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
        fig2.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])

    elif timeframe == 'Weekly':
        x_axis_start = datetime.datetime.now() - datetime.timedelta(days=365)

    elif timeframe == 'Monthly':
        x_axis_start = datetime.datetime.now() - datetime.timedelta(days=365*3)

    fig.update_xaxes(range=[x_axis_start, datetime.datetime.now()])
    fig2.update_xaxes(range=[x_axis_start, datetime.datetime.now()])
    #fig.update_yaxes(type="log")
    #fig2.update_yaxes(type="log")

    # Update y-axis based on the range of x-axis
    y_min = df.loc[(df.index >= x_axis_start) & (df.index <= datetime.datetime.now()), 'Low'].min()
    y_max = df.loc[(df.index >= x_axis_start) & (df.index <= datetime.datetime.now()), 'High'].max()
    
    # You might want to add some buffer to the min and max values for better plot aesthetics.
    y_buffer = (y_max - y_min) * 0.1  # 10% of the price range
    y_min = max(0, y_min - y_buffer)  # Prices can't go below 0
    y_max = y_max + y_buffer

    fig.update_yaxes(range=[y_min, y_max])
    
    return fig, fig2

"""         aliceblue, antiquewhite, aqua, aquamarine, azure,
            beige, bisque, black, blanchedalmond, blue,
            blueviolet, brown, burlywood, cadetblue,
            chartreuse, chocolate, coral, cornflowerblue,
            cornsilk, crimson, cyan, darkblue, darkcyan,
            darkgoldenrod, darkgray, darkgrey, darkgreen,
            darkkhaki, darkmagenta, darkolivegreen, darkorange,
            darkorchid, darkred, darksalmon, darkseagreen,
            darkslateblue, darkslategray, darkslategrey,
            darkturquoise, darkviolet, deeppink, deepskyblue,
            dimgray, dimgrey, dodgerblue, firebrick,
            floralwhite, forestgreen, fuchsia, gainsboro,
            ghostwhite, gold, goldenrod, gray, grey, green,
            greenyellow, honeydew, hotpink, indianred, indigo,
            ivory, khaki, lavender, lavenderblush, lawngreen,
            lemonchiffon, lightblue, lightcoral, lightcyan,
            lightgoldenrodyellow, lightgray, lightgrey,
            lightgreen, lightpink, lightsalmon, lightseagreen,
            lightskyblue, lightslategray, lightslategrey,
            lightsteelblue, lightyellow, lime, limegreen,
            linen, magenta, maroon, mediumaquamarine,
            mediumblue, mediumorchid, mediumpurple,
            mediumseagreen, mediumslateblue, mediumspringgreen,
            mediumturquoise, mediumvioletred, midnightblue,
            mintcream, mistyrose, moccasin, navajowhite, navy,
            oldlace, olive, olivedrab, orange, orangered,
            orchid, palegoldenrod, palegreen, paleturquoise,
            palevioletred, papayawhip, peachpuff, peru, pink,
            plum, powderblue, purple, red, rosybrown,
            royalblue, rebeccapurple, saddlebrown, salmon,
            sandybrown, seagreen, seashell, sienna, silver,
            skyblue, slateblue, slategray, slategrey, snow,
            springgreen, steelblue, tan, teal, thistle, tomato,
            turquoise, violet, wheat, white, whitesmoke,
            yellow, yellowgreen
"""