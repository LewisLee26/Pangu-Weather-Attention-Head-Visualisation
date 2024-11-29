import dash
from dash import dcc
from dash import html
import dash.dependencies
import plotly.graph_objs as go
import numpy as np

# Sample data: Replace these with your actual 12x12 numpy arrays
arr1 = np.random.rand(12,12)
arr2 = np.random.rand(12,12)

app = dash.Dash(__name__)

# Common layout for both figures
common_layout = go.Layout(
    hovermode='closest',
    xaxis=dict(
        range=[-0.5, 11.5],
        showgrid=False,
        zeroline=False,
        constrain='domain'
    ),
    yaxis=dict(
        range=[-0.5, 11.5],
        showgrid=False,
        zeroline=False,
        scaleratio=1,
        autorange='reversed',  # Place origin at the top-left corner
        constrain='domain'
    ),
    margin=dict(l=40, b=40, t=40, r=40),
    height=400,
    width=400
)

# Figure for the first image
fig1 = go.Figure(
    data=[
        go.Heatmap(
            z=arr1,
            colorscale='Viridis',
            showscale=False,
            hoverinfo='x+y+z',
            x=np.arange(12),
            y=np.arange(12)
        ),
        go.Scatter(
            x=[],  # Empty initially; will be updated on hover
            y=[],
            mode='markers',
            marker=dict(
                color='red',
                size=12,
                symbol='square',
                opacity=0.7
            ),
            showlegend=False,
            hoverinfo='skip'  # Skip hover info for the marker
        )
    ],
    layout=common_layout.update(title='Image 1')
)

# Figure for the second image
fig2 = go.Figure(
    data=[
        go.Heatmap(
            z=arr2,
            colorscale='Viridis',
            showscale=False,
            hoverinfo='x+y+z',
            x=np.arange(12),
            y=np.arange(12)
        ),
        go.Scatter(
            x=[],
            y=[],
            mode='markers',
            marker=dict(
                color='red',
                size=12,
                symbol='square',
                opacity=0.7
            ),
            showlegend=False,
            hoverinfo='skip'
        )
    ],
    layout=common_layout.update(title='Image 2')
)

# Layout of the Dash app with two graphs
app.layout = html.Div([
    html.Div([
        dcc.Graph(id='graph1', figure=fig1)
    ], style={'display': 'inline-block'}),
    html.Div([
        dcc.Graph(id='graph2', figure=fig2)
    ], style={'display': 'inline-block'})
])

# Callback to update Image 2 when hovering over Image 1
@app.callback(
    dash.dependencies.Output('graph2', 'figure'),
    [dash.dependencies.Input('graph1', 'hoverData')],
    [dash.dependencies.State('graph2', 'figure')]
)
def update_graph2(hoverData, figure):
    if hoverData is None:
        # Clear the marker if not hovering
        figure['data'][1]['x'] = []
        figure['data'][1]['y'] = []
    else:
        # Get the x and y indices of the hovered cell
        point = hoverData['points'][0]
        x = point['x']
        y = point['y']
        # Update the marker position
        figure['data'][1]['x'] = [x]
        figure['data'][1]['y'] = [y]
    return figure

# Callback to update Image 1 when hovering over Image 2
@app.callback(
    dash.dependencies.Output('graph1', 'figure'),
    [dash.dependencies.Input('graph2', 'hoverData')],
    [dash.dependencies.State('graph1', 'figure')]
)
def update_graph1(hoverData, figure):
    if hoverData is None:
        figure['data'][1]['x'] = []
        figure['data'][1]['y'] = []
    else:
        point = hoverData['points'][0]
        x = point['x']
        y = point['y']
        figure['data'][1]['x'] = [x]
        figure['data'][1]['y'] = [y]
    return figure

if __name__ == '__main__':
    app.run_server(debug=True)

