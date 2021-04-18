# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 22:26:16 2021

@author: kalan
"""
# import plotly.graph_objects as go


# fig = go.Figure(go.Indicator(
#     mode = "number+delta",
#     value = 400,
#     number = {'prefix': "$"},
#     delta = {'position': "top", 'reference': 320},
#     domain = {'x': [0, 1], 'y': [0, 1]}))

# fig.update_layout(paper_bgcolor = "lightgray")

# fig.show()



import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go

fig = go.Figure(go.Indicator(
    mode = "number+delta",
    value = 400,
    number = {'prefix': "$"},
    delta = {'position': "top", 'reference': 320},
    domain = {'x': [0, 1], 'y': [0, 1]}))

app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id="graph", figure=fig),
])

app.run_server(debug=True)