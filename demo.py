import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_table

import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.figure_factory as ff


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
    html.H1(children='Hello Dash'),

    html.Div(children='''
        Dash: A web application framework for Python.
    '''),
    dcc.Textarea(
        id='mytext',
        placeholder='Enter a value...',
        value='This is a TextArea component',
        style={'width': '100%'}
        ),
    html.H1(id='op'),
])

@app.callback(
    Output('op','children'),
    [Input('mytext','value')]
)
def print_op(mytext_value):
    return 'The text in the above field is : {}'.format(mytext_value)

if __name__ == '__main__':
    app.run_server(debug=True)
