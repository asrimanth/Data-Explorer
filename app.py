import base64
import datetime
import io

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

import linear_regression as lin_reg


external_stylesheets = ['assets/css_app.css', dbc.themes.BOOTSTRAP]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div([
    dcc.Tabs(id="tabs", children=[
        dcc.Tab(label='Data Input and Preprocessing', children=[
            # Upload area for dataset as an input. Covers the top dashed space.
            dcc.Upload(
                id='upload_dataset_button',
                children=html.Div([
                    'Upload your dataset as a .csv of .xls ',
                    html.A('here')
                ]),
                style={
                    'width': '99%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'
                },
                # Allow multiple files to be uploaded
                multiple=True
            ),
            html.Hr(),
            html.Div(id='alert_display_main_div', style={'textAlign': 'center'}),
            html.Hr(),
            # A div to display the head of the dataset to the user.
            html.Div(id='original_dataset_display_div', children=[
                dash_table.DataTable(
                    id='original_dataset_table',
                    style_header={
                        'backgroundColor': 'white',
                        'fontWeight': 'bold',
                        'fontFamily': 'Ubuntu',
                        'whiteSpace': 'no-wrap',
                        'overflow': 'hidden',
                        'textOverflow': 'ellipsis',
                        'maxWidth': 500,
                    },
                    style_cell={'textAlign': 'left', 'fontFamily': 'Ubuntu'},
                    # style_cell_conditional=[
                    #     {
                    #         'if': {'row_index': 'odd'},
                    #         'backgroundColor': 'rgb(248, 248, 248)'
                    #     }
                    # ],
                    style_table={
                        'maxHeight': '250px',
                        'overflowY': 'scroll'
                    },
                    editable=True,
                    # filtering=True,
                    # sorting=True,
                    # sorting_type="multi",
                    row_selectable="multi",
                    row_deletable=True,
                ),
            ]),

            html.Div(id='alert_display_save_div', style={
                     'width': '49%', 'display': 'inline-block', 'float': 'right'}),

            html.Div(id='save_changes_button_div', children=[
                html.H6('Upload your dataset, make your changes in the table and \
                 save the changes by clicking the button below.'),
                html.Button("Save changes in dataset", id='save_changes_button',
                            n_clicks=0),
            ], style={'width': '49%', 'display': 'inline-block'}),

            # A hidden div to store a dataset in json format.        if(jsonified_data)
            html.Div(id='original_dataset_div', style={'display': 'none'}),

            html.Div(id='processed_dataset_div', style={'display': 'none'}),

        ]),
        dcc.Tab(label='Data Cleaning', children=[
            html.Div(id='processed_dataset_display_div', style={'textAlign': 'center'}),
            html.Hr(),
            html.Div([
                dbc.CardDeck(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader("Header"),
                                dbc.CardBody(
                                    [
                                        dcc.RadioItems(
                                            options=[
                                                {'label': 'New York City', 'value': 'NYC'},
                                                {'label': 'Montréal', 'value': 'MTL'},
                                                {'label': 'San Francisco', 'value': 'SF'}
                                            ],
                                            value='MTL',
                                        )
                                    ]
                                ),
                            ],
                            outline=True,
                            color="primary",
                        ),
                        dbc.Card(
                            [
                                dbc.CardBody(
                                    [
                                        dcc.Dropdown(
                                            options=[
                                                {'label': 'New York City', 'value': 'NYC'},
                                                {'label': 'Montréal', 'value': 'MTL'},
                                                {'label': 'San Francisco', 'value': 'SF'}
                                            ],
                                            value='MTL',
                                            style={'textAlign': 'left'},
                                            multi=True
                                        )
                                    ]
                                ),
                                dbc.CardFooter("Footer"),
                            ],
                            outline=True,
                            color="primary",
                        ),
                    ]
                )
            ], style={'margin': '10px', 'textAlign': 'center'}),
        ]),
        dcc.Tab(label='Exploratory Data Analysis', children=[
            # Two Dropdown boxes to take input regarding the X and Y axes for the barplot
            html.Div([
                html.Div([
                    dcc.Dropdown(id='x_axis_dropdown_bar_plot',
                                 placeholder='Select the column to be plotted on X axis'),
                ],
                    style={'width': '49%', 'display': 'inline-block'}),
                html.Div([
                    dcc.Dropdown(id='y_axis_dropdown_bar_plot',
                                 placeholder='Select the column to be plotted on Y axis'),
                ],
                    style={'width': '49%', 'display': 'inline-block', 'float': 'right'})
            ], style={'margin': '10px'}),

            # The dash core component for a Bar Graph
            dcc.Graph(id='bar_graph'),

            # Two Dropdown boxes to take input regarding the X and Y axes for the Scatter Plot
            html.Div([
                html.Div([
                    dcc.Dropdown(id='x_axis_dropdown_scatter_plot',
                                 placeholder='Select the column to be plotted on X axis'),
                ],
                    style={'width': '49%', 'display': 'inline-block'}),
                html.Div([
                    dcc.Dropdown(id='y_axis_dropdown_scatter_plot',
                                 placeholder='Select the column to be plotted on Y axis'),
                ],
                    style={'width': '49%', 'display': 'inline-block', 'float': 'right'})
            ], style={'margin': '10px'}),

            # The dash core component for a Scatter Plot
            dcc.Graph(id='scatter_plot'),

            dcc.Graph(id='correlation_heatmap'
                      ),


            html.Div([
                html.Div([
                    dcc.Dropdown(id='x_axis_dropdown_box_plot',
                                 placeholder='Select the column to be plotted on X axis'),
                ],
                    style={'width': '49%', 'display': 'inline-block'}),
                html.Div([
                    dcc.Dropdown(id='y_axis_dropdown_box_plot',
                                 placeholder='Select the column to be plotted on Y axis'),
                ],
                    style={'width': '49%', 'display': 'inline-block', 'float': 'right'})
            ], style={'margin': '10px'}),

            dcc.Graph(id='box_plot'),

        ]),
        dcc.Tab(label='Machine Learning', children=[
            # Dropdown to choose main and sub algorithms
            html.Div([
                html.Div([
                    dcc.Dropdown(id='main_algorithm_dropdown',
                                 placeholder='Choose the algorithm',
                                 ),
                ],
                    style={'width': '49%', 'display': 'inline-block'}),
                html.Div([
                    dcc.Dropdown(id='sub_algorithm_dropdown',
                                 placeholder='Choose the sub algorithm'),
                ],
                    style={'width': '49%', 'display': 'inline-block', 'float': 'right'})
            ], style={'margin': '10px'}),
            # A Dropdown to take the list of feature names for Machine Learning
            html.Div([
                html.Div([
                    dcc.Dropdown(
                        id='feature_select_dropdown',
                        multi=True,
                        placeholder='Select the features for the model.'
                    ),
                ],
                    style={'width': '49%', 'display': 'inline-block'}),
                html.Div([
                    dcc.Dropdown(id='target_select_dropdown',
                                 placeholder='Select the target variable'),
                ],
                    style={'width': '49%', 'display': 'inline-block', 'float': 'right'})
            ], style={'margin': '10px'}),

            # A div to display the output for Machine Learning
            html.Div(id='lin_reg_output'),
        ]),
    ], colors={
        "border": "white",
        "primary": "#119DFF",
        "background": "#81D4FA"
    }),
])


# A function which parses the raw data.
def display_alert(alert_message, color):
    return html.Div([dbc.Alert(alert_message, color=color),
                     ], style={'width': '75%', 'display': 'inline-block'}),


def display_dataset(dataset):

    table = dbc.Table.from_dataframe(dataset, striped=True, bordered=True, hover=True)
    return table
    # return html.Div([
    #     dash_table.DataTable(
    #         columns=[{"name": i, "id": i} for i in dataset.columns],
    #         data=dataset.to_dict("rows"),
    #     )
    # ])


def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    dataset = pd.DataFrame()
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            dataset = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            dataset = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return dataset
    return dataset
    return[[{'name': i, 'id': i, 'deletable': True} for i in dataset.columns],
           dataset.to_dict('rows')]

# A function to process the raw data and
# To give the output to return a dash table with the original dataset.


@app.callback([Output('original_dataset_table', 'columns'),
               Output('original_dataset_table', 'data'),
               Output('alert_display_main_div', 'children')],
              [Input('upload_dataset_button', 'contents')],
              [State('upload_dataset_button', 'filename'),
               State('upload_dataset_button', 'last_modified')])
def update_output(list_of_contents, list_of_names, list_of_dates):
    dataset = pd.DataFrame()
    col_list = list(dataset.columns)
    data = dataset.to_dict('rows')
    if list_of_contents is not None:
        try:
            dataset = pd.DataFrame()
            filename = list_of_names[-1]
            content_type, content_string = ''.join(list_of_contents[-1]).split(',')
            decoded = base64.b64decode(content_string)
            if 'csv' in str(filename):
                # Assume that the user uploaded a CSV file
                dataset = pd.read_csv(
                    io.StringIO(decoded.decode('utf-8')))
            elif 'xls' in str(filename):
                # Assume that the user uploaded an excel file
                dataset = pd.read_excel(io.BytesIO(decoded))
            else:
                file_ext = filename.split('.')[-1]
                return [
                    [{'name': i, 'id': i, 'deletable': True} for i in col_list], data,
                    display_alert('.{} file is not supported.'.format(file_ext), color='danger')
                ]
            col_list = list(dataset.columns)
            if('Unnamed: 0' in col_list):
                dataset.drop(columns=['Unnamed: 0'], axis=1, inplace=True)
                col_list.remove('Unnamed: 0')

            data = dataset.to_dict('rows')
            return [
                [{'name': i, 'id': i, 'deletable': True} for i in col_list], data,
                html.Div([dbc.Alert(
                    [
                        html.H6("Dataset upload sucessful.", className="alert-heading"),
                        html.P(
                            "Filename: {}".format(filename),
                            style={'width': '49%', 'display': 'inline-block'}
                        ),
                        html.P(
                            "Last modified : {}".format(
                                datetime.datetime.fromtimestamp(list_of_dates[-1])),
                            style={'width': '49%', 'display': 'inline-block', 'float': 'right'}
                        ),
                        html.P(
                            "Number of rows : {}".format(dataset.shape[0]),
                            style={'width': '49%', 'display': 'inline-block'}
                        ),
                        html.P(
                            "Number of columns : {}".format(dataset.shape[1]),
                            style={'width': '49%', 'display': 'inline-block', 'float': 'right'}
                        )
                    ]
                )], style={'width': '75%', 'display': 'inline-block'})

            ]

        except Exception as e:
            print("Hi")
    return [
        [{'name': i, 'id': i, 'deletable': True} for i in col_list], data,
        display_alert('Upload a dataset to get started.', color='primary')
    ]


# A function to parse and store the original dataset as a json string in a hidden div.


@app.callback(Output('original_dataset_div', 'children'),
              [Input('upload_dataset_button', 'contents'),
               Input('upload_dataset_button', 'filename')])
def store_original_dataset(contents, filename):
    try:
        dataset = pd.DataFrame()
        content_type, content_string = 'original_dataset_display_div'.join(contents).split(',')
        decoded = base64.b64decode(content_string)
        if 'csv' in str(filename):
            # Assume that the user uploaded a CSV file
            dataset = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in str(filename):
            # Assume that the user uploaded an excel file
            dataset = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)

    if(dataset is not None):
        col_list = list(dataset.columns)
        if('Unnamed: 0' in col_list):
            dataset.drop(columns=['Unnamed: 0'], axis=1, inplace=True)
            col_list.remove('Unnamed: 0')
        return dataset.to_json(date_format='iso', orient='split')


@app.callback(
    [Output('alert_display_save_div', "children"),
     Output('processed_dataset_div', "children")],
    [Input('save_changes_button', 'n_clicks')],
    [State('original_dataset_table', 'derived_virtual_data')])
def store_modified_dataset(n_clicks, data_div_list):
    dataset = pd.DataFrame()
    col_list = list(dataset.columns)
    print(col_list)
    try:
        if(n_clicks == 0 and data_div_list is None):
            return[display_alert('Save your changes by clicking the button.', color='primary'),
                   dataset.to_json(date_format='iso', orient='split')
                   ]
        elif(len(data_div_list) == 0 and n_clicks > 0):
            return [display_alert('No dataset uploaded! Upload a dataset and then save it.', color='danger'),
                    dataset.to_json(date_format='iso', orient='split')]
        elif(n_clicks is not None and data_div_list is not None and len(data_div_list) != 0):
            dataset = pd.DataFrame(data_div_list)
            return [display_alert('Save success!', color='success'),
                    dataset.to_json(date_format='iso', orient='split')]
    except Exception as e:
        print(e)
    return[display_alert('Upload a dataset to get started.', color='primary'),
           dataset.to_json(date_format='iso', orient='split')
           ]


# Displaying modified dataset.


@app.callback(Output(component_id='processed_dataset_display_div', component_property='children'),
              [Input('original_dataset_div', 'children'),
               Input('processed_dataset_div', 'children')])
def modified_dataset_display(jsonified_data, processed_jsonified_data):
    if('[]' in str(processed_jsonified_data) and '[]' in str(jsonified_data) or processed_jsonified_data is None):
        return display_alert('Upload a dataset to get started.', color='primary')
    elif('[]' in str(processed_jsonified_data)):
        dataset = pd.read_json(jsonified_data, orient='split')
        return html.Div([dbc.Alert('Taking the original(unmodified) dataset.', color="secondary", dismissable=True, is_open=True),
                         display_dataset(dataset.head())
                         ], style={'width': '75%', 'display': 'inline-block'}),
    else:
        dataset = pd.read_json(processed_jsonified_data, orient='split')
        return html.Div([dbc.Alert('Taking the last saved dataset.', color="info", dismissable=True, is_open=True),
                         display_dataset(dataset.head())
                         ], style={'width': '75%', 'display': 'inline-block'}),


def list_of_columns_dropdown(column_names_dropdown):
    return [
        {'label': i, 'value': i} for i in column_names_dropdown
    ]

# A function which returns numerical and categorical column names seperately.


def sep_cat_and_num(jsonified_data):

    dataset = pd.read_json(jsonified_data, orient='split')
    column_names = list(dataset.columns)
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numerical_column_names = list(dataset.select_dtypes(include=numerics).columns)
    categorical_column_names = list(set(column_names) - set(numerical_column_names))
    # categorical_column_names = []
    # for column_name in tbd_categorical_column_names:
    #     if(len(dataset[column_name].unique()) <= 30):
    #         categorical_column_names.append(column_name)
    return column_names, numerical_column_names, categorical_column_names


# Populate the dropdowns for the Bar Graph


@app.callback([Output(component_id='x_axis_dropdown_bar_plot', component_property='options'),
               Output(component_id='y_axis_dropdown_bar_plot', component_property='options')],
              [Input('original_dataset_div', 'children')])
def update_bar_graph_dropdowns(jsonified_data):

    column_names, numerical_column_names, categorical_column_names = sep_cat_and_num(jsonified_data)
    if(numerical_column_names is not None and categorical_column_names is not None):
        return [list_of_columns_dropdown(categorical_column_names),
                list_of_columns_dropdown(column_names)]


# A Bar Graph which plots the data from the dataset stored in hidden div and the column names from the two dropdowns.


@app.callback(
    Output(component_id='bar_graph', component_property='figure'),
    [Input(component_id='x_axis_dropdown_bar_plot', component_property='value'),
     Input(component_id='y_axis_dropdown_bar_plot', component_property='value'),
     Input('original_dataset_div', 'children')])
def update_bar_graph(selected_column_x, selected_column_y, jsonified_data):
    dataset = pd.read_json(jsonified_data, orient='split')
    if('[]' in jsonified_data):
        return{
            'layout': {
                'title': 'Please Upload a dataset to get started.'
            }
        }
    elif(selected_column_x is None or selected_column_y is None):
        return{
            'layout': {
                'title': 'Please select two unique columns from the above dropdown boxes.'
            }
        }
    elif(selected_column_x == selected_column_y):
        return{
            'layout': {
                'title': 'Bar Plot cannot be plotted between {} and {}'.format(selected_column_x, selected_column_y)
            }
        }
    else:

        return {
            'data': [
                {'x': dataset[selected_column_x], 'y': dataset[selected_column_y],
                 'type': 'bar', 'name': 'My Plot'},
            ],
            'layout': {
                'title': 'Bar Plot between {} and {}'.format(selected_column_x, selected_column_y),
                'hovermode': 'closest'
            }
        }


# Populate the dropdowns for the Scatter Plot.


@app.callback([Output(component_id='x_axis_dropdown_scatter_plot', component_property='options'),
               Output(component_id='y_axis_dropdown_scatter_plot', component_property='options')],
              [Input('original_dataset_div', 'children')])
def update_bar_graph_dropdowns(jsonified_data):

    column_names, numerical_column_names, categorical_column_names = sep_cat_and_num(jsonified_data)
    return [list_of_columns_dropdown(numerical_column_names), list_of_columns_dropdown(numerical_column_names)]


@app.callback(
    Output('scatter_plot', 'figure'),
    [Input(component_id='x_axis_dropdown_scatter_plot', component_property='value'),
     Input(component_id='y_axis_dropdown_scatter_plot', component_property='value'),
     Input('original_dataset_div', 'children')])
def update_scatter_plot(selected_column_x, selected_column_y, jsonified_data):

    if('[]' in jsonified_data):
        return{
            'layout': go.Layout(
                title='Please Upload a dataset to get started.'
            )
        }
    elif(selected_column_x is None or selected_column_y is None):
        return{
            'layout': go.Layout(
                title='Please select two unique columns from the above dropdown boxes.'
            )
        }
    elif(selected_column_x == selected_column_y):
        return{
            'layout': go.Layout(
                title='Scatter plot cannot be plotted between {} and {}'.format(
                    selected_column_x, selected_column_y)
            )
        }
    dataset = pd.read_json(jsonified_data, orient='split')
    filtered_dataset = dataset[[selected_column_x, selected_column_y]]

    traces = []
    for i in filtered_dataset:
        traces.append(go.Scatter(
            x=filtered_dataset[selected_column_x],
            y=filtered_dataset[selected_column_y],
            mode='markers',
            opacity=0.7,
            marker={
                'size': 10,
                'line': {'width': 0.5, 'color': 'white'}
            },
            name=i
        ))
    return {
        'data': traces,
        'layout': go.Layout(
            xaxis={'type': 'linear', 'title': selected_column_x},
            yaxis={'title': selected_column_y},
            hovermode='closest',
            showlegend=False
        )
    }


@app.callback(
    Output('correlation_heatmap', 'figure'),
    [Input(component_id='x_axis_dropdown_scatter_plot', component_property='value'),
     Input(component_id='y_axis_dropdown_scatter_plot', component_property='value'),
     Input('original_dataset_div', 'children')])
def update_correlation_heatmap(selected_column_x, selected_column_y, jsonified_data):

    if('[]' in jsonified_data):
        return{
            'layout': go.Layout(
                title='Please Upload a dataset to get started.'
            )
        }
    else:
        dataset = pd.read_json(jsonified_data, orient='split')
        column_names, numerical_column_names, categorical_column_names = sep_cat_and_num(
            jsonified_data)
        correlation_dataset = dataset.corr().round(3)
        correlations = list(correlation_dataset.values)
        data_list = []
        correlation_matrix = [list(correlations[i]) for i in range(len(correlations))]
        x_axis_labels = numerical_column_names
        y_axis_labels = numerical_column_names

        fig = ff.create_annotated_heatmap(
            z=correlation_matrix, x=x_axis_labels, y=y_axis_labels, annotation_text=correlation_matrix, zauto=False, zmin=-1, zmax=1)

        fig.layout.title = 'Correlation Heatmap'
        fig.layout.margin.update({
            "l": 100,
            "r": 20,
            "b": 10,
            "t": 60,
            "pad": 4
        })

        return fig


@app.callback([Output(component_id='x_axis_dropdown_box_plot', component_property='options'),
               Output(component_id='y_axis_dropdown_box_plot', component_property='options')],
              [Input('original_dataset_div', 'children')])
def update_box_plot_dropdowns(jsonified_data):

    column_names, numerical_column_names, categorical_column_names = sep_cat_and_num(jsonified_data)
    return [list_of_columns_dropdown(categorical_column_names), list_of_columns_dropdown(numerical_column_names)]


@app.callback(
    Output('box_plot', 'figure'),
    [Input(component_id='x_axis_dropdown_box_plot', component_property='value'),
     Input(component_id='y_axis_dropdown_box_plot', component_property='value'),
     Input('original_dataset_div', 'children')])
def update_box_plot(selected_column_x, selected_column_y, jsonified_data):

    if('[]' in jsonified_data):
        return{
            'layout': go.Layout(
                title='Please Upload a dataset to get started.'
            )
        }
    elif(selected_column_x is None or selected_column_y is None):
        return{
            'layout': go.Layout(
                title='Please select two unique columns from the above dropdown boxes.'
            )
        }
    elif(selected_column_x == selected_column_y):
        return{
            'layout': go.Layout(
                title='Scatter plot cannot be plotted between {} and {}'.format(
                    selected_column_x, selected_column_y)
            )
        }

    else:
        colors = ['rgb(239, 83, 80)', 'rgb(129, 199, 132)', 'rgb(100, 181, 246)', 'rgb(236, 64, 122)', 'rgb(171, 71, 188)',
                  'rgb(149, 117, 205)', 'rgb(92, 107, 192)', 'rgb(41, 182, 246)', 'rgb(38, 198, 218)', 'rgb(38, 166, 154)',
                  'rgb(156, 204, 101)', 'rgb(212, 225, 87)', 'rgb(255, 171, 0)', 'rgb(255, 202, 40)', 'rgb(255, 167, 38)',
                  'rgb(255, 87, 34)', 'rgb(141, 110, 99)', 'rgb(120, 144, 156)', 'rgb(136, 14, 79)', 'rgb(142, 36, 170)',
                  'rgb(0, 121, 107)', 'rgb(41, 121, 255)', 'rgb(26, 35, 126)', 'rgb(69, 39, 160)', 'rgb(30, 136, 229)',
                  'rgb(0, 145, 234)', 'rgb(46, 125, 50)', 'rgb(85, 139, 47)', 'rgb(255, 214, 0)', 'rgb(97, 97, 97)']
        dataset = pd.read_json(jsonified_data, orient='split')
        categories_list = dataset[selected_column_x].unique()
        traces = []
        j = 0
        for i in range(len(categories_list)):
            subset = dataset.loc[dataset[selected_column_x] == categories_list[i]]
            traces.append(go.Box(
                y=subset[selected_column_y],
                name=categories_list[i],
                boxpoints='suspectedoutliers',
                marker=dict(
                    color='rgb(0, 0, 0)',
                    outliercolor='rgba(0, 0, 0)',
                    line=dict(
                        outliercolor='rgba(0, 0, 0)',
                        outlierwidth=1)),

                line=dict(
                    color=colors[j])
            ))
            j = j+1
            if(j >= 30):
                j = 0

        return {
            'data': traces,
            'layout': go.Layout(
                hovermode='closest',
                title='Box plot of {} for classes of {}'.format(
                    selected_column_y, selected_column_x),
                showlegend=True,
            )
        }


@app.callback(Output(component_id='main_algorithm_dropdown', component_property='options'),
              [Input('original_dataset_div', 'children')])
def main_algorithm_select(jsonified_data):
    column_names, numerical_column_names, categorical_column_names = sep_cat_and_num(jsonified_data)
    if(len(column_names) > 0):
        return list_of_columns_dropdown(['Linear Regression', 'SVM'])
    else:
        return []  # list_of_columns_dropdown(['Upload a dataset to choose an algorithm'])


@app.callback(Output(component_id='sub_algorithm_dropdown', component_property='options'),
              [Input('original_dataset_div', 'children'),
               Input(component_id='main_algorithm_dropdown', component_property='value')])
def sub_algorithm_select(jsonified_data, selected_main_algorithm):
    column_names, numerical_column_names, categorical_column_names = sep_cat_and_num(jsonified_data)
    if(selected_main_algorithm is not None):
        print(selected_main_algorithm)
        if(selected_main_algorithm == 'linear_regression' or selected_main_algorithm == 'Linear Regression'):
            items = [{'label': 'Ordinary Least Squares(OLS) in sklearn', 'value': 'OLS_sklearn'},
                     {'label': 'Ridge Regression',  'value': 'ridge_regression'},
                     {'label': 'LASSO Regression', 'value': 'lasso_regression'},
                     {'label': 'Ordinary Least Squares(OLS) in statsmodels', 'value': 'OLS_statsmodels'}]

            return items
    else:
        return []


@app.callback(Output(component_id='feature_select_dropdown', component_property='options'),
              [Input('original_dataset_div', 'children')])
def feature_select(jsonified_data):
    column_names, numerical_column_names, categorical_column_names = sep_cat_and_num(jsonified_data)
    return list_of_columns_dropdown(column_names)

@app.callback(Output(component_id='target_select_dropdown', component_property='options'),
              [Input('original_dataset_div', 'children')])
def target_select(jsonified_data):
    column_names,  numerical_column_names, categorical_column_names = sep_cat_and_num(
        jsonified_data)
    return list_of_columns_dropdown(column_names)


@app.callback(Output('lin_reg_output', 'children'),
              [Input('original_dataset_div', 'children'),
               Input(component_id='main_algorithm_dropdown', component_property='value'),
               Input(component_id='sub_algorithm_dropdown', component_property='value'),
               Input(component_id='feature_select_dropdown', component_property='value'),
               Input(component_id='target_select_dropdown', component_property='value')])
def linear_regression(jsonified_data, selected_main_algorithm, selected_sub_algorithm, selected_features, selected_target):
    dataset = pd.read_json(jsonified_data, orient='split')

    target = selected_target
    feature_list = selected_features
    if(feature_list is not None and target is not None):
        reg = lin_reg.LinearReg()
        reg.set_dataset(dataset, feature_list, target)
        result1 = getattr(reg, selected_sub_algorithm)()
        return result1


if __name__ == '__main__':
    app.run_server(debug=True)
