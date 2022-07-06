import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import plotly.graph_objs as go
import pandas as pd
import random
import pickle
import sklearn

from helpers.key_finder import api_key
from helpers.api_call import *


########### Define a few variables ######

tabtitle = 'Horror!'
sourceurl = 'https://www.kaggle.com/tmdb/tmdb-movie-metadata'
sourceurl2 = 'https://developers.themoviedb.org/3/getting-started/introduction'
githublink = 'https://github.com/austinlasseter/tmdb-rf-classifier'

# pickled vectorizer
file = open('analysis/vectorizer.pkl', 'rb')
vectorizer=pickle.load(file)
file.close()

# open the pickled RF model file
file = open(f'analysis/trained_rf_model.pkl', 'rb')
rf_model_pickled=pickle.load(file)
file.close()

######## Define the figure

top20=pd.read_csv('analysis/top20.csv')

# Define the color palette (19 colors).
# colors= ['#8EAA90', '#738C7F', '#536869',  '#31414E', '#1B2536',  '#360707', '#720f0f', '#9f1111','#c52525', '#f43d3d',   ]
# horror_colors=[val for val in colors for _ in (0, 1)]
horror_colors=['#536869','#536869','#536869','#536869','#536869','#536869','#536869', '#f43d3d','#536869','#536869','#536869','#f43d3d','#536869','#f43d3d','#536869','#536869','#f43d3d','#536869','#f43d3d','#536869']
mydata = [go.Bar(
    x=top20['feature'],
    y=top20['importance'],
    marker=dict(color=horror_colors)
)]

mylayout = go.Layout(
    title='What words make it a horror movie?',
    xaxis = dict(title = 'Average % contribution of each word to the overall prediction (red words are more highly associated non-horror predictions)'),
    yaxis = dict(title = 'Feature Importance'),

)
fig = go.Figure(data=mydata, layout=mylayout)

## Confusion Matrix
cm = pd.read_csv('analysis/conf_matrix.csv')

########### Initiate the app
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.title=tabtitle

########### Layout

app.layout = html.Div(children=[
    dcc.Store(id='tmdb-store', storage_type='session'),
    dcc.Store(id='summary-store', storage_type='session'),
    html.Div([
        html.H1(['Horror Movie Predictor']),
        html.Div([
            html.Div([
                html.Div('Randomly select a movie summary'),
                html.Button(id='eek-button', n_clicks=0, children='EEK!', style={'color': 'rgb(255, 255, 255)'}),
                html.Div(id='movie-title', children=[]),
                html.Div(id='movie-release', children=[]),
                html.Div(id='movie-overview', children=[]),

            ], style={ 'padding': '12px',
                    'font-size': '22px',
                    # 'height': '400px',
                    'border': 'thick red solid',
                    'color': 'rgb(255, 255, 255)',
                    'backgroundColor': '#536869',
                    'textAlign': 'left',
                    },
            className='six columns'),
            html.Div([
                html.Div('Enter your summary here (or try adding words from the chart below)'),
                dcc.Input(
                    id='summary-input',
                    type='text',
                    size='60',
                    placeholder='Type or paste your movie summary here',
                ),
                html.Button(id='boo-button', n_clicks=0, children='BOO!', style={'color': 'rgb(255, 255, 255)'}),
                html.Div(id='summary-output', children='Press the button!'),
            ], style={ 'padding': '12px',
                    'font-size': '22px',
                    # 'height': '120px',
                    'border': 'thick red solid',
                    'color': 'rgb(255, 255, 255)',
                    'backgroundColor': '#536869',
                    'textAlign': 'left',
                    },
            className='six columns'),
        ], className='twelve columns'),
        html.Br(),

        html.H2(id='prediction-div', style={'textAlign': 'right'}),

        dcc.Graph(id='top20', figure=fig),
        html.Br(),
        html.Div([
            html.Div([
                html.Div(id='cm', children=['Confusion Matrix: Random Forest Classifier']),
                dash_table.DataTable(
                    id='table',
                    columns=[{"name": i, "id": i} for i in cm.columns],
                    data=cm.to_dict('records'),
                    style_header={'backgroundColor': 'rgb(30, 30, 30)'},
                    style_cell={
                        'backgroundColor': 'rgb(50, 50, 50)',
                        'color': 'white'
                    }),
                ], className='six columns'),
            html.Div([
                html.Br(),
                html.Div('Recall: 37%'),
                html.Div('Precision: 44%'),
                html.Div('ROC AUC: 72%'),
                html.Div('Accuracy: 74%'),
            ], className='six columns'),
        ], className='twelve columns'),
        html.Div('Random Forest classifier trained on a labeled dataset of 600 negative/536 positive cases. Testing data included 779 cases.')
    ], className='twelve columns'),


        # Output
    html.Div([
        # Footer
        html.Br(),
        html.A('Code on Github', href=githublink, target="_blank"),
        html.Br(),
        html.A("Data Source: Kaggle", href=sourceurl, target="_blank"),
        html.Br(),
        html.A("Data Source: TMDB", href=sourceurl2, target="_blank"),
    ], className='twelve columns'),



    ]
)

########## Callbacks

# TMDB API call
@app.callback(Output('tmdb-store', 'data'),
              [Input('eek-button', 'n_clicks')],
              [State('tmdb-store', 'data')])
def on_click(n_clicks, data):
    if n_clicks is None:
        raise PreventUpdate
    elif n_clicks==0:
        data = {'title':' ', 'release_date':' ', 'overview':' '}
    elif n_clicks>0:
        data = api_pull(random.choice(ids_list))
    return data

@app.callback([Output('movie-title', 'children'),
                Output('movie-release', 'children'),
                Output('movie-overview', 'children'),
                ],
              [Input('tmdb-store', 'modified_timestamp')],
              [State('tmdb-store', 'data')])
def on_data(ts, data):
    if ts is None:
        raise PreventUpdate
    else:
        return data['title'], data['release_date'], data['overview']

# User writes their own summary

@app.callback(Output('summary-store', 'data'),
              [Input('boo-button', 'n_clicks')],
              [State('summary-input', 'value')]
              )
def on_click(n_clicks, value):
    if n_clicks is None:
        raise PreventUpdate
    elif n_clicks==0:
        data = 'high school zombie night!'
    elif n_clicks>0:
        data = str(value)
    return data

@app.callback([Output('summary-output', 'children'),
               Output('prediction-div', 'children')],
              [Input('summary-store', 'modified_timestamp')],
              [State('summary-store', 'data')])
def on_data(ts, data):
    if ts is None:
        raise PreventUpdate
    else:
        vectorized_text=vectorizer.transform([data])
        probability=100*rf_model_pickled.predict_proba(vectorized_text)[:,1]
        return data, str(f'Probability of being a horror movie: {probability[0]}%')



############ Deploy
if __name__ == '__main__':
    app.run_server(debug=True)
