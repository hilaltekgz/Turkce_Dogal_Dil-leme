# -*- coding: cp1252 -*-
import dash
import re
from dash.dependencies import Output, Input, State
import dash_core_components as dcc
import dash_html_components as html
import plotly
import dash_core_components as dcc
import random
import plotly.graph_objs as go
from collections import deque
import sqlite3
import pandas as pd
# import dash_table_experiments as dte
from keys import KeyConf
from tweetsSideCounter import TweetsSideCounter
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import csv

# pip install dash==0.34.0  # The core dash backend
# pip install dash-html-components==0.13.4  # HTML components
# pip install dash-core-components==0.41.0  # Supercharged components
# pip install dash-table==3.1.11


app_colors = {
    'pageBackground': '#f2f2f2',
    'background': '#f8f8ff',
    'text': '#000000',
    'sentiment-plot': '#41EAD4',
    'volume-bar': '#f2f2f2',
    'someothercolor': '#80aaff',
    'papercolor': '#f2f2f2',
    'plotcolor': '#f2f2f2',
    'fillcolor': '#f2f2f2',
    'gridcolor': '#f2f2f2',
    'backgroundTableHeaders': '#f2f2f2',
    'backgroundTableRows': '#002266'
}

tweetsCounter = TweetsSideCounter()
PositiveNegativeThreshold = KeyConf.positiveNegativeThreshold

currentKeyWordsString = ",".join(KeyConf.keyWords)
global gvalue
gvalue = 1000

global resampleValue
resampleValue = "300L"


def quick_color(s):
    # Pozitif yorumlari yesil highlitr
    if s >= PositiveNegativeThreshold:
        # positive
        return "#66CDAA"
    elif s <= -PositiveNegativeThreshold:
        # negative:
        return "#b30000"

    else:
        return app_colors['background']


app = dash.Dash(__name__)
app.layout = html.Div(

    [html.Div(className='container-fluid', children=[html.H2('Canli Twitter Analizi', style={'color': "#f2f2f2"}),
                                                     html.H5('Sentiment Term:',
                                                             style={'color': '#f2f2f2', 'margin-top': 0,
                                                                    'margin-bottom': 0}),
                                                     dcc.Input(id='sentiment_term', value=currentKeyWordsString,
                                                               type='text',
                                                               style={'width': 300, 'color': '#f2f2f2', 'margin-top': 0,
                                                                      'margin-bottom': 0}),
                                                     # html.Button('Submit', id='buttonKeyWords'),
                                                     ],
              style={'width': '98%', 'margin-left': 15, 'margin-right': 15, 'max-width': 50000}
              ),

     html.Div(className='container-fluid',
              children=[html.H5('Window:', style={'color': '#f2f2f2', 'margin-top': 0, 'margin-bottom': 0}),
                        html.Div(className='row', children=
                        [dcc.Input(id='window', value=str(gvalue), type='text',
                                   style={'width': 50, 'color': '#f2f2f2', 'margin-top': 0, 'margin-bottom': 10}),
                         html.Button('Submit', id='buttonWindow', hidden=True),
                         html.Div(id='output-container-button', children='', hidden=True)
                         ]
                                 ),
                        ],
              style={'width': '98%', 'margin-left': 15, 'margin-right': 15, 'max-width': 50000}
              ),

     html.Div(className='two columns', children=[html.Div(dcc.Graph(id='live-graph', figure={'layout': go.Layout(
         xaxis={'showgrid': False},
         yaxis={'title': 'Volume', 'side': 'right'},
         yaxis2={'side': 'left', 'overlaying': 'y',
                 'title': 'Sentiment', 'gridcolor': app_colors['gridcolor']},
         font={'color': app_colors['text'], 'size': 18},
         plot_bgcolor=app_colors['plotcolor'],
         paper_bgcolor=app_colors['papercolor'],
         showlegend=False,
     )}, animate=False), style={'display': 'inline-block',
                                'width': '66%', 'margin-right': -15}
                                                          ),
                                                 html.Div(dcc.Graph(id='pie-graph', figure={'layout': go.Layout(
                                                     xaxis={'showgrid': False},
                                                     yaxis={'gridcolor': app_colors['gridcolor']},
                                                     font={'color': app_colors['text'], 'size': 18},
                                                     plot_bgcolor=app_colors['plotcolor'],
                                                     paper_bgcolor=app_colors['papercolor'],
                                                     showlegend=False,
                                                 )}, animate=False), style={'display': 'inline-block', 'width': '34%',
                                                                            'margin-left': -40, 'margin-right': 0}

                                                          )
                                                 ],
              style={'display': 'inline-block', 'height': 400, 'width': '100%', 'margin-left': 10, 'margin-right': 10,
                     'max-width': 50000}
              ),

     html.Div(id="recent-tweets-table", children=[
         html.Thead(html.Tr(children=[], style={'color': app_colors['text']})),
         html.Tbody([html.Tr(children=[], style={'color': app_colors['text'],
                                                 'background-color': app_colors['backgroundTableRows'],
                                                 'border': '0.2px', 'font - size': '0.7rem'}
                             )])],
              className='col s12 m6 l6', style={'height': 400, 'width': '100%', 'margin-top': 30, 'margin-left': 15,
                                                'margin-right': 15, 'max-width': 500000}),

     dcc.Interval(
         id='graph-update',
         interval=1 * 1000
     ),

     dcc.Interval(
         id='pie-update',
         interval=5 * 1000
     ),

     dcc.Interval(
         id='recent-table-update',
         interval=2 * 1000
     ),

     dcc.Interval(
         id='dashTableUpdate',
         interval=2 * 1000
     ),
     dcc.Interval(
         id='wordcloudUpdate',
         interval=2 * 1000
     ),
     dcc.Interval(
         id='map-update',
         interval=2 * 1000
     ),
     ], style={'backgroundColor': app_colors['pageBackground'], 'margin-top': '-20px',
               'margin-left': -10, 'margin-right': -10, 'height': '2000px', },
)


@app.callback(dash.dependencies.Output('output-container-button', 'children'),
              [dash.dependencies.Input('buttonWindow', 'n_clicks')],
              [dash.dependencies.State('window', 'value')], )
def update_output(n_clicks, value):
    global gvalue
    gvalue = int(value)

    global resampleValue
    resampleValue = "300L"
    if (gvalue > 0) and (gvalue <= 100):
        resampleValue = "300L"
    if (gvalue > 100) and (gvalue <= 200):
        resampleValue = "500L"
    elif (gvalue > 200) and (gvalue <= 1000):
        resampleValue = "1s"
    elif (gvalue > 1000) and (gvalue <= 2000):
        resampleValue = "4s"
    elif (gvalue > 2000) and (gvalue <= 5000):
        resampleValue = "10s"
    elif (gvalue > 5000) and (gvalue <= 10000):
        resampleValue = "15s"
    elif (gvalue > 10000):
        resampleValue = "30s"

    print("Window: " + str(gvalue) + " Resample: " + str(resampleValue))


@app.callback(Output('live-graph', 'figure'),
              [Input(component_id='sentiment_term', component_property='value'),
               Input(component_id='window', component_property='value')
               ],
              )
def update_graph_scatter(sentiment_term, window):
    try:
        train_df = pd.DataFrame()
        conn = sqlite3.connect(KeyConf.dbName)
        df = pd.read_sql("SELECT * FROM %s ORDER BY UnixTime DESC LIMIT %s" % (KeyConf.tableName, gvalue), conn)
        df = df.drop_duplicates(subset=['Tweet'], keep=False)
        if len(df) > 0:
            df.sort_values('UnixTime', inplace=True)
            df['sentiment_smoothed'] = df['Polarity'].rolling(int(len(df) / 5)).mean()
            df['Date'] = pd.to_datetime(df['UnixTime'], unit='ms')
            df.set_index('Date', inplace=True)

            df["Volume"] = 1
            df = df.resample(resampleValue).agg({'Polarity': 'mean', 'sentiment_smoothed': 'mean', 'Volume': 'sum'})
            df.Polarity = df.Polarity.fillna(method="ffill")  # nan degerler i�in polarity
            df.sentiment_smoothed = df.sentiment_smoothed.fillna(method="ffill")  # nan degerler i�in timelar
            df.Volume = df.Volume.fillna(0)
            df.dropna(inplace=True)
            X = df.index
            Y = df.sentiment_smoothed
            Y2 = df.Volume
            import plotly.express as px
            fig = px.scatter(df,x=X, y=Y2,color=Y)
            fig.show()
            plt.show()
            dataScatter = plotly.graph_objs.Scatter(
                x=X,
                y=Y,
                name='Scatter',
                mode='lines+markers',
                marker={'size': 1, 'opacity': 1},
                yaxis='y2',
                line={'width': 1}
            )
            dataVolume = plotly.graph_objs.Bar(
                x=X,
                y=Y2,
                name='Volume',
                marker=dict(color=app_colors['volume-bar'], opacity=0.5),
            )
            return {'data': [dataScatter, dataVolume], 'layout': go.Layout(
                title='Twitter Analizi',
                xaxis={'range': [min(X), max(X)], 'showgrid': False},
                yaxis={'range': [min(Y),max(Y)], 'gridcolor': app_colors['gridcolor']},
                #yaxis={'range': [min(Y2), max(Y2 * 4)], 'title': 'Volume', 'side': 'right'},
                yaxis2={'range': [min(Y), max(Y)], 'side': 'left', 'overlaying': 'y', 'title': 'Sentiment',
                        'gridcolor': app_colors['gridcolor']},
                font={'color': app_colors['text']},
                plot_bgcolor=app_colors['plotcolor'],
                paper_bgcolor=app_colors['papercolor'],
                showlegend=False,
            )}

    except Exception as e:
        with open('errors.txt', 'a') as f:
            f.write("update_graph_scatter: " + str(e))
            f.write('\n')


def generate_table(df, max_rows=20):
    return html.Table(className="responsive-table",
                      children=[
                          html.Thead(
                              html.Tr(
                                  children=[
                                      html.Th(col.title()) for col in df.columns.values],
                                  style={'color': app_colors['text'],
                                         'background-color': app_colors['backgroundTableHeaders']}
                              )
                          ),
                          html.Tbody(
                              [
                                  html.Tr(
                                      children=[
                                          html.Td(data) for data in d
                                      ], style={'color': app_colors['text'],
                                                'background-color': quick_color(d[2]),
                                                'border': '0.2px', 'font - size': '0.7rem'}
                                  )
                                  for d in df.values.tolist()])
                      ]
                      )


# def generateDashDataTable2(df):
#     dte.DataTable(
#         data=df.to_dict("rows"),
#         columns=[
#             {"name": i, "id": i} for i in df.columns
#         ],
#         style_data_conditional=[{
#             "if": {"row_index": 4},
#             "backgroundColor": "#3D9970",
#             'color': 'white'
#         }]
#     )

def generateDashDataTable(df):
    return df.to_dict("rows")


@app.callback(Output('recent-tweets-table', 'children'),
              [Input(component_id='sentiment_term', component_property='value')],
              )
def update_recent_tweets(sentiment_term):
    genTable = html.Table()
    try:
        conn = sqlite3.connect(KeyConf.dbName)
        df = pd.read_sql(
            "SELECT UnixTime, Tweet, Polarity FROM %s ORDER BY UnixTime DESC LIMIT 20" % (KeyConf.tableName), conn)
        if len(df) > 0:
            df['Date'] = pd.to_datetime(df['UnixTime'], unit='ms')
            df = df.drop(['UnixTime'], axis=1)
            df = df[['Date', 'Tweet', 'Polarity']]
            df.Polarity = df.Polarity.round(3)
            # df = df.drop_duplicates(df)
            genTable = generate_table(df, max_rows=20)
    except Exception as e:
        with open('errors.txt', 'a') as f:
            f.write("update_recent_tweets: " + str(e))
            f.write('\n')

    return genTable


@app.callback(Output('pie-graph', 'figure'),
              [Input(component_id='sentiment_term', component_property='value')],
              )
def updatePieChart(sentiment_term):

    try:
        df = pd.DataFrame()
        df_n = pd.DataFrame()
        conn = sqlite3.connect(KeyConf.dbName)
        df = pd.read_sql(
            "SELECT count(CASE WHEN Polarity == 1.0 THEN 1 ELSE NULL END) AS Positive,count(CASE WHEN Polarity == 0.0 THEN 1 ELSE NULL END) AS Negative FROM Tweets11 WHERE (SELECT Polarity FROM Tweets11 WHERE abs(Polarity)>=0.001 ORDER BY UnixTime DESC LIMIT 1000 )",
            conn)
        if len(df) > 0:
            values = [round(100 * df.Positive.iloc[0] / (df.Positive.iloc[0] + df.Negative.iloc[0]), 2),
                      round(100 * df.Negative.iloc[0] / (df.Positive.iloc[0] + df.Negative.iloc[0]), 2)]
        else:
            values = [50, 50]
        colors = ['#66CDAA', '#800000']
        labels = ['Positive', 'Negative']

        trace = go.Pie(labels=labels, values=values,
                       hoverinfo='label+percent', textinfo='value',
                       textfont=dict(size=20, color=app_colors['text']),
                       marker=dict(colors=colors,
                                   line=dict(color=app_colors['background'], width=2)))

        return {"data": [trace], 'layout': go.Layout(
            title='Positive vs Negative (Count)',
            font={'color': app_colors['text']},
            plot_bgcolor=app_colors['plotcolor'],
            paper_bgcolor=app_colors['papercolor'],
            showlegend=True)}

    except Exception as e:
        with open('errors.txt','a') as f:
            f.write("updatePieChart: " + str(e))
            f.write('\n')

    return df


app.config.suppress_callback_exceptions = True



# @app.callback(Output('dashTable', 'rows'),
#               [Input(component_id='sentiment_term', component_property='value')],
#               events=[Event('dashTableUpdate', 'interval')])
# def update_recent_tweets(sentiment_term):
#     conn = sqlite3.connect(RunConfig.dbName)
#     df = pd.read_sql("SELECT UnixTime, Tweet, Polarity FROM %s ORDER BY UnixTime DESC LIMIT 20" % (RunConfig.tableName), conn)
#
#     df['Date'] = pd.to_datetime(df['UnixTime'], unit='ms')
#
#     df = df.drop(['UnixTime'], axis=1)
#     df = df[['Date', 'Tweet', 'Polarity']]
#
#     df = df.to_dict("rows")
#
#     return df


if __name__ == '__main__':
    app.run_server(debug=True)
