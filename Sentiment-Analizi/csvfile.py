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

#
train_df = pd.DataFrame()
conn = sqlite3.connect(KeyConf.dbName)
df = pd.read_sql("SELECT * FROM %s ORDER BY UnixTime DESC" % (KeyConf.tableName), conn)
df = df.drop_duplicates(subset=['Tweet'], keep=False)
print(df)
import plotly.express as px
if len(df) > 0:
    df.sort_values('UnixTime', inplace=True)
    df['sentiment_smoothed'] = df['Polarity'].rolling(int(len(df) / 5)).mean()
    df['Date'] = pd.to_datetime(df['UnixTime'], unit='ms')
    df.set_index('Date', inplace=True)
    df["Volume"] = 1
    df.Polarity = df.Polarity.fillna(method="ffill")  # nan degerler için polarity
    df.sentiment_smoothed = df.sentiment_smoothed.fillna(method="ffill")  # nan degerler için timelar
    df.Volume = df.Volume.fillna(0)
    df.dropna(inplace=True)
    X = df.index
    print(X)
    Y = df.sentiment_smoothed
    print(Y)
    Y2 = df.Volume
    print(Y2)
    for template in ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white", "none"]:
        fig = px.scatter(
                         x=X, y=Y,
                         log_x=True,
                         template=template)
        fig.show()
plt.show()