# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd

app = dash.Dash(__name__)

df = pd.read_csv("data.txt", delimiter=",")

#fig = px.bar(df, x="temp", y="hum")
fig = px.line(df, x=range(len(df)), y="temp", title='Temperatur')
fig2 = px.line(df, x=range(len(df)), y="hum", title='Feuchtigkeit')

app.layout = html.Div(children=[
    html.H1(children='Dash Humidity Visualisation'),

    html.Div(children='''
        Dash: A web application framework for your data.
    '''),

    dcc.Graph(
        id='example-graph',
        figure=fig
    ),
    dcc.Graph(
        id='example-graph2',
        figure=fig2
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)
