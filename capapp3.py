# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
import base64
import datetime
import io

from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output, State
from tensorflow.keras.models import load_model
from tensorflow.nn import softmax
import plotly.express as px
import pydicom as pdc
from pandas import DataFrame
from PIL import Image

from pipeline import extract_data, transform_data

app = Dash(__name__)

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

app.layout = html.Div([
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or',
            html.A('Select Files')
        ]),
    style={
        'width': '100%',
        'height': '60px',
        'lineHeight': '60px',
        'borderWidth': '1px',
        'borderStyle': 'dashed',
        'borderRadius': '5px',
        'textAlign': 'center',
        'margin': '10px'
    },
    multiple=True
    ),
    html.Div(id='output-data-upload')
])

def parse_contents(contents, filename, date):
    """Load the content
    
    Function set to load the uploaded content and
    transform the data into a readable format for
    the machine learning model to make predictions.

    Parameters
    ----------
    contents : Unkown
        Contains the uploaded file encoded into its
        64-bit version. This is then decoded and read
        using the pydicom library for data processing.
    
    filename : string
        The name of the uploaded file. this does
        not impact the data processing aspect or
        the predictions, but it is necessary to
        point out which file the user is looking
        at.
    
    date : datetime
        Used to let the user know the date in which
        the file was uploaded. Will be used within
        report to download.
    """
    content_type, content = contents.split(',')
    decoded = base64.b64decode(content)
    try:
        ds = pdc.dcmread(io.BytesIO(decoded))
        datapoint = extract_data(ds)
        datapoint = transform_data(datapoint)
        img_arr = ds.pixel_array
        img = Image.fromarray(img_arr)
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])
    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),
        html.Img(src=img),
        html.Hr(), #Horizontal Line

        # For debugging
        html.Div('Raw Content'),
        html.Pre(contents[0:200] + '...', style={
            'whiteSpace': 'pre-wrap',
            'wordBreak': 'break-all'
        })
    ])

@app.callback(
    Output('output-data-upload', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('upload-data', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates): #Need to change this to collect all data.
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in zip(list_of_contents, list_of_names, list_of_dates)
        ]
        return children

if __name__ == '__main__':
    app.run_server(debug=True)