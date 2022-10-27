"""Create online dash application."""
import base64
import datetime
import io

from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output, State
from tensorflow.keras.models import load_model
from tensorflow.nn import softmax
import plotly.express as px
from plotly.subplots import make_subplots
import pydicom as pdc
from pandas import DataFrame
from PIL import Image
from numpy import asarray, argmax

from pipeline import extract_data, transform_data

#Base data
modalities = {
    0: 'MR',
    1: 'CT',
    2: 'PT',
    3: 'MG'
}

sides = {
    0: 'L',
    1: 'R'
}

sex = {
    0: 'F',
    1: 'M'
}

class_names = {
    0: 'Benign',
    1: 'Malignant'
}

app = Dash(__name__)

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

app.layout = html.Div([
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
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
    """Load the content.

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
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])
    return datapoint

def predict(data:DataFrame) -> DataFrame:
    """Make predictions based on dataset.

    Extracts the image data and required categories
    for loading into the model.
    """
    model = load_model('./tclass_V1')
    predictions = model.predict({'image': asarray(data['Image'].to_list()), 'cat':asarray(data[['age', 'side']])})
    data['sex'] = data['sex'].map(sex)
    data['modality'] = data['modality'].map(modalities)
    data['side'] = data['side'].map(sides)
    if len(predictions) < 2 and len(predictions) > 0:
        predictions = predictions[0]
        data['score'] = [softmax(predictions).numpy()]
        data['pred_class'] = class_names[argmax(data['score'])]
    elif len(predictions) >= 2:
        predictions = predictions
        pred_data = list()
        for pred in predictions:
            score = softmax(pred)
            pclass = class_names[argmax(score)]
            pred_data.append({'score':score, 'pred_class':pclass})
        _df = DataFrame(pred_data)
        data = data.join(_df)
    return data

@app.callback(
    Output('output-data-upload', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('upload-data', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates): #Need to change this to collect all data.
    """Load the main dashboard."""
    if list_of_contents is not None:
        data = DataFrame([
            parse_contents(c, n, d) for c, n, d in zip(list_of_contents, list_of_names, list_of_dates)
        ])
        data = predict(data)
        data = data.drop(columns='Image')
        first_column = data.pop('Subject ID')
        data.insert(0, 'Suject ID', first_column)
        data['score'] = data['score'].astype(str)
        #Plotting the data.
        fig = make_subplots(
            rows=2, cols=2
        )
        fig1 = px.pie(data, names='pred_class')
        fig2 = px.pie(data, names='sex')
        fig3 = px.pie(data, names='side')
        fig4 = px.bar(data, x='age')
        fig5 = px.bar(data, x='sex', y='pred_class')
        return html.Div([
            html.H1(children='Main Dashboard', style={'textAlign': 'center'}),
            html.H5('Data Table'),
            html.H6(datetime.datetime.now()),
            html.Div([
                html.Div([
                    dcc.Graph(id='g1', figure=fig1, style={'display': 'inline-block', 'width':'33vw'}),
                    dcc.Graph(id='g2', figure=fig2, style={'display': 'inline-block', 'width':'33vw'}),
                    dcc.Graph(id='g3', figure=fig3, style={'display': 'inline-block', 'width':'33vw'}),
                ]),
                html.Div(children=[
                    dcc.Graph(id='g4', figure=fig4, style={'display': 'inline-block'}),
                    dcc.Graph(id='g5', figure=fig5, style={'display': 'inline-block'}),
                ])
            ], className="row"),
            #dcc.Graph(figure=fig1, style={'display': 'inline-block'}),
            #dcc.Graph(figure=fig2, style={'display': 'inline-block'}),

            dash_table.DataTable(
                data.to_dict('records'),
                [{'name':i, 'id':i} for i in data.columns],
                export_format="csv"
            ),
            html.Hr(),
        ])


if __name__ == '__main__':
    app.run_server(debug=True)
