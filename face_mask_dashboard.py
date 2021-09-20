from textwrap import dedent

import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import zmq
import base64
import pandas as pd 
import json 

from dash.dependencies import Input, Output, State
from flask import Flask, Response
from collections import deque


DEBUG = True
FRAMERATE = 24.0

counter = "0"
json_data = {}

class cctvSSD(object):
        
    def get_ori_frame(self):
        
        global counter
        global json_data

        context_count = zmq.Context()
        count_socket = context_count.socket(zmq.SUB)
        count_socket.bind('tcp://*:6100')
        count_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        counter = count_socket.recv_string()

        context_json = zmq.Context()
        json_socket = context_json.socket(zmq.SUB)
        json_socket.bind('tcp://*:6500')
        json_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        json_data = json_socket.recv_json()
        json_data = json.loads(json_data)
        
        contextSSD = zmq.Context()
        footage_socketSSD = contextSSD.socket(zmq.SUB)
        footage_socketSSD.bind('tcp://*:7000')
        footage_socketSSD.setsockopt_string(zmq.SUBSCRIBE, str(''))
      
        frameSSD = footage_socketSSD.recv_string()
        img = base64.b64decode(frameSSD)
        return img

def gen_ori(camera):
    while True:
        frame = camera.get_ori_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


server = Flask(__name__)
app = dash.Dash(__name__, server=server)

app.scripts.config.serve_locally = True
app.config['suppress_callback_exceptions'] = True

@server.route('/cctv_ssd')
def cctv_yolo():
    return Response(gen_ori(cctvSSD()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
                    

def markdown_popup():
    return html.Div(
        id='markdown',
        className="model",
        style={'display': 'none'},
        children=(
            html.Div(
                className="markdown-container",
                children=[
                    html.Div(
                        className='close-container',
                        children=html.Button(
                            "Close",
                            id="markdown_close",
                            n_clicks=0,
                            className="closeButton",
                            style={'border': 'none', 'height': '100%'}
                        )
                    ),
                    html.Div(
                        className='markdown-text',
                        children=[dcc.Markdown(
                            children=dedent(
                                '''
                                ##### What am I looking at?
                                
                                This app enhances visualization of face mask detected using state-of-the-art Mobile Vision Neural Networks.
                                Most user generated videos are dynamic and fast-paced, which might be hard to interpret. A confidence
                                heatmap stays consistent through the video and intuitively displays the model predictions. The pie chart
                                lets you interpret how the object classes are divided, which is useful when analyzing videos with numerous
                                and differing objects.

                                ##### More about this dash app
                                
                                The purpose of this demo is to explore alternative visualization methods for Face Mask Detection. Therefore,
                                the visualizations, predictions and videos are not generated in real time, but done beforehand.
                                '''
                            ))
                        ]
                    )
                ]
            )
        )
    )


# Main App
app.layout = html.Div(
    children=[
        html.Div(
            id='top-bar',
            className='row',
            style={'backgroundColor': '#790b94',
                   'height': '5px',
                   }
        ),
        html.Div(
            className='container',
            children=[
        html.Div(
            id='left-side-column',
            className='eight columns',
            style={'display': 'flex',
                   'flexDirection': 'column',
                   'flex': 1,
                   'height': 'calc(100vh - 5px)',
                   'backgroundColor': '#F2F2F2',
                   'overflow-y': 'scroll',
                   'marginLeft': '0px',
                   'justifyContent': 'flex-start',
                   'alignItems': 'center'},
            children=[
                html.Div(
                    id='header-section',
                    children=[
                        html.H4(
                            'Face Mask Detection Example Use Case'
                        ),
                        html.P(
                            'To get started, select a footage you want to view, and choose the display mode (live graph report or'
                            'detection mode). Then, you can start playing the video, and the visualization will '
                            'be displayed depending on the current time.'
                        ),
                        html.Button("Learn More", id="learn-more-button", n_clicks=0)
                    ]
                ),
                html.Div(
                    className='video-outer-container',
                    children=html.Div(
                        style={'width': '100%', 'paddingBottom': '10%', 'position': 'relative'},
                        children= html.Div(
                            id = "video-display",
                            style={'position': 'relative', 'width': '100%',
                                    'height': '100%', 'top': '50px', 'left': '0px', 'bottom': '0', 'right': '0'}, 
                        
                        )
                    )
                ),
                html.Div(
                    className='control-section',
                    children=[
                        html.Div(
                            className='control-element',
                            children=[
                                html.Div(children=["Footage Selection:"], style={'width': '40%'}),
                                dcc.Dropdown(
                                    id="dropdown-footage-selection",
                                    options=[
                                        {'label': 'Source 1',
                                         'value': '1'},
                                        {'label': 'Source 2', 'value': '2'}
                                    ],
                                    value='1',
                                    clearable=False,
                                    style={'width': '60%'}
                                )
                            ]
                        ),

                        html.Div(
                            className='control-element',
                            children=[
                                html.Div(children=["Graph View Mode:"], style={'width': '40%'}),
                                dcc.Dropdown(
                                    id="dropdown-graph-view-mode",
                                    options=[
                                        {'label': 'Live Mode', 'value': 'live'},
                                        {'label': 'Detection Mode', 'value': 'detection'}
                                    ],
                                    value='detection',
                                    searchable=False,
                                    clearable=False,
                                    style={'width': '60%'}
                                )
                            ]
                        )
                    ]
                )
            ]
        ),
        html.Div(
            id='right-side-column',
            className='four columns',
            style={
                'height': 'calc(100vh - 5px)',
                'overflow-y': 'scroll',
                'marginLeft': '1%',
                'display': 'flex',
                'backgroundColor': '#F9F9F9',
                'flexDirection': 'column'
            },
            children=[
                html.Div(
                    className='img-container',
                    children=html.Img(
                        style={'height': '100%', 'margin': '2px'},
                        src="https://le-cdn.website-editor.net/bbf7a5ffe90a4adabeb56412afa00deb/dms3rep/multi/opt/logo-text-vertical-color-1920w.png"
                        )
                ),
                html.Div(id="div-live-mode"),
                html.Div(id="div-detection-mode")
            ]
        )]),
        markdown_popup()
    ]
)

# # Footage Selection
# context = zmq.Context()
# select_socket = context.socket(zmq.PUB) 
# select_socket.connect('tcp://localhost:9000')

@app.callback(Output("video-display", "children"),
              [Input('dropdown-footage-selection', 'value')])
               
def select_footage(dropdown_value):
    # Find desired footage and update player video
    global select_socket
    if dropdown_value=='1':
        return html.Img(src="/cctv_ssd", style={'height':'466px','width':'800px', 'margin-top':'0px', 'margin-left':'0px', 'margin-right':'0px', 'margin-bottom':'0px'})
    elif dropdown_value=='2':
        return html.Img(src="/cctv_ssd", style={'height':'466px','width':'800px', 'margin-top':'0px', 'margin-left':'0px', 'margin-right':'0px', 'margin-bottom':'0px'})

# Learn more popup
@app.callback(Output("markdown", "style"),
              [Input("learn-more-button", "n_clicks"), Input("markdown_close", "n_clicks")])
def update_click_output(button_click, close_click):
    if button_click > close_click:
        return {"display": "block"}
    else:
        return {"display": "none"}


# live mode: 1. Live graph
#            2. Pie graph

@app.callback(Output("div-live-mode", "children"),
              [Input("dropdown-graph-view-mode", "value")])
def update_output(dropdown_value):
    if dropdown_value == "live":
        return [
            dcc.Interval(
                id="interval-live-mode",
                interval=700,
                n_intervals=1
            ),
            html.Div(
                children=[
                    html.P(children="Live Graph",
                           className='plot-title'),
                    dcc.Graph(
                        id="graph-counter",
                        style={'height': '45vh', 'width': '100%'}),

                    html.P(children="Data History",
                           className='plot-title'),
                    dcc.Graph(
                        id="pie-object-count",
                        style={'height': '40vh', 'width': '100%'}
                    )

                ]
            )
        ]
    else:
        return []

# Detection Mode
# 1. Confidence level (bar score graph)
# 2. Object detected (heatmap)

@app.callback(Output("div-detection-mode", "children"),
              [Input("dropdown-graph-view-mode", "value")])
def update_detection_mode(value):
    if value == "detection":
        return [
            dcc.Interval(
                id="interval-detection-mode",
                interval=700,
                n_intervals=1
            ),
            html.Div(
                children=[
                    html.P(children="Confidence Level of Object Presence",
                           className='plot-title'),
                    dcc.Graph(
                        id="heatmap-confidence",
                        style={'height': '45vh', 'width': '100%'}),

                    html.P(children="Detection Score of Most Probable Objects",
                           className='plot-title'),
                    dcc.Graph(
                        id="bar-score-graph",
                        style={'height': '45vh'}
                    )
                ]
            )
        ]
    else:
        return []


X = deque(maxlen=20)
X.append(0)
# X.append()
Y = deque(maxlen=20)
Y.append(0)

@app.callback(Output("graph-counter", "figure"),
              [Input("interval-live-mode", "n_intervals")])
            
def update_graph_counter(n_intervals):

    layout = go.Layout(
        showlegend=False,
        paper_bgcolor='rgb(249,249,249)',
        plot_bgcolor='rgb(249,249,249)',
        xaxis={
            'automargin': True,
        },
        yaxis={
            'title': 'Number of Person',
            'automargin': True
        }
    )

    global X, Y, counter

    counter = str(counter)
    counter = int(counter)

    if n_intervals > 0:

        X.append(X[-1]+1)
        Y.append(counter)

        data = go.Scatter(
            x=list(X),
            y=list(Y),
            name= 'Right',
            mode= 'lines+markers',
            fill = 'tozeroy',
            line_color='indigo'
            )

        return {'data': [data],'layout' : go.Layout(xaxis=dict(range=[min(X), max(X)], type='category'),
                                                yaxis=dict(range=[min(Y), max(Y)], dtick=1))}
    
    return go.Figure(data=[go.Pie()], layout=layout)

# Updating Figures
@app.callback(Output("bar-score-graph", "figure"),
              [Input("interval-detection-mode", "n_intervals")])
            
def update_score_bar(n_intervals):
    layout = go.Layout(
        showlegend=False,
        paper_bgcolor='rgb(249,249,249)',
        plot_bgcolor='rgb(249,249,249)',
        xaxis={
            'automargin': True,
        },
        yaxis={
            'title': 'Score',
            'automargin': True,
            'range': [0, 1]
        }
    )

    global json_data

    frame_df = pd.DataFrame(json_data)   
    
    if n_intervals > 0 and not frame_df.empty:

        frame_df = frame_df[:min(8, frame_df.shape[0])]
        
        objects = frame_df["class_str"].tolist()
        object_count_dict = {x: 0 for x in set(objects)} 

        objects_wc = []  # Object renamed with counts
        for object in objects:
            object_count_dict[object] += 1  # Increment count
            objects_wc.append("{} {}".format(object, object_count_dict[object]))

        colors = list('rgb(81, 11, 146)' for i in range(len(objects_wc)))

        # Add text information
        y_text = ["{}% confidence".format(round(value * 100)) for value in frame_df['score'].tolist()]

        figure = go.Figure({
            'data': [{'hoverinfo': 'x+text',
                        'name': 'Detection Scores',
                        'text': y_text,
                        'type': 'bar',
                        'x': objects_wc,
                        'marker': {'color': colors},
                        'y': frame_df["score"].tolist()}],
            'layout': {'showlegend': False,
                        'autosize': False,
                        'paper_bgcolor': 'rgb(249,249,249)',
                        'plot_bgcolor': 'rgb(249,249,249)',
                        'xaxis': {'automargin': True, 'tickangle': -45},
                        'yaxis': {'automargin': True, 'range': [0, 1], 'title': {'text': 'Score'}}}
            }
        )
        return figure

    return go.Figure(data=[go.Bar()], layout=layout)  # Returns empty bar


@app.callback(Output("pie-object-count", "figure"),
              [Input("interval-live-mode", "n_intervals")])
              
def update_object_count_pie(n_intervals):
    layout = go.Layout(
        showlegend=True,
        paper_bgcolor='rgb(249,249,249)',
        plot_bgcolor='rgb(249,249,249)',
        autosize=False,
        margin=go.layout.Margin(
            l=10,
            r=10,
            t=15,
            b=15
        )
    )

    global json_data

    frame_df = pd.DataFrame(json_data)

    if n_intervals > 0 and not frame_df.empty:
        
        class_counts = frame_df["class_str"].value_counts()
        classes = class_counts.index.tolist()
        counts = class_counts.tolist()  # List of each count

        text = ["{} detected".format(count) for count in counts]
        
        colorscale = ['#560b94', '#760b94', '#940b94', '#940b6f', '#e848b7', '#c63daf', '#dc6fbd', '#dc6fb6', '#ffffff']

        pie = go.Pie(
            labels=classes,
            values=counts,
            text=text,
            hoverinfo="text+percent",
            textinfo="label+percent",
            marker={'colors': colorscale[:len(classes)]}
        )
        return go.Figure(data=[pie], layout=layout)
    
    return go.Figure(data=[go.Pie()], layout=layout)  # Returns empty pie chart


@app.callback(Output("heatmap-confidence", "figure"),
              [Input("interval-detection-mode", "n_intervals")])
              
def update_heatmap_confidence(n_intervals):
    layout = go.Layout(
        showlegend=False,
        paper_bgcolor='rgb(249,249,249)',
        plot_bgcolor='rgb(249,249,249)',
        autosize=False,
        margin=go.layout.Margin(
            l=10,
            r=10,
            b=20,
            t=20,
            pad=4
        )
    )

    fp = open('labels-face-mask.txt', "r")
    classes_list = fp.read().split("\n")[:-1]
    n_classes = len(classes_list)
    root_round = np.ceil(np.sqrt(len(classes_list)))

    total_size = root_round ** 2
    padding_value = int(total_size - n_classes)
    classes_padded = np.pad(classes_list, (0, padding_value), mode='constant')

    # The padded matrix containing all the classes inside a matrix
    classes_matrix = np.reshape(classes_padded, (int(root_round), int(root_round)))

    # Flip it for better looks
    classes_matrix = np.flip(classes_matrix, axis=0)

    global json_data

    frame_df = pd.DataFrame(json_data)

    if n_intervals > 0 and not frame_df.empty:
        
        frame_no_dup = frame_df[["class_str", "score"]].drop_duplicates("class_str")
        frame_no_dup.set_index("class_str", inplace=True)

        # The list of scores
        score_list = []
        for el in classes_padded:
            if el in frame_no_dup.index.values:
                score_list.append(frame_no_dup.loc[el][0])
            else:
                score_list.append(0)
            
        # Generate the score matrix, and flip it for visual
        score_matrix = np.reshape(score_list, (-1, int(root_round)))
        score_matrix = np.flip(score_matrix, axis=0)

        # We set the color scale to white if there's nothing in the frame_no_dup
        if frame_no_dup.shape != (0, 1):
            colorscale = [[0, '#f9f9f9'], [1, '#560b94']]
        else:
            colorscale = [[0, '#f9f9f9'], [1, '#f9f9f9']]

        hover_text = ["{}:.2f% confidence".format(score) for score in score_list]
        hover_text = np.reshape(hover_text, (-1, int(root_round)))
        hover_text = np.flip(hover_text, axis=0)

        # Add linebreak for multi-word annotation
        classes_matrix = classes_matrix.astype(dtype='|U40')

        for index, row in enumerate(classes_matrix):
            row = list(map(lambda x: '<br>'.join(x.split()), row))
            classes_matrix[index] = row

        # Set up annotation text
        annotation = []
        for y_cord in range(int(root_round)):
            for x_cord in range(int(root_round)):
                annotation_dict = dict(
                    showarrow=False,
                    text=classes_matrix[y_cord][x_cord],
                    xref='x',
                    yref='y',
                    x=x_cord,
                    y=y_cord
                )
                if score_matrix[y_cord][x_cord] > 0:
                    annotation_dict['font'] = {'color': '#F9F9F9', 'size': '11'}
                else:
                    annotation_dict['font'] = {'color': '#606060', 'size': '11'}
                annotation.append(annotation_dict)

        # Generate heatmap figure
        figure = {
            'data': [
                {'colorscale': colorscale,
                    'showscale': False,
                    'hoverinfo': 'text',
                    'text': hover_text,
                    'type': 'heatmap',
                    'zmin': 0,
                    'zmax': 1,
                    'xgap': 1,
                    'ygap': 1,
                    'z': score_matrix}],
            'layout':
                {'showlegend': False,
                    'autosize': False,
                    'paper_bgcolor': 'rgb(249,249,249)',
                    'plot_bgcolor': 'rgb(249,249,249)',
                    'margin': {'l': 10, 'r': 10, 'b': 20, 't': 20, 'pad': 2},
                    'annotations': annotation,
                    'xaxis': {'showticklabels': False, 'showgrid': False, 'side': 'top', 'ticks': ''},
                    'yaxis': {'showticklabels': False, 'showgrid': False, 'side': 'left', 'ticks': ''}
                    }
        }

        return figure

    # Returns empty figure
    return go.Figure(data=[go.Pie()], layout=layout)

if __name__ == '__main__':
    app.run_server(port=8200, debug=True)