from dash import Dash, html, dcc, page_container, page_registry
import dash_bootstrap_components as dbc
import dash_uploader as du
import os

UPLOAD_FOLDER_ROOT = "uploads"

app = Dash(__name__, use_pages=True, 
           suppress_callback_exceptions=True,
           external_stylesheets=[dbc.themes.BOOTSTRAP],
           external_scripts=['/assets/01_config.js', 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js'])

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "backgroundColor": "#f8f9fa",
}

CONTENT_STYLE = {
    "marginLeft": "16rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div([
    html.H3("Navigation"),
    html.Hr(),
    dbc.Nav([dbc.NavLink(page["name"], href=page["path"], active="exact") for page in page_registry.values()],
            vertical=True, pills=True)
], style=SIDEBAR_STYLE)

app.layout = html.Div([sidebar, html.Div(page_container, style=CONTENT_STYLE)])

# dash-uploader 설정
du.configure_upload(app, UPLOAD_FOLDER_ROOT)

import socket

def get_port():
    ports = range(8050, 8550)
    for port in ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', port)) != 0:
                return port

if __name__ == '__main__':
    app.run(debug=True, port=get_port())
