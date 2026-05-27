import dash
from dash import dcc, html, register_page, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from scipy import stats
import os
import tempfile
import re
from html2image import Html2Image
import base64
import io

register_page(__name__)

# --- 헬퍼 함수: Dash 컴포넌트를 HTML 문자열로 변환 ---
def camel_to_kebab(name):
    return re.sub(r'(?<!^)(?=[A-Z])', '-', name).lower()

def component_to_html(d):
    if d is None: return ""
    if isinstance(d, (str, int, float)): return str(d)
    if isinstance(d, list): return "".join([component_to_html(c) for c in d])
    props = d.get('props', {})
    tag_name = d.get('type', 'div').lower()
    
    # 그래프와 버튼은 HTML 리포트에서 제외
    if 'graph' in tag_name or 'button' in tag_name: return ""
    
    style_str = ""
    style = props.get('style')
    if style:
        style_parts = [f"{camel_to_kebab(k)}: {v}" for k, v in style.items()]
        style_str = f' style="{"; ".join(style_parts)}"'
    
    class_name = props.get('className', '')
    class_str = f' class="{class_name}"' if class_name else ""
    
    children = props.get('children', [])
    return f"<{tag_name}{class_str}{style_str}>{component_to_html(children)}</{tag_name}>"

# --- 반복적 Grubbs 검정 함수 ---
def find_multiple_outliers(data, alpha=0.05, side='both'):
    current_data = list(data); original_data = list(data); outliers = []
    while len(current_data) >= 3:
        n = len(current_data); mean = np.mean(current_data); std = np.std(current_data, ddof=1)
        if std == 0: break
        if side == 'smallest': val = np.min(current_data); g_stat = (mean - val) / std
        elif side == 'largest': val = np.max(current_data); g_stat = (val - mean) / std
        else: # both
            idx = np.argmax(np.abs(np.array(current_data) - mean))
            val = current_data[idx]; g_stat = np.abs(val - mean) / std
        try:
            t_sq = (n * (n - 2) * (g_stat**2)) / ((n - 1)**2 - n * (g_stat**2))
            if t_sq < 0: break
            p_val = n * (1 - stats.t.cdf(np.sqrt(t_sq), n - 2))
            if p_val < alpha:
                for i, v in enumerate(original_data):
                    if v == val and i not in [o['idx'] for o in outliers]:
                        outliers.append({'val': val, 'idx': i, 'g': g_stat, 'p': p_val}); break
                current_data.remove(val)
            else: break
        except: break
    return outliers

# --- 레이아웃 ---
layout = html.Div([
    dbc.Container([
        html.H2("특이치 검정 (Outlier Test)", className="mb-4", style={'fontWeight': 'bold', 'color': '#2c3e50'}),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("데이터 입력", style={'fontWeight': 'bold'}),
                    dbc.CardBody([
                        html.Label("데이터 (숫자)"),
                        dcc.Textarea(id='outlier-data-input', style={'width': '100%', 'height': '200px'}, placeholder="숫자를 줄바꿈이나 공백으로 구분..."),
                        html.Label("대립 가설", className="mt-3"),
                        dcc.Dropdown(id='outlier-side-input', value='both', options=[
                            {'label': '가장 크거나 가장 작은 값이 특이치', 'value': 'both'},
                            {'label': '가장 작은 데이터 값이 특이치', 'value': 'smallest'},
                            {'label': '가장 큰 데이터 값이 특이치', 'value': 'largest'}
                        ], clearable=False),
                        html.Label("유의 수준 (α)", className="mt-3"),
                        dcc.Input(id='outlier-alpha-input', type='number', value=0.05, step=0.01, style={'width': '100%'}),
                        dbc.Button("분석 실행", id='run-outlier-btn', color="primary", className="mt-4 w-100"),
                    ])
                ])
            ], width=4),
            dbc.Col([dcc.Loading(children=html.Div(id='outlier-result-area'))], width=8)
        ]),
        dcc.Download(id="download-outlier-image")
    ], fluid=True, style={'padding': '20px'})
])

@callback(
    Output('outlier-result-area', 'children'),
    Input('run-outlier-btn', 'n_clicks'),
    State('outlier-data-input', 'value'),
    State('outlier-side-input', 'value'),
    State('outlier-alpha-input', 'value'),
    prevent_initial_call=True
)
def update_outlier_analysis(n, data_str, side, alpha):
    if not data_str: return ""
    try: data = [float(x) for x in re.split(r'[\s,]+', data_str.strip()) if x]
    except: return dbc.Alert("올바른 숫자 데이터를 입력하세요.", color="danger")
    
    n_count = len(data); mean = np.mean(data); std = np.std(data, ddof=1)
    min_v, max_v = np.min(data), np.max(data)
    outliers = find_multiple_outliers(data, alpha, side)
    
    # 1. 방법 박스
    method_box = html.Div([
        html.H5("방법", style={'color': '#0056b3', 'fontWeight': 'bold'}),
        html.P(f"귀무 가설: 모든 데이터 값이 동일한 정규 모집단에서 추출됩니다."),
        html.P(f"대립 가설: {side if side != 'both' else 'both'} 값이 특이치 (반복적 Grubbs 검정)"),
        html.P(f"유의 수준: α = {alpha}")
    ], style={'backgroundColor': '#f8f9fa', 'padding': '15px', 'borderLeft': '5px solid #0056b3', 'marginBottom': '20px'})

    # 2. 통계 테이블
    g_val = outliers[0]['g'] if outliers else 0
    p_val = outliers[0]['p'] if outliers else 1.0
    stats_table = html.Div([
        html.H5("Grubbs의 검정 결과", style={'color': '#0056b3', 'fontWeight': 'bold'}),
        dbc.Table([
            html.Thead(html.Tr([html.Th("변수"), html.Th("N"), html.Th("평균"), html.Th("표준 편차"), html.Th("최소값"), html.Th("최대값"), html.Th("G (최대)"), html.Th("P (최대)")])),
            html.Tbody(html.Tr([html.Td("데이터"), html.Td(n_count), html.Td(f"{mean:.2f}"), html.Td(f"{std:.2f}"), html.Td(f"{min_v:.2f}"), html.Td(f"{max_v:.2f}"), html.Td(f"{g_val:.2f}"), html.Td(f"{p_val:.3f}")]))
        ], bordered=True, size="sm")
    ], style={'marginBottom': '20px'})

    # 3. 특이치 테이블
    rows = [html.Tr([html.Td(o['idx']+1), html.Td(f"{o['val']:.2f}")]) for o in outliers] if outliers else [html.Tr([html.Td("-"), html.Td("없음")])]
    outlier_box = html.Div([
        html.H5("발견된 특이치", style={'color': '#0056b3', 'fontWeight': 'bold'}),
        dbc.Table([html.Thead(html.Tr([html.Th("행"), html.Th("특이치")])), html.Tbody(rows)], bordered=True, size="sm")
    ], style={'marginBottom': '20px'})

    fig = go.Figure()
    fig.add_trace(go.Box(x=data, name='데이터', boxpoints='all', marker_color='#3498db'))
    if outliers: fig.add_trace(go.Scatter(x=[o['val'] for o in outliers], y=['데이터']*len(outliers), mode='markers', marker=dict(size=10, color='red', symbol='square'), name='특이치'))
    fig.update_layout(height=400, plot_bgcolor='white', margin=dict(l=20,r=20,t=40,b=40), title="데이터의 특이치 박스 플롯")

    return html.Div([
        dbc.Button("📷 리포트 저장 (텍스트/표)", id='download-outlier-img-btn', color="success", className="mb-3"),
        html.Div([method_box, stats_table, outlier_box], id='outlier-report-content'),
        dcc.Graph(id='outlier-plot', figure=fig, config={'displayModeBar': True})
    ])

@callback(
    Output("download-outlier-image", "data"),
    Input("download-outlier-img-btn", "n_clicks"),
    State('outlier-report-content', 'children'),
    prevent_initial_call=True
)
def download_outlier_report(n, children):
    if not n: return dash.no_update
    inner = component_to_html(children)
    html_c = f"<html><body style='padding:20px; background:white; font-family:sans-serif;'>{inner}</body></html>"
    with tempfile.TemporaryDirectory() as d:
        hti = Html2Image(output_path=d)
        hti.screenshot(html_str=html_c, save_as="out.png", size=(800, 600))
        with open(os.path.join(d, "out.png"), "rb") as f: return dcc.send_bytes(f.read(), "outlier_report.png")
