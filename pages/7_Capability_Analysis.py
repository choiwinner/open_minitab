import dash
from dash import dcc, html, register_page
from dash.dependencies import Input, Output, State, ALL
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats
import pandas as pd
import os
import tempfile
import re
import math
from html2image import Html2Image

register_page(__name__)

# --- 헬퍼 함수: 편향 수정 상수 c4(n) 계산 ---
def get_c4(n):
    if n <= 1: return 1.0
    try:
        val = math.sqrt(2.0 / (n - 1)) * (math.gamma(n / 2.0) / math.gamma((n - 1) / 2.0))
        return val
    except:
        # 감마 함수 계산 중 오버플로우 발생 시 Stirling 근사
        return 1.0 - 1.0 / (4.0 * n)

# --- 헬퍼 함수: Dash 컴포넌트를 HTML 문자열로 변환 ---
def camel_to_kebab(name):
    return re.sub(r'(?<!^)(?=[A-Z])', '-', name).lower()

def component_to_html(d):
    if d is None: return ""
    if isinstance(d, (str, int, float)): return str(d)
    if isinstance(d, list): return "".join([component_to_html(c) for c in d])
    props = d.get('props', {})
    tag_name = d.get('type', 'div').lower()
    style_str = ""
    style = props.get('style')
    if style:
        style_parts = [f"{camel_to_kebab(k)}: {v}" for k, v in style.items()]
        style_str = f' style="{"; ".join(style_parts)}"'
    class_str = ""
    class_name = props.get('className')
    if class_name:
        class_str = f' class="{class_name}"'
    children = props.get('children')
    return f"<{tag_name}{style_str}{class_str}>{component_to_html(children)}</{tag_name}>"

# --- 앱 레이아웃 ---
layout = html.Div(
    style={'fontFamily': 'Arial, sans-serif', 'maxWidth': '1000px', 'margin': 'auto', 'padding': '20px'},
    children=[
        dcc.Store(id='capa-stats-download-store'),
        dcc.Download(id='capa-stats-download-component'),

        html.H1("정규 공정 능력 분석 (Normal Capability Analysis)", style={'textAlign': 'center', 'color': '#333'}),

        html.P("공정이 규격 요구사항을 얼마나 잘 만족하는지 단기 및 장기 관점에서 분석합니다. 엑셀 등에서 데이터 열을 복사하여 입력해 주세요.",
               style={'textAlign': 'center', 'color': '#555', 'marginBottom': '30px'}),

        # 1. 표본 데이터 입력
        html.Div([
            html.Label("측정 데이터 입력 (Excel 복사/붙여넣기 지원):", style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': '5px'}),
            html.P("한 줄에 하나의 측정값을 입력하세요. (공백 제외)", style={'fontSize': '12px', 'color': '#666', 'marginTop': '0px'}),
            dcc.Textarea(
                id='capa-data-input',
                placeholder="예 (피스톤 링 지름):\n74.03\n74.00\n74.01\n73.98\n74.02\n...",
                value="74.0075\n73.9979\n74.0097\n74.0228\n73.9965\n73.9965\n74.0237\n74.0115\n73.9930\n74.0081\n73.9930\n73.9930\n74.0036\n73.9713\n73.9741\n73.9916\n73.9848\n74.0047\n73.9864\n73.9788\n74.0220\n73.9966\n74.0010\n73.9786\n73.9918\n74.0017\n73.9827\n74.0056\n73.9910\n73.9956\n73.9910\n74.0278\n73.9998\n73.9841\n74.0123\n73.9817\n74.0031\n73.9706\n73.9801\n74.0030\n74.0111\n74.0026\n73.9983\n73.9955\n73.9778\n73.9892\n73.9931\n74.0159\n74.0052\n73.9736\n74.0049\n73.9942\n73.9898\n74.0092\n74.0155\n74.0140\n73.9874\n73.9954\n74.0050\n74.0146\n73.9928\n73.9972\n73.9834\n73.9821\n74.0122\n74.0203\n73.9989\n74.0151\n74.0054\n73.9903\n74.0054\n74.0231\n73.9995\n74.0235\n73.9607\n74.0123\n74.0013\n73.9955\n74.0014\n73.9702\n73.9967\n74.0054\n74.0222\n73.9922\n73.9879\n73.9925\n74.0137\n74.0049\n73.9921\n74.0077\n74.0015\n74.0145\n73.9895\n73.9951\n73.9941\n73.9780\n74.0044\n74.0039\n74.0001\n73.9965",
                style={'width': '100%', 'height': '200px', 'fontSize': '16px', 'borderRadius': '5px', 'border': '1px solid #ccc'}
            )
        ], style={'marginBottom': '25px'}),

        # 2. 부분군 및 규격 설정 영역
        html.Div([
            # 부분군 크기
            html.Div([
                html.Label("부분군 크기 (Subgroup Size):", style={'fontWeight': 'bold'}),
                html.P("기본값 1은 개별 측정치 기준입니다. (2 이상 시 부분군간/내 변동 연산)", style={'fontSize': '12px', 'color': '#666', 'margin': '2px 0'}),
                dcc.Input(id='capa-subgroup-size', type='number', value=1, min=1, style={'width': '100%', 'padding': '8px', 'borderRadius': '5px', 'border': '1px solid #ccc'})
            ], style={'width': '22%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginRight': '4%'}),

            # 규격 하한 (LSL)
            html.Div([
                html.Label("규격 하한 (LSL):", style={'fontWeight': 'bold'}),
                html.P("제품이 만족해야 하는 최소 한계 규격치입니다.", style={'fontSize': '12px', 'color': '#666', 'margin': '2px 0'}),
                dcc.Input(id='capa-lsl-value', type='number', value=73.95, style={'width': '100%', 'padding': '8px', 'borderRadius': '5px', 'border': '1px solid #ccc'})
            ], style={'width': '22%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginRight': '4%'}),

            # 규격 상한 (USL)
            html.Div([
                html.Label("규격 상한 (USL):", style={'fontWeight': 'bold'}),
                html.P("제품이 만족해야 하는 최대 한계 규격치입니다.", style={'fontSize': '12px', 'color': '#666', 'margin': '2px 0'}),
                dcc.Input(id='capa-usl-value', type='number', value=74.05, style={'width': '100%', 'padding': '8px', 'borderRadius': '5px', 'border': '1px solid #ccc'})
            ], style={'width': '22%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginRight': '4%'}),

            # 목표값 (Target, Optional)
            html.Div([
                html.Label("목표값 (Target - 선택사항):", style={'fontWeight': 'bold'}),
                html.P("공정이 지향하는 타겟 중심값 (Cpm 연산에 활용)", style={'fontSize': '12px', 'color': '#666', 'margin': '2px 0'}),
                dcc.Input(id='capa-target-value', type='number', value=74.00, style={'width': '100%', 'padding': '8px', 'borderRadius': '5px', 'border': '1px solid #ccc'})
            ], style={'width': '22%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        ], style={'marginBottom': '30px'}),

        # 3. 공정 능력 기준 설정 영역
        html.Div([
            html.H4("⚙️ 공정 능력 판정 기준 설정", style={'fontWeight': 'bold', 'borderBottom': '1px solid #ccc', 'paddingBottom': '5px', 'marginBottom': '15px'}),
            
            # 기준 선택 라디오 버튼
            html.Div([
                html.Label("기준 방식 선택:", style={'fontWeight': 'bold', 'marginRight': '15px'}),
                dcc.RadioItems(
                    id='capa-threshold-mode',
                    options=[
                        {'label': ' 기본 기준 사용 (Default)', 'value': 'default'},
                        {'label': ' 사용자 정의 기준 사용 (Custom)', 'value': 'custom'}
                    ],
                    value='default',
                    labelStyle={'display': 'inline-block', 'marginRight': '20px', 'cursor': 'pointer'},
                    style={'display': 'inline-block'}
                )
            ], style={'marginBottom': '20px'}),
            
            # 가로 배치용 Row
            html.Div([
                # 입력 필드 컬럼
                html.Div([
                    html.Label("지수 임계값 설정:", style={'fontWeight': 'bold', 'marginBottom': '10px', 'display': 'block'}),
                    
                    html.Div([
                        html.Span("매우 우수 (Very Good) 이상: ", style={'display': 'inline-block', 'width': '180px'}),
                        dcc.Input(id='capa-th-very-good', type='number', value=2.00, step=0.01, style={'width': '80px', 'padding': '5px', 'borderRadius': '4px', 'border': '1px solid #ccc'})
                    ], style={'marginBottom': '8px'}),
                    
                    html.Div([
                        html.Span("우수 (Good) 이상: ", style={'display': 'inline-block', 'width': '180px'}),
                        dcc.Input(id='capa-th-good', type='number', value=1.67, step=0.01, style={'width': '80px', 'padding': '5px', 'borderRadius': '4px', 'border': '1px solid #ccc'})
                    ], style={'marginBottom': '8px'}),
                    
                    html.Div([
                        html.Span("양호 (Fair) 이상: ", style={'display': 'inline-block', 'width': '180px'}),
                        dcc.Input(id='capa-th-fair', type='number', value=1.33, step=0.01, style={'width': '80px', 'padding': '5px', 'borderRadius': '4px', 'border': '1px solid #ccc'})
                    ], style={'marginBottom': '8px'}),
                    
                    html.Div([
                        html.Span("개선 필요 (Poor) 이상: ", style={'display': 'inline-block', 'width': '180px'}),
                        dcc.Input(id='capa-th-poor', type='number', value=1.00, step=0.01, style={'width': '80px', 'padding': '5px', 'borderRadius': '4px', 'border': '1px solid #ccc'})
                    ], style={'marginBottom': '8px'}),
                    
                ], style={'width': '45%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                
                # 기준 안내 표 컬럼
                html.Div([
                    html.Label("현재 적용된 공정 능력 기준:", style={'fontWeight': 'bold', 'marginBottom': '10px', 'display': 'block'}),
                    html.Table([
                        html.Thead(html.Tr([
                            html.Th("지수 값 범위", style={'textAlign': 'center', 'padding': '8px', 'backgroundColor': '#f2f2f2'}),
                            html.Th("해석 (의미)", style={'textAlign': 'center', 'padding': '8px', 'backgroundColor': '#f2f2f2'})
                        ])),
                        html.Tbody([
                            html.Tr([
                                html.Td(id='lbl-vg-range', style={'textAlign': 'center', 'padding': '8px', 'border': '1px solid #ddd'}),
                                html.Td("매우 우수한 공정", style={'textAlign': 'center', 'padding': '8px', 'border': '1px solid #ddd', 'fontWeight': 'bold', 'color': '#28A745'})
                            ]),
                            html.Tr([
                                html.Td(id='lbl-g-range', style={'textAlign': 'center', 'padding': '8px', 'border': '1px solid #ddd'}),
                                html.Td("우수", style={'textAlign': 'center', 'padding': '8px', 'border': '1px solid #ddd', 'color': '#17A2B8'})
                            ]),
                            html.Tr([
                                html.Td(id='lbl-f-range', style={'textAlign': 'center', 'padding': '8px', 'border': '1px solid #ddd'}),
                                html.Td("양호", style={'textAlign': 'center', 'padding': '8px', 'border': '1px solid #ddd', 'color': '#0056b3'})
                            ]),
                            html.Tr([
                                html.Td(id='lbl-p-range', style={'textAlign': 'center', 'padding': '8px', 'border': '1px solid #ddd'}),
                                html.Td("개선 필요", style={'textAlign': 'center', 'padding': '8px', 'border': '1px solid #ddd', 'color': '#FFC107', 'fontWeight': 'bold'})
                            ]),
                            html.Tr([
                                html.Td(id='lbl-bad-range', style={'textAlign': 'center', 'padding': '8px', 'border': '1px solid #ddd'}),
                                html.Td("불량 공정, 즉각 개선 필요", style={'textAlign': 'center', 'padding': '8px', 'border': '1px solid #ddd', 'color': '#DC3545', 'fontWeight': 'bold'})
                            ])
                        ])
                    ], style={'width': '100%', 'borderCollapse': 'collapse', 'border': '1px solid #ddd'})
                ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '5%'})
            ])
        ], style={'backgroundColor': '#ffffff', 'padding': '20px', 'borderRadius': '6px', 'border': '1px solid #e0e0e0', 'marginBottom': '25px'}),

        # 4. 히스토그램 및 시각화 설정 영역
        html.Div([
            html.H4("📊 히스토그램 및 시각화 설정", style={'fontWeight': 'bold', 'borderBottom': '1px solid #ccc', 'paddingBottom': '5px', 'marginBottom': '15px'}),
            
            # 가로 배치를 위한 레이아웃
            html.Div([
                # 왼쪽: 계급 구간 설정
                html.Div([
                    html.Label("계급 구간 설정:", style={'fontWeight': 'bold', 'marginBottom': '10px', 'display': 'block'}),
                    
                    html.Div([
                        html.Span("구간 방식: ", style={'marginRight': '10px'}),
                        dcc.RadioItems(
                            id='capa-bin-mode',
                            options=[
                                {'label': ' 자동 생성 (Auto)', 'value': 'auto'},
                                {'label': ' 사용자 지정 (Custom)', 'value': 'custom'}
                            ],
                            value='auto',
                            labelStyle={'display': 'inline-block', 'marginRight': '15px', 'cursor': 'pointer'},
                            style={'display': 'inline-block'}
                        )
                    ], style={'marginBottom': '10px'}),
                    
                    html.Div([
                        html.Div([
                            html.Span("계급 최소값 (Min): ", style={'display': 'inline-block', 'width': '120px', 'marginRight': '10px'}),
                            dcc.Input(
                                id='capa-bin-min',
                                type='number',
                                placeholder="자동 (비워둠)",
                                disabled=True,
                                style={'width': '110px', 'padding': '5px', 'borderRadius': '4px', 'border': '1px solid #ccc'}
                            )
                        ], style={'marginBottom': '8px'}),
                        
                        html.Div([
                            html.Span("계급 최대값 (Max): ", style={'display': 'inline-block', 'width': '120px', 'marginRight': '10px'}),
                            dcc.Input(
                                id='capa-bin-max',
                                type='number',
                                placeholder="자동 (비워둠)",
                                disabled=True,
                                style={'width': '110px', 'padding': '5px', 'borderRadius': '4px', 'border': '1px solid #ccc'}
                            )
                        ], style={'marginBottom': '8px'}),
                        
                        html.Div([
                            html.Span("계급 개수 (Bins): ", style={'display': 'inline-block', 'width': '120px', 'marginRight': '10px'}),
                            dcc.Input(
                                id='capa-bin-count',
                                type='number',
                                value=25,
                                min=2,
                                step=1,
                                disabled=True,
                                style={'width': '110px', 'padding': '5px', 'borderRadius': '4px', 'border': '1px solid #ccc'}
                            ),
                            html.Span(" (2 이상의 정수)", style={'fontSize': '12px', 'color': '#666', 'marginLeft': '5px'})
                        ])
                    ])
                ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                
                # 오른쪽: Y축 스케일 설정
                html.Div([
                    html.Label("Y축 스케일 (척도) 설정:", style={'fontWeight': 'bold', 'marginBottom': '10px', 'display': 'block'}),
                    
                    html.Div([
                        dcc.RadioItems(
                            id='capa-y-mode',
                            options=[
                                {'label': ' 확률 밀도 (Probability Density)', 'value': 'density'},
                                {'label': ' 계급별 표본수 / 빈도수 (Frequency Count)', 'value': 'count'}
                            ],
                            value='density',
                            labelStyle={'display': 'block', 'marginBottom': '10px', 'cursor': 'pointer'}
                        )
                    ])
                ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '4%'})
            ])
        ], style={'backgroundColor': '#ffffff', 'padding': '20px', 'borderRadius': '6px', 'border': '1px solid #e0e0e0', 'marginBottom': '25px'}),

        # 분석 실행 버튼
        html.Button(
            '공정 능력 분석 실행',
            id='capa-run-button',
            n_clicks=0,
            style={'width': '100%', 'padding': '12px', 'fontSize': '18px', 'fontWeight': 'bold',
                   'backgroundColor': '#28A745', 'color': 'white', 'border': 'none', 'borderRadius': '5px',
                   'cursor': 'pointer', 'marginBottom': '20px'}
        ),

        # 결과 출력 영역
        dcc.Loading(
            id="capa-loading",
            type="circle",
            children=[
                html.Div([
                    html.Button(
                        "📷 통계 결과 이미지로 다운로드",
                        id="capa-stats-download-btn",
                        style={'marginBottom': '10px', 'padding': '8px 15px', 'backgroundColor': '#6C757D',
                               'color': 'white', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer', 'display': 'none'}
                    )
                ]),
                html.Div(id='capa-results-output', style={'marginTop': '20px', 'padding': '20px', 'backgroundColor': '#f9f9f9', 'borderRadius': '5px', 'border': '1px solid #eee'}),
                dcc.Graph(id='capa-plot-graph', style={'marginTop': '20px'})
            ]
        )
    ]
)

# --- 콜백: 공정 능력 분석 수행 ---
@dash.callback(
    [Output('capa-plot-graph', 'figure'),
     Output('capa-results-output', 'children'),
     Output('capa-stats-download-store', 'data'),
     Output('capa-stats-download-btn', 'style')],
    [Input('capa-run-button', 'n_clicks')],
    [State('capa-data-input', 'value'),
     State('capa-subgroup-size', 'value'),
     State('capa-lsl-value', 'value'),
     State('capa-usl-value', 'value'),
     State('capa-target-value', 'value'),
     State('capa-threshold-mode', 'value'),
     State('capa-th-very-good', 'value'),
     State('capa-th-good', 'value'),
     State('capa-th-fair', 'value'),
     State('capa-th-poor', 'value'),
     State('capa-bin-mode', 'value'),
     State('capa-bin-min', 'value'),
     State('capa-bin-max', 'value'),
     State('capa-bin-count', 'value'),
     State('capa-y-mode', 'value')]
)
def run_capability_analysis(n_clicks, data_str, d_size, lsl, usl, target, th_mode, th_vg, th_g, th_f, th_p, bin_mode, bin_min, bin_max, bin_count, y_mode):
    if n_clicks == 0 or not data_str:
        fig = go.Figure()
        fig.update_layout(title="데이터를 입력하고 '공정 능력 분석 실행' 버튼을 클릭하세요.")
        return fig, "분석 대기 중...", None, {'display': 'none'}

    # 1. 파라미터 기본 검사
    if lsl is None or usl is None:
        return go.Figure(), html.Div("오류: 규격 하한(LSL)과 상한(USL)을 모두 지정해 주세요.", style={'color': 'red', 'fontWeight': 'bold'}), None, {'display': 'none'}
    if lsl >= usl:
        return go.Figure(), html.Div("오류: 규격 하한은 규격 상한보다 작아야 합니다.", style={'color': 'red', 'fontWeight': 'bold'}), None, {'display': 'none'}
    if d_size is None or d_size < 1:
        d_size = 1

    # 2. 데이터 파싱
    try:
        data = np.array([float(line.strip()) for line in data_str.strip().split('\n') if line.strip()])
    except:
        return go.Figure(), html.Div("오류: 숫자 데이터만 한 줄에 하나씩 올바르게 입력해 주세요.", style={'color': 'red', 'fontWeight': 'bold'}), None, {'display': 'none'}

    n = len(data)
    if n < 2:
        return go.Figure(), html.Div(f"오류: 공정 능력 분석을 위해서는 최소 2개 이상의 관측 데이터가 필요합니다. (현재 {n}개)", style={'color': 'red', 'fontWeight': 'bold'}), None, {'display': 'none'}

    # 3. 통계 연산
    mean = np.mean(data)
    sd_overall = np.std(data, ddof=1) # 장기 표준편차

    # 단기(Within) 표준편차 계산
    if d_size == 1:
        # 개별 관측치 기준: 이동 범위(Moving Range) 평균 추정량 적용
        moving_ranges = np.abs(np.diff(data))
        mr_bar = np.mean(moving_ranges)
        d2_2 = 1.128379167 # d2(2) 상수
        sd_within = mr_bar / d2_2
    else:
        # 부분군 기준: 합동 표준편차(Pooled SD) 및 c4 보정치 적용
        k = n // d_size
        if k < 1:
            return go.Figure(), html.Div(f"오류: 전체 데이터 개수({n})가 부분군 크기({d_size})보다 작습니다.", style={'color': 'red', 'fontWeight': 'bold'}), None, {'display': 'none'}
        
        variances = []
        for j in range(k):
            sub_data = data[j*d_size : (j+1)*d_size]
            v = np.var(sub_data, ddof=1)
            variances.append(v)
        
        pooled_var = np.mean(variances)
        pooled_sd = np.sqrt(pooled_var)
        c4_val = get_c4(d_size)
        sd_within = pooled_sd / c4_val

    # 4. 공정 능력 지수 연산 (단기: Within / 장기: Overall)
    # Cp, Cpk (Within)
    cp = (usl - lsl) / (6.0 * sd_within) if sd_within > 0 else np.nan
    cpl = (mean - lsl) / (3.0 * sd_within) if sd_within > 0 else np.nan
    cpu = (usl - mean) / (3.0 * sd_within) if sd_within > 0 else np.nan
    cpk = min(cpl, cpu) if not np.isnan(cpl) and not np.isnan(cpu) else np.nan

    # Pp, Ppk (Overall)
    pp = (usl - lsl) / (6.0 * sd_overall) if sd_overall > 0 else np.nan
    ppl = (mean - lsl) / (3.0 * sd_overall) if sd_overall > 0 else np.nan
    ppu = (usl - mean) / (3.0 * sd_overall) if sd_overall > 0 else np.nan
    ppk = min(ppl, ppu) if not np.isnan(ppl) and not np.isnan(ppu) else np.nan

    # Cpm (Target이 정의된 경우)
    cpm = np.nan
    if target is not None:
        cpm_denominator = 6.0 * np.sqrt(sd_overall**2 + (mean - target)**2)
        if cpm_denominator > 0:
            cpm = (usl - lsl) / cpm_denominator

    # 5. 불량률 예측 및 관측 성능 연산 (PPM 단위)
    # Within 불량률 예측
    ppm_lsl_within = stats.norm.cdf((lsl - mean) / sd_within) * 1e6 if sd_within > 0 else np.nan
    ppm_usl_within = (1.0 - stats.norm.cdf((usl - mean) / sd_within)) * 1e6 if sd_within > 0 else np.nan
    ppm_total_within = ppm_lsl_within + ppm_usl_within

    # Overall 불량률 예측
    ppm_lsl_overall = stats.norm.cdf((lsl - mean) / sd_overall) * 1e6 if sd_overall > 0 else np.nan
    ppm_usl_overall = (1.0 - stats.norm.cdf((usl - mean) / sd_overall)) * 1e6 if sd_overall > 0 else np.nan
    ppm_total_overall = ppm_lsl_overall + ppm_usl_overall

    # 관측 실제 성능 (Observed Performance)
    obs_lsl_count = np.sum(data < lsl)
    obs_usl_count = np.sum(data > usl)
    ppm_lsl_obs = (obs_lsl_count / n) * 1e6
    ppm_usl_obs = (obs_usl_count / n) * 1e6
    ppm_total_obs = ppm_lsl_obs + ppm_usl_obs

    # 2.5. 판정 기준값 확정
    if th_mode == 'default' or th_vg is None or th_g is None or th_f is None or th_p is None:
        vg, g, f, p = 2.00, 1.67, 1.33, 1.00
    else:
        vg, g, f, p = float(th_vg), float(th_g), float(th_f), float(th_p)

    # 6. 결과 텍스트 해석 구성
    def get_eval_grade(val):
        if np.isnan(val): return "N/A", "판정 불가", "#666"
        if val >= vg:
            return "매우 우수", f"매우 우수한 공정 (지수 {val:.2f} ≥ {vg:.2f})", "#28A745"
        elif val >= g:
            return "우수", f"우수한 공정 ({g:.2f} ≤ 지수 {val:.2f} < {vg:.2f})", "#17A2B8"
        elif val >= f:
            return "양호", f"양호한 공정 ({f:.2f} ≤ 지수 {val:.2f} < {g:.2f})", "#0056b3"
        elif val >= p:
            return "개선 필요", f"개선이 필요한 공정 ({p:.2f} ≤ 지수 {val:.2f} < {f:.2f})", "#D97706"
        else:
            return "불량", f"불량 공정, 즉각 개선 필요 (지수 {val:.2f} < {p:.2f})", "#DC3545"

    cpk_grade, cpk_desc, cpk_color = get_eval_grade(cpk)
    ppk_grade, ppk_desc, ppk_color = get_eval_grade(ppk)

    eval_elements = []
    if not np.isnan(cpk):
        eval_elements.append(
            html.Div([
                html.Span("• 단기 공정능력 지수(Cpk) 판정: ", style={'fontWeight': 'bold', 'fontSize': '17px', 'color': '#333'}),
                html.Span(cpk_desc, style={'fontWeight': 'bold', 'fontSize': '21px', 'color': cpk_color})
            ], style={'marginBottom': '12px'})
        )
    if not np.isnan(ppk):
        eval_elements.append(
            html.Div([
                html.Span("• 장기 공정능력 지수(Ppk) 판정: ", style={'fontWeight': 'bold', 'fontSize': '17px', 'color': '#333'}),
                html.Span(ppk_desc, style={'fontWeight': 'bold', 'fontSize': '21px', 'color': ppk_color})
            ])
        )

    # Cpk 값 기준으로 테두리 색상 결정
    border_color = '#DC3545'  # 기본 빨강 (불량)
    if not np.isnan(cpk):
        if cpk >= f:
            border_color = '#28A745'  # 양호 이상 (초록)
        elif cpk >= p:
            border_color = '#FFC107'  # 개선 필요 (노랑)

    interpretation_div = html.Div([
        html.H4("🔍 공정 능력 판정 및 최종 해석", style={'borderBottom': '2px solid #ddd', 'paddingBottom': '8px', 'marginTop': '20px', 'fontWeight': 'bold', 'color': '#333', 'fontSize': '22px'}),
        html.Div(eval_elements if eval_elements else "해석할 정보가 부족합니다.", style={'lineHeight': '1.6', 'marginTop': '15px'})
    ], style={'padding': '20px', 'backgroundColor': '#fff', 'border': f'3px solid {border_color}', 'borderRadius': '8px', 'marginTop': '20px'})

    # 7. UI 표시용 결과 리포트 HTML 작성 (Minitab 스타일 구조화)
    stats_div = html.Div([
        html.H3("정규 공정 능력 분석 결과 보고서", style={'textAlign': 'center', 'color': '#0056b3', 'marginBottom': '25px'}),
        
        # 1행: 요약 정보 및 규격
        html.Div([
            html.Div([
                html.H5("설정 사양 및 규격", style={'borderBottom': '1px solid #ccc', 'paddingBottom': '3px'}),
                html.P(f"규격 하한 (LSL): {lsl:.4f}"),
                html.P(f"규격 상한 (USL): {usl:.4f}"),
                html.P(f"목표값 (Target): {f'{target:.4f}' if target is not None else '지정 안 됨'}"),
                html.P(f"부분군 크기: {d_size}")
            ], style={'flex': '1', 'marginRight': '20px'}),
            
            html.Div([
                html.H5("공정 기본 기술통계량", style={'borderBottom': '1px solid #ccc', 'paddingBottom': '3px'}),
                html.P(f"전체 관측수 (N): {n}"),
                html.P(f"공정 평균 (Mean): {mean:.4f}"),
                html.P(f"장기 표준편차 (Overall SD): {sd_overall:.5f}"),
                html.P(f"단기 표준편차 (Within SD): {sd_within:.5f}")
            ], style={'flex': '1'})
        ], style={'display': 'flex', 'marginBottom': '25px'}),

        # 2행: 공정 능력 지수 요약 테이블
        html.H4("공정 능력 지수 (Capability Indices)", style={'borderBottom': '1px solid #ddd', 'paddingBottom': '5px'}),
        html.Table([
            html.Thead(html.Tr([
                html.Th("구분 (Index)", style={'textAlign': 'left'}),
                html.Th("단기 능력 (Within / 잠재적)", style={'textAlign': 'center'}),
                html.Th("장기 능력 (Overall / 실제적)", style={'textAlign': 'center'})
            ])),
            html.Tbody([
                html.Tr([
                    html.Td("공정 산포 비율 (Cp / Pp)", style={'fontWeight': 'bold'}),
                    html.Td(f"{cp:.2f}" if not np.isnan(cp) else "N/A", style={'textAlign': 'center'}),
                    html.Td(f"{pp:.2f}" if not np.isnan(pp) else "N/A", style={'textAlign': 'center'})
                ]),
                html.Tr([
                    html.Td("하한 공정 능력 (Cpl / Ppl)"),
                    html.Td(f"{cpl:.2f}" if not np.isnan(cpl) else "N/A", style={'textAlign': 'center'}),
                    html.Td(f"{ppl:.2f}" if not np.isnan(ppl) else "N/A", style={'textAlign': 'center'})
                ]),
                html.Tr([
                    html.Td("상한 공정 능력 (Cpu / Ppu)"),
                    html.Td(f"{cpu:.2f}" if not np.isnan(cpu) else "N/A", style={'textAlign': 'center'}),
                    html.Td(f"{ppu:.2f}" if not np.isnan(ppu) else "N/A", style={'textAlign': 'center'})
                ]),
                html.Tr([
                    html.Td("종합 공정 능력 지수 (Cpk / Ppk)", style={'fontWeight': 'bold', 'color': '#0056b3'}),
                    html.Td(f"{cpk:.2f}" if not np.isnan(cpk) else "N/A", style={'textAlign': 'center', 'fontWeight': 'bold'}),
                    html.Td(f"{ppk:.2f}" if not np.isnan(ppk) else "N/A", style={'textAlign': 'center', 'fontWeight': 'bold'})
                ]),
                html.Tr([
                    html.Td("목표치 대비 공정 성능 (Cpm)"),
                    html.Td("-", style={'textAlign': 'center'}),
                    html.Td(f"{cpm:.2f}" if not np.isnan(cpm) else "N/A", style={'textAlign': 'center', 'fontWeight': 'bold', 'color': '#28A745'})
                ])
            ])
        ], className="table table-bordered table-striped", style={'width': '100%', 'marginBottom': '25px'}),

        # 3행: 불량률 예측 리포트 (PPM 단위)
        html.H4("예측 공정 불량률 (PPM)", style={'borderBottom': '1px solid #ddd', 'paddingBottom': '5px'}),
        html.Table([
            html.Thead(html.Tr([
                html.Th("불량 유형", style={'textAlign': 'left'}),
                html.Th("관측 성능 (Observed)", style={'textAlign': 'center'}),
                html.Th("단기 예측 (Within)", style={'textAlign': 'center'}),
                html.Th("장기 예측 (Overall)", style={'textAlign': 'center'})
            ])),
            html.Tbody([
                html.Tr([
                    html.Td("규격 하한선 미달 (LSL 미달)"),
                    html.Td(f"{ppm_lsl_obs:,.2f}", style={'textAlign': 'center'}),
                    html.Td(f"{ppm_lsl_within:,.2f}" if not np.isnan(ppm_lsl_within) else "N/A", style={'textAlign': 'center'}),
                    html.Td(f"{ppm_lsl_overall:,.2f}" if not np.isnan(ppm_lsl_overall) else "N/A", style={'textAlign': 'center'})
                ]),
                html.Tr([
                    html.Td("규격 상한선 초과 (USL 초과)"),
                    html.Td(f"{ppm_usl_obs:,.2f}", style={'textAlign': 'center'}),
                    html.Td(f"{ppm_usl_within:,.2f}" if not np.isnan(ppm_usl_within) else "N/A", style={'textAlign': 'center'}),
                    html.Td(f"{ppm_usl_overall:,.2f}" if not np.isnan(ppm_usl_overall) else "N/A", style={'textAlign': 'center'})
                ]),
                html.Tr([
                    html.Td("종합 불량률 합계 (Total)", style={'fontWeight': 'bold', 'color': '#dc3545'}),
                    html.Td(f"{ppm_total_obs:,.2f}", style={'textAlign': 'center', 'fontWeight': 'bold'}),
                    html.Td(f"{ppm_total_within:,.2f}" if not np.isnan(ppm_total_within) else "N/A", style={'textAlign': 'center', 'fontWeight': 'bold'}),
                    html.Td(f"{ppm_total_overall:,.2f}" if not np.isnan(ppm_total_overall) else "N/A", style={'textAlign': 'center', 'fontWeight': 'bold'})
                ])
            ])
        ], className="table table-bordered", style={'width': '100%', 'marginBottom': '25px'}),

        # 4행: 적용된 공정 능력 평가 기준
        html.H4("공정 능력 평가 기준 (Capability Criteria)", style={'borderBottom': '1px solid #ddd', 'paddingBottom': '5px'}),
        html.Table([
            html.Thead(html.Tr([
                html.Th("지수 값 범위", style={'textAlign': 'center', 'backgroundColor': '#f1f3f5'}),
                html.Th("판정 (해석)", style={'textAlign': 'center', 'backgroundColor': '#f1f3f5'})
            ])),
            html.Tbody([
                html.Tr([
                    html.Td(f"≥ {vg:.2f}", style={'textAlign': 'center'}),
                    html.Td("매우 우수한 공정", style={'textAlign': 'center', 'color': '#28A745', 'fontWeight': 'bold'})
                ]),
                html.Tr([
                    html.Td(f"{g:.2f} ~ {vg - 0.01:.2f}" if vg > g else f"{g:.2f} ~ {vg:.2f} 미만", style={'textAlign': 'center'}),
                    html.Td("우수", style={'textAlign': 'center', 'color': '#17A2B8'})
                ]),
                html.Tr([
                    html.Td(f"{f:.2f} ~ {g - 0.01:.2f}" if g > f else f"{f:.2f} ~ {g:.2f} 미만", style={'textAlign': 'center'}),
                    html.Td("양호", style={'textAlign': 'center', 'color': '#0056b3'})
                ]),
                html.Tr([
                    html.Td(f"{p:.2f} ~ {f - 0.01:.2f}" if f > p else f"{p:.2f} ~ {f:.2f} 미만", style={'textAlign': 'center'}),
                    html.Td("개선 필요", style={'textAlign': 'center', 'color': '#FFC107', 'fontWeight': 'bold'})
                ]),
                html.Tr([
                    html.Td(f"< {p:.2f}", style={'textAlign': 'center'}),
                    html.Td("불량 공정, 즉각 개선 필요", style={'textAlign': 'center', 'color': '#DC3545', 'fontWeight': 'bold'})
                ])
            ])
        ], className="table table-bordered", style={'width': '100%'}),

        interpretation_div
    ])

    # 8. Plotly 차트 구성
    # 계급 구간 결정 우선 수행
    if bin_mode == 'custom':
        bin_min_val = float(bin_min) if bin_min is not None else np.min(data)
        bin_max_val = float(bin_max) if bin_max is not None else np.max(data)
        bin_count_val = int(bin_count) if bin_count is not None and int(bin_count) >= 2 else 25
        bin_edges = np.linspace(bin_min_val, bin_max_val, bin_count_val + 1)
    else:
        # auto 모드
        _, bin_edges = np.histogram(data, bins='auto')
        
    bin_width = bin_edges[1] - bin_edges[0]
    bin_centers = bin_edges[:-1] + bin_width / 2

    # 데이터 분포에 따른 X축 영역 계산 (계급 구간 범위도 함께 고려하여 여유 확보)
    x_min = min(lsl, np.min(data), bin_edges[0])
    x_max = max(usl, np.max(data), bin_edges[-1])
    span = x_max - x_min
    margin = span * 0.15 if span > 0 else 1.0
    x_range = [x_min - margin, x_max + margin]
    x_eval = np.linspace(x_range[0], x_range[1], 300)

    # 각 계급의 실제 빈도수 계산
    counts, _ = np.histogram(data, bins=bin_edges)

    # Subplots 또는 단일 플롯 위에 히스토그램 및 정규곡선 중첩
    fig = go.Figure()

    # 단기 (Within) 및 장기 (Overall) 분포 계산
    y_within = stats.norm.pdf(x_eval, mean, sd_within) if sd_within > 0 else np.zeros_like(x_eval)
    y_overall = stats.norm.pdf(x_eval, mean, sd_overall) if sd_overall > 0 else np.zeros_like(x_eval)

    # 빈도수 모드일 경우 정규분포 적합 곡선을 데이터 개수와 빈 너비에 비례하게 스케일링
    if y_mode == 'count':
        scale_factor = n * bin_width
        y_within = y_within * scale_factor
        y_overall = y_overall * scale_factor

    if y_mode == 'count':
        # go.Bar 추가 (빈도수)
        fig.add_trace(go.Bar(
            x=bin_centers,
            y=counts,
            width=[bin_width] * len(bin_centers),
            name='데이터 분포 (빈도수)',
            marker=dict(color='#d1ecf1', line=dict(color='#0056b3', width=1.5)),
            opacity=0.85,
            text=[str(c) if c > 0 else "" for c in counts],
            textposition='outside',
            textfont=dict(size=9, color='#555555'),
            customdata=np.vstack((bin_edges[:-1], bin_edges[1:], counts)).T,
            hovertemplate="계급 구간: [%{customdata[0]:.4f}, %{customdata[1]:.4f})<br>빈도수: %{customdata[2]:d}<extra></extra>"
        ))
    else:
        # 확률 밀도 계산
        density, _ = np.histogram(data, bins=bin_edges, density=True)
        
        # go.Bar 추가 (확률 밀도)
        fig.add_trace(go.Bar(
            x=bin_centers,
            y=density,
            width=[bin_width] * len(bin_centers),
            name='데이터 분포 (확률밀도)',
            marker=dict(color='#d1ecf1', line=dict(color='#0056b3', width=1.5)),
            opacity=0.85,
            text=[f"{d:.2f}" if d > 0 else "" for d in density],
            textposition='outside',
            textfont=dict(size=9, color='#555555'),
            customdata=np.vstack((bin_edges[:-1], bin_edges[1:], counts)).T,
            hovertemplate="계급 구간: [%{customdata[0]:.4f}, %{customdata[1]:.4f})<br>밀도: %{y:.4f}<br>빈도수: %{customdata[2]:d}<extra></extra>"
        ))
        
    # 단기 및 장기 분포 곡선 추가
    fig.add_trace(go.Scatter(
        x=x_eval,
        y=y_within,
        mode='lines',
        name='단기 (Within) 분포',
        line=dict(color='#fd7e14', width=2, dash='dash')
    ))
    
    fig.add_trace(go.Scatter(
        x=x_eval,
        y=y_overall,
        mode='lines',
        name='장기 (Overall) 분포',
        line=dict(color='#0056b3', width=2.5)
    ))

    # LSL, USL, Target 수직 가이드라인 추가 (텍스트 가로로 변경 및 그래프 상단 외부 배치)
    fig.add_vline(x=lsl, line_dash="dash", line_color="#dc3545", line_width=1.5,
                  annotation=dict(text=f"LSL\n({lsl})", font=dict(color="#dc3545", size=10), 
                                  yref="paper", y=1.01, yanchor="bottom", textangle=0))
    fig.add_vline(x=usl, line_dash="dash", line_color="#dc3545", line_width=1.5,
                  annotation=dict(text=f"USL\n({usl})", font=dict(color="#dc3545", size=10), 
                                  yref="paper", y=1.01, yanchor="bottom", textangle=0))
    
    if target is not None:
        fig.add_vline(x=target, line_color="#28a745", line_width=1.5,
                      annotation=dict(text=f"Target\n({target})", font=dict(color="#28a745", size=10), 
                                      yref="paper", y=1.01, yanchor="bottom", textangle=0))

    fig.update_layout(
        title="공정 능력 히스토그램 & 정규분포 적합 곡선",
        title_x=0.5,
        xaxis_title="측정 치수",
        xaxis=dict(
            range=x_range,
            tickmode='array',
            tickvals=list(bin_edges),
            ticks='outside',
            ticklen=12,
            tickcolor='red',
            tickwidth=2,
            tickangle=-45
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500,
        margin=dict(l=40, r=40, t=80, b=60),
        bargap=0.03
    )

    if y_mode == 'count':
        fig.update_layout(yaxis_title="계급별 표본수 (Frequency Count)")
    else:
        fig.update_layout(yaxis_title="확률 밀도 (Density)")

    # 다운로드용 스토어 데이터 준비 및 버튼 노출 처리
    btn_style = {'marginBottom': '10px', 'padding': '8px 15px', 'backgroundColor': '#6C757D', 'color': 'white', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer', 'display': 'inline-block'}

    return fig, stats_div, stats_div, btn_style

# --- 콜백: 결과 보고서 이미지 다운로드 ---
@dash.callback(
    Output("capa-stats-download-component", "data"),
    Input("capa-stats-download-btn", "n_clicks"),
    State('capa-stats-download-store', 'data'),
    prevent_initial_call=True
)
def download_capa_report_image(n_clicks, component_dict):
    if not component_dict: return None
    
    inner_html = component_to_html(component_dict)
    html_content = f"""
    <html>
    <head>
        <meta charset='utf-8'>
        <link rel='stylesheet' href='https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css'>
        <style>
            body {{ 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                padding: 40px; 
                background-color: white; 
                width: 750px; 
            }}
            .table th {{ background-color: #f1f3f5; }}
        </style>
    </head>
    <body>
        <div style="border: 1px solid #dee2e6; padding: 30px; border-radius: 8px; background-color: #fdfdfd;">
            {inner_html}
        </div>
    </body>
    </html>
    """

    hti = Html2Image()
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_name = "capa_report.png"
        # 넉넉한 사이즈로 캡처
        hti.screenshot(html_str=html_content, save_as=output_name, size=(820, 1800))
        
        if os.path.exists(output_name):
            with open(output_name, "rb") as f:
                content = f.read()
            os.remove(output_name)
            return dcc.send_bytes(content, "normal_capability_analysis_report.png")
    return None


# --- 콜백: 설정 모드에 따른 임계값 활성화/비활성화 처리 및 기본값 세팅 ---
@dash.callback(
    [Output('capa-th-very-good', 'disabled'),
     Output('capa-th-good', 'disabled'),
     Output('capa-th-fair', 'disabled'),
     Output('capa-th-poor', 'disabled'),
     Output('capa-th-very-good', 'value'),
     Output('capa-th-good', 'value'),
     Output('capa-th-fair', 'value'),
     Output('capa-th-poor', 'value')],
    [Input('capa-threshold-mode', 'value')],
    [State('capa-th-very-good', 'value'),
     State('capa-th-good', 'value'),
     State('capa-th-fair', 'value'),
     State('capa-th-poor', 'value')]
)
def toggle_threshold_inputs(mode, vg, g, f, p):
    if mode == 'default':
        return True, True, True, True, 2.00, 1.67, 1.33, 1.00
    else:
        return False, False, False, False, vg, g, f, p


# --- 콜백: 임계값 설정에 따른 안내 테이블 지수 값 범위 실시간 갱신 ---
@dash.callback(
    [Output('lbl-vg-range', 'children'),
     Output('lbl-g-range', 'children'),
     Output('lbl-f-range', 'children'),
     Output('lbl-p-range', 'children'),
     Output('lbl-bad-range', 'children')],
    [Input('capa-th-very-good', 'value'),
     Input('capa-th-good', 'value'),
     Input('capa-th-fair', 'value'),
     Input('capa-th-poor', 'value')]
)
def update_threshold_table_labels(vg, g, f, p):
    vg_val = vg if vg is not None else 2.00
    g_val = g if g is not None else 1.67
    f_val = f if f is not None else 1.33
    p_val = p if p is not None else 1.00
    
    return (
        f"≥ {vg_val:.2f}",
        f"{g_val:.2f} ~ {vg_val - 0.01:.2f}" if vg_val > g_val else f"{g_val:.2f} ~ {vg_val:.2f} 미만",
        f"{f_val:.2f} ~ {g_val - 0.01:.2f}" if g_val > f_val else f"{f_val:.2f} ~ {g_val:.2f} 미만",
        f"{p_val:.2f} ~ {f_val - 0.01:.2f}" if f_val > p_val else f"{p_val:.2f} ~ {f_val:.2f} 미만",
        f"< {p_val:.2f}"
    )


# --- 콜백: 계급 구간 설정 모드에 따른 인풋 활성화/비활성화 처리 ---
@dash.callback(
    [Output('capa-bin-min', 'disabled'),
     Output('capa-bin-max', 'disabled'),
     Output('capa-bin-count', 'disabled')],
    Input('capa-bin-mode', 'value')
)
def toggle_bin_inputs(mode):
    is_disabled = (mode == 'auto')
    return is_disabled, is_disabled, is_disabled
