import dash
from dash import dcc, html, register_page
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import plotly.express as px
from scipy import stats

register_page(__name__)

# 앱 레이아웃 정의
layout = html.Div(
    style={'fontFamily': 'Arial, sans-serif', 'maxWidth': '1000px', 'margin': 'auto', 'padding': '20px'},
    children=[
        html.H1(
            "2-표본 콜모고로프-스미르노프 검정 (2-Sample KS-Test)",
            style={'textAlign': 'center', 'color': '#333'}
        ),

        html.P(
            "두 독립적인 데이터 열(column)을 각각 붙여넣으세요. 두 데이터의 분포가 동일한지 검정합니다.",
            style={'textAlign': 'center', 'color': '#555'}
        ),

        # 데이터 입력을 위한 텍스트 영역 (두 개)
        html.Div([
            html.Div([
                html.Label("데이터 1 (Sample 1):", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                dcc.Textarea(
                    id='ks-data-input-area-1',
                    placeholder="예:\n10.2\n11.5\n9.8\n...",
                    style={'width': '100%', 'height': '150px', 'fontSize': '16px', 'borderRadius': '5px', 'border': '1px solid #ccc'}
                ),
            ], style={'flex': '1', 'marginRight': '10px'}), # Use flexbox for side-by-side

            html.Div([
                html.Label("데이터 2 (Sample 2):", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                dcc.Textarea(
                    id='ks-data-input-area-2',
                    placeholder="예:\n9.5\n10.8\n10.0\n...",
                    style={'width': '100%', 'height': '150px', 'fontSize': '16px', 'borderRadius': '5px', 'border': '1px solid #ccc'}
                ),
            ], style={'flex': '1', 'marginLeft': '10px'}), # Use flexbox for side-by-side
        ], style={'display': 'flex', 'marginBottom': '15px'}), # Flex container for textareas

        # 신뢰수준 및 가설 선택을 위한 Div
        html.Div([
            html.Div([
                html.Label("유의 수준 (α):", style={'fontWeight': 'bold'}),
                dcc.Slider(
                    id='ks-significance-level-input',
                    min=0.01,
                    max=0.1,
                    step=0.01,
                    value=0.05, # 기본값 0.05
                    marks={
                        val: f'{val:.2f}' for val in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
                    },
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%', 'verticalAlign': 'top'}),
            
            html.Div([
                html.Label("귀무 가설 (Null Hypothesis):", style={'marginRight': '10px', 'fontWeight': 'bold'}),
                dcc.RadioItems(
                    id='ks-hypothesis-selection-radio',
                    options=[
                        {'label': '분포 1 = 분포 2', 'value': 'two-sided'},
                        {'label': '분포 1 ≤ 분포 2 (stochastically)', 'value': 'greater'},
                        {'label': '분포 1 ≥ 분포 2 (stochastically)', 'value': 'less'}
                    ],
                    value='two-sided', # Default selection
                    inline=True,
                    style={'marginTop': '5px'}
                )
            ], style={'display': 'block', 'marginTop': '10px'}), # Use 'block' for better layout
        ], style={'marginTop': '15px', 'marginBottom': '10px'}),

        # 분석 실행 버튼
        html.Div([
            html.Label("데이터 전처리:", style={'marginRight': '10px', 'fontWeight': 'bold'}),
            dcc.RadioItems(
                id='ks-data-preprocessing-radio',
                options=[
                    {'label': '원본 데이터 사용', 'value': 'raw'},
                    {'label': '평균 제거 (위치 보정)', 'value': 'remove_mean'}
                ],
                value='raw',
                inline=True
            )
        ], style={'display': 'block', 'marginTop': '15px'}),

        html.Button(
            '분석 실행',
            id='ks-run-analysis-button',
            n_clicks=0,
            style={
                'width': '100%',
                'padding': '10px',
                'fontSize': '18px',
                'fontWeight': 'bold',
                'backgroundColor': '#007BFF',
                'color': 'white',
                'border': 'none',
                'borderRadius': '5px',
                'marginTop': '10px',
                'cursor': 'pointer'
            }
        ),

        # 로딩 스피너
        dcc.Loading(
            id="loading-spinner",
            type="circle",
            children=[
                # 통계 결과 출력 영역
                html.Div(id='ks-stats-results-output', style={'marginTop': '20px', 'padding': '15px', 'backgroundColor': '#f9f9f9', 'borderRadius': '5px'}),
                # 그래프 출력 영역
                dcc.Graph(id='ks-plot-graph')
            ]
        ) # This closes the dcc.Loading component
    ]
)

# 콜백: 버튼 클릭 시 그래프 및 통계 업데이트
@dash.callback(
    [Output('ks-plot-graph', 'figure'),
     Output('ks-stats-results-output', 'children')],
    [Input('ks-run-analysis-button', 'n_clicks')],
    [State('ks-data-input-area-1', 'value'),
     State('ks-data-input-area-2', 'value'),
     State('ks-hypothesis-selection-radio', 'value'),
     State('ks-significance-level-input', 'value'),
     State('ks-data-preprocessing-radio', 'value')]
)
def update_ks_test_analysis(n_clicks, data_string_1, data_string_2, alternative_hypothesis, alpha_level, preprocessing):
    # 버튼이 클릭되지 않았거나 입력이 없으면 빈 상태 반환
    if n_clicks == 0 or not (data_string_1 or data_string_2):
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="두 데이터를 입력하고 '분석 실행' 버튼을 클릭하세요.",
            xaxis={'visible': False},
            yaxis={'visible': False},
            annotations=[{
                'text': '결과가 여기에 표시됩니다.',
                'xref': 'paper',
                'yref': 'paper',
                'showarrow': False,
                'font': {'size': 16, 'color': '#888'}
            }]
        )
        return empty_fig, "분석 대기 중..."

    # 입력된 텍스트 데이터를 파싱
    def parse_data(data_str, sample_name):
        try:
            lines = data_str.strip().split('\n')
            data = [float(line.strip()) for line in lines if line.strip()]
            if not data:
                raise ValueError(f"{sample_name} 데이터가 없습니다.")
            return np.array(data)
        except ValueError as e:
            raise ValueError(f"오류: {sample_name} 데이터를 파싱할 수 없습니다. 숫자만 입력하세요. ({e})")

    try:
        data1 = parse_data(data_string_1, "Sample 1")
        data2 = parse_data(data_string_2, "Sample 2")
    except ValueError as e:
        error_fig = go.Figure()
        error_fig.update_layout(title="입력 오류", annotations=[{'text': str(e), 'showarrow': False}])
        return error_fig, html.Div(str(e), style={'color': 'red', 'fontWeight': 'bold'})

    # 최소 1개 이상의 데이터 필요
    if len(data1) < 1 or len(data2) < 1:
        error_fig = go.Figure()
        error_fig.update_layout(title="데이터 부족", annotations=[{'text': '분석을 위해 각 샘플에 1개 이상의 데이터가 필요합니다.', 'showarrow': False}])
        return error_fig, html.Div(f"오류: 각 샘플에 1개 이상의 데이터가 필요합니다. (데이터 1: {len(data1)}개, 데이터 2: {len(data2)}개)", style={'color': 'red', 'fontWeight': 'bold'})

    # 유의수준 파싱
    try:
        if alpha_level is None:
            alpha_level = 0.05 # 기본값
        alpha_level_float = float(alpha_level)
        if not (0 < alpha_level_float < 1):
            raise ValueError("유의수준은 0과 1 사이여야 합니다.")
    except (ValueError, TypeError) as e:
        error_fig = go.Figure()
        error_fig.update_layout(title="입력 오류", annotations=[{'text': f'유의 수준 값 오류: {e}', 'showarrow': False}])
        return error_fig, html.Div(f"오류: 유의 수준이 올바르지 않습니다. (입력값: {alpha_level})", style={'color': 'red', 'fontWeight': 'bold'})

    # 데이터 전처리
    data1_processed = np.copy(data1)
    data2_processed = np.copy(data2)
    preprocessing_text = "원본 데이터 사용"
    if preprocessing == 'remove_mean':
        data1_processed -= np.mean(data1)
        data2_processed -= np.mean(data2)
        preprocessing_text = "평균 제거 (위치 보정)"

    # --- 2-Sample KS-Test ---
    ks_stat, ks_p_value = stats.ks_2samp(data1_processed, data2_processed, alternative=alternative_hypothesis)

    # --- 통계 결과 Div 생성 ---
    stats_div_children = [
        html.H4("요약 통계량 (Sample 1)", style={'borderBottom': '1px solid #ddd', 'paddingBottom': '5px'}),
        html.Table([
            html.Tr([html.Td("데이터 개수 (N)"), html.Td(f"{len(data1)}", style={'textAlign': 'right', 'fontWeight': 'bold'})]),
            html.Tr([html.Td("평균 (Mean)"), html.Td(f"{np.mean(data1):.4f}", style={'textAlign': 'right', 'fontWeight': 'bold'})]),
            html.Tr([html.Td("표준편차 (StDev)"), html.Td(f"{np.std(data1, ddof=1):.4f}", style={'textAlign': 'right', 'fontWeight': 'bold'})]),
            html.Tr([html.Td("최소값 (Min)"), html.Td(f"{np.min(data1):.4f}", style={'textAlign': 'right', 'fontWeight': 'bold'})]),
            html.Tr([html.Td("최대값 (Max)"), html.Td(f"{np.max(data1):.4f}", style={'textAlign': 'right', 'fontWeight': 'bold'})]),
        ], style={'width': '300px'}),

        html.H4("요약 통계량 (Sample 2)", style={'marginTop': '20px', 'borderBottom': '1px solid #ddd', 'paddingBottom': '5px'}),
        html.Table([
            html.Tr([html.Td("데이터 개수 (N)"), html.Td(f"{len(data2)}", style={'textAlign': 'right', 'fontWeight': 'bold'})]),
            html.Tr([html.Td("평균 (Mean)"), html.Td(f"{np.mean(data2):.4f}", style={'textAlign': 'right', 'fontWeight': 'bold'})]),
            html.Tr([html.Td("표준편차 (StDev)"), html.Td(f"{np.std(data2, ddof=1):.4f}", style={'textAlign': 'right', 'fontWeight': 'bold'})]),
            html.Tr([html.Td("최소값 (Min)"), html.Td(f"{np.min(data2):.4f}", style={'textAlign': 'right', 'fontWeight': 'bold'})]),
            html.Tr([html.Td("최대값 (Max)"), html.Td(f"{np.max(data2):.4f}", style={'textAlign': 'right', 'fontWeight': 'bold'})]),
        ], style={'width': '300px'}),

        html.H4("가설 검정 결과", style={'marginTop': '20px', 'borderBottom': '1px solid #ddd', 'paddingBottom': '5px'}),
    ]

    # 가설 정의
    null_hypo_text = {
        'two-sided': "귀무가설 (H₀): 두 샘플의 분포는 동일하다.",
        'greater': "귀무가설 (H₀): 샘플1의 분포가 샘플2의 분포보다 작거나 같다 (stochastically).",
        'less': "귀무가설 (H₀): 샘플1의 분포가 샘플2의 분포보다 크거나 같다 (stochastically)."
    }
    alt_hypo_text = {
        'two-sided': "대립가설 (H₁): 두 샘플의 분포는 동일하지 않다.",
        'greater': "대립가설 (H₁): 샘플1의 분포가 샘플2의 분포보다 크다 (stochastically).",
        'less': "대립가설 (H₁): 샘플1의 분포가 샘플2의 분포보다 작다 (stochastically)."
    }
    stats_div_children.append(html.P(null_hypo_text[alternative_hypothesis]))
    stats_div_children.append(html.P(alt_hypo_text[alternative_hypothesis]))
    stats_div_children.append(html.P(f"유의수준 (α): {alpha_level_float}"))
    stats_div_children.append(html.P(f"데이터 전처리: {preprocessing_text}", style={'fontStyle': 'italic', 'color': '#555'}))

    # 가설 검정 결론
    if ks_p_value < alpha_level_float:
        conclusion_text = f"결론: P-값 ({ks_p_value:.4f}) < α ({alpha_level_float}) 이므로, 귀무가설을 기각합니다. (두 분포는 통계적으로 유의미하게 다릅니다)"
        conclusion_style = {'color': 'red', 'fontWeight': 'bold', 'marginTop': '10px'}
    else:
        conclusion_text = f"결론: P-값 ({ks_p_value:.4f}) ≥ α ({alpha_level_float}) 이므로, 귀무가설을 기각할 수 없습니다. (두 분포가 다르다는 통계적 증거가 부족합니다)"
        conclusion_style = {'color': 'green', 'fontWeight': 'bold', 'marginTop': '10px'}
    
    stats_div_children.append(html.Table([
        html.Tr([html.Td("KS Statistic"), html.Td(f"{ks_stat:.4f}", style={'textAlign': 'right', 'fontWeight': 'bold'})]),
        html.Tr([html.Td("P-Value"), html.Td(f"{ks_p_value:.4f}", style={'textAlign': 'right', 'fontWeight': 'bold'})]),
    ], style={'width': '300px', 'marginTop': '10px'}))

    stats_div_children.append(html.P(conclusion_text, style=conclusion_style))

    stats_div = html.Div(stats_div_children)

    # --- Plotly 그래프 생성 ---
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "누적 분포 함수 (ECDF)", "데이터 비교 박스 플롯 (Box Plot)"
        )
    )

    # ECDF Plot
    def ecdf(data):
        x = np.sort(data)
        y = np.arange(1, len(x) + 1) / len(x)
        return x, y

    x1_ecdf, y1_ecdf = ecdf(data1_processed)
    x2_ecdf, y2_ecdf = ecdf(data2_processed)

    fig.add_trace(go.Scatter(x=x1_ecdf, y=y1_ecdf, mode='lines', name='Sample 1 ECDF', line=dict(color='#007BFF', shape='hv')), row=1, col=1)
    fig.add_trace(go.Scatter(x=x2_ecdf, y=y2_ecdf, mode='lines', name='Sample 2 ECDF', line=dict(color='#28A745', shape='hv')), row=1, col=1)
    fig.update_yaxes(title_text="누적 확률", row=1, col=1)
    fig.update_xaxes(title_text="데이터 값", row=1, col=1)

    # Box Plot for comparing two samples
    df_box = pd.DataFrame({
        'Value': np.concatenate([data1_processed, data2_processed]),
        'Sample': ['Sample 1'] * len(data1_processed) + ['Sample 2'] * len(data2_processed)
    })
    box_fig = px.box(df_box, x='Sample', y='Value', color='Sample',
                     color_discrete_map={'Sample 1': '#007BFF', 'Sample 2': '#28A745'})
    for trace in box_fig.data:
        fig.add_trace(trace, row=1, col=2)
    fig.update_yaxes(title_text="데이터 값", row=1, col=2)
    fig.update_xaxes(title_text="", row=1, col=2)

    fig.update_layout(
        title_text=f"<b>2-표본 KS 검정 결과</b><br>(P-value: {ks_p_value:.4f})",
        title_x=0.5,
        showlegend=True,
        height=500,
        bargap=0.01
    )

    return fig, stats_div