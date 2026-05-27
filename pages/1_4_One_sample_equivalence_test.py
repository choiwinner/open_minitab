import dash
from dash import dcc, html, register_page
from dash.dependencies import Input, Output, State, ALL
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats
import plotly.express as px
import pandas as pd
import os
import tempfile
import re
from html2image import Html2Image

register_page(__name__)

# --- 헬퍼 함수: Dash 컴포넌트를 HTML 문자열로 변환 (디자인 100% 일치용) ---
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

# --- 통계 함수: 1-Sample TOST 검정력 계산 ---
def calculate_one_sample_tost_power(n, delta, sd, alpha, lower, upper):
    if n < 2 or sd <= 0: return 0
    se = sd / np.sqrt(n)
    df = n - 1
    
    # 비중심 t-분포 파라미터 (ncp)
    ncp1 = (delta - lower) / se
    ncp2 = (delta - upper) / se
    
    if df > 500:
        z_crit = stats.norm.ppf(1 - alpha)
        power = stats.norm.cdf((upper - delta)/se - z_crit) - stats.norm.cdf(z_crit - (delta - lower)/se)
    else:
        try:
            t_crit = stats.t.ppf(1 - alpha, df)
            p_lower = stats.nct.cdf(t_crit, df, ncp1)
            p_upper = stats.nct.cdf(-t_crit, df, ncp2)
            power = p_upper - p_lower
            
            if np.isnan(power) or power < -1e-5:
                z_crit = stats.norm.ppf(1 - alpha)
                power = stats.norm.cdf((upper - delta)/se - z_crit) - stats.norm.cdf(z_crit - (delta - lower)/se)
        except:
            z_crit = stats.norm.ppf(1 - alpha)
            power = stats.norm.cdf((upper - delta)/se - z_crit) - stats.norm.cdf(z_crit - (delta - lower)/se)
            
    return max(0, power)

def find_required_sample_size_one_sample(target_power, delta, sd, alpha, lower, upper):
    n = 2
    max_iter = 15
    found_upper = False
    for _ in range(max_iter):
        if calculate_one_sample_tost_power(n, delta, sd, alpha, lower, upper) >= target_power:
            found_upper = True
            break
        n *= 2
    
    if not found_upper: return 5000 
    
    low = n // 2 if n > 2 else 2
    high = n
    ans = n
    while low <= high:
        mid = (low + high) // 2
        if calculate_one_sample_tost_power(mid, delta, sd, alpha, lower, upper) >= target_power:
            ans = mid
            high = mid - 1
        else:
            low = mid + 1
    return ans

# --- 앱 레이아웃 정의 ---
layout = html.Div(
    style={'fontFamily': 'Arial, sans-serif', 'maxWidth': '1000px', 'margin': 'auto', 'padding': '20px'},
    children=[
        dcc.Store(id='one-equiv-stats-download-store'),
        dcc.Download(id='one-equiv-stats-download-component'),
        dcc.Store(id='one-equiv-analysis-results-store'),

        html.H1("1표본 동등성 검정 (1-Sample Equivalence Test)", style={'textAlign': 'center', 'color': '#333'}),

        html.P("단일 집단의 평균이 목표값과 실질적으로 동등한 수준인지 검증합니다. 엑셀 등에서 데이터 열을 복사하여 입력해 주세요.",
               style={'textAlign': 'center', 'color': '#555', 'marginBottom': '30px'}),

        # 1. 데이터 입력 (1_One_sample_t-test.py 와 동일한 UI 형태)
        html.Div([
            html.Label("표본 데이터 입력 (Excel 복사/붙여넣기 지원):", style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': '5px'}),
            html.P("한 줄에 숫자 하나씩 입력하세요. 공백 라인은 무시됩니다.", style={'fontSize': '12px', 'color': '#666', 'marginTop': '0px'}),
            dcc.Textarea(
                id='one-equiv-data-input-area',
                placeholder="예:\n150.2\n149.8\n151.1\n148.9\n150.5\n...",
                style={'width': '100%', 'height': '200px', 'fontSize': '16px', 'borderRadius': '5px', 'border': '1px solid #ccc'}
            )
        ], style={'marginBottom': '25px'}),

        # 2. 파라미터 입력 영역
        html.Div([
            # 목표값 입력
            html.Div([
                html.Label("목표값 (Target):", style={'fontWeight': 'bold', 'display': 'block'}),
                html.P("비교의 기준이 되는 이상적인 값입니다.", style={'fontSize': '12px', 'color': '#666', 'margin': '2px 0'}),
                dcc.Input(id='one-equiv-target-value', type='number', value=150.0, style={'width': '100%', 'padding': '8px', 'borderRadius': '5px', 'border': '1px solid #ccc'})
            ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginRight': '5%'}),

            # 유의 수준 설정
            html.Div([
                html.Label("유의 수준 (α):", style={'fontWeight': 'bold', 'display': 'block'}),
                html.P("동등하지 않은데 동등하다고 오판할 최대 허용 확률입니다. (기본 0.05)", style={'fontSize': '12px', 'color': '#666', 'margin': '2px 0'}),
                dcc.Slider(
                    id='one-equiv-alpha-slider',
                    min=0.01,
                    max=0.1,
                    step=0.01,
                    value=0.05,
                    marks={
                        val: f'{val:.2f}' for val in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
                    },
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], style={'width': '65%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        ], style={'marginBottom': '25px'}),

        # 3. 동등성 한계 입력 영역
        html.Div([
            html.Label("동등성 한계(Equivalence Limits) 설정 방식:", style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': '5px'}),
            html.P("표본 평균과 목표값의 차이가 '실질적으로 차이가 없다'고 인정할 수 있는 허용 범위의 상/하한선입니다.", style={'fontSize': '12px', 'color': '#666', 'marginTop': '0px'}),
            
            dcc.RadioItems(
                id='one-equiv-limit-mode',
                options=[
                    {'label': ' 직접 수치로 입력 (목표값과의 차이 기준)', 'value': 'value'},
                    {'label': ' 목표값 대비 비율 (%)로 입력', 'value': 'percent'}
                ],
                value='value',
                inline=True,
                style={'marginBottom': '15px'}
            ),
            
            html.Div([
                html.Div([
                    html.Label(id='one-equiv-lower-label', children="하한차이 (Lower Limit):", style={'fontWeight': 'bold'}),
                    dcc.Input(id='one-equiv-lower-limit', type='number', value=-5.0, style={'width': '100%', 'padding': '8px', 'borderRadius': '5px', 'border': '1px solid #ccc'})
                ], style={'flex': '1', 'marginRight': '20px'}),
                html.Div([
                    html.Label(id='one-equiv-upper-label', children="상한차이 (Upper Limit):", style={'fontWeight': 'bold'}),
                    dcc.Input(id='one-equiv-upper-limit', type='number', value=5.0, style={'width': '100%', 'padding': '8px', 'borderRadius': '5px', 'border': '1px solid #ccc'})
                ], style={'flex': '1'}),
            ], style={'display': 'flex'})
        ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '8px', 'border': '1px solid #e9ecef', 'marginBottom': '30px'}),

        # 분석 실행 버튼
        html.Button(
            '분석 실행',
            id='one-equiv-run-button',
            n_clicks=0,
            style={'width': '100%', 'padding': '12px', 'fontSize': '18px', 'fontWeight': 'bold',
                   'backgroundColor': '#28A745', 'color': 'white', 'border': 'none', 'borderRadius': '5px',
                   'cursor': 'pointer', 'marginBottom': '20px'}
        ),

        # 로딩 및 결과 출력
        dcc.Loading(
            id="one-equiv-loading",
            type="circle",
            children=[
                html.Div([
                    html.Button(
                        "📷 통계 결과 이미지로 다운로드",
                        id="one-equiv-stats-download-btn",
                        style={'marginBottom': '10px', 'padding': '8px 15px', 'backgroundColor': '#6C757D',
                               'color': 'white', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer', 'display': 'none'}
                    )
                ]),
                html.Div(id='one-equiv-results-output', style={'marginTop': '20px', 'padding': '20px', 'backgroundColor': '#f9f9f9', 'borderRadius': '5px', 'border': '1px solid #eee'}),
                dcc.Graph(id='one-equiv-plot-graph', style={'marginTop': '20px'}),
                
                # 검정력 분석 영역 (2_2_Two_samples_equivalence_test.py 패턴 적용)
                html.Hr(style={'marginTop': '45px', 'marginBottom': '40px'}),
                html.Div([
                    html.H2("검정력 및 표본 크기 (Power and Sample Size)", style={'textAlign': 'center', 'color': '#333'}),
                    html.P("동등성을 입증하기 위해 필요한 표본 크기나 특정 표본 수에서의 검정력을 모의 계산합니다.", style={'textAlign': 'center', 'color': '#666', 'fontSize': '14px', 'marginBottom': '25px'}),
                    
                    html.Div([
                        html.Div([
                            html.Label("단일 집단 표준편차 (SD):", style={'fontWeight': 'bold'}),
                            dcc.Input(id='one-power-sd-input', type='number', placeholder="분석 시 자동 완성됨", style={'width': '100%', 'padding': '6px', 'borderRadius': '4px', 'border': '1px solid #ccc'}),
                        ], style={'flex': '1', 'marginRight': '20px'}),
                        html.Div([
                            html.Label("목표 검정력 (Target Power):", style={'fontWeight': 'bold'}),
                            dcc.Input(id='one-power-target-input', type='number', value=0.9, step=0.01, min=0.1, max=0.99, style={'width': '100%', 'padding': '6px', 'borderRadius': '4px', 'border': '1px solid #ccc'}),
                        ], style={'flex': '1', 'marginRight': '20px'}),
                        html.Div([
                            html.Label("비교할 차이값(Difference) 개수:", style={'fontWeight': 'bold'}),
                            dcc.Dropdown(id='one-power-diff-count-dropdown', options=[{'label': f'{i}개', 'value': i} for i in range(1, 6)],
                                         value=3, clearable=False, style={'width': '100%'})
                        ], style={'flex': '1'}),
                    ], style={'display': 'flex', 'marginBottom': '20px'}),
                    
                    html.Label("가설 평균과 실제 평균의 참 차이값 입력 (Differences):", style={'fontWeight': 'bold', 'marginBottom': '10px', 'display': 'block'}),
                    html.Div(id='one-power-diff-inputs-container', style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '10px', 'marginBottom': '20px'}),
                    
                    html.Button(
                        '표본 크기 계산',
                        id='one-power-calc-button',
                        n_clicks=0,
                        style={'width': '100%', 'padding': '10px', 'fontSize': '16px', 'fontWeight': 'bold',
                               'backgroundColor': '#007BFF', 'color': 'white', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'}
                    ),
                    
                    html.Div(id='one-power-calc-results-output', style={'marginTop': '25px'}),
                    dcc.Graph(id='one-power-sample-size-graph', style={'marginTop': '20px'}),
                ], id='one-power-analysis-section', style={'padding': '25px', 'border': '1px solid #ddd', 'borderRadius': '10px', 'backgroundColor': '#fff'})
            ]
        )
    ]
)


# --- 콜백: 동등성 한계 모드 선택에 따른 레이벨 문구 변경 ---
@dash.callback(
    [Output('one-equiv-lower-label', 'children'),
     Output('one-equiv-upper-label', 'children')],
    Input('one-equiv-limit-mode', 'value')
)
def update_limit_labels(mode):
    if mode == 'percent':
        return "하한 비율 (%):", "상한 비율 (%):"
    return "하한차이 (Lower Limit):", "상한차이 (Upper Limit):"


# --- 콜백: 분석 결과에서 구한 SD 값을 검정력 분석 입력폼에 동기화 ---
@dash.callback(
    Output('one-power-sd-input', 'value'),
    Input('one-equiv-analysis-results-store', 'data')
)
def update_power_sd_val(analysis_data):
    if analysis_data and 'sd' in analysis_data:
        return round(analysis_data['sd'], 4)
    return None


# --- 콜백: 차이값 입력 박스 렌더링 ---
@dash.callback(
    Output('one-power-diff-inputs-container', 'children'),
    Input('one-power-diff-count-dropdown', 'value')
)
def render_diff_inputs_one_sample(count):
    # 기본 디폴트 값 제공
    defaults = [0.0, 0.5, 1.0, 1.5, 2.0]
    return [
        html.Div([
            html.Label(f"차이 {i+1}:", style={'fontSize': '13px'}),
            dcc.Input(
                id={'type': 'one-power-diff-input', 'index': i},
                type='number',
                value=defaults[i] if i < len(defaults) else 0.0,
                style={'width': '90px', 'padding': '6px', 'borderRadius': '4px', 'border': '1px solid #ccc'}
            )
        ]) for i in range(count)
    ]


# --- 콜백: 1표본 동등성 검정 실행 ---
@dash.callback(
    [Output('one-equiv-plot-graph', 'figure'),
     Output('one-equiv-results-output', 'children'),
     Output('one-equiv-stats-download-store', 'data'),
     Output('one-equiv-stats-download-btn', 'style'),
     Output('one-equiv-analysis-results-store', 'data')],
    [Input('one-equiv-run-button', 'n_clicks')],
    [State('one-equiv-data-input-area', 'value'),
     State('one-equiv-target-value', 'value'),
     State('one-equiv-alpha-slider', 'value'),
     State('one-equiv-limit-mode', 'value'),
     State('one-equiv-lower-limit', 'value'),
     State('one-equiv-upper-limit', 'value')]
)
def run_one_sample_equivalence_test(n_clicks, data_str, target_val, alpha, limit_mode, lower_input, upper_input):
    if n_clicks == 0 or not data_str:
        fig = go.Figure()
        fig.update_layout(title="데이터를 입력하고 '분석 실행' 버튼을 클릭하세요.")
        return fig, "분석 대기 중...", None, {'display': 'none'}, None

    # 1. 파라미터 유효성 검사
    if target_val is None:
        return go.Figure(), html.Div("오류: 목표값을 입력해 주세요.", style={'color': 'red', 'fontWeight': 'bold'}), None, {'display': 'none'}, None
    if lower_input is None or upper_input is None:
        return go.Figure(), html.Div("오류: 동등성 상/하한 한계값을 입력해 주세요.", style={'color': 'red', 'fontWeight': 'bold'}), None, {'display': 'none'}, None
    if lower_input >= upper_input:
        return go.Figure(), html.Div("오류: 동등성 하한은 상한보다 작아야 합니다.", style={'color': 'red', 'fontWeight': 'bold'}), None, {'display': 'none'}, None
    if limit_mode == 'percent' and target_val == 0:
        return go.Figure(), html.Div("오류: 목표값이 0일 경우 비율(%) 동등성 한계를 계산할 수 없습니다. 수치 직접 입력 방식을 사용하세요.", style={'color': 'red', 'fontWeight': 'bold'}), None, {'display': 'none'}, None

    # 2. 데이터 파싱
    try:
        data = np.array([float(line.strip()) for line in data_str.strip().split('\n') if line.strip()])
    except:
        return go.Figure(), html.Div("오류: 숫자 데이터만 한 줄에 하나씩 올바르게 입력해 주세요.", style={'color': 'red', 'fontWeight': 'bold'}), None, {'display': 'none'}, None

    n = len(data)
    if n < 2:
        return go.Figure(), html.Div(f"오류: 분석을 위해서는 최소 2개 이상의 표본 데이터가 필요합니다. (현재 {n}개)", style={'color': 'red', 'fontWeight': 'bold'}), None, {'display': 'none'}, None

    # 3. 정규성 검정 (Shapiro-Wilk)
    shapiro_p = None
    shapiro_conclusion = "표본 크기가 너무 작아(N < 3) 정규성 검정을 진행할 수 없습니다."
    shapiro_style = {'color': 'gray', 'fontStyle': 'italic'}
    if n >= 3:
        _, shapiro_p = stats.shapiro(data)
        if shapiro_p > alpha:
            shapiro_conclusion = f"정규성 검정 결과 P-Value({shapiro_p:.4f}) > α({alpha}) 이므로, 데이터의 정규성 가정을 만족합니다."
            shapiro_style = {'color': 'green', 'fontWeight': 'bold'}
        else:
            # 정규성을 만족하지 못할 경우 경고 및 종료
            err_fig = go.Figure()
            err_fig.update_layout(title="정규성 가정이 충족되지 않음", annotations=[{'text': f'정규성 가정 실패 (Shapiro-Wilk p={shapiro_p:.4f})', 'showarrow': False}])
            err_msg = html.Div([
                html.Span("오류: 데이터의 정규성 가정을 만족하지 않습니다.", style={'fontWeight': 'bold'}),
                html.P(f"Shapiro-Wilk P-Value: {shapiro_p:.4f} (설정된 유의수준 α={alpha} 이하)"),
                html.P("동등성 검정은 표본의 정규성을 가정하고 동작합니다. 데이터 분포를 확인해 주세요.")
            ], style={'color': 'red'})
            return err_fig, err_msg, None, {'display': 'none'}, None

    # 4. 통계 계산 진행
    mean = np.mean(data)
    sd = np.std(data, ddof=1)
    se = sd / np.sqrt(n)
    df = n - 1
    mean_diff = mean - target_val

    # 동등성 한계 실측값 변환 (차이값으로 환산)
    if limit_mode == 'percent':
        lower_limit = target_val * (lower_input / 100.0)
        upper_limit = target_val * (upper_input / 100.0)
    else:
        lower_limit = lower_input
        upper_limit = upper_input

    # 5. TOST (Two One-Sided Tests) 가설 검정
    # Test 1: H01: 차이 <= 하한  vs H11: 차이 > 하한
    # Test 2: H02: 차이 >= 상한  vs H12: 차이 < 상한
    t1 = (mean_diff - lower_limit) / se
    t2 = (mean_diff - upper_limit) / se

    p1 = 1.0 - stats.t.cdf(t1, df)
    p2 = stats.t.cdf(t2, df)
    overall_p = max(p1, p2)

    # 6. 신뢰구간 (1 - 2*alpha) * 100 % CI 계산
    t_crit = stats.t.ppf(1.0 - alpha, df)
    ci_lower = mean_diff - t_crit * se
    ci_upper = mean_diff + t_crit * se

    # 미니탭 스타일 동등성 CI 표기용 값 변환 (양수 차이 시 하한 0, 음수 차이 시 상한 0)
    if mean_diff >= 0:
        ci_lower_display = 0.0
        ci_upper_display = ci_upper
    else:
        ci_lower_display = ci_lower
        ci_upper_display = 0.0

    # 동등성 여부 판정
    is_equivalent = (ci_lower > lower_limit) and (ci_upper < upper_limit)

    # 7. 귀무가설/대립가설 한글 풀이 텍스트 구성
    hypo_section = html.Div([
        html.H5("💡 1표본 동등성 검정 가설 설명", style={'marginTop': '15px', 'fontWeight': 'bold'}),
        html.Ul([
            html.Li([
                html.B("귀무 가설 (H₀): "),
                html.Span(f"표본 평균과 목표값의 차이가 동등성 한계를 벗어납니다. (차이 ≤ {lower_limit:.4f} 또는 차이 ≥ {upper_limit:.4f})"),
                html.Br(),
                html.Small("👉 즉, 표본 집단의 수준이 목표값과 실질적으로 동등하지(유사하지) 않습니다.", style={'color': '#777'})
            ], style={'marginBottom': '8px'}),
            html.Li([
                html.B("대립 가설 (H₁): "),
                html.Span(f"표본 평균과 목표값의 차이가 동등성 한계 내에 존재합니다. ({lower_limit:.4f} < 차이 < {upper_limit:.4f})"),
                html.Br(),
                html.Small("👉 즉, 표본 집단의 수준이 목표값과 오차 범위 내에서 실질적으로 동등합니다.", style={'color': '#777'})
            ])
        ])
    ], style={'padding': '15px', 'backgroundColor': '#e9ecef', 'borderRadius': '6px', 'marginBottom': '20px'})

    # 8. 최종 결과 해석 텍스트 구성
    ci_percent = (1.0 - alpha) * 100.0
    if is_equivalent:
        conclusion_color = 'green'
        conclusion_main = f"동등성 입증 성공 (유의수준 {alpha:.2f}에서 표본 평균은 목표값과 동등합니다.)"
        conclusion_desc = (
            f"평균 차이({mean_diff:.4f})에 대한 {ci_percent:.0f}% 동등성 신뢰구간 [{ci_lower_display:.4f}, {ci_upper_display:.4f}]이 "
            f"동등성 한계 범위 [{lower_limit:.4f}, {upper_limit:.4f}] 내에 완전히 포함됩니다. "
            f"따라서 귀무가설을 기각하고, 통계적으로 유의미하게 동등성이 증명되었습니다."
        )
    else:
        conclusion_color = 'red'
        conclusion_main = f"동등성 입증 실패 (유의수준 {alpha:.2f}에서 동등성을 주장할 수 없습니다.)"
        conclusion_desc = (
            f"평균 차이({mean_diff:.4f})에 대한 {ci_percent:.0f}% 동등성 신뢰구간 [{ci_lower_display:.4f}, {ci_upper_display:.4f}]이 "
            f"동등성 한계 범위 [{lower_limit:.4f}, {upper_limit:.4f}]를 벗어나거나 완전히 포함되지 않습니다. "
            f"따라서 귀무가설을 기각할 수 없으며, 두 값이 동등하다고 판단할 충분한 통계적 근거가 없습니다."
        )

    interpretation_section = html.Div([
        html.H4("🔍 분석 결과 해석 및 결론", style={'borderBottom': '2px solid #ddd', 'paddingBottom': '5px', 'marginTop': '20px'}),
        html.P(conclusion_main, style={'color': conclusion_color, 'fontWeight': 'bold', 'fontSize': '16px'}),
        html.P(conclusion_desc, style={'lineHeight': '1.5', 'color': '#333'}),
    ], style={'padding': '15px', 'backgroundColor': '#fff', 'border': f'1px solid {conclusion_color}', 'borderRadius': '6px', 'marginTop': '20px'})

    # 9. UI 표시용 결과 리포트 HTML
    stats_div = html.Div([
        html.H3("1표본 동등성 검정 결과 요약", style={'textAlign': 'center', 'color': '#0056b3', 'marginBottom': '20px'}),
        
        # 기본 요약 표
        html.H4("요약 통계량", style={'borderBottom': '1px solid #ddd', 'paddingBottom': '5px'}),
        html.Table([
            html.Thead(html.Tr([
                html.Th("N (표본수)", style={'textAlign': 'center'}),
                html.Th("평균 (Mean)", style={'textAlign': 'center'}),
                html.Th("표준편차 (StDev)", style={'textAlign': 'center'}),
                html.Th("표준오차 (SE Mean)", style={'textAlign': 'center'})
            ])),
            html.Tbody([
                html.Tr([
                    html.Td(f"{n}", style={'textAlign': 'center', 'fontWeight': 'bold'}),
                    html.Td(f"{mean:.4f}", style={'textAlign': 'center'}),
                    html.Td(f"{sd:.4f}", style={'textAlign': 'center'}),
                    html.Td(f"{se:.4f}", style={'textAlign': 'center'})
                ])
            ])
        ], className="table table-bordered", style={'width': '100%', 'marginBottom': '20px'}),

        # 차이 및 신뢰구간 표
        html.H4("평균 차이 및 동등성 구간", style={'borderBottom': '1px solid #ddd', 'paddingBottom': '5px'}),
        html.Table([
            html.Tr([html.Td("비교 대상 목표값 (Target)"), html.Td(f"{target_val:.4f}", style={'fontWeight': 'bold', 'textAlign': 'right'})]),
            html.Tr([html.Td("평균 차이 (표본평균 - 목표값)"), html.Td(f"{mean_diff:.4f}", style={'fontWeight': 'bold', 'textAlign': 'right'})]),
            html.Tr([html.Td(f"{ci_percent:.0f}% 동등성 CI"), html.Td(f"({ci_lower_display:.4f}, {ci_upper_display:.4f})", style={'fontWeight': 'bold', 'color': conclusion_color, 'textAlign': 'right'})]),
            html.Tr([
                html.Td("허용 오차 동등성 한계 (Limits)"), 
                html.Td(f"({lower_limit:.4f}, {upper_limit:.4f})" + (f" (목표값의 {lower_input}% ~ {upper_input}%)" if limit_mode == 'percent' else ""), 
                        style={'fontWeight': 'bold', 'color': '#0056b3', 'textAlign': 'right'})
            ])
        ], className="table table-striped", style={'width': '100%', 'marginBottom': '20px'}),

        # TOST 상세 검정 결과 표
        html.H4("TOST(단측 검정 결합) 결과", style={'borderBottom': '1px solid #ddd', 'paddingBottom': '5px'}),
        html.Table([
            html.Thead(html.Tr([
                html.Th("귀무가설 (H₀)", style={'textAlign': 'left'}),
                html.Th("자유도 (DF)", style={'textAlign': 'center'}),
                html.Th("T-Value", style={'textAlign': 'center'}),
                html.Th("P-Value", style={'textAlign': 'center'})
            ])),
            html.Tbody([
                html.Tr([
                    html.Td(f"차이 ≤ {lower_limit:.4f} (하한선 미달)"),
                    html.Td(f"{df}", style={'textAlign': 'center'}),
                    html.Td(f"{t1:.3f}", style={'textAlign': 'center'}),
                    html.Td(f"{p1:.4f}", style={'textAlign': 'center'})
                ]),
                html.Tr([
                    html.Td(f"차이 ≥ {upper_limit:.4f} (상한선 초과)"),
                    html.Td(f"{df}", style={'textAlign': 'center'}),
                    html.Td(f"{t2:.3f}", style={'textAlign': 'center'}),
                    html.Td(f"{p2:.4f}", style={'textAlign': 'center'})
                ])
            ])
        ], className="table table-bordered", style={'width': '100%'}),
        html.P(f"결합된 종합 P-Value: {overall_p:.4f}", style={'fontWeight': 'bold', 'fontSize': '15px', 'marginTop': '10px'}),

        # 정규성 검정
        html.H4("정규성 검정 (Shapiro-Wilk)", style={'marginTop': '20px', 'borderBottom': '1px solid #ddd', 'paddingBottom': '5px'}),
        html.P(f"P-Value: {shapiro_p:.4f}" if shapiro_p is not None else "N < 3"),
        html.P(shapiro_conclusion, style=shapiro_style),

        hypo_section,
        interpretation_section
    ])

    # 10. 시각화 그래프 구성 (2행 2열 Subplots)
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "히스토그램 (Normal Fit)", "정규 확률도 (Q-Q Plot)",
            "표본 데이터 박스 플롯 (Box Plot)", f"동등성 도표 ({ci_percent:.0f}% 동등성 CI vs 한계선)"
        ),
        vertical_spacing=0.18,
        horizontal_spacing=0.12
    )

    # (1, 1) 히스토그램
    fig.add_trace(go.Histogram(x=data, name='표본 분포', histnorm='probability density', marker_color='#007BFF', opacity=0.7), row=1, col=1)
    if sd > 0:
        x_curve = np.linspace(data.min() - sd, data.max() + sd, 150)
        y_curve = stats.norm.pdf(x_curve, mean, sd)
        fig.add_trace(go.Scatter(x=x_curve, y=y_curve, mode='lines', name='정규분포 곡선', line=dict(color='red', width=2)), row=1, col=1)
    fig.update_yaxes(title_text="밀도 (Density)", row=1, col=1)
    fig.update_xaxes(title_text="측정값", row=1, col=1)

    # (1, 2) Q-Q Plot
    if n >= 2:
        res = stats.probplot(data, dist="norm")
        fig.add_trace(go.Scatter(x=res[0][0], y=res[0][1], mode='markers', name='데이터 포인트', marker_color='#007BFF', showlegend=False), row=1, col=2)
        fit_y = res[1][0] * res[0][0] + res[1][1]
        fig.add_trace(go.Scatter(x=res[0][0], y=fit_y, mode='lines', name='적합선', line=dict(color='red', dash='dash', width=2), showlegend=False), row=1, col=2)
    fig.update_yaxes(title_text="정렬된 데이터 값", row=1, col=2)
    fig.update_xaxes(title_text="이론적 분위수", row=1, col=2)

    # (2, 1) Box Plot
    fig.add_trace(go.Box(y=data, name="표본", marker_color='#007BFF', boxpoints='outliers'), row=2, col=1)
    # 가설 목표값 수평선 추가
    fig.add_hline(y=target_val, line_dash="dash", line_color="green", annotation_text="목표치", row=2, col=1)
    fig.update_yaxes(title_text="측정값", row=2, col=1)

    # (2, 2) 동등성 도표 (평균 차이 및 신뢰구간, 그리고 한계선 표시)
    # y축은 범주 단일 항목, x축은 평균 차이값
    fig.add_trace(go.Scatter(
        x=[mean_diff],
        y=["평균 차이"],
        error_x=dict(
            type='data',
            symmetric=False,
            array=[ci_upper_display - mean_diff],
            arrayminus=[mean_diff - ci_lower_display],
            thickness=3,
            color=conclusion_color
        ),
        mode='markers',
        marker=dict(size=14, color=conclusion_color, symbol='circle'),
        name=f'{ci_percent:.0f}% 동등성 CI'
    ), row=2, col=2)
    
    # 상한/하한 수직 점선 추가
    fig.add_vline(x=lower_limit, line_dash="dash", line_color="red", row=2, col=2,
                  annotation=dict(text=f"하한 ({lower_limit:.4f})", font=dict(color="red", size=10), textangle=-90, yshift=20))
    fig.add_vline(x=upper_limit, line_dash="dash", line_color="red", row=2, col=2,
                  annotation=dict(text=f"상한 ({upper_limit:.4f})", font=dict(color="red", size=10), textangle=-90, yshift=20))
    
    # 차이 = 0 기준선 (목표값과 완벽 일치하는 점)
    fig.add_vline(x=0, line_color="black", row=2, col=2,
                  annotation=dict(text="기준 (차이=0)", font=dict(color="black", size=9), textangle=-90, yshift=-20))

    # X축 범위 계산 (여유있게 설정)
    all_vals = [lower_limit, upper_limit, ci_lower_display, ci_upper_display, 0.0, mean_diff]
    min_val = min(all_vals)
    max_val = max(all_vals)
    span = max_val - min_val
    margin = span * 0.25 if span > 0 else 1.0
    x_range = [min_val - margin, max_val + margin]

    fig.update_xaxes(title_text="평균 차이 (표본평균 - 목표치)", range=x_range, row=2, col=2)
    fig.update_yaxes(showticklabels=False, row=2, col=2)

    fig.update_layout(height=800, title_text="1표본 동등성 분석 차트", showlegend=False)

    # 저장된 결과 스토어 데이터 준비 (다운로드 기능용)
    results_store = {
        'n': n,
        'mean': mean,
        'sd': sd,
        'se': se,
        'target_val': target_val,
        'mean_diff': mean_diff,
        'ci_percent': ci_percent,
        'ci_lower': ci_lower_display,
        'ci_upper': ci_upper_display,
        'lower_limit': lower_limit,
        'upper_limit': upper_limit,
        't1': t1,
        'p1': p1,
        't2': t2,
        'p2': p2,
        'overall_p': overall_p,
        'is_equivalent': is_equivalent,
        'shapiro_p': shapiro_p,
        'shapiro_conclusion': shapiro_conclusion,
        'limit_mode': limit_mode,
        'lower_input': lower_input,
        'upper_input': upper_input,
        'alpha': alpha
    }

    # 다운로드 버튼 보이기 처리
    btn_style = {'marginBottom': '10px', 'padding': '8px 15px', 'backgroundColor': '#6C757D', 'color': 'white', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer', 'display': 'inline-block'}

    return fig, stats_div, stats_div, btn_style, results_store


# --- 콜백: 검정력 및 표본 크기 계산 작동 ---
@dash.callback(
    [Output('one-power-sample-size-graph', 'figure'),
     Output('one-power-calc-results-output', 'children')],
    [Input('one-power-calc-button', 'n_clicks')],
    [State('one-power-sd-input', 'value'),
     State({'type': 'one-power-diff-input', 'index': ALL}, 'value'),
     State('one-power-target-input', 'value'),
     State('one-equiv-target-value', 'value'),
     State('one-equiv-alpha-slider', 'value'),
     State('one-equiv-limit-mode', 'value'),
     State('one-equiv-lower-limit', 'value'),
     State('one-equiv-upper-limit', 'value'),
     State('one-equiv-analysis-results-store', 'data')]
)
def run_one_sample_power_calculation(n_clicks, input_sd, differences, target_power, target_val, alpha, limit_mode, lower_input, upper_input, analysis_data):
    if n_clicks == 0:
        return go.Figure(), ""

    sd = input_sd if input_sd is not None else (analysis_data['sd'] if analysis_data else None)
    if sd is None or sd <= 0:
        return go.Figure(), html.Div("오류: 유효한 표준편차(SD) 값을 입력해 주세요.", style={'color': 'red', 'fontWeight': 'bold'})
    if lower_input is None or upper_input is None:
        return go.Figure(), html.Div("오류: 동등성 한계값을 먼저 지정해 주세요.", style={'color': 'red', 'fontWeight': 'bold'})
    if target_power is None or not (0.1 <= target_power <= 0.99):
        return go.Figure(), html.Div("오류: 목표 검정력은 0.1에서 0.99 사이의 값이어야 합니다.", style={'color': 'red', 'fontWeight': 'bold'})

    # 동등성 범위 한계 환산
    if limit_mode == 'percent':
        lower_limit = target_val * (lower_input / 100.0)
        upper_limit = target_val * (upper_input / 100.0)
    else:
        lower_limit = lower_input
        upper_limit = upper_input

    results = []
    for diff in differences:
        if diff is None: continue
        # 하한 ~ 상한 영역 밖의 차이값에 대해서는 검정력을 계산하지 않음
        if diff <= lower_limit or diff >= upper_limit:
            continue
        n_req = find_required_sample_size_one_sample(target_power, diff, sd, alpha, lower_limit, upper_limit)
        actual_power = calculate_one_sample_tost_power(n_req, diff, sd, alpha, lower_limit, upper_limit)
        results.append({'Diff': diff, 'N': n_req, 'Target': target_power, 'Actual': actual_power})

    if not results:
        return go.Figure(), html.Div("오류: 동등성 한계 범위 내의 유효한 차이값(Difference)을 1개 이상 입력해 주세요.", style={'color': 'red', 'fontWeight': 'bold'})

    # 방법 설명 카드
    method_div = html.Div([
        html.H4("검정력 계산 조건 정보", style={'color': '#0056b3', 'borderBottom': '2px solid #0056b3', 'paddingBottom': '5px'}),
        html.P(f"기반 가설: 1표본 TOST (Two One-Sided Tests) 동등성 검정"),
        html.P(f"귀무가설(H₀): 차이 ≤ {lower_limit:.4f} 또는 차이 ≥ {upper_limit:.4f} (동등하지 않음)"),
        html.P(f"대립가설(H₁): {lower_limit:.4f} < 차이 < {upper_limit:.4f} (동등함)"),
        html.P(f"알파(α) 유의수준: {alpha}  |  설정된 가정 표준편차(SD): {sd:.4f}"),
    ], style={'backgroundColor': '#f0f4f8', 'padding': '15px', 'borderRadius': '6px', 'marginBottom': '20px'})

    # 결과 테이블
    table = html.Div([
        html.H4("산출 결과 및 권장 표본 크기", style={'color': '#0056b3', 'borderBottom': '2px solid #0056b3', 'paddingBottom': '5px'}),
        html.Table([
            html.Thead(html.Tr([
                html.Th("평균 차이 (Difference)", style={'textAlign': 'center'}),
                html.Th("필요 표본 크기 (N)", style={'textAlign': 'center', 'fontWeight': 'bold'}),
                html.Th("목표 검정력", style={'textAlign': 'center'}),
                html.Th("산출된 실제 검정력 (Power)", style={'textAlign': 'center'})
            ])),
            html.Tbody([
                html.Tr([
                    html.Td(f"{r['Diff']:.4f}", style={'textAlign': 'center'}),
                    html.Td(f"{r['N']}", style={'textAlign': 'center', 'fontWeight': 'bold', 'color': '#0056b3'}),
                    html.Td(f"{r['Target']}", style={'textAlign': 'center'}),
                    html.Td(f"{r['Actual']:.6f}", style={'textAlign': 'center'})
                ]) for r in results
            ])
        ], className="table table-bordered table-striped", style={'width': '100%'}),
        html.P("※ 필요한 최소한의 표본 크기(N)를 그룹당 수가 아닌 단일 집단 전체 수치로 출력하였습니다.", style={'fontStyle': 'italic', 'fontSize': '12px', 'marginTop': '10px', 'color': '#666'})
    ], style={'padding': '10px 0'})

    # 검정력 곡선 시각화
    fig = go.Figure()
    x_range = np.linspace(lower_limit * 1.1, upper_limit * 1.1, 400)
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd']

    for i, res in enumerate(results):
        y_vals = [calculate_one_sample_tost_power(res['N'], x, sd, alpha, lower_limit, upper_limit) for x in x_range]
        fig.add_trace(go.Scatter(x=x_range, y=y_vals, mode='lines', name=f"N={res['N']} (Diff={res['Diff']:.2f})", line=dict(color=colors[i % len(colors)], width=2.5)))
        # 실제 계산점 표시
        fig.add_trace(go.Scatter(x=[res['Diff']], y=[res['Actual']], mode='markers', marker=dict(color='black', size=10, symbol='x'), showlegend=False))

    fig.add_vline(x=lower_limit, line_dash="dash", line_color="red", annotation_text="하한선")
    fig.add_vline(x=upper_limit, line_dash="dash", line_color="red", annotation_text="상한선")
    fig.add_vline(x=0, line_color="gray", line_width=1)

    fig.update_layout(
        title="1표본 동등성 검정에 대한 검정력 곡선 (Power Curve)",
        xaxis_title="가설 평균과 실제 평균의 참 차이",
        yaxis_title="검정력 (Power)",
        yaxis=dict(range=[0, 1.05]),
        hovermode="x unified"
    )

    return fig, html.Div([method_div, table])


# --- 콜백: 결과 보고서 이미지 다운로드 ---
@dash.callback(
    Output("one-equiv-stats-download-component", "data"),
    Input("one-equiv-stats-download-btn", "n_clicks"),
    State('one-equiv-stats-download-store', 'data'),
    prevent_initial_call=True
)
def download_one_equiv_report_image(n_clicks, component_dict):
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
        output_name = "one_equiv_report.png"
        # 스냅샷의 크기를 넉넉하게 설정하여 텍스트가 잘리는 것을 방지
        hti.screenshot(html_str=html_content, save_as=output_name, size=(820, 1500))
        
        if os.path.exists(output_name):
            with open(output_name, "rb") as f:
                content = f.read()
            os.remove(output_name)
            return dcc.send_bytes(content, "1_sample_equivalence_test_report.png")
    return None
