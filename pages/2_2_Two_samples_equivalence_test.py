import dash
from dash import dcc, html, register_page
from dash.dependencies import Input, Output, State, ALL
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats
from scipy.optimize import brentq
import plotly.express as px
import pandas as pd
import os
import tempfile
import re
from html2image import Html2Image

register_page(__name__)

# minitab url : https://support.minitab.com/ko-kr/minitab/help-and-how-to/statistics/power-and-sample-size/how-to/equivalence-tests/power-and-sample-size-for-2-sample-equivalence-test/interpret-the-results/all-statistics-and-graphs/

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

# --- 통계 함수: TOST 검정력 계산 ---
def calculate_tost_power(n, delta, sd, alpha, lower, upper):
    if n < 2: return 0
    se = sd * np.sqrt(2/n)
    df = 2*n - 2
    
    # 1. 비중심 t-분포 파라미터
    ncp1 = (delta - lower) / se
    ncp2 = (delta - upper) / se
    
    # 2. 수치적 안정성을 위한 로직
    # df가 매우 크거나 nct 계산에 문제가 생길 가능성이 있을 때 정규분포 근사 사용
    if df > 500:
        z_crit = stats.norm.ppf(1 - alpha)
        # Power approx: Phi((U-delta)/se - z) - Phi(z - (delta-L)/se)
        power = stats.norm.cdf((upper - delta)/se - z_crit) - stats.norm.cdf(z_crit - (delta - lower)/se)
    else:
        try:
            t_crit = stats.t.ppf(1 - alpha, df)
            p_lower = stats.nct.cdf(t_crit, df, ncp1)
            p_upper = stats.nct.cdf(-t_crit, df, ncp2)
            power = p_upper - p_lower
            
            # NaN이나 수치 오류 발생 시 정규분포로 백업
            if np.isnan(power) or power < -1e-5:
                z_crit = stats.norm.ppf(1 - alpha)
                power = stats.norm.cdf((upper - delta)/se - z_crit) - stats.norm.cdf(z_crit - (delta - lower)/se)
        except:
            z_crit = stats.norm.ppf(1 - alpha)
            power = stats.norm.cdf((upper - delta)/se - z_crit) - stats.norm.cdf(z_crit - (delta - lower)/se)
            
    return max(0, power)

def find_required_sample_size(target_power, delta, sd, alpha, lower, upper):
    # n=2부터 시작해서 power >= target_power인 최소 n 찾기
    n = 2
    max_iter = 15
    found_upper = False
    for _ in range(max_iter):
        if calculate_tost_power(n, delta, sd, alpha, lower, upper) >= target_power:
            found_upper = True
            break
        n *= 2
    
    if not found_upper: return 5000 
    
    low = n // 2 if n > 2 else 2
    high = n
    ans = n
    while low <= high:
        mid = (low + high) // 2
        if calculate_tost_power(mid, delta, sd, alpha, lower, upper) >= target_power:
            ans = mid
            high = mid - 1
        else:
            low = mid + 1
    return ans

# 앱 레이아웃 정의
layout = html.Div(
    style={'fontFamily': 'Arial, sans-serif', 'maxWidth': '1000px', 'margin': 'auto', 'padding': '20px'},
    children=[
        dcc.Store(id='equiv-stats-download-store'),
        dcc.Download(id='equiv-stats-download-component'),
        dcc.Store(id='equiv-analysis-results-store'),

        html.H1("2표본 동등성 검정", style={'textAlign': 'center', 'color': '#333'}),

        html.P("두 개의 데이터 열(column)을 복사한 후, 아래 텍스트 상자에 각각 붙여넣으세요. 동등성 한계값(하한, 상한)을 입력하세요.",
               style={'textAlign': 'center', 'color': '#555'}),

        html.Div([
            html.Div([
                html.Label("데이터 1 (Sample 1):", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                dcc.Textarea(id='equiv-data-input-area-1', placeholder="예:\n10.2\n11.5\n9.8\n...",
                            style={'width': '100%', 'height': '150px', 'fontSize': '16px', 'borderRadius': '5px', 'border': '1px solid #ccc'}),
            ], style={'flex': '1', 'marginRight': '10px'}),
            html.Div([
                html.Label("데이터 2 (Sample 2):", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                dcc.Textarea(id='equiv-data-input-area-2', placeholder="예:\n9.5\n10.8\n10.0\n...",
                            style={'width': '100%', 'height': '150px', 'fontSize': '16px', 'borderRadius': '5px', 'border': '1px solid #ccc'}),
            ], style={'flex': '1', 'marginLeft': '10px'}),
        ], style={'display': 'flex', 'marginBottom': '15px'}),

        html.Div([
            html.Div([
                html.Label("동등성 한계 (Equivalence Limits):", style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': '10px'}),
                html.Div([
                    html.Label("하한 (Lower):", style={'marginRight': '5px'}),
                    dcc.Input(id='equiv-lower-limit', type='number', value=-1.0, style={'width': '80px', 'marginRight': '20px'}),
                    html.Label("상한 (Upper):", style={'marginRight': '5px'}),
                    dcc.Input(id='equiv-upper-limit', type='number', value=1.0, style={'width': '80px'}),
                ], style={'display': 'flex', 'alignItems': 'center'})
            ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '4%'}),
            html.Div([
                html.Label("유의 수준 (α):", style={'fontWeight': 'bold'}),
                dcc.Slider(id='equiv-significance-level-input', min=0.01, max=0.1, step=0.01, value=0.05,
                          marks={val: f'{val:.2f}' for val in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]},
                          tooltip={"placement": "bottom", "always_visible": True})
            ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        ], style={'marginTop': '15px', 'marginBottom': '20px'}),

        html.Button('분석 실행', id='equiv-run-analysis-button', n_clicks=0,
                    style={'width': '100%', 'padding': '10px', 'fontSize': '18px', 'fontWeight': 'bold',
                           'backgroundColor': '#28A745', 'color': 'white', 'border': 'none', 'borderRadius': '5px',
                           'marginTop': '10px', 'cursor': 'pointer'}),

        dcc.Loading(id="equiv-loading-spinner", type="circle", children=[
            html.Div([html.Button("📷 통계 결과 이미지로 다운로드", id="equiv-stats-download-btn",
                                style={'marginBottom': '10px', 'padding': '8px 15px', 'backgroundColor': '#6C757D',
                                       'color': 'white', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer', 'display': 'none'})]),
            html.Div(id='equiv-stats-results-output', style={'marginTop': '20px', 'padding': '15px', 'backgroundColor': '#f9f9f9', 'borderRadius': '5px'}),
            dcc.Graph(id='equiv-normality-plot-graph'),
            
            html.Hr(style={'marginTop': '40px', 'marginBottom': '40px'}),
            html.Div([
                html.H2("검정력 및 표본 크기 (Power and Sample Size)", style={'textAlign': 'center', 'color': '#333'}),
                html.Div([
                    html.Div([
                        html.Label("표준편차 (SD):", style={'fontWeight': 'bold'}),
                        dcc.Input(id='power-sd-input', type='number', placeholder="분석 결과에서 자동 입력됨", style={'width': '100%', 'padding': '5px'}),
                    ], style={'flex': '1', 'marginRight': '20px'}),
                        html.Div([
                            html.Label("목표 검정력 (Power):", style={'fontWeight': 'bold'}),
                            dcc.Input(id='power-target-input', type='number', value=0.9, step=0.01, min=0.1, max=0.99, style={'width': '100%', 'padding': '5px'}),
                        ], style={'flex': '1', 'marginRight': '20px'}),
                        
                        html.Div([
                            html.Label("비교할 차이값 개수:", style={'fontWeight': 'bold'}),
                            dcc.Dropdown(id='power-diff-count-dropdown', options=[{'label': f'{i}개', 'value': i} for i in range(1, 6)],
                                    value=3, clearable=False, style={'width': '100%'})
                        ], style={'flex': '1'}),
                ], style={'display': 'flex', 'marginBottom': '20px'}),
                
                html.Label("차이값 입력 (Differences):", style={'fontWeight': 'bold', 'marginBottom': '10px', 'display': 'block'}),
                html.Div(id='power-diff-inputs-container', style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '10px', 'marginBottom': '20px'}),
                
                html.Button('표본 크기 계산', id='power-calc-button', n_clicks=0,
                            style={'width': '100%', 'padding': '10px', 'fontSize': '16px', 'fontWeight': 'bold',
                                   'backgroundColor': '#007BFF', 'color': 'white', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'}),
                
                html.Div(id='power-calc-results-output', style={'marginTop': '20px'}),
                dcc.Graph(id='power-sample-size-graph', style={'marginTop': '20px'}),
            ], id='power-analysis-section', style={'padding': '20px', 'border': '1px solid #ddd', 'borderRadius': '10px', 'backgroundColor': '#fff'})
        ])
    ]
)

@dash.callback(Output('power-sd-input', 'value'), Input('equiv-analysis-results-store', 'data'))
def update_power_sd(analysis_data):
    if analysis_data and 'pooled_sd' in analysis_data: return round(analysis_data['pooled_sd'], 4)
    return None

@dash.callback(Output('power-diff-inputs-container', 'children'), Input('power-diff-count-dropdown', 'value'))
def render_diff_inputs(count):
    return [html.Div([html.Label(f"차이 {i+1}:", style={'fontSize': '13px'}),
                      dcc.Input(id={'type': 'power-diff-input', 'index': i}, type='number', value=0.5 if i==0 else 0.7 if i==1 else 0.9 if i==2 else 0, style={'width': '80px', 'padding': '5px'})]) for i in range(count)]

@dash.callback(
    [Output('equiv-normality-plot-graph', 'figure'), Output('equiv-stats-results-output', 'children'),
     Output('equiv-stats-download-store', 'data'), Output('equiv-stats-download-btn', 'style'), Output('equiv-analysis-results-store', 'data')],
    [Input('equiv-run-analysis-button', 'n_clicks')],
    [State('equiv-data-input-area-1', 'value'), State('equiv-data-input-area-2', 'value'),
     State('equiv-lower-limit', 'value'), State('equiv-upper-limit', 'value'), State('equiv-significance-level-input', 'value')]
)
def update_equivalence_analysis(n_clicks, data_str1, data_str2, lower_limit, upper_limit, alpha):
    if n_clicks == 0 or not (data_str1 and data_str2):
        fig = go.Figure(); fig.update_layout(title="데이터를 입력하고 '분석 실행' 버튼을 클릭하세요.")
        return fig, "분석 대기 중...", None, {'display': 'none'}, None

    if lower_limit is None or upper_limit is None or lower_limit >= upper_limit:
        return go.Figure(), html.Div("오류: 동등성 하한은 상한보다 작아야 합니다.", style={'color': 'red'}), None, {'display': 'none'}, None

    def parse_data(s): return np.array([float(l.strip()) for l in s.strip().split('\n') if l.strip()])
    try: d1, d2 = parse_data(data_str1), parse_data(data_str2)
    except: return go.Figure(), html.Div("오류: 숫자 데이터만 입력하세요.", style={'color': 'red'}), None, {'display': 'none'}, None

    if len(d1) < 2 or len(d2) < 2: return go.Figure(), html.Div("오류: 각 샘플에 2개 이상의 데이터가 필요합니다.", style={'color': 'red'}), None, {'display': 'none'}, None

    n1, n2 = len(d1), len(d2)
    m1, m2 = np.mean(d1), np.mean(d2)
    v1, v2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
    sd1, sd2 = np.sqrt(v1), np.sqrt(v2)
    sem1, sem2 = sd1 / np.sqrt(n1), sd2 / np.sqrt(n2)
    pooled_sd = np.sqrt(((n1-1)*v1 + (n2-1)*v2) / (n1+n2-2))
    mean_diff = m1 - m2
    se_diff = np.sqrt(v1/n1 + v2/n2)
    df = (v1/n1 + v2/n2)**2 / ((v1/n1)**2/(n1-1) + (v2/n2)**2/(n2-1))
    
    t1, t2 = (mean_diff - lower_limit) / se_diff, (mean_diff - upper_limit) / se_diff
    p1, p2 = 1 - stats.t.cdf(t1, df), stats.t.cdf(t2, df)
    overall_p = max(p1, p2)
    t_crit = stats.t.ppf(1 - alpha, df)
    ci_lower, ci_upper = mean_diff - t_crit * se_diff, mean_diff + t_crit * se_diff
    is_equivalent = (ci_lower > lower_limit) and (ci_upper < upper_limit)

    stats_div = [
        html.H4("요약 통계량", style={'borderBottom': '1px solid #ddd'}),
        html.Table([html.Thead(html.Tr([html.Th("샘플"), html.Th("N"), html.Th("평균"), html.Th("표준편차"), html.Th("SE Mean")])),
                    html.Tbody([html.Tr([html.Td("Sample 1"), html.Td(n1), html.Td(f"{m1:.4f}"), html.Td(f"{sd1:.4f}"), html.Td(f"{sem1:.4f}")]),
                                html.Tr([html.Td("Sample 2"), html.Td(n2), html.Td(f"{m2:.4f}"), html.Td(f"{sd2:.4f}"), html.Td(f"{sem2:.4f}")])])], className="table"),
        html.H4("차이 및 동등성 구간", style={'borderBottom': '1px solid #ddd'}),
        html.Table([html.Tr([html.Td("평균 차이"), html.Td(f"{mean_diff:.4f}", style={'fontWeight': 'bold'})]),
                    html.Tr([html.Td("SE Diff"), html.Td(f"{se_diff:.4f}")]),
                    html.Tr([html.Td(f"{(1-2*alpha)*100:.0f}% CI"), html.Td(f"({ci_lower:.4f}, {ci_upper:.4f})", style={'fontWeight': 'bold'})]),
                    html.Tr([html.Td("동등성 구간"), html.Td(f"({lower_limit:.4f}, {upper_limit:.4f})", style={'color': '#28A745'})])], style={'width': '400px'}),
        html.H4("가설 검정 (TOST)", style={'borderBottom': '1px solid #ddd'}),
        html.Table([html.Thead(html.Tr([html.Th("검정"), html.Th("귀무 가설"), html.Th("DF"), html.Th("T-Value"), html.Th("P-Value")])),
                    html.Tbody([html.Tr([html.Td("Test 1"), html.Td(f"차이 ≤ {lower_limit}"), html.Td(f"{df:.2f}"), html.Td(f"{t1:.3f}"), html.Td(f"{p1:.4f}")]),
                                html.Tr([html.Td("Test 2"), html.Td(f"차이 ≥ {upper_limit}"), html.Td(f"{df:.2f}"), html.Td(f"{t2:.3f}"), html.Td(f"{p2:.4f}")])])], className="table"),
        html.P(f"결합된 P-Value: {overall_p:.4f}", style={'fontWeight': 'bold'}),
        html.H4("결론", style={'borderBottom': '1px solid #ddd'}),
        html.P(f"결론: 유의수준 {alpha}에서 {'동등성을 주장할 수 있습니다.' if is_equivalent else '동등성을 주장할 수 없습니다.'}", style={'color': 'green' if is_equivalent else 'red', 'fontWeight': 'bold'})
    ]
    
    fig = make_subplots(rows=3, cols=2, subplot_titles=("Sample 1 히스토그램", "Sample 1 정규 확률도", "Sample 2 히스토그램", "Sample 2 정규 확률도", "두 샘플 박스 플롯", "동등성 도표"), vertical_spacing=0.1)
    def add_std_plots(f, data, r, name, col):
        f.add_trace(go.Histogram(x=data, name=name, marker_color=col, opacity=0.7), row=r, col=1)
        res = stats.probplot(data, dist="norm")
        f.add_trace(go.Scatter(x=res[0][0], y=res[0][1], mode='markers', marker_color=col, showlegend=False), row=r, col=2)
        f.add_trace(go.Scatter(x=res[0][0], y=res[1][0]*res[0][0]+res[1][1], mode='lines', line=dict(color='red', dash='dash'), showlegend=False), row=r, col=2)
    add_std_plots(fig, d1, 1, "Sample 1", "#007BFF"); add_std_plots(fig, d2, 2, "Sample 2", "#28A745")
    df_box = pd.DataFrame({'Val': np.concatenate([d1, d2]), 'Grp': ['S1']*n1 + ['S2']*n2})
    for t in px.box(df_box, x='Grp', y='Val', color='Grp', color_discrete_map={'S1':'#007BFF', 'S2':'#28A745'}).data: fig.add_trace(t, row=3, col=1)
    fig.add_trace(go.Scatter(x=[mean_diff], y=["평균 차이"], error_x=dict(type='data', symmetric=False, array=[ci_upper-mean_diff], arrayminus=[mean_diff-ci_lower], thickness=3, color='#007BFF'), mode='markers+text', text=[f" 차이: {mean_diff:.4f}"], textposition="top center", marker=dict(size=12, color='#007BFF')), row=3, col=2)
    fig.add_vline(x=lower_limit, line_dash="dash", line_color="red", row=3, col=2); fig.add_vline(x=upper_limit, line_dash="dash", line_color="red", row=3, col=2); fig.add_vline(x=0, line_color="black", row=3, col=2)
    fig.update_layout(height=1000, title_text="2표본 동등성 분석 결과", showlegend=False)

    return fig, html.Div(stats_div), html.Div(stats_div), {'display': 'inline-block'}, {'pooled_sd': pooled_sd, 'lower_limit': lower_limit, 'upper_limit': upper_limit, 'alpha': alpha}

@dash.callback(
    [Output('power-sample-size-graph', 'figure'), Output('power-calc-results-output', 'children')],
    [Input('power-calc-button', 'n_clicks')],
    [State('power-sd-input', 'value'), State({'type': 'power-diff-input', 'index': ALL}, 'value'),
     State('power-target-input', 'value'),
     State('equiv-lower-limit', 'value'), State('equiv-upper-limit', 'value'), State('equiv-significance-level-input', 'value'), State('equiv-analysis-results-store', 'data')]
)
def run_power_calculation(n_clicks, input_sd, differences, target_power, lower, upper, alpha, analysis_data):
    if n_clicks == 0: return go.Figure(), ""
    sd = input_sd if input_sd else (analysis_data['pooled_sd'] if analysis_data else None)
    if sd is None or lower is None or upper is None or target_power is None: return go.Figure(), html.Div("오류: 필요한 정보가 부족합니다.", style={'color': 'red'})
    
    results = []
    for diff in differences:
        if diff is None: continue
        n_req = find_required_sample_size(target_power, diff, sd, alpha, lower, upper)
        actual_power = calculate_tost_power(n_req, diff, sd, alpha, lower, upper)
        results.append({'Diff': diff, 'N': n_req, 'Target': target_power, 'Actual': actual_power})
    
    if not results: return go.Figure(), html.Div("오류: 유효한 차이값을 입력하세요.", style={'color': 'red'})
    
    # Method Summary
    method_div = html.Div([
        html.H3("방법", style={'color': '#0056b3', 'borderBottom': '2px solid #0056b3', 'paddingBottom': '5px'}),
        html.P(f"차이에 대한 검정력: 검정 평균 - 기준 평균"),
        html.P(f"귀무 가설: 차이 ≤ {lower} 또는 차이 ≥ {upper}"),
        html.P(f"대립 가설: {lower} < 차이 < {upper}"),
        html.P(f"α 수준: {alpha}"),
        html.P(f"가정된 표준 편차: {sd}"),
    ], style={'backgroundColor': '#f0f4f8', 'padding': '15px', 'borderRadius': '5px', 'marginBottom': '20px'})

    # Results Table
    table = html.Div([
        html.H3("결과", style={'color': '#0056b3', 'borderBottom': '2px solid #0056b3', 'paddingBottom': '5px'}),
        html.Table([
            html.Thead(html.Tr([html.Th("차이"), html.Th("표본 크기"), html.Th("목표 검정력"), html.Th("실제 검정력")])),
            html.Tbody([html.Tr([html.Td(f"{r['Diff']}"), html.Td(f"{r['N']}"), html.Td(f"{r['Target']}"), html.Td(f"{r['Actual']:.6f}")]) for r in results])
        ], className="table", style={'width': '100%'}),
        html.P("각 그룹에 대한 표본 크기입니다.", style={'fontStyle': 'italic', 'fontSize': '12px', 'marginTop': '10px'})
    ], style={'padding': '15px'})

    # Power Curve Graph
    fig = go.Figure()
    x_range = np.linspace(lower * 1.1, upper * 1.1, 500) # 해상도 증가
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd']
    
    for i, res in enumerate(results):
        y_vals = [calculate_tost_power(res['N'], x, sd, alpha, lower, upper) for x in x_range]
        fig.add_trace(go.Scatter(x=x_range, y=y_vals, mode='lines', name=f"N={res['N']}", line=dict(color=colors[i % len(colors)])))
        fig.add_trace(go.Scatter(x=[res['Diff']], y=[res['Actual']], mode='markers', marker=dict(color='black', size=8), showlegend=False))

    fig.add_vline(x=lower, line_dash="dash", line_color="red", annotation_text="동등성 하한")
    fig.add_vline(x=upper, line_dash="dash", line_color="red", annotation_text="동등성 상한")
    
    fig.update_layout(title="2-표본 동등성 검정에 대한 검정력 곡선", xaxis_title="차이", yaxis_title="검정력", yaxis=dict(range=[0, 1.05]), hovermode="x unified")
    
    return fig, html.Div([method_div, table])

@dash.callback(Output("equiv-stats-download-component", "data"), Input("equiv-stats-download-btn", "n_clicks"), State('equiv-stats-download-store', 'data'), prevent_initial_call=True)
def download_equiv_image(n_clicks, component_dict):
    if not component_dict: return None
    inner_html = component_to_html(component_dict)
    html_content = f"<html><head><meta charset='utf-8'><link rel='stylesheet' href='https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css'><style>body {{ padding: 30px; width: 700px; }}</style></head><body>{inner_html}</body></html>"
    hti = Html2Image()
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_name = "equiv_report.png"; hti.screenshot(html_str=html_content, save_as=output_name, size=(750, 1200))
        if os.path.exists(output_name):
            with open(output_name, "rb") as f: content = f.read()
            os.remove(output_name); return dcc.send_bytes(content, "equivalence_test_report.png")
    return None
