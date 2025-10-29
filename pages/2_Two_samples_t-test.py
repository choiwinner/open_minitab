import dash
from dash import dcc, html, register_page
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats
import plotly.express as px
import pandas as pd

register_page(__name__)

# 앱 레이아웃 정의
layout = html.Div(
    style={'fontFamily': 'Arial, sans-serif', 'maxWidth': '1000px', 'margin': 'auto', 'padding': '20px'},
    children=[
        html.H1(
            "2표본 t-검정 및 정규성 검정", # Updated title
            style={'textAlign': 'center', 'color': '#333'}
        ),

        html.P(
            "두 개의 데이터 열(column)을 복사한 후, 아래 텍스트 상자에 각각 붙여넣으세요. (숫자만, 한 줄에 하나씩)", # Updated instruction
            style={'textAlign': 'center', 'color': '#555'}
        ),

        # 데이터 입력을 위한 텍스트 영역 (두 개)
        html.Div([
            html.Div([
                html.Label("데이터 1 (Sample 1):", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                dcc.Textarea(
                    id='data-input-area-1',
                    placeholder="예:\n10.2\n11.5\n9.8\n...",
                    style={'width': '100%', 'height': '150px', 'fontSize': '16px', 'borderRadius': '5px', 'border': '1px solid #ccc'}
                ),
            ], style={'flex': '1', 'marginRight': '10px'}), # Use flexbox for side-by-side

            html.Div([
                html.Label("데이터 2 (Sample 2):", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                dcc.Textarea(
                    id='data-input-area-2',
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
                    id='significance-level-input',
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
                    id='hypothesis-selection-radio',
                    options=[
                        {'label': '평균 1 = 평균 2', 'value': 'equal'},
                        {'label': '평균 1 ≤ 평균 2', 'value': 'less_or_equal'},
                        {'label': '평균 1 ≥ 평균 2', 'value': 'greater_or_equal'}
                    ],
                    value='equal', # Default selection
                    inline=True,
                    style={'marginTop': '5px'}
                )
            ], style={'display': 'block', 'marginTop': '10px'}), # Use 'block' for better layout

            html.Div([
                html.Label("t-검정 유형:", style={'marginRight': '10px', 'fontWeight': 'bold'}),
                dcc.RadioItems(
                    id='ttest-type-radio',
                    options=[
                        {'label': "Welch's t-test (등분산 가정 안함)", 'value': 'welch'},
                        {'label': 'Pooled t-test (등분산 가정함)', 'value': 'pooled'},
                        {'label': 'F-검정 후 자동 선택', 'value': 'auto'}
                    ],
                    value='pooled', # Default to pooled test
                    inline=True
                )
            ], style={'display': 'block', 'marginTop': '15px'})
        ], style={'marginTop': '15px', 'marginBottom': '10px'}),

        # 분석 실행 버튼
        html.Button(
            '분석 실행',
            id='run-analysis-button',
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
                html.Div(id='stats-results-output', style={'marginTop': '20px', 'padding': '15px', 'backgroundColor': '#f9f9f9', 'borderRadius': '5px'}),
                # 그래프 출력 영역
                dcc.Graph(id='normality-plot-graph')
            ]
        ) # This closes the dcc.Loading component
    ]
)

# 콜백: 버튼 클릭 시 그래프 및 통계 업데이트
@dash.callback(
    [Output('normality-plot-graph', 'figure'),
     Output('stats-results-output', 'children')],
    [Input('run-analysis-button', 'n_clicks')],
    [State('data-input-area-1', 'value'),
     State('data-input-area-2', 'value'),
     State('hypothesis-selection-radio', 'value'),
     State('significance-level-input', 'value'),
     State('ttest-type-radio', 'value')]
)
def update_two_sample_analysis(n_clicks, data_string_1, data_string_2, null_hypothesis, alpha_level, ttest_type):
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

    # t-검정 및 신뢰구간을 위해 최소 2개 이상의 데이터 필요
    if len(data1) < 2 or len(data2) < 2:
        error_fig = go.Figure()
        error_fig.update_layout(title="데이터 부족", annotations=[{'text': '각 샘플에 대해 2개 이상의 데이터가 필요합니다.', 'showarrow': False}])
        return error_fig, html.Div(f"오류: 각 샘플에 2개 이상의 데이터가 필요합니다. (Sample 1: {len(data1)}개, Sample 2: {len(data2)}개)", style={'color': 'red', 'fontWeight': 'bold'})

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

    conf_level_float = (1 - alpha_level_float) * 100


    # --- 통계량 계산 ---
    results = {}
    for i, data in enumerate([data1, data2]):
        sample_name = f"Sample {i+1}"
        n = len(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1) # 표본 표준편차
        sem = stats.sem(data) # 표준 오차

        # Shapiro-Wilk 정규성 검정
        shapiro_p = None
        shapiro_conclusion = "N < 3 이므로 Shapiro-Wilk 검정을 생략합니다."
        shapiro_conclusion_style = {'color': 'gray', 'fontStyle': 'italic'}

        if n >= 3:
            shapiro_stat, shapiro_p = stats.shapiro(data)
            if shapiro_p > 0.05:
                shapiro_conclusion = f"P-value ({shapiro_p:.3f})가 0.05보다 크므로, 데이터는 정규분포를 따릅니다."
                shapiro_conclusion_style = {'color': 'green', 'fontWeight': 'bold'}
            # 정규성 검정 실패 시 분석 중단
            else:
                error_fig = go.Figure()
                error_fig.update_layout(title="정규성 검정 실패", annotations=[{'text': f'{sample_name} 데이터가 정규분포를 따르지 않습니다 (p={shapiro_p:.4f}).', 'showarrow': False}])
                error_message = html.Div(f"오류: {sample_name} 데이터가 정규분포를 따르지 않아 분석을 중단합니다 (Shapiro-Wilk p-value: {shapiro_p:.4f}). t-검정은 데이터의 정규성을 가정합니다.", style={'color': 'red', 'fontWeight': 'bold'})
                
                return error_fig, error_message
            
        results[sample_name] = {
            'n': n, 'mean': mean, 'std': std, 'sem': sem,
            'shapiro_p': shapiro_p, 'shapiro_conclusion': shapiro_conclusion,
            'shapiro_conclusion_style': shapiro_conclusion_style
        }

    # --- 2-Sample t-Test ---
    # Minitab은 기본적으로 등분산 가정을 하지 않는 Welch's t-test를 사용합니다.
    # scipy.stats.ttest_ind의 equal_var=False가 Welch's t-test입니다.
    # alternative 인자는 'two-sided', 'less', 'greater' 중 하나입니다.

    # 사용자가 선택한 귀무가설을 scipy의 대립가설 파라미터로 변환
    hypothesis_map = {
        'equal': 'two-sided',  # H0: μ1 = μ2  => H1: μ1 ≠ μ2
        'less_or_equal': 'greater', # H0: μ1 ≤ μ2  => H1: μ1 > μ2
        'greater_or_equal': 'less'   # H0: μ1 ≥ μ2  => H1: μ1 < μ2
    }
    alternative_hypothesis = hypothesis_map[null_hypothesis]
    
    # t-검정 유형 결정

    variance_test_results = []
    equal_var_flag = False # 기본값: Welch's
    ttest_method_str = "Welch's t-test (등분산 가정 안함)"

    if ttest_type == 'pooled':
        equal_var_flag = True
        ttest_method_str = "Pooled t-test (등분산 가정함)"
    elif ttest_type == 'auto':
        # F-test for equality of variances
        var1, var2 = results['Sample 1']['std']**2, results['Sample 2']['std']**2
        n1, n2 = results['Sample 1']['n'], results['Sample 2']['n']
        
        # F-statistic is the ratio of the larger variance to the smaller one
        if var1 >= var2:
            f_stat = var1 / var2 if var2 > 0 else np.inf
            dfn, dfd = n1 - 1, n2 - 1
        else:
            f_stat = var2 / var1 if var1 > 0 else np.inf
            dfn, dfd = n2 - 1, n1 - 1
        
        f_p_value = stats.f.sf(f_stat, dfn, dfd) * 2 # Two-tailed test

        variance_test_results.append(html.H5("등분산 검정 (F-Test)", style={'marginTop': '15px'}))
        variance_test_results.append(html.P(f"F-Value: {f_stat:.3f}, P-Value: {f_p_value:.4f}"))

        if f_p_value < alpha_level_float:
            equal_var_flag = False # Variances are different, use Welch's
            ttest_method_str = "Welch's t-test (F-검정 결과: 등분산 가정 안함)"
            variance_test_results.append(html.P(f"결론: P-값({f_p_value:.4f}) < α({alpha_level_float}) 이므로 등분산 가정을 기각합니다.", style={'color': 'red'}))
        else:
            equal_var_flag = True # Variances are equal, use Pooled
            ttest_method_str = "Pooled t-test (F-검정 결과: 등분산 가정함)"
            variance_test_results.append(html.P(f"결론: P-값({f_p_value:.4f}) ≥ α({alpha_level_float}) 이므로 등분산 가정을 기각할 수 없습니다.", style={'color': 'green'}))

    # Perform t-test using scipy.stats.ttest_ind
    t_stat, t_p_value = stats.ttest_ind(data1, data2, equal_var=equal_var_flag, alternative=alternative_hypothesis)

    # Calculate degrees of freedom (DF) and confidence interval (CI)
    n1, n2 = len(data1), len(data2)
    mean1, mean2 = np.mean(data1), np.mean(data2)
    std1, std2 = np.std(data1, ddof=1), np.std(data2, ddof=1)
    mean_diff = mean1 - mean2

    if equal_var_flag:
        df_test = n1 + n2 - 2
        sp = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / df_test)
        se_diff = sp * np.sqrt(1/n1 + 1/n2)
    else: # Welch's t-test
        s1_sq_n1 = std1**2 / n1
        s2_sq_n2 = std2**2 / n2
        se_diff = np.sqrt(s1_sq_n1 + s2_sq_n2)
        # Welch-Satterthwaite equation for DF
        numerator = (s1_sq_n1 + s2_sq_n2)**2
        denominator = (s1_sq_n1**2 / (n1 - 1)) + (s2_sq_n2**2 / (n2 - 1))
        df_test = numerator / denominator if denominator > 0 else np.nan

    t_critical = stats.t.ppf(1 - alpha_level_float / 2, df_test)
    margin_of_error = t_critical * se_diff
    ci_diff = (mean_diff - margin_of_error, mean_diff + margin_of_error)

    ci_diff_note = f"차이에 대한 {conf_level_float}% CI (평균1 - 평균2): "

    # --- 통계 결과 Div 생성 ---
    stats_div_children = []

    # Summary Statistics for Sample 1
    stats_div_children.append(html.H4("요약 통계량 (Sample 1)", style={'borderBottom': '1px solid #ddd', 'paddingBottom': '5px'}))
    stats_div_children.append(html.Table([
        html.Tr([html.Td("데이터 개수 (N)"), html.Td(f"{results['Sample 1']['n']}", style={'textAlign': 'right', 'fontWeight': 'bold'})]),
        html.Tr([html.Td("평균 (Mean)"), html.Td(f"{results['Sample 1']['mean']:.4f}", style={'textAlign': 'right', 'fontWeight': 'bold'})]),
        html.Tr([html.Td("표준편차 (StDev)"), html.Td(f"{results['Sample 1']['std']:.4f}", style={'textAlign': 'right', 'fontWeight': 'bold'})]),
    ], style={'width': '300px'}))
    shapiro_p_text_1 = f"{results['Sample 1']['shapiro_p']:.4f}" if results['Sample 1']['shapiro_p'] is not None else 'N < 3'
    stats_div_children.append(html.H5("정규성 검정 (Shapiro-Wilk)", style={'marginTop': '10px'}))
    stats_div_children.append(html.P(f"P-value: {shapiro_p_text_1}"))
    stats_div_children.append(html.P(results['Sample 1']['shapiro_conclusion'], style=results['Sample 1']['shapiro_conclusion_style']))

    # Summary Statistics for Sample 2
    stats_div_children.append(html.H4("요약 통계량 (Sample 2)", style={'marginTop': '20px', 'borderBottom': '1px solid #ddd', 'paddingBottom': '5px'}))
    stats_div_children.append(html.Table([
        html.Tr([html.Td("데이터 개수 (N)"), html.Td(f"{results['Sample 2']['n']}", style={'textAlign': 'right', 'fontWeight': 'bold'})]),
        html.Tr([html.Td("평균 (Mean)"), html.Td(f"{results['Sample 2']['mean']:.4f}", style={'textAlign': 'right', 'fontWeight': 'bold'})]),
        html.Tr([html.Td("표준편차 (StDev)"), html.Td(f"{results['Sample 2']['std']:.4f}", style={'textAlign': 'right', 'fontWeight': 'bold'})]),
    ], style={'width': '300px'}))
    shapiro_p_text_2 = f"{results['Sample 2']['shapiro_p']:.4f}" if results['Sample 2']['shapiro_p'] is not None else 'N < 3'
    stats_div_children.append(html.H5("정규성 검정 (Shapiro-Wilk)", style={'marginTop': '10px'}))
    stats_div_children.append(html.P(f"P-value: {shapiro_p_text_2}"))
    stats_div_children.append(html.P(results['Sample 2']['shapiro_conclusion'], style=results['Sample 2']['shapiro_conclusion_style']))

    # Hypothesis Test Results
    stats_div_children.append(html.H4("가설 검정 결과 (Hypothesis Test Result)", style={'marginTop': '20px', 'borderBottom': '1px solid #ddd', 'paddingBottom': '5px'}))
    
    # 가설 정의
    null_hypo_text = {
        'equal': "귀무가설 (H0): 평균 1 = 평균 2",
        'less_or_equal': "귀무가설 (H0): 평균 1 ≤ 평균 2",
        'greater_or_equal': "귀무가설 (H0): 평균 1 ≥ 평균 2"
    }
    alt_hypo_text = {
        'equal': "대립가설 (H1): 평균 1 ≠ 평균 2",
        'less_or_equal': "대립가설 (H1): 평균 1 > 평균 2",
        'greater_or_equal': "대립가설 (H1): 평균 1 < 평균 2"
    }
    stats_div_children.append(html.P(null_hypo_text[null_hypothesis]))
    stats_div_children.append(html.P(alt_hypo_text[null_hypothesis]))
    stats_div_children.append(html.P(f"유의수준 (α): {alpha_level_float}"))

    # 등분산 검정 결과가 있으면 추가
    stats_div_children.extend(variance_test_results)

    # 가설 검정 결론
    if t_p_value < alpha_level_float:
        conclusion_text = f"결론: P-값 ({t_p_value:.4f})이 유의수준 ({alpha_level_float})보다 작으므로, '{null_hypo_text[null_hypothesis]}' 을 기각합니다."
        conclusion_style = {'color': 'red', 'fontWeight': 'bold', 'marginTop': '10px'}
    else:
        conclusion_text = f"결론: P-값 ({t_p_value:.4f})이 유의수준 ({alpha_level_float})보다 크거나 같으므로, '{null_hypo_text[null_hypothesis]}' 을 기각할 수 없습니다."
        conclusion_style = {'color': 'green', 'fontWeight': 'bold', 'marginTop': '10px'}
    
    stats_div_children.append(html.P(f"수행된 검정: {ttest_method_str}", style={'fontStyle': 'italic', 'color': '#555', 'marginTop': '10px'}))
    stats_div_children.append(html.Table([
        html.Tr([html.Td("평균 1"), html.Td(f"{results['Sample 1']['mean']:.4f}", style={'textAlign': 'right'})]),
        html.Tr([html.Td("평균 2"), html.Td(f"{results['Sample 2']['mean']:.4f}", style={'textAlign': 'right'})]),
        html.Tr([html.Td("평균 차이 (Mean Diff)"), html.Td(f"{results['Sample 1']['mean'] - results['Sample 2']['mean']:.4f}", style={'textAlign': 'right', 'fontWeight': 'bold'})]),
        html.Tr([html.Td("t-Value"), html.Td(f"{t_stat:.3f}", style={'textAlign': 'right', 'fontWeight': 'bold'})]),
        html.Tr([html.Td("P-Value"), html.Td(f"{t_p_value:.4f}", style={'textAlign': 'right', 'fontWeight': 'bold'})]),
        html.Tr([html.Td("자유도 (DF)"), html.Td(f"{df_test:.2f}", style={'textAlign': 'right', 'fontWeight': 'bold'})]),
    ], style={'width': '300px', 'marginTop': '10px'}))

    stats_div_children.append(html.P(conclusion_text, style=conclusion_style))

    stats_div_children.append(html.H5(f"{conf_level_float}% 신뢰 구간 (Confidence Interval for Difference)", style={'marginTop': '15px'}))
    stats_div_children.append(html.P([
        ci_diff_note,
        html.B(f"({ci_diff[0]:.4f}, {ci_diff[1]:.4f})") if not np.isnan(ci_diff[0]) else ""
    ]))

    stats_div = html.Div(stats_div_children)


    # --- Plotly 그래프 생성 (2x2 subplots) ---
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "Sample 1 히스토그램 (Fitted Normal Distribution)", "Sample 1 정규 확률도 (Normal Probability Plot)",
            "Sample 2 히스토그램 (Fitted Normal Distribution)", "Sample 2 정규 확률도 (Normal Probability Plot)",
            "데이터 비교 박스 플롯 (Box Plot)", f"{conf_level_float}% 평균 차이의 신뢰 구간"
        ),
        vertical_spacing=0.1
    )

    # Helper function to add plots for a single sample
    def add_sample_plots(fig, data, sample_name, row_offset, color):
        n = len(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1)

        # Histogram
        fig.add_trace(
            go.Histogram(
                x=data,
                name=f'{sample_name} 데이터',
                histnorm='probability density',
                marker_color=color,
                opacity=0.7,
                showlegend=False
            ),
            row=row_offset, col=1
        )
        if std > 0 and not np.isnan(std):
            x_curve = np.linspace(data.min() - std, data.max() + std, 100)
            y_curve = stats.norm.pdf(x_curve, mean, std)
            fig.add_trace(
                go.Scatter(
                    x=x_curve,
                    y=y_curve,
                    mode='lines',
                    name=f'{sample_name} 정규분포 곡선',
                    line=dict(color='red', width=2),
                    showlegend=False
                ),
                row=row_offset, col=1
            )
        fig.update_yaxes(title_text="밀도 (Density)", row=row_offset, col=1)
        fig.update_xaxes(title_text="데이터 값", row=row_offset, col=1)

        # Q-Q Plot
        if n >= 2:
            prob_plot_data = stats.probplot(data, dist="norm")
            theoretical_quantiles = prob_plot_data[0][0]
            ordered_values = prob_plot_data[0][1]
            slope, intercept, r = prob_plot_data[1]

            fig.add_trace(
                go.Scatter(
                    x=theoretical_quantiles,
                    y=ordered_values,
                    mode='markers',
                    name=f'{sample_name} 데이터 포인트',
                    marker_color=color,
                    showlegend=False
                ),
                row=row_offset, col=2
            )
            fit_line_y = slope * theoretical_quantiles + intercept
            fig.add_trace(
                go.Scatter(
                    x=theoretical_quantiles,
                    y=fit_line_y,
                    mode='lines',
                    name=f'{sample_name} 적합선',
                    line=dict(color='red', width=2, dash='dash'),
                    showlegend=False
                ),
                row=row_offset, col=2
            )
        fig.update_yaxes(title_text="데이터 값 (Ordered Values)", row=row_offset, col=2)
        fig.update_xaxes(title_text="이론적 분위수 (Theoretical Quantiles)", row=row_offset, col=2)

    add_sample_plots(fig, data1, "Sample 1", 1, '#007BFF')
    add_sample_plots(fig, data2, "Sample 2", 2, '#28A745') # Different color for Sample 2

    # Box Plot for comparing two samples
    df_box = pd.DataFrame({
        'Value': np.concatenate([data1, data2]),
        'Sample': ['Sample 1'] * len(data1) + ['Sample 2'] * len(data2)
    })
    box_fig = px.box(df_box, x='Sample', y='Value', color='Sample',
                     color_discrete_map={'Sample 1': '#007BFF', 'Sample 2': '#28A745'})
    for trace in box_fig.data:
        fig.add_trace(trace, row=3, col=1)
    fig.update_yaxes(title_text="데이터 값", row=3, col=1)
    fig.update_xaxes(title_text="", row=3, col=1)

    # Confidence Interval Plot for the difference
    if not np.isnan(ci_diff[0]):
        fig.add_trace(go.Scatter(
            x=[mean_diff],
            y=['평균 차이'],
            error_x=dict(
                type='data', symmetric=False,
                array=[ci_diff[1] - mean_diff],
                arrayminus=[mean_diff - ci_diff[0]],
                thickness=1.5, color='#EF553B'
            ),
            mode='markers',
            marker=dict(size=12, color='#EF553B', symbol='circle'),
        ), row=3, col=2)

        # Add annotations for CI values
        fig.add_annotation(x=ci_diff[0], y='평균 차이', text=f"{ci_diff[0]:.4f}", showarrow=False, yshift=-20, font=dict(color="black"), row=3, col=2)
        fig.add_annotation(x=ci_diff[1], y='평균 차이', text=f"{ci_diff[1]:.4f}", showarrow=False, yshift=-20, font=dict(color="black"), row=3, col=2)

        # Add a vertical line at x=0 for reference
        fig.add_vline(x=mean_diff, line_dash="dot", line_color="gray", 
                      annotation_text=f"두 표본의 차이: {mean_diff:.4f}", 
                      annotation_position="top right",
                      row=3, col=2)

    fig.update_yaxes(showticklabels=False, title_text="", row=3, col=2)
    fig.update_xaxes(title_text="신뢰 구간 (평균1 - 평균2)", row=3, col=2)

    fig.update_layout(
        title_text=f"<b>2표본 t-검정 및 정규성 검정 결과</b>",
        title_x=0.5,
        showlegend=False,
        height=1200, # Increased height for 3x2 plots
        bargap=0.01
    )

    return fig, stats_div