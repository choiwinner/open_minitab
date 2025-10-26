import dash
from dash import dcc, html, register_page
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats

register_page(__name__)

# 앱 레이아웃 정의
layout = html.Div(
    style={'fontFamily': 'Arial, sans-serif', 'maxWidth': '1000px', 'margin': 'auto', 'padding': '20px'},
    children=[
        html.H1(
            "대응표본 t-검정 (Paired t-test)",
            style={'textAlign': 'center', 'color': '#333'}
        ),

        html.P(
            "쌍으로 이루어진 두 데이터 열(column)을 각각 붙여넣으세요. 두 데이터의 개수는 반드시 같아야 합니다.",
            style={'textAlign': 'center', 'color': '#555'}
        ),

        # 데이터 입력을 위한 텍스트 영역 (두 개)
        html.Div([
            html.Div([
                html.Label("데이터 1 (Sample 1):", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                dcc.Textarea(
                    id='paired-data-input-area-1',
                    placeholder="예:\n10.2\n11.5\n9.8\n...",
                    style={'width': '100%', 'height': '150px', 'fontSize': '16px', 'borderRadius': '5px', 'border': '1px solid #ccc'}
                ),
            ], style={'flex': '1', 'marginRight': '10px'}), # Use flexbox for side-by-side

            html.Div([
                html.Label("데이터 2 (Sample 2):", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                dcc.Textarea(
                    id='paired-data-input-area-2',
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
                    id='paired-significance-level-input',
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
                    id='paired-hypothesis-selection-radio',
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
        ], style={'marginTop': '15px', 'marginBottom': '10px'}),

        # 분석 실행 버튼
        html.Button(
            '분석 실행',
            id='paired-run-analysis-button',
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
                html.Div(id='paired-stats-results-output', style={'marginTop': '20px', 'padding': '15px', 'backgroundColor': '#f9f9f9', 'borderRadius': '5px'}),
                # 그래프 출력 영역
                dcc.Graph(id='paired-normality-plot-graph')
            ]
        ) # This closes the dcc.Loading component
    ]
)

# 콜백: 버튼 클릭 시 그래프 및 통계 업데이트
@dash.callback(
    [Output('paired-normality-plot-graph', 'figure'),
     Output('paired-stats-results-output', 'children')],
    [Input('paired-run-analysis-button', 'n_clicks')],
    [State('paired-data-input-area-1', 'value'),
     State('paired-data-input-area-2', 'value'),
     State('paired-hypothesis-selection-radio', 'value'),
     State('paired-significance-level-input', 'value')]
)
def update_paired_sample_analysis(n_clicks, data_string_1, data_string_2, null_hypothesis, alpha_level):
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

    # 대응표본 t-검정을 위해 두 샘플의 크기가 같아야 함
    if len(data1) != len(data2):
        error_fig = go.Figure()
        error_fig.update_layout(title="데이터 개수 불일치", annotations=[{'text': '대응표본 t-검정을 위해 두 데이터의 개수가 같아야 합니다.', 'showarrow': False}])
        return error_fig, html.Div(f"오류: 두 데이터의 개수가 일치하지 않습니다. (데이터 1: {len(data1)}개, 데이터 2: {len(data2)}개)", style={'color': 'red', 'fontWeight': 'bold'})

    # 최소 2개 이상의 데이터 쌍 필요
    if len(data1) < 2:
        error_fig = go.Figure()
        error_fig.update_layout(title="데이터 부족", annotations=[{'text': '분석을 위해 2개 이상의 데이터 쌍이 필요합니다.', 'showarrow': False}])
        return error_fig, html.Div(f"오류: 2개 이상의 데이터 쌍이 필요합니다. (현재 {len(data1)}개)", style={'color': 'red', 'fontWeight': 'bold'})

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

    conf_level_float = round((1 - alpha_level_float) * 100, 1)

    # --- 통계량 계산 ---
    differences = data1 - data2
    n = len(differences)
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    df = n - 1
    sem_diff = stats.sem(differences)

    # 차이(differences)에 대한 정규성 검정
    shapiro_p = None
    shapiro_conclusion = "N < 3 이므로 Shapiro-Wilk 검정을 생략합니다."
    shapiro_conclusion_style = {'color': 'gray', 'fontStyle': 'italic'}
    if n >= 3:
        shapiro_stat, shapiro_p = stats.shapiro(differences)
        if shapiro_p > alpha_level_float:
            shapiro_conclusion = f"P-value ({shapiro_p:.3f}) > α ({alpha_level_float}) 이므로, 데이터의 차이는 정규분포를 따릅니다."
            shapiro_conclusion_style = {'color': 'green', 'fontWeight': 'bold'}
        else:
            error_fig = go.Figure()
            error_fig.update_layout(title="정규성 검정 실패", annotations=[{'text': f'데이터의 차이가 정규분포를 따르지 않습니다 (p={shapiro_p:.4f}).', 'showarrow': False}])
            error_message = html.Div(f"오류: 데이터의 차이가 정규분포를 따르지 않아 분석을 중단합니다 (Shapiro-Wilk p-value: {shapiro_p:.4f}). 대응표본 t-검정은 차이의 정규성을 가정합니다.", style={'color': 'red', 'fontWeight': 'bold'})
            return error_fig, error_message

    # --- Paired t-Test ---

    # 사용자가 선택한 귀무가설을 scipy의 대립가설 파라미터로 변환
    hypothesis_map = {
        'equal': 'two-sided',      # H₀: μ_diff = 0  => H₁: μ_diff ≠ 0
        'less_or_equal': 'greater',# H₀: μ_diff ≤ 0  => H₁: μ_diff > 0
        'greater_or_equal': 'less' # H₀: μ_diff ≥ 0  => H₁: μ_diff < 0
    }
    alternative_hypothesis = hypothesis_map[null_hypothesis]
    
    # Perform paired t-test using scipy.stats.ttest_rel
    t_stat, t_p_value = stats.ttest_rel(data1, data2, alternative=alternative_hypothesis)

    # 신뢰구간 계산
    ci_diff = (np.nan, np.nan)
    ci_note = "모든 차이값이 동일하여 신뢰구간을 계산할 수 없습니다."
    if sem_diff > 0:
        ci_diff = stats.t.interval(confidence=conf_level_float / 100.0, df=df, loc=mean_diff, scale=sem_diff)
        ci_note = f"평균 차이에 대한 {conf_level_float}% CI: "
    elif n > 1:
        ci_diff = (mean_diff, mean_diff)
        ci_note = f"모든 차이값이 동일합니다. {conf_level_float}% CI: "
    else:
        ci_note = "N=1 이므로 신뢰구간을 계산할 수 없습니다."

    # --- 통계 결과 Div 생성 ---
    stats_div_children = [
        html.H4("요약 통계량 (차이 = 데이터1 - 데이터2)", style={'borderBottom': '1px solid #ddd', 'paddingBottom': '5px'}),
        html.Table([
            html.Tr([html.Td("데이터 쌍 개수 (N)"), html.Td(f"{n}", style={'textAlign': 'right', 'fontWeight': 'bold'})]),
            html.Tr([html.Td("평균 차이 (Mean)"), html.Td(f"{mean_diff:.4f}", style={'textAlign': 'right', 'fontWeight': 'bold'})]),
            html.Tr([html.Td("표준편차 (StDev)"), html.Td(f"{std_diff:.4f}", style={'textAlign': 'right', 'fontWeight': 'bold'})]),
            html.Tr([html.Td("표준 오차 (SE Mean)"), html.Td(f"{sem_diff:.4f}", style={'textAlign': 'right', 'fontWeight': 'bold'})]),
        ], style={'width': '300px'}),

        html.H5(f"평균 차이에 대한 {conf_level_float}% 신뢰 구간", style={'marginTop': '15px'}),
        html.P([ci_note, html.B(f"({ci_diff[0]:.4f}, {ci_diff[1]:.4f})") if not np.isnan(ci_diff[0]) else ""]),

        html.H4("차이의 정규성 검정 (Shapiro-Wilk)", style={'marginTop': '20px', 'borderBottom': '1px solid #ddd', 'paddingBottom': '5px'}),
        html.P(f"P-value: {shapiro_p:.4f}" if shapiro_p is not None else "N < 3"),
        html.P(shapiro_conclusion, style=shapiro_conclusion_style),

        html.H4("가설 검정 결과", style={'marginTop': '20px', 'borderBottom': '1px solid #ddd', 'paddingBottom': '5px'}),
    ]

    # 가설 정의
    null_hypo_text = {
        'equal': "귀무가설 (H₀): 평균 차이 = 0",
        'less_or_equal': "귀무가설 (H₀): 평균 차이 ≤ 0",
        'greater_or_equal': "귀무가설 (H₀): 평균 차이 ≥ 0"
    }
    alt_hypo_text = {
        'equal': "대립가설 (H₁): 평균 차이 ≠ 0",
        'less_or_equal': "대립가설 (H₁): 평균 차이 > 0",
        'greater_or_equal': "대립가설 (H₁): 평균 차이 < 0"
    }
    stats_div_children.append(html.P(null_hypo_text[null_hypothesis]))
    stats_div_children.append(html.P(alt_hypo_text[null_hypothesis]))
    stats_div_children.append(html.P(f"유의수준 (α): {alpha_level_float}"))

    # 가설 검정 결론
    if t_p_value < alpha_level_float:
        conclusion_text = f"결론: P-값 ({t_p_value:.4f}) < α ({alpha_level_float}) 이므로, 귀무가설을 기각합니다."
        conclusion_style = {'color': 'red', 'fontWeight': 'bold', 'marginTop': '10px'}
    else:
        conclusion_text = f"결론: P-값 ({t_p_value:.4f}) ≥ α ({alpha_level_float}) 이므로, 귀무가설을 기각할 수 없습니다."
        conclusion_style = {'color': 'green', 'fontWeight': 'bold', 'marginTop': '10px'}
    
    stats_div_children.append(html.Table([
        html.Tr([html.Td("t-Value"), html.Td(f"{t_stat:.3f}", style={'textAlign': 'right', 'fontWeight': 'bold'})]),
        html.Tr([html.Td("P-Value"), html.Td(f"{t_p_value:.4f}", style={'textAlign': 'right', 'fontWeight': 'bold'})]),
        html.Tr([html.Td("자유도 (DF)"), html.Td(f"{df}", style={'textAlign': 'right', 'fontWeight': 'bold'})]),
    ], style={'width': '300px', 'marginTop': '10px'}))

    stats_div_children.append(html.P(conclusion_text, style=conclusion_style))

    stats_div = html.Div(stats_div_children)

    # --- Plotly 그래프 생성 (차이에 대한 그래프) ---
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "차이의 히스토그램 (Fitted Normal Distribution)", "차이의 정규 확률도 (Normal Probability Plot)"
        )
    )

    # 히스토그램
    fig.add_trace(go.Histogram(x=differences, name='차이', histnorm='probability density', marker_color='#636EFA', opacity=0.7), row=1, col=1)
    if std_diff > 0 and not np.isnan(std_diff):
        x_curve = np.linspace(differences.min() - std_diff, differences.max() + std_diff, 100)
        y_curve = stats.norm.pdf(x_curve, mean_diff, std_diff)
        fig.add_trace(go.Scatter(x=x_curve, y=y_curve, mode='lines', name='정규분포 곡선', line=dict(color='red', width=2)), row=1, col=1)
    fig.update_yaxes(title_text="밀도 (Density)", row=1, col=1)
    fig.update_xaxes(title_text="차이 값 (Data1 - Data2)", row=1, col=1)

    # Q-Q Plot
    if n >= 2:
        prob_plot_data = stats.probplot(differences, dist="norm")
        theoretical_quantiles, ordered_values = prob_plot_data[0]
        slope, intercept, r = prob_plot_data[1]
        fig.add_trace(go.Scatter(x=theoretical_quantiles, y=ordered_values, mode='markers', name='데이터 포인트', marker_color='#636EFA'), row=1, col=2)
        fit_line_y = slope * theoretical_quantiles + intercept
        fig.add_trace(go.Scatter(x=theoretical_quantiles, y=fit_line_y, mode='lines', name='적합선', line=dict(color='red', width=2, dash='dash')), row=1, col=2)
    fig.update_yaxes(title_text="차이 값 (Ordered Values)", row=1, col=2)
    fig.update_xaxes(title_text="이론적 분위수 (Theoretical Quantiles)", row=1, col=2)

    fig.update_layout(
        title_text=f"<b>차이의 정규성 검정 결과</b><br>(Shapiro-Wilk P-value: {f'{shapiro_p:.4f}' if shapiro_p is not None else 'N/A'})",
        title_x=0.5,
        showlegend=False,
        height=500,
        bargap=0.01
    )

    return fig, stats_div