
import dash
from dash import dcc, html, register_page
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats
import plotly.express as px

register_page(__name__, path='/')

# 앱 레이아웃 정의
layout = html.Div(
    style={'fontFamily': 'Arial, sans-serif', 'maxWidth': '1000px', 'margin': 'auto', 'padding': '20px'},
    children=[
        html.H1(
            "1표본 t-검정 및 정규성 검정",
            style={'textAlign': 'center', 'color': '#333'}
        ),

        html.P(
            "데이터 열(column)을 복사한 후, 아래 텍스트 상자에 붙여넣으세요. (숫자만, 한 줄에 하나씩)",
            style={'textAlign': 'center', 'color': '#555'}
        ),

        # 데이터 입력을 위한 텍스트 영역
        dcc.Textarea(
            id='home-data-input-area',
            placeholder="예:\n10.2\n11.5\n9.8\n...",
            style={'width': '100%', 'height': '200px', 'fontSize': '16px', 'borderRadius': '5px', 'border': '1px solid #ccc'}
        ),

        # t-검정 및 옵션 입력을 위한 Div
        html.Div([
            html.Div([
                html.Label("가설 평균 (H₀):", style={'marginRight': '10px'}),
                dcc.Input(
                    id='home-hypothesized-mean-input',
                    type='number',
                    placeholder='검정할 평균값',
                    style={'width': '150px'}
                )
            ], style={'display': 'inline-block', 'marginRight': '30px', 'marginTop': '10px'}),

            html.Div([
                html.Label("유의 수준 (α):", style={'fontWeight': 'bold'}),
                dcc.Slider(
                    id='home-significance-level-input',
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
                html.Label("귀무 가설 (H₀):", style={'marginRight': '10px', 'fontWeight': 'bold'}),
                dcc.RadioItems(
                    id='home-hypothesis-selection-radio',
                    options=[
                        {'label': '평균 = 가설 평균', 'value': 'equal'},
                        {'label': '평균 ≥ 가설 평균', 'value': 'greater_or_equal'},
                        {'label': '평균 ≤ 가설 평균', 'value': 'less_or_equal'}
                    ],
                    value='equal', # Default selection
                    inline=True,
                    style={'marginTop': '5px'}
                )
            ], style={'display': 'block', 'marginTop': '10px'}),
        ], style={'marginTop': '15px', 'marginBottom': '10px'}),

        # 분석 실행 버튼
        html.Button(
            '분석 실행',
            id='home-run-analysis-button',
            n_clicks=0,
            style={'width': '100%', 'padding': '10px', 'fontSize': '18px', 'fontWeight': 'bold', 'backgroundColor': '#007BFF', 'color': 'white', 'border': 'none', 'borderRadius': '5px', 'marginTop': '10px', 'cursor': 'pointer'}
        ),

        # 로딩 스피너
        dcc.Loading(
            id="loading-spinner",
            type="circle",
            children=[html.Div(id='home-stats-results-output', style={'marginTop': '20px', 'padding': '15px', 'backgroundColor': '#f9f9f9', 'borderRadius': '5px'}), dcc.Graph(id='home-normality-plot-graph')]
        )
    ]
)

# 콜백: 버튼 클릭 시 그래프 및 통계 업데이트
@dash.callback(
    [Output('home-normality-plot-graph', 'figure'), Output('home-stats-results-output', 'children')],
    [Input('home-run-analysis-button', 'n_clicks')],
    [State('home-data-input-area', 'value'),
     State('home-hypothesized-mean-input', 'value'),
     State('home-hypothesis-selection-radio', 'value'),
     State('home-significance-level-input', 'value')]
)
def update_one_sample_analysis(n_clicks, data_string, hypo_mean, null_hypothesis, alpha_level):
    # 버튼이 클릭되지 않았거나 입력이 없으면 빈 상태 반환
    if n_clicks == 0 or not data_string:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="데이터를 입력하고 '분석 실행' 버튼을 클릭하세요.",
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
    try:
        lines = data_string.strip().split('\n')
        data = [float(line.strip()) for line in lines if line.strip()]
        if not data:
             raise ValueError("데이터가 없습니다.")
        data = np.array(data)
    except ValueError as e:
        error_fig = go.Figure()
        error_fig.update_layout(title="입력 오류", annotations=[{'text': f'데이터 파싱 오류: {e}', 'showarrow': False}])
        return error_fig, html.Div(str(e), style={'color': 'red', 'fontWeight': 'bold'})

    # t-검정 및 신뢰구간을 위해 최소 2개 이상의 데이터 필요
    if len(data) < 2:
        error_fig = go.Figure()
        error_fig.update_layout(title="데이터 부족", annotations=[{'text': '분석을 위해 2개 이상의 데이터가 필요합니다.', 'showarrow': False}])
        return error_fig, html.Div(f"오류: 2개 이상의 데이터가 필요합니다. (현재 {len(data)}개)", style={'color': 'red', 'fontWeight': 'bold'})

    # 유의수준 파싱
    try:
        if alpha_level is None:
            alpha_level = 0.05 # 기본값
        alpha_level_float = float(alpha_level)
        if not (0 < alpha_level_float < 1.0):
            raise ValueError("유의수준은 0과 1 사이여야 합니다.")
    except (ValueError, TypeError) as e:
        error_fig = go.Figure()
        error_fig.update_layout(title="입력 오류", annotations=[{'text': f'유의 수준 값 오류: {e}', 'showarrow': False}])
        return error_fig, html.Div(f"오류: 유의 수준이 올바르지 않습니다. (입력값: {alpha_level})", style={'color': 'red', 'fontWeight': 'bold'})

    conf_level_float = round((1 - alpha_level_float) * 100, 1)

    # --- 통계량 계산 ---
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    df = n - 1
    sem = stats.sem(data)

    # 신뢰구간 계산
    ci_mean = (np.nan, np.nan)
    ci_note = "모든 데이터 값이 동일하여 신뢰구간을 계산할 수 없습니다."
    if sem > 0:
        ci_mean = stats.t.interval(confidence=conf_level_float / 100.0, df=df, loc=mean, scale=sem)
        ci_note = f"평균에 대한 {conf_level_float}% CI: "
    elif n > 1:
        ci_mean = (mean, mean)
        ci_note = f"모든 데이터 값이 동일합니다. {conf_level_float}% CI: "
    else:
        ci_note = "N=1 이므로 신뢰구간을 계산할 수 없습니다."

    # Shapiro-Wilk 정규성 검정
    shapiro_p = None
    shapiro_conclusion = "N < 3 이므로 Shapiro-Wilk 검정을 생략합니다."
    shapiro_conclusion_style = {'color': 'gray', 'fontStyle': 'italic'}
    if n >= 3:
        shapiro_stat, shapiro_p = stats.shapiro(data)
        if shapiro_p > alpha_level_float:
            shapiro_conclusion = f"P-value ({shapiro_p:.3f}) > α ({alpha_level_float}) 이므로, 데이터는 정규분포를 따릅니다."
            shapiro_conclusion_style = {'color': 'green', 'fontWeight': 'bold'}
        else:
            error_fig = go.Figure()
            error_fig.update_layout(title="정규성 검정 실패", annotations=[{'text': f'데이터가 정규분포를 따르지 않습니다 (p={shapiro_p:.4f}).', 'showarrow': False}])
            error_message = html.Div(f"오류: 데이터가 정규분포를 따르지 않아 분석을 중단합니다 (Shapiro-Wilk p-value: {shapiro_p:.4f}). t-검정은 데이터의 정규성을 가정합니다.", style={'color': 'red', 'fontWeight': 'bold'})
            return error_fig, error_message

    # 1-Sample t-Test
    t_test_results_div = []
    if hypo_mean is not None and str(hypo_mean).strip() != '':
        try:
            hypo_mean_float = float(hypo_mean)
            if std == 0 or np.isnan(std):
                t_test_results_div = [html.P("모든 데이터 값이 동일하여 t-검정을 수행할 수 없습니다.", style={'color': 'orange'})]
            else:
                # 사용자가 선택한 귀무가설을 scipy의 대립가설 파라미터로 변환
                hypothesis_map = {
                    'equal': 'two-sided',      # H₀: μ = μ₀  => H₁: μ ≠ μ₀
                    'less_or_equal': 'greater',# H₀: μ ≤ μ₀  => H₁: μ > μ₀
                    'greater_or_equal': 'less' # H₀: μ ≥ μ₀  => H₁: μ < μ₀
                }
                alternative_hypothesis = hypothesis_map[null_hypothesis]

                t_stat, t_p_value = stats.ttest_1samp(data, hypo_mean_float, alternative=alternative_hypothesis)

                null_hypo_text_map = {
                    'equal': f"귀무가설 (H₀): 평균 = {hypo_mean_float}",
                    'greater_or_equal': f"귀무가설 (H₀): 평균 ≥ {hypo_mean_float}",
                    'less_or_equal': f"귀무가설 (H₀): 평균 ≤ {hypo_mean_float}"
                }
                alt_hypo_text = f"대립가설 (H₁): {null_hypo_text_map[null_hypothesis].replace('H₀', 'H₁').replace('=', '≠').replace('≥', '<').replace('≤', '>')}"

                if t_p_value < alpha_level_float:
                    conclusion_text = f"결론: P-값 ({t_p_value:.4f}) < α ({alpha_level_float}) 이므로, 귀무가설을 기각합니다."
                    conclusion_style = {'color': 'red', 'fontWeight': 'bold', 'marginTop': '10px'}
                else:
                    conclusion_text = f"결론: P-값 ({t_p_value:.4f}) ≥ α ({alpha_level_float}) 이므로, 귀무가설을 기각할 수 없습니다."
                    conclusion_style = {'color': 'green', 'fontWeight': 'bold', 'marginTop': '10px'}

                t_test_results_div = [
                    html.H4("1표본 t-검정 결과", style={'marginTop': '20px', 'borderBottom': '1px solid #ddd', 'paddingBottom': '5px'}),
                    html.P(null_hypo_text_map[null_hypothesis]),
                    html.P(alt_hypo_text),
                    html.P(f"유의수준 (α): {alpha_level_float}"),
                    html.Table([
                        html.Tr([html.Td("t-Value"), html.Td(f"{t_stat:.3f}", style={'textAlign': 'right', 'fontWeight': 'bold'})]),
                        html.Tr([html.Td("P-Value"), html.Td(f"{t_p_value:.4f}", style={'textAlign': 'right', 'fontWeight': 'bold'})]),
                        html.Tr([html.Td("자유도 (DF)"), html.Td(f"{df}", style={'textAlign': 'right', 'fontWeight': 'bold'})]),
                    ], style={'width': '300px', 'marginTop': '10px'}),
                    html.P(conclusion_text, style=conclusion_style)
                ]
        except (ValueError, TypeError):
            t_test_results_div = [html.P(f"오류: 가설 평균 '{hypo_mean}'이(가) 올바른 숫자가 아닙니다.", style={'color': 'red'})]

    # --- 통계 결과 Div 생성 ---
    stats_div = html.Div([
        html.H4("요약 통계량", style={'borderBottom': '1px solid #ddd', 'paddingBottom': '5px'}),
        html.Table([
            html.Tr([html.Td("데이터 개수 (N)"), html.Td(f"{n}", style={'textAlign': 'right', 'fontWeight': 'bold'})]),
            html.Tr([html.Td("평균 (Mean)"), html.Td(f"{mean:.4f}", style={'textAlign': 'right', 'fontWeight': 'bold'})]),
            html.Tr([html.Td("표준편차 (StDev)"), html.Td(f"{std:.4f}", style={'textAlign': 'right', 'fontWeight': 'bold'})]),
        ], style={'width': '300px'}),

        html.H5(f"평균에 대한 {conf_level_float}% 신뢰 구간", style={'marginTop': '15px'}),
        html.P([ci_note, html.B(f"({ci_mean[0]:.4f}, {ci_mean[1]:.4f})") if not np.isnan(ci_mean[0]) else ""]),

        html.H4("정규성 검정 (Shapiro-Wilk)", style={'marginTop': '20px', 'borderBottom': '1px solid #ddd', 'paddingBottom': '5px'}),
        html.P(f"P-value: {shapiro_p:.4f}" if shapiro_p is not None else "N < 3"),
        html.P(shapiro_conclusion, style=shapiro_conclusion_style),

        *t_test_results_div,
    ])

    # --- Plotly 그래프 생성 (1x2 subplots) ---
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "히스토그램 (Fitted Normal Distribution)", "정규 확률도 (Normal Probability Plot)",
            "박스 플롯 (Box Plot)", f"{conf_level_float}% 평균의 신뢰 구간 (Confidence Interval)"
        ),
        vertical_spacing=0.15
    )

    # 히스토그램
    fig.add_trace(go.Histogram(x=data, name='데이터', histnorm='probability density', marker_color='#007BFF', opacity=0.7), row=1, col=1)
    if std > 0 and not np.isnan(std):
        x_curve = np.linspace(data.min() - std, data.max() + std, 100)
        y_curve = stats.norm.pdf(x_curve, mean, std)
        fig.add_trace(go.Scatter(x=x_curve, y=y_curve, mode='lines', name='정규분포 곡선', line=dict(color='red', width=2)), row=1, col=1)
    fig.update_yaxes(title_text="밀도 (Density)", row=1, col=1)
    fig.update_xaxes(title_text="데이터 값", row=1, col=1)

    # Q-Q Plot
    if n >= 2:
        prob_plot_data = stats.probplot(data, dist="norm")
        theoretical_quantiles, ordered_values = prob_plot_data[0]
        slope, intercept, r = prob_plot_data[1]
        fig.add_trace(go.Scatter(x=theoretical_quantiles, y=ordered_values, mode='markers', name='데이터 포인트', marker_color='#007BFF'), row=1, col=2)
        fit_line_y = slope * theoretical_quantiles + intercept
        fig.add_trace(go.Scatter(x=theoretical_quantiles, y=fit_line_y, mode='lines', name='적합선', line=dict(color='red', width=2, dash='dash')), row=1, col=2)
    fig.update_yaxes(title_text="데이터 값 (Ordered Values)", row=1, col=2)
    fig.update_xaxes(title_text="이론적 분위수 (Theoretical Quantiles)", row=1, col=2)

    # Box Plot (요청 1)
    # plotly.express를 사용하여 box plot 생성
    #box_fig = px.box(y=data, points="all", color_discrete_sequence=['#007BFF'])
    box_fig = px.box(y=data, color_discrete_sequence=['#007BFF'])
    #box_fig.update_traces(name='데이터', boxmean='sd') # 평균 및 표준편차 표시, 이름 설정
    fig.add_trace(box_fig.data[0], row=2, col=1)
    fig.update_yaxes(title_text="데이터 값", row=2, col=1)

    # Confidence Interval Plot (요청 2)
    if not np.isnan(ci_mean[0]):
        # Minitab 스타일로 신뢰구간을 x축에 표시
        fig.add_trace(go.Scatter(
            x=[mean],
            y=['평균'],
            error_x=dict(
                type='data',
                symmetric=False,
                array=[ci_mean[1] - mean],
                arrayminus=[mean - ci_mean[0]],
                thickness=1.5,
                color='#007BFF'
            ),
            mode='markers', # 마커만 표시하도록 변경
            marker=dict(size=12, color='#007BFF', symbol='circle'),
            name=f'{conf_level_float}% CI'
        ), row=2, col=2)

        # 신뢰구간 양 끝에 텍스트 추가 (add_annotation 사용)
        # 하한값
        fig.add_annotation(x=ci_mean[0], y='평균', text=f"{ci_mean[0]:.4f}",
                           showarrow=False, yshift=-20, font=dict(color="black"), row=2, col=2)
        # 상한값
        fig.add_annotation(x=ci_mean[1], y='평균', text=f"{ci_mean[1]:.4f}",
                           showarrow=False, yshift=-20, font=dict(color="black"), row=2, col=2)

    fig.update_yaxes(showticklabels=False, title_text="", row=2, col=2)
    fig.update_xaxes(title_text="평균 신뢰 구간(Confidence Interval)", row=2, col=2)

    # 가설 평균이 있을 경우, 박스 플롯과 신뢰구간 그래프에 선 추가
    if hypo_mean is not None and str(hypo_mean).strip() != '' and not np.isnan(float(hypo_mean)):
        try:
            hypo_mean_float = float(hypo_mean)
            fig.add_hline(y=hypo_mean_float, line_dash="dot", line_color="green",
                          annotation=dict(
                              text=f"가설 평균: {hypo_mean}",
                              font=dict(color="green")
                          ),
                          annotation_position="bottom right",
                          row=2, col=1)
            # 신뢰구간 플롯에는 수직선(vline)으로 가설 평균 표시
            fig.add_vline(x=hypo_mean_float, line_dash="dot", line_color="green", 
                          annotation=dict(
                              text=f"가설 평균: {hypo_mean}",
                              font=dict(color="green")
                          ),
                          annotation_position="top right",
                          row=2, col=2)
        except (ValueError, TypeError):
            pass # hypo_mean이 숫자가 아니면 무시

    fig.update_layout(
        title_text=f"<b>데이터 정규성 검정 결과</b><br>(Shapiro-Wilk P-value: {f'{shapiro_p:.4f}' if shapiro_p is not None else 'N/A'})",
        title_x=0.5,
        showlegend=False,
        height=800, # 그래프 높이 조정
        bargap=0.01
    )

    return fig, stats_div