import dash
from dash import dcc, html, register_page, Input, Output, State, callback, ALL
import dash_bootstrap_components as dbc
import json
import math
import plotly.graph_objects as go

register_page(__name__, path='/')

# --- 통계 도구 데이터 정의 ---
TOOLS = {
    '1-sample-t': {'name': '1-Sample t-test', 'desc': '단일 집단 평균 비교', 'logic': '목표치와 현재 수준의 차이를 검증합니다.', 'link': '/1-one-sample-t-test', 'icon': '📊'},
    '1-sample-equiv': {'name': '1-Sample Equivalence Test', 'desc': '단일 집단 동등성 검정', 'logic': '단일 집단의 평균이 목표치와 동등성 범위 내에 있는지 검증합니다.', 'link': '/1-4-one-sample-equivalence-test', 'icon': '⚖️'},
    '2-sample-t': {'name': '2-Sample t-test', 'desc': '독립적 두 집단 평균 비교', 'logic': '서로 다른 두 조건의 성능 차이를 검증합니다.', 'link': '/2-two-samples-t-test', 'icon': '👬'},
    'paired-t': {'name': 'Paired t-test', 'desc': '대응 샘플(전/후) 비교', 'logic': '동일 대상의 변화량을 정밀하게 분석합니다.', 'link': '/3-paired-samples-t-test', 'icon': '🔄'},
    'equivalence': {'name': 'Equivalence Test', 'desc': '동등성 증명', 'logic': '오차 범위 내에 데이터가 존재함을 증명합니다.', 'link': '/2-2-two-samples-equivalence-test', 'icon': '⚖️'},
    'anova': {'name': 'One-way ANOVA', 'desc': '다수 집단 평균 비교', 'logic': '3개 이상의 조건 중 유의미한 인자를 찾습니다.', 'link': '/anova', 'icon': '📉'},
    'capability': {'name': '공정 능력 (Cpk)', 'desc': '규격 만족 능력 평가', 'logic': '공정 품질 안정성을 지수화하여 관리합니다.', 'link': '/7-capability-analysis', 'icon': '🏭'},
    'msa': {'name': '측정 시스템 (MSA)', 'desc': '계측기 신뢰성 분석', 'logic': '측정 오차가 분석에 미치는 영향을 검증합니다.', 'link': '/msa', 'icon': '📏'},
    'regression': {'name': '상관·회귀 분석', 'desc': '변수 간 인과 관계 분석', 'logic': 'X에 따른 Y의 변화를 수식화하여 예측합니다.', 'link': '/regression', 'icon': '📈'},
    'outlier-test': {'name': '이상점 탐지', 'desc': '특이치 검정 및 정제', 'logic': '비정상적인 데이터를 탐지하여 왜곡을 방지합니다.', 'link': '/1-3-outlier-test', 'icon': '🔍'},
    'doe': {'name': '실험 계획 (DOE)', 'desc': '공정 조건 최적화 설계', 'logic': '최소 실험으로 최적의 품질 조건을 도출합니다.', 'link': '/doe', 'icon': '🧪'}
}

# --- 계층 구조 및 질문 정의 ---
TREE_HIERARCHY = {
    'start': ['comp', 'quality', 'rel'],
    'comp': ['1-sample', '2-sample', 'multi-sample'],
    'quality': ['capability-flow', 'msa-flow', 'outlier-test'],
    'rel': ['reg-flow', 'doe-flow'],
    '1-sample': ['1-sample-t', '1-sample-equiv', 'outlier-test'],
    '2-sample': ['2-sample-t', 'equivalence', 'paired-t'],
    'multi-sample': ['anova'],
    'capability-flow': ['capability'],
    'msa-flow': ['msa'],
    'reg-flow': ['regression'],
    'doe-flow': ['doe']
}

QUESTIONS = {
    'start': "분석의 목적이 무엇인가요?",
    'comp': "비교하고자 하는 집단의 개수가 몇 개인가요?",
    'quality': "품질 관리의 구체적인 목적이 무엇인가요?",
    'rel': "인과 관계 분석 또는 실험 설계 중 선택하세요.",
    '1-sample': "단일 집단에 대해 어떤 분석을 수행할까요?",
    '2-sample': "두 집단 데이터의 성격이 어떠한가요?",
    'multi-sample': "다수 집단 간의 평균 차이를 분석합니다.",
    'capability-flow': "공정의 품질 능력을 지수로 평가합니다.",
    'msa-flow': "측정 시스템의 신뢰도를 검증합니다.",
    'reg-flow': "변수 간의 관계를 수식화하여 분석합니다.",
    'doe-flow': "최적 조건을 찾기 위한 실험을 설계합니다."
}

FRIENDLY_NAMES = {
    'start': '분석 시작', 'comp': '차이/검정', 'quality': '공정/품질', 'rel': '관계/예측',
    '1-sample': '1개 집단', '2-sample': '2개 집단', 'multi-sample': '3개 이상',
    'capability-flow': '공정 능력', 'msa-flow': '측정 신뢰도', 'reg-flow': '상관/회귀', 'doe-flow': '실험 설계'
}

# --- CSS 및 다크 테마 설정 ---
THEME = {
    'bg': '#0f172a',
    'card_bg': '#1e293b',
    'card_border': '#334155',
    'active_border': '#3b82f6',
    'text_primary': '#f8fafc',
    'text_secondary': '#94a3b8',
    'accent': '#3b82f6',
    'line': '#334155',
    'active_line': '#3b82f6'
}

# --- 레이아웃 ---
layout = html.Div([
    dcc.Store(id='selector-state', data={'path': ['start'], 'mode': 'guide'}),
    
    # 1. 상단 트리 영역 (NotebookLM 수평형 + Plotly Arrows)
    html.Div([
        html.H1("Statistics Decision Tree", style={'textAlign': 'center', 'fontWeight': '900', 'marginBottom': '40px', 'color': THEME['text_primary']}),
        
        html.Div([
            # Plotly Graph 레이어 (화살표 라인용)
            dcc.Graph(id='tree-graph-layer', 
                     config={'displayModeBar': False},
                     style={'position': 'absolute', 'top': 0, 'left': 0, 'width': '100%', 'height': '100%', 'zIndex': 1}),
            # 노드 레이어
            html.Div(id='horizontal-tree-nodes', style={'position': 'relative', 'zIndex': 2, 'padding': '20px', 'height': '100%'})
        ], id='tree-viewport', style={'position': 'relative', 'width': '100%', 'height': '500px', 'overflowX': 'auto', 'backgroundColor': THEME['bg'], 'borderRadius': '20px', 'boxShadow': 'inset 0 0 20px rgba(0,0,0,0.5)'})
    ], style={'padding': '40px'}),

    # 2. 하단 듀얼 패널 (컨트롤 패널)
    dbc.Container([
        dbc.Card([
            dbc.Tabs([
                dbc.Tab(label="📋 선택형 가이드", tab_id="guide-tab", label_style={'color': THEME['text_secondary']}, active_label_style={'color': THEME['accent'], 'fontWeight': 'bold'}),
                dbc.Tab(label="🤖 AI 도구 추천 엔진", tab_id="ai-tab", label_style={'color': THEME['text_secondary']}, active_label_style={'color': THEME['accent'], 'fontWeight': 'bold'}),
            ], id="panel-mode-tabs", active_tab="guide-tab", style={'borderBottom': f"1px solid {THEME['card_border']}"}),
            
            dbc.CardBody(id='panel-content', style={'minHeight': '300px', 'padding': '40px'})
        ], style={'backgroundColor': THEME['card_bg'], 'border': f"1px solid {THEME['card_border']}", 'borderRadius': '20px', 'boxShadow': '0 20px 40px rgba(0,0,0,0.3)'})
    ], fluid=True, style={'maxWidth': '1200px', 'marginBottom': '100px'}),

    # 승인 모달
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("도구 추천 승인"), style={'backgroundColor': THEME['card_bg'], 'border': 'none', 'color': '#fff'}),
        dbc.ModalBody(id='approval-content', style={'backgroundColor': THEME['bg'], 'color': '#fff'}),
        dbc.ModalFooter([
            dbc.Button("취소", id="close-modal", color="secondary", outline=True),
            dbc.Button("분석 시작 (승인)", id="approve-btn", color="primary")
        ], style={'backgroundColor': THEME['card_bg'], 'border': 'none'})
    ], id="approval-modal", size="lg", centered=True, is_open=False),

    dcc.Location(id='url-redirect', refresh=True)
], style={'backgroundColor': THEME['bg'], 'minHeight': '100vh', 'fontFamily': '"Inter", sans-serif'})

# --- 트리 및 Plotly 라인 생성 함수 ---
def build_tree_layout(path):
    columns_html = []
    
    col_width = 300
    node_height = 80
    center_y = 250
    
    # Plotly Figure 초기화
    fig = go.Figure()
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 1500]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 500]),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        hovermode=False
    )
    
    current_nodes = ['start']
    depth = 0
    
    while current_nodes:
        next_nodes = []
        total_nodes = len(current_nodes)
        
        for idx, node_id in enumerate(current_nodes):
            is_active = node_id in path
            is_leaf = node_id in TOOLS
            
            # 노드 좌표
            node_x = depth * col_width + 110
            node_y = center_y + (idx - (total_nodes-1)/2) * node_height
            
            # 카드 렌더링
            label = TOOLS[node_id]['name'] if is_leaf else FRIENDLY_NAMES.get(node_id, node_id)
            card_style = {
                'position': 'absolute', 'left': f'{depth * col_width}px', 'top': f'{500 - node_y - 25}px', # Plotly Y축은 아래가 0
                'width': '220px', 'padding': '12px 18px', 'borderRadius': '12px',
                'backgroundColor': THEME['card_bg'] if not is_active else '#1e3a8a',
                'border': f"2px solid {(THEME['active_border'] if is_active else THEME['card_border'])}",
                'color': THEME['text_primary'] if is_active else THEME['text_secondary'],
                'cursor': 'pointer', 'transition': 'all 0.3s ease', 'zIndex': 3,
                'display': 'flex', 'alignItems': 'center', 'justifyContent': 'space-between',
                'boxShadow': '0 4px 6px rgba(0,0,0,0.2)' if not is_active else f"0 0 15px {THEME['accent']}44"
            }
            
            columns_html.append(html.Div([
                html.Span(label, style={'fontWeight': '600' if is_active else '400'}),
                html.Span(">" if not is_leaf else "✓", style={'opacity': 0.5})
            ], id={'type': 'tree-node', 'index': node_id}, style=card_style))
            
            # 연결선 그리기
            if depth > 0:
                parent_id = find_grandparent(node_id)
                if parent_id in path:
                    # 부모 좌표 계산 (단순화를 위해 이전 컬럼의 활성 노드 위치 재계산)
                    prev_col_nodes = get_nodes_at_depth(depth - 1, path)
                    p_idx = prev_col_nodes.index(parent_id)
                    p_y = center_y + (p_idx - (len(prev_col_nodes)-1)/2) * node_height
                    p_x = (depth-1) * col_width + 220
                    
                    # 베지에 곡선 좌표 생성 (S-curve)
                    fig.add_trace(go.Scatter(
                        x=[p_x, p_x + 40, (p_x + depth * col_width)/2, depth * col_width - 40, depth * col_width],
                        y=[p_y, p_y, (p_y + node_y)/2, node_y, node_y],
                        mode='lines',
                        line=dict(color=THEME['active_line'] if is_active else THEME['line'], 
                                 width=2 if is_active else 1,
                                 shape='spline'),
                        showlegend=False
                    ))

            if is_active and node_id in TREE_HIERARCHY:
                next_nodes.extend(TREE_HIERARCHY[node_id])
        
        current_nodes = next_nodes
        depth += 1
        if depth > 5: break

    return columns_html, fig

def find_grandparent(node):
    for p, children in TREE_HIERARCHY.items():
        if node in children: return p
    return None

def get_nodes_at_depth(depth, path):
    if depth == 0: return ['start']
    nodes = ['start']
    for _ in range(depth):
        next_nodes = []
        for n in nodes:
            if n in path and n in TREE_HIERARCHY:
                next_nodes.extend(TREE_HIERARCHY[n])
        nodes = next_nodes
    return nodes

# --- 콜백: 트리 및 패널 렌더링 ---
@callback(
    [Output('horizontal-tree-nodes', 'children'),
     Output('tree-graph-layer', 'figure'),
     Output('panel-content', 'children')],
    [Input('selector-state', 'data'),
     Input('panel-mode-tabs', 'active_tab')]
)
def update_ui(state, active_tab):
    path = state['path']
    current_node = path[-1]
    
    # 1. 트리 렌더링
    nodes, fig = build_tree_layout(path)
    
    # 2. 하단 패널 렌더링
    if active_tab == 'ai-tab':
        panel = html.Div([
            html.H3("🤖 AI 통계 전문가 가이드", style={'color': THEME['accent'], 'marginBottom': '20px'}),
            html.P("분석 목표나 데이터의 특성을 자유롭게 입력해 주세요. 최적의 통계 도구를 추천해 드립니다.", style={'color': THEME['text_secondary']}),
            dbc.InputGroup([
                dbc.Input(id="ai-input", placeholder="예: '공정 변경 전후의 평균 차이를 증명하고 싶어요'", style={'backgroundColor': THEME['bg'], 'color': '#fff', 'border': f"1px solid {THEME['card_border']}"}),
                dbc.Button("추천받기", id="ai-submit", color="primary")
            ], size="lg"),
            html.Div(id='ai-response-area', style={'marginTop': '30px'})
        ])
    else:
        # 가이드 모드
        if current_node in TOOLS:
            panel = html.Div([
                html.H2("✅ 분석 도구가 선택되었습니다", style={'color': '#10b981'}),
                html.P(f"추천된 도구: {TOOLS[current_node]['name']}", style={'fontSize': '1.2rem'}),
                dbc.Button("다시 시작하기", id={'type': 'tree-node', 'index': 'start'}, color="secondary", outline=True, style={'marginTop': '20px'})
            ], style={'textAlign': 'center'})
        else:
            question = QUESTIONS.get(current_node, "다음 단계를 선택하세요.")
            options = TREE_HIERARCHY.get(current_node, [])
            panel = html.Div([
                html.H3(f"❓ {question}", style={'color': THEME['text_primary'], 'marginBottom': '30px'}),
                html.Div([
                    dbc.Button(
                        FRIENDLY_NAMES.get(opt, TOOLS.get(opt, {}).get('name', opt)),
                        id={'type': 'tree-node', 'index': opt},
                        color="primary", outline=True, size="lg",
                        style={'borderRadius': '10px', 'minWidth': '200px'}
                    ) for opt in options
                ], style={'display': 'flex', 'gap': '20px', 'flexWrap': 'wrap'})
            ])
            
    return nodes, fig, panel

# --- 콜백: 노드 클릭 및 모달 제어 ---
@callback(
    [Output('selector-state', 'data', allow_duplicate=True),
     Output('approval-modal', 'is_open'),
     Output('approval-content', 'children')],
    [Input({'type': 'tree-node', 'index': ALL}, 'n_clicks'),
     Input('close-modal', 'n_clicks')],
    [State('selector-state', 'data')],
    prevent_initial_call=True
)
def handle_interaction(n_clicks, close_clicks, state):
    ctx = dash.callback_context
    if not ctx.triggered: return dash.no_update
    
    trigger_id = ctx.triggered[0]['prop_id']
    if 'close-modal' in trigger_id: return dash.no_update, False, dash.no_update

    # 노드 클릭
    btn_id = json.loads(trigger_id.split('.')[0])
    node_id = btn_id['index']
    
    if node_id == 'start':
        state['path'] = ['start']
        if 'selected' in state: del state['selected']
        return state, False, None

    # 경로 재구성
    new_path = [node_id]
    temp = node_id
    while True:
        parent = find_grandparent(temp)
        if not parent: break
        new_path.insert(0, parent)
        temp = parent
    
    # 최종 도구 도달 시
    if node_id in TOOLS:
        tool = TOOLS[node_id]
        content = html.Div([
            html.Div([
                html.Span(tool['icon'], style={'fontSize': '4rem', 'marginRight': '20px'}),
                html.Div([
                    html.H2(tool['name'], style={'color': THEME['accent']}),
                    html.P(tool['desc'], style={'color': THEME['text_secondary']}),
                ])
            ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '30px'}),
            dbc.Alert([
                html.H5("💡 통계적 추천 근거"),
                html.P(tool['logic'])
            ], color="info", style={'backgroundColor': '#1e3a8a', 'border': 'none', 'color': '#fff'}),
            html.Small("분석을 시작하시겠습니까? 승인 시 관련 분석 페이지로 이동합니다.")
        ])
        state['path'] = new_path
        state['selected'] = node_id
        return state, True, content

    state['path'] = new_path
    if 'selected' in state: del state['selected']
    return state, False, None

# --- AI 엔진 (이전 로직 재활용) ---
@callback(
    [Output('ai-response-area', 'children'),
     Output('approval-modal', 'is_open', allow_duplicate=True),
     Output('approval-content', 'children', allow_duplicate=True),
     Output('selector-state', 'data', allow_duplicate=True)],
    Input('ai-submit', 'n_clicks'),
    State('ai-input', 'value'),
    State('selector-state', 'data'),
    prevent_initial_call=True
)
def handle_ai(n_clicks, text, state):
    if not text: return dash.no_update
    text_lower = text.lower()
    match = None
    if any(k in text_lower for k in ['비교', '차이', '평균', '검정']):
        if any(k in text_lower for k in ['전후', '대응', '웨이퍼']): match = 'paired-t'
        elif any(k in text_lower for k in ['동등', '오차범위']):
            if any(k in text_lower for k in ['1개', '단일', '1표본']): match = '1-sample-equiv'
            else: match = 'equivalence'
        elif any(k in text_lower for k in ['3개', '세 개', '여러']): match = 'anova'
        elif any(k in text_lower for k in ['1개', '목표값']): match = '1-sample-t'
        else: match = '2-sample-t'
    elif any(k in text_lower for k in ['능력', 'cpk', 'ppk', '규격']): match = 'capability'
    elif any(k in text_lower for k in ['측정', '계측기', 'msa', 'gage', 'r&r']): match = 'msa'
    elif any(k in text_lower for k in ['상관', '회귀', '예측', '영향']): match = 'regression'
    elif any(k in text_lower for k in ['실험', '최적', 'doe']): match = 'doe'
    elif any(k in text_lower for k in ['이상치', '특이치', 'outlier', '튀는']): match = 'outlier-test'
    
    if match:
        tool = TOOLS[match]
        # 승인 모달용 내용 생성 (handle_interaction과 유사)
        content = html.Div([
            html.H4("🤖 AI 추천 결과", style={'color': THEME['accent'], 'marginBottom': '20px'}),
            html.Div([
                html.Span(tool['icon'], style={'fontSize': '4rem', 'marginRight': '20px'}),
                html.Div([
                    html.H2(tool['name'], style={'color': THEME['accent']}),
                    html.P(tool['desc']),
                ])
            ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '30px'}),
            dbc.Alert([html.H5("💡 추천 근거"), html.P(tool['logic'])], color="info"),
            html.Small("승인 시 해당 페이지로 이동합니다.")
        ])
        # AI 추천 시 트리 경로도 업데이트
        new_path = [match]
        temp = match
        while True:
            p = find_grandparent(temp)
            if not p: break
            new_path.insert(0, p); temp = p
            
        state['path'] = new_path
        state['selected'] = match
        return None, True, content, state
    return dbc.Alert("분석 의도를 파악하기 어렵습니다. 조금 더 구체적으로 설명해 주세요.", color="warning"), False, None, state

@callback(
    Output('url-redirect', 'href'),
    Input('approve-btn', 'n_clicks'),
    State('selector-state', 'data'),
    prevent_initial_call=True
)
def redirect_to_tool(n_clicks, state):
    if n_clicks and 'selected' in state:
        return TOOLS[state['selected']]['link']
    return dash.no_update
