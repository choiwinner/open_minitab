import base64
import time
import io
import cv2
import numpy as np
import dash
from dash import dcc, html, Input, Output, State, register_page, callback_context, no_update, ALL
from PIL import Image
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

import torch
import gc

register_page(__name__)

# 1. Real-ESRGAN 모델 설정
# 전역 변수로 모델 캐싱
global_upsampler = None
global_device_type = None

def get_upsampler(device_type):
    global global_upsampler, global_device_type

    # 설정 변경 시 모델 재생성
    if global_upsampler is None or global_device_type != device_type:
        # 메모리 정리
        if global_upsampler is not None:
            del global_upsampler
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        model_path = 'RealESRGAN_x4plus.pth'
        
        device = 'cuda' if device_type == 'gpu' else 'cpu'
        half = True if device_type == 'gpu' else False

        global_upsampler = RealESRGANer(
            scale=4,
            model_path=model_path,
            model=model,
            tile=400,
            tile_pad=10,
            pre_pad=0,
            half=half,
            device=device
        )
        global_device_type = device_type
    
    return global_upsampler

# 2. Dash 앱 초기화

layout = html.Div(
    style={'fontFamily': 'Arial, sans-serif', 'maxWidth': '1000px', 'margin': 'auto', 'padding': '20px'},
    children=[
        html.H1(
            "Super Resolution (Image Upscaling)",
            style={'textAlign': 'center', 'color': '#333'}
        ),

        html.P(
            "이미지를 업로드하면 AI 모델(Real-ESRGAN)을 사용하여 해상도를 4배 향상시킵니다.",
            style={'textAlign': 'center', 'color': '#555'}
        ),

        # 이미지 업로드 섹션
        dcc.Upload(
            id='upload-image',
            children=html.Div([
                '이미지를 드래그하거나 ',
                html.A('클릭해서 선택하세요')
            ]),
            style={
                'width': '100%', 'height': '100px', 'lineHeight': '100px',
                'borderWidth': '1px', 'borderStyle': 'dashed',
                'borderRadius': '5px', 'textAlign': 'center', 'marginBottom': '20px',
                'borderColor': '#ccc'
            },
            multiple=False
        ),

        # 장치 선택 (CPU/GPU)
        html.Div([
            html.Label("처리 장치 (Device):", style={'fontWeight': 'bold', 'marginRight': '10px'}),
            dcc.RadioItems(
                id='device-selector',
                options=[
                    {'label': 'CPU', 'value': 'cpu'},
                    {'label': 'GPU (CUDA)', 'value': 'gpu'}
                ],
                value='gpu' if torch.cuda.is_available() else 'cpu',
                inline=True
            )
        ], style={'marginBottom': '15px', 'textAlign': 'center'}),

        # 변환 실행 버튼
        html.Button(
            '변환 실행 (Upscale)',
            id='run-upscale-button',
            n_clicks=0,
            style={
                'width': '100%', 'padding': '10px', 'fontSize': '18px',
                'fontWeight': 'bold', 'backgroundColor': '#007BFF',
                'color': 'white', 'border': 'none', 'borderRadius': '5px',
                'marginBottom': '20px', 'cursor': 'pointer'
            }
        ),
        
        # 소요 시간 표시
        html.Div(id='processing-time-display', style={'textAlign': 'center', 'marginBottom': '10px', 'fontWeight': 'bold', 'color': '#333'}),

        # 진행 상태 및 결과 출력
        html.Div([
            html.Div([
                html.H4("Original Image", style={'textAlign': 'center', 'marginBottom': '10px', 'borderBottom': '1px solid #ddd', 'paddingBottom': '5px'}),
                html.Div(id='original-image-display', style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center', 'minHeight': '200px'})
            ], style={'flex': '1', 'marginRight': '10px'}),

            html.Div([
                html.H4("Upscaled Image", style={'textAlign': 'center', 'marginBottom': '10px', 'borderBottom': '1px solid #ddd', 'paddingBottom': '5px'}),
                dcc.Loading(
                    id="loading-upscale",
                    type="circle",
                    children=[html.Div(id='upscaled-image-display', style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center', 'minHeight': '200px'})]
                )
            ], style={'flex': '1', 'marginLeft': '10px'}),
        ], style={'display': 'flex', 'marginBottom': '15px'}),
    ]
)

# 3. 이미지 처리 헬퍼 함수
def process_image(contents, device_type):
    # Base64 데이터를 numpy 배열(OpenCV 포맷)로 변환
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    image = Image.open(io.BytesIO(decoded)).convert('RGB')
    img_array = np.array(image)
    img_cv2 = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    upsampler = get_upsampler(device_type)

    # Real-ESRGAN 업스케일링 실행
    output, _ = upsampler.enhance(img_cv2, outscale=4)
    
    # 결과를 다시 Base64로 인코딩
    _, buffer = cv2.imencode('.png', output)
    encoded_image = base64.b64encode(buffer).decode('utf-8')

    # 결과 반환 전 메모리 정리
    gc.collect()
    torch.cuda.empty_cache()

    return f"data:image/png;base64,{encoded_image}"

# 4. 콜백 함수
@dash.callback(
    [Output('original-image-display', 'children'),
     Output('upscaled-image-display', 'children'),
     Output('processing-time-display', 'children')],
    [Input('upload-image', 'contents'),
     Input('run-upscale-button', 'n_clicks'),
     Input('device-selector', 'value')],
    State('upload-image', 'filename') 
)
def update_output(contents, n_clicks, device_mode, filename):
    ctx = callback_context
    if not ctx.triggered:
        return no_update, no_update, no_update

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if contents is None:
        return None, None, ""

    # 원본 이미지 표시
    original_img_element = html.Img(src=contents, style={'width': '100%'})

    if trigger_id == 'upload-image':
        return original_img_element, None, ""

    elif trigger_id == 'run-upscale-button':
        # GPU 선택 시 가용성 체크
        if device_mode == 'gpu' and not torch.cuda.is_available():
            return original_img_element, html.Div("오류: GPU(CUDA)를 사용할 수 없습니다. CPU를 선택해주세요.", style={'color': 'red', 'fontWeight': 'bold'}), ""

        # 업스케일링 처리
        try:
            start_time = time.time()
            upscaled_src = process_image(contents, device_mode)
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            upscaled_img_element = html.Img(src=upscaled_src, style={'width': '100%'})
            return original_img_element, upscaled_img_element, f"소요 시간: {elapsed_time:.2f}초"
        except Exception as e:
            return original_img_element, html.Div(f"Error: {str(e)}"), ""

    return no_update, no_update, no_update