# pip pip install dash-extensions

import os
import sys
import re
import base64
import io
import zipfile
import shutil
import uuid
from datetime import datetime

import dash
from dash_extensions.enrich import dcc, html, Input, Output, State, register_page, callback_context, no_update

register_page(__name__)

# --- 기존 로직 (main.py에서 가져옴) ---
def process_file_recursively(filepath, output_file, processed_files, base_dir):
    """
    파일을 재귀적으로 읽고 'INSERT' 지시어를 처리하여 output_file에 씁니다.
    웹 앱 환경에 맞게 base_dir 인자를 추가했습니다.
    """
    abs_filepath = os.path.abspath(filepath)

    if abs_filepath in processed_files:
        print(f"경고: 순환 참조가 감지되어 '{filepath}' 파일을 다시 삽입하지 않습니다.", file=sys.stderr)
        return

    if not os.path.exists(abs_filepath):
        print(f"오류: '{filepath}' 파일을 찾을 수 없습니다.", file=sys.stderr)
        return

    print(f"처리 중: '{filepath}'")
    processed_files.add(abs_filepath)

    current_dir = os.path.dirname(abs_filepath)

    try:
        with open(abs_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                stripped_line = line.strip()
                comment_pos = stripped_line.find(';')
                effective_line = stripped_line.split(';', 1)[0].strip() if comment_pos != -1 else stripped_line

                match = re.match(r'^INSERT\s+([^\s]+)', effective_line)
                if match:
                    filename_to_insert = match.group(1).strip()
                    if '.' not in filename_to_insert:
                        filename_to_insert += '.asc'
                    
                    path_to_insert = os.path.join(base_dir, filename_to_insert)
                    
                    process_file_recursively(path_to_insert, output_file, processed_files, base_dir)
                    output_file.write('\n')
                else:
                    output_file.write(line)
    except Exception as e:
        print(f"오류: '{filepath}' 파일을 읽는 중 예외가 발생했습니다: {e}", file=sys.stderr)

    processed_files.remove(abs_filepath)

def merge_files(main_input_file, output_filename, base_dir):
    """
    메인 함수: 파일 병합 프로세스를 시작합니다.
    """
    processed_files = set()
    try:
        with open(output_filename, 'w', encoding='utf-8') as output_f:
            print(f"병합 시작: '{main_input_file}' -> '{output_filename}'")
            process_file_recursively(main_input_file, output_f, processed_files, base_dir)
        print("병합 완료!")
        return True
    except Exception as e:
        print(f"오류: 최종 파일을 쓰는 중 예외가 발생했습니다: {e}", file=sys.stderr)
        return False

def process_jni_remap(file_content):
    """
    JNI9~15를 JNI1~8로, IDX9~15를 IDX1~8로 재매핑하는 함수.
    """
    lines = file_content.splitlines()
    
    # 1. 파일 전체를 스캔하여 사용 중인 JNI와 재매핑이 필요한 JNI를 찾습니다.
    used_jni_low = set()
    to_remap_jni_high = set()
    
    for line in lines:
        effective_line = line.split(';', 1)[0]
        # JNI 토큰 찾기
        jni_tokens = re.findall(r'\bJNI(\d+)\b', effective_line)
        for num_str in jni_tokens:
            num = int(num_str)
            if 1 <= num <= 8:
                used_jni_low.add(f"JNI{num}")
            elif 9 <= num <= 15:
                to_remap_jni_high.add(f"JNI{num}")

    # 2. 재매핑 규칙을 생성합니다.
    available_jni_low = {f"JNI{i}" for i in range(1, 9)}
    available_slots = sorted(list(available_jni_low - used_jni_low), key=lambda s: int(s[3:]))
    
    to_remap_sorted = sorted(list(to_remap_jni_high), key=lambda s: int(s[3:]))

    if len(available_slots) < len(to_remap_sorted):
        raise ValueError(f"JNI 재매핑 실패: JNI1-8에 할당 가능한 공간({len(available_slots)}개)이 부족합니다. (필요: {len(to_remap_sorted)}개)")

    # 예제와 같이 높은 번호부터 매핑합니다.
    jni_map = {old: new for old, new in zip(reversed(to_remap_sorted), reversed(available_slots))}
    
    # IDX 맵 생성 (JNI 맵 기반)
    idx_map = {f"IDX{k[3:]}": f"IDX{v[3:]}" for k, v in jni_map.items()}
    
    # 3. 새로운 파일 내용을 생성합니다.
    new_lines = []
    for line in lines:
        # 주석과 코드 부분을 분리
        parts = line.split(';', 1)
        effective_line = parts[0]
        comment = f";{parts[1]}" if len(parts) > 1 else ""

        # JNI와 IDX를 동시에 치환하기 위해 정규식 사용
        # 단어 경계(\b)를 사용하여 JNI1이 JNI10의 일부로 인식되는 것을 방지
        def replace_token(match):
            token = match.group(0)
            if token in jni_map:
                return jni_map[token]
            if token in idx_map:
                return idx_map[token]
            return token

        modified_line = re.sub(r'\b(JNI\d+|IDX\d+)\b', replace_token, effective_line)
        new_lines.append(modified_line + comment)
        
    return "\n".join(new_lines)

# --- Dash 애플리케이션 설정 ---

#app = dash.Dash(__name__)
#server = app.server

layout = html.Div(
    style={'fontFamily': 'Arial, sans-serif', 'maxWidth': '1000px', 'margin': 'auto', 'padding': '20px'},
    id='app-container', children=[
    # 이 페이지에만 특정 CSS 파일을 적용하기 위해 html.Link를 추가합니다.
    # CSS 파일은 'assets' 폴더 내에 있어야 합니다.
    html.Link(
        rel='stylesheet',
        href='/assets/style.css'
    ),

    html.Div(className='header', children=[
        html.H1("ATL Pattern Converter", style={'textAlign': 'center', 'color': '#333'}),
        html.P("파일 병합을 위해 ZIP 파일을 업로드하고 Pattern 이름(ex: pattern.asc)을 입력하세요."),
    ]),

    html.Div(className='controls-container', children=[
        dcc.Upload(
            id='upload-zip',
            children=html.Div([
                'ZIP 파일을 드래그 앤 드롭하거나 ',
                html.A('파일을 선택하세요')
            ]),
            multiple=False
        ),
        html.Div(id='upload-status'),
        
        dcc.Input(
            id='main-file-name',
            type='text',
            placeholder='변경할 Pattern 파일 이름 (예: pattern.asc)',
        ),
        
        html.Div([
            html.Label("JNI,IDX 재매핑(JNI9-15 -> JNI1-8):", style={'marginRight': '10px', 'fontWeight': 'bold'}),
            dcc.RadioItems(
                id='remap-jni-radio',
                options=[{'label': '실행', 'value': 'yes'}, {'label': '실행 안함', 'value': 'no'}],
                value='no',
                inline=True
            )
        ], style={'marginTop': '15px', 'marginBottom': '15px'}),

        html.Button('병합 실행', id='submit-button', n_clicks=0),
    ]),
    dcc.Loading(
        id="loading-spinner",
        children=html.Div(id='output-status', className='status-container'),
        type="circle",
    ),
    dcc.Download(id="download-result"),
])

@dash.callback(
    Output('upload-status', 'children'),
    Input('upload-zip', 'filename')
)
def update_upload_status(filename):
    if filename:
        return html.Div(f"'{filename}' 파일이 로드되었습니다.", className='upload-success')
    return ""

@dash.callback(
    Output("download-result", "data"),
    Output("output-status", "children"),
    Input("submit-button", "n_clicks"),
    [
        State('upload-zip', 'contents'),
        State('upload-zip', 'filename'),
        State('main-file-name', 'value'),
        State('remap-jni-radio', 'value')
    ],
    prevent_initial_call=True,
)
def run_merge_process(n_clicks, content, zip_filename, main_filename, remap_jni_choice):
    if not content or not main_filename:
        return no_update, html.Div("ZIP 파일과 Pattern 파일 이름을 모두 입력하세요.", className='status-error')

    # 임시 작업 디렉토리 생성
    session_id = str(uuid.uuid4())
    work_dir = os.path.join(os.getcwd(), "temp", session_id)
    os.makedirs(work_dir, exist_ok=True)

    try:
        # ZIP 파일 처리
        content_type, content_string = content.split(',')
        decoded = base64.b64decode(content_string)
        
        with zipfile.ZipFile(io.BytesIO(decoded), 'r') as zip_ref:
            zip_ref.extractall(work_dir)

        # ZIP 파일 내에 최상위 폴더가 있는 경우 처리
        extracted_items = os.listdir(work_dir)
        if len(extracted_items) == 1 and os.path.isdir(os.path.join(work_dir, extracted_items[0])):
            # 압축 해제된 내용이 단일 폴더 안에 있을 경우, 내용물을 work_dir의 루트로 이동
            nested_dir = os.path.join(work_dir, extracted_items[0])
            for item in os.listdir(nested_dir):
                shutil.move(os.path.join(nested_dir, item), work_dir)
            os.rmdir(nested_dir)

        # 메인 파일이 ZIP 파일의 루트에 있는지 대소문자를 구분하여 확인
        file_list_in_zip = os.listdir(work_dir)
        if main_filename not in file_list_in_zip:
            # ZIP 파일 내의 파일 목록을 보여주어 디버깅에 도움을 줌
            file_list_for_error = [f for f in file_list_in_zip if os.path.isfile(os.path.join(work_dir, f))]
            raise ValueError(f"오류: ZIP 파일 내에서 '{main_filename}'을(를) 찾을 수 없습니다. (대소문자 구분) 사용 가능한 파일: {', '.join(file_list_for_error)}")

        # 병합할 파일 경로 설정
        main_file_path = os.path.join(work_dir, main_filename)

        # 출력 파일 경로 설정
        # rsplit을 사용하여 오른쪽부터 '.'을 기준으로 1번만 분리합니다.
        parts = main_filename.rsplit('.', 1)
        output_filename = parts[0] + '_new' + '.' + parts[1]
        output_path = os.path.join(work_dir, output_filename)

        # 파일 병합 실행
        success = merge_files(main_file_path, output_path, work_dir)

        if success and os.path.exists(output_path):
            # JNI 재매핑 기능 실행 여부 확인
            if remap_jni_choice == 'yes':
                with open(output_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                
                modified_content = process_jni_remap(file_content)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(modified_content)

            status_message = html.Div([
                html.P(f"병합 성공! '{output_filename}' 파일을 다운로드합니다."),
                html.P(f"(업로드된 파일: {zip_filename}, 메인 파일: {main_filename})")
            ], className='status-success')
            return dcc.send_file(output_path, filename=output_filename), status_message
        else:
            raise RuntimeError("파일 병합 중 오류가 발생했습니다. 서버 로그를 확인하세요.")

    except Exception as e:
        return no_update, html.Div(f"처리 중 예외가 발생했습니다: {e}", className='status-error')
    finally:
        # 임시 디렉토리 정리 (선택적: 디버깅을 위해 남겨둘 수 있음)
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)